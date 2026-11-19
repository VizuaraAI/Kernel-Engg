"""
FlashAttention-3 Forward Pass in CuTe DSL Python (Hopper SM90).

Implements the core FlashAttention-3 techniques from:
  "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"
  (Shah et al., 2024)

Key features implemented:
  1. Producer-Consumer Warp Specialization:
     - Producer warpgroup (128 threads, 1 active TMA issuer) loads Q, K, V tiles
       from HBM to SMEM via Hopper TMA into multi-stage circular buffers managed
       by `cutlass.pipeline.PipelineTmaAsync`.
     - Consumer warpgroup (128 threads) executes asynchronous WGMMA tensor core
       operations and online softmax, with dynamic register reallocation via
       `setmaxnreg` (`warpgroup_reg_dealloc` / `warpgroup_reg_alloc`).
  2. 2-Stage Intra-Warpgroup GEMM-Softmax Pipelining (Algorithm 2):
     - Overlaps the second WGMMA (P_cur @ V_{j-1}) with the online softmax
       computation on the next score tile (S_next = Q @ K_j^T) using
       `wgmma.wait_group(1)` / `wgmma.wait_group(0)`.
  3. Supports FP16 / BF16, head dimensions 64 / 128 / 256, and optional causal masking.

Note: Requires an NVIDIA Hopper (SM90) GPU and `nvidia-cutlass-dsl >= 4.8.1`.
"""

import math
import time
from typing import Type

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
import torch.nn.functional as F
from cutlass.cute.nvgpu.warpgroup import (
    MmaF16BF16Op,
    OperandMajorMode,
    OperandSource,
)
from cutlass.cute.runtime import from_dlpack

LOG2_E = math.log2(math.e)
NUM_PRODUCER_THREADS = 128
NUM_CONSUMER_THREADS = 128
TOTAL_THREADS = NUM_PRODUCER_THREADS + NUM_CONSUMER_THREADS


@cute.jit
def _warp_reduce_max_4(val: cute.Float32) -> cute.Float32:
    """Butterfly max-reduction across the 4 threads sharing a WGMMA row."""
    val = cute.arch.fmax(val, cute.arch.shuffle_sync_bfly(val, offset=2))
    val = cute.arch.fmax(val, cute.arch.shuffle_sync_bfly(val, offset=1))
    return val


@cute.jit
def _warp_reduce_sum_4(val: cute.Float32) -> cute.Float32:
    """Butterfly sum-reduction across the 4 threads sharing a WGMMA row."""
    val = val + cute.arch.shuffle_sync_bfly(val, offset=2)
    val = val + cute.arch.shuffle_sync_bfly(val, offset=1)
    return val


class FlashAttention3Forward:
    """CuTe DSL Hopper (SM90) FlashAttention-3 forward pass kernel."""

    def __init__(
            self,
            dtype: Type[cutlass.Numeric],
            head_dim: int,
            tile_m: int = 64,
            tile_n: int = 64,
            num_stages: int = 2,
            is_causal: bool = False,
    ):
        self.dtype = dtype
        self.head_dim = head_dim
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_stages = num_stages
        self.is_causal = is_causal

    def _make_smem_layouts(self):
        # GEMM0 tiler: (M, N, K) = (tile_m, tile_n, head_dim)
        mma_tiler_qk = (self.tile_m, self.tile_n, self.head_dim)
        # GEMM1 tiler: (M, N, K) = (tile_m, head_dim, tile_n)
        mma_tiler_pv = (self.tile_m, self.head_dim, self.tile_n)

        # Q: (tile_m, head_dim), K-major (ROW_MAJOR in (M, K))
        sQ_layout = sm90_utils.make_smem_layout_a(
            utils.LayoutEnum.ROW_MAJOR,
            mma_tiler_qk,
            self.dtype,
            1,
        )
        # K: (tile_n, head_dim), K-major (ROW_MAJOR in (N, K))
        sK_layout = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.ROW_MAJOR,
            mma_tiler_qk,
            self.dtype,
            self.num_stages,
        )
        # V: (head_dim, tile_n) view of (tile_n, head_dim) in GMEM -> MN-major
        # (COL_MAJOR in (N_pv, K_pv) = (head_dim, tile_n))
        sV_layout = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.COL_MAJOR,
            mma_tiler_pv,
            self.dtype,
            self.num_stages,
        )
        return sQ_layout, sK_layout, sV_layout

    def _make_tiled_mmas(self):
        # GEMM0: S = Q_i @ K_j^T  (SS-WGMMA: both A and B from SMEM)
        mma_op_qk = MmaF16BF16Op(
            ab_dtype=self.dtype,
            acc_dtype=cutlass.Float32,
            instruction_shape=(64, self.tile_n, 16),
            a_src=OperandSource.SMEM,
            a_major_mode=OperandMajorMode.K,
            b_major_mode=OperandMajorMode.K,
        )
        tiled_mma_qk = sm90_utils.make_trivial_tiled_mma(mma_op_qk)

        # GEMM1: O_i += P_tilde @ V_j  (RS-WGMMA: A=P_tilde from RMEM, B=V_j from SMEM)
        mma_op_pv = MmaF16BF16Op(
            ab_dtype=self.dtype,
            acc_dtype=cutlass.Float32,
            instruction_shape=(64, self.head_dim, 16),
            a_src=OperandSource.RMEM,
            a_major_mode=OperandMajorMode.K,
            b_major_mode=OperandMajorMode.MN,
        )
        tiled_mma_pv = sm90_utils.make_trivial_tiled_mma(mma_op_pv)
        return tiled_mma_qk, tiled_mma_pv

    @cute.jit
    def __call__(
            self,
            mQ: cute.Tensor,  # (B, H, N, D)
            mK: cute.Tensor,  # (B, H, N, D)
            mV: cute.Tensor,  # (B, H, N, D)
            mO: cute.Tensor,  # (B, H, N, D)
            mL: cute.Tensor,  # (B, H, N)
            softmax_scale: cutlass.Float32,
    ):
        batch_size, num_heads, seq_len, _ = mQ.shape
        sQ_layout, sK_layout, sV_layout = self._make_smem_layouts()
        tiled_mma_qk, tiled_mma_pv = self._make_tiled_mmas()

        # TMA Copy Atoms (Global -> Shared)
        tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        sQ_stage0 = cute.slice_(sQ_layout, (None, None, 0))
        sK_stage0 = cute.slice_(sK_layout, (None, None, 0))
        sV_stage0 = cute.slice_(sV_layout, (None, None, 0))

        tma_atom_q, tma_mQ = cute.nvgpu.cpasync.make_tma_tile_atom(
            op=tma_op,
            gmem_tensor=mQ,
            smem_layout=sQ_stage0,
            cta_tiler=(self.tile_m, self.head_dim),
        )
        tma_atom_k, tma_mK = cute.nvgpu.cpasync.make_tma_tile_atom(
            op=tma_op,
            gmem_tensor=mK,
            smem_layout=sK_stage0,
            cta_tiler=(self.tile_n, self.head_dim),
        )
        # Permute V global tensor so (N, D) becomes (D, N) for MN-major TMA load
        mV_transposed = cute.make_tensor(
            mV.iterator,
            cute.select(mV.layout, mode=[0, 1, 3, 2]),
        )
        tma_atom_v, tma_mV = cute.nvgpu.cpasync.make_tma_tile_atom(
            op=tma_op,
            gmem_tensor=mV_transposed,
            smem_layout=sV_stage0,
            cta_tiler=(self.head_dim, self.tile_n),
        )

        num_q_blocks = cute.ceil_div(seq_len, self.tile_m)
        softmax_scale_log2 = softmax_scale * LOG2_E

        self.kernel(
            tma_atom_q,
            tma_mQ,
            tma_atom_k,
            tma_mK,
            tma_atom_v,
            tma_mV,
            mO,
            mL,
            sQ_layout,
            sK_layout,
            sV_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            softmax_scale_log2,
            seq_len,
        ).launch(
            grid=(num_q_blocks, num_heads, batch_size),
            block=(TOTAL_THREADS, 1, 1),
        )

    @cute.kernel
    def kernel(
            self,
            tma_atom_q: cute.CopyAtom,
            tma_mQ: cute.Tensor,
            tma_atom_k: cute.CopyAtom,
            tma_mK: cute.Tensor,
            tma_atom_v: cute.CopyAtom,
            tma_mV: cute.Tensor,
            mO: cute.Tensor,
            mL: cute.Tensor,
            sQ_layout: cute.ComposedLayout,
            sK_layout: cute.ComposedLayout,
            sV_layout: cute.ComposedLayout,
            tiled_mma_qk: cute.TiledMma,
            tiled_mma_pv: cute.TiledMma,
            softmax_scale_log2: cutlass.Float32,
            seq_len: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, head_idx, batch_idx = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        is_producer = warp_idx < 4

        # ------------------------------------------------------------------
        # 1. Shared Memory Allocation & Pipeline Initialization
        # ------------------------------------------------------------------
        smem = utils.SmemAllocator()
        sQ = smem.allocate_tensor(self.dtype, sQ_layout.outer, 128, sQ_layout.inner)
        sK = smem.allocate_tensor(self.dtype, sK_layout.outer, 128, sK_layout.inner)
        sV = smem.allocate_tensor(self.dtype, sV_layout.outer, 128, sV_layout.inner)

        # Allocate mbarriers: 1 stage for Q, num_stages each for K and V (2 * 64-bit per stage)
        mbar_q = smem.allocate_array(cutlass.Int64, 2)
        mbar_k = smem.allocate_array(cutlass.Int64, self.num_stages * 2)
        mbar_v = smem.allocate_array(cutlass.Int64, self.num_stages * 2)

        elem_bytes = self.dtype.width // 8
        q_tx_bytes = self.tile_m * self.head_dim * elem_bytes
        kv_tx_bytes = self.tile_n * self.head_dim * elem_bytes

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Warp, NUM_CONSUMER_THREADS // 32
        )

        pipe_q = pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_q,
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=q_tx_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        pipe_k = pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_k,
            num_stages=self.num_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=kv_tx_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        pipe_v = pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_v,
            num_stages=self.num_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=kv_tx_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        cute.arch.sync_threads()

        # Compute number of K/V blocks to iterate over (with causal early-exit)
        total_kv_blocks = cute.ceil_div(seq_len, self.tile_n)
        if cutlass.const_expr(self.is_causal):
            max_q_row = (bidx + 1) * self.tile_m - 1
            causal_kv_blocks = cute.ceil_div(max_q_row + 1, self.tile_n)
            num_kv_blocks = min(total_kv_blocks, causal_kv_blocks)
        else:
            num_kv_blocks = total_kv_blocks

        # ------------------------------------------------------------------
        # 2. Producer Warpgroup (TMA Loads)
        # ------------------------------------------------------------------
        if is_producer:
            cute.arch.warpgroup_reg_dealloc(40)

            if tidx == 0:
                # Tile global tensors for this (batch, head)
                gQ_bh = tma_mQ[(batch_idx, head_idx, None, None)]
                gK_bh = tma_mK[(batch_idx, head_idx, None, None)]
                gV_bh = tma_mV[(batch_idx, head_idx, None, None)]

                gQ_tiled = cute.local_tile(
                    gQ_bh, (self.tile_m, self.head_dim), (bidx, 0)
                )
                tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_q,
                    0,
                    cute.make_layout((1,)),
                    cute.group_modes(sQ, 0, 2),
                    cute.group_modes(gQ_tiled, 0, 2),
                )

                # Load Q_i once
                q_prod_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, 1
                )
                pipe_q.producer_acquire(q_prod_state)
                cute.copy(
                    tma_atom_q,
                    tQgQ,
                    tQsQ[(None, 0)],
                    tma_bar_ptr=pipe_q.producer_get_barrier(q_prod_state),
                )
                pipe_q.producer_commit(q_prod_state)

                k_prod_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_stages
                )
                v_prod_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_stages
                )

                # Stream K_j and V_j tiles into circular SMEM buffers
                for j in range(num_kv_blocks):
                    gK_j = cute.local_tile(
                        gK_bh, (self.tile_n, self.head_dim), (j, 0)
                    )
                    tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_k,
                        0,
                        cute.make_layout((1,)),
                        cute.group_modes(sK, 0, 2),
                        cute.group_modes(gK_j, 0, 2),
                    )
                    pipe_k.producer_acquire(k_prod_state)
                    cute.copy(
                        tma_atom_k,
                        tKgK,
                        tKsK[(None, k_prod_state.index)],
                        tma_bar_ptr=pipe_k.producer_get_barrier(k_prod_state),
                    )
                    pipe_k.producer_commit(k_prod_state)
                    k_prod_state = k_prod_state.advance()

                    gV_j = cute.local_tile(
                        gV_bh, (self.head_dim, self.tile_n), (0, j)
                    )
                    tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v,
                        0,
                        cute.make_layout((1,)),
                        cute.group_modes(sV, 0, 2),
                        cute.group_modes(gV_j, 0, 2),
                    )
                    pipe_v.producer_acquire(v_prod_state)
                    cute.copy(
                        tma_atom_v,
                        tVgV,
                        tVsV[(None, v_prod_state.index)],
                        tma_bar_ptr=pipe_v.producer_get_barrier(v_prod_state),
                    )
                    pipe_v.producer_commit(v_prod_state)
                    v_prod_state = v_prod_state.advance()

        # ------------------------------------------------------------------
        # 3. Consumer Warpgroup (2-Stage WGMMA-Softmax Pipeline, Algorithm 2)
        # ------------------------------------------------------------------
        else:
            cute.arch.warpgroup_reg_alloc(232)
            consumer_tidx = tidx - NUM_PRODUCER_THREADS

            thr_mma_qk = tiled_mma_qk.get_slice(consumer_tidx)
            thr_mma_pv = tiled_mma_pv.get_slice(consumer_tidx)

            tCsQ = thr_mma_qk.partition_A(sQ[(None, None, 0)])
            tCrQ = thr_mma_qk.make_fragment_A(tCsQ)
            tCsK = thr_mma_qk.partition_B(sK)
            tCrK = thr_mma_qk.make_fragment_B(tCsK)

            tCsV = thr_mma_pv.partition_B(sV)
            tCrV = thr_mma_pv.make_fragment_B(tCsV)

            # Accumulator fragments in RMEM: S_cur, S_next (for 2-stage overlap), and O
            tSrS_cur = thr_mma_qk.make_fragment_C(
                thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
            )
            tSrS_next = thr_mma_qk.make_fragment_C(
                thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
            )
            tOrO = thr_mma_pv.make_fragment_C(
                thr_mma_pv.partition_shape_C((self.tile_m, self.head_dim))
            )
            tOrP = thr_mma_pv.make_fragment_A(
                thr_mma_pv.partition_shape_A((self.tile_m, self.tile_n))
            )
            cute.clear(tOrO)

            # Identity coordinates for causal & boundary masking
            cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
            tScS = thr_mma_qk.partition_C(cS)

            # Per-thread row statistics (each WGMMA thread holds 2 rows per 64-row tile)
            num_m_rows = cute.size(tSrS_cur, mode=[0, 1]) * cute.size(tSrS_cur, mode=[1])
            row_max = cute.make_fragment(num_m_rows, cutlass.Float32)
            row_sum = cute.make_fragment(num_m_rows, cutlass.Float32)
            row_max.fill(-cutlass.Float32.inf)
            row_sum.fill(0.0)

            q_cons_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            k_cons_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages
            )
            v_cons_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages
            )

            # Wait for Q_i and K_0
            pipe_q.consumer_wait(q_cons_state)
            pipe_k.consumer_wait(k_cons_state)

            # Prologue GEMM0: S_cur = Q_i @ K_0^T (commit & wait)
            cute.clear(tSrS_cur)
            cute.nvgpu.warpgroup.fence()
            cute.gemm(tiled_mma_qk, tSrS_cur, tCrQ, tCrK[(None, None, None, k_cons_state.index)], tSrS_cur)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            pipe_k.consumer_release(k_cons_state)
            k_cons_state = k_cons_state.advance()

            # Prologue softmax on S_cur -> P_tilde_cur (stored in tOrP)
            self._online_softmax(
                tSrS_cur,
                tScS,
                tOrP,
                row_max,
                row_sum,
                tOrO,
                softmax_scale_log2,
                bidx * self.tile_m,
                0,
                seq_len,
                rescale_o=False,
            )

            # Mainloop (2-stage GEMM-Softmax pipelining, Algorithm 2 lines 7-17)
            for j in range(1, num_kv_blocks):
                # 1. Issue S_next = Q_i @ K_j^T (SS-WGMMA, commit but DO NOT wait)
                pipe_k.consumer_wait(k_cons_state)
                cute.clear(tSrS_next)
                cute.nvgpu.warpgroup.fence()
                cute.gemm(
                    tiled_mma_qk,
                    tSrS_next,
                    tCrQ,
                    tCrK[(None, None, None, k_cons_state.index)],
                    tSrS_next,
                )
                cute.nvgpu.warpgroup.commit_group()

                # 2. Issue O_i += P_tilde_cur @ V_{j-1} (RS-WGMMA, commit but DO NOT wait)
                pipe_v.consumer_wait(v_cons_state)
                cute.nvgpu.warpgroup.fence()
                cute.gemm(
                    tiled_mma_pv,
                    tOrO,
                    tOrP,
                    tCrV[(None, None, None, v_cons_state.index)],
                    tOrO,
                )
                cute.nvgpu.warpgroup.commit_group()

                # 3. Wait for S_next WGMMA (at most 1 group, i.e., O_i WGMMA, still in flight)
                cute.nvgpu.warpgroup.wait_group(1)

                # 4. Compute online softmax on S_next OVERLAPPED with O_i WGMMA!
                scale_o = self._online_softmax(
                    tSrS_next,
                    tScS,
                    tOrP,
                    row_max,
                    row_sum,
                    tOrO,
                    softmax_scale_log2,
                    bidx * self.tile_m,
                    j * self.tile_n,
                    seq_len,
                    rescale_o=False,
                )

                # 5. Wait for O_i WGMMA, then rescale O_i by exp2((m_old - m_new) * log2_e)
                cute.nvgpu.warpgroup.wait_group(0)
                self._rescale_output(tOrO, scale_o)

                # 6. Release K_j and V_{j-1} SMEM stages
                pipe_k.consumer_release(k_cons_state)
                k_cons_state = k_cons_state.advance()
                pipe_v.consumer_release(v_cons_state)
                v_cons_state = v_cons_state.advance()

            # Tail: O_i += P_tilde_last @ V_{T_c - 1} (Algorithm 2 lines 18-19)
            pipe_v.consumer_wait(v_cons_state)
            cute.nvgpu.warpgroup.fence()
            cute.gemm(
                tiled_mma_pv,
                tOrO,
                tOrP,
                tCrV[(None, None, None, v_cons_state.index)],
                tOrO,
            )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            pipe_v.consumer_release(v_cons_state)
            pipe_q.consumer_release(q_cons_state)

            # ------------------------------------------------------------------
            # 4. Epilogue: Normalize O_i by l_i and Write O_i, L_i to HBM
            # ------------------------------------------------------------------
            gO_bh = mO[(batch_idx, head_idx, None, None)]
            gO_tile = cute.local_tile(
                gO_bh, (self.tile_m, self.head_dim), (bidx, 0)
            )
            tOgO = thr_mma_pv.partition_C(gO_tile)
            cO = cute.make_identity_tensor((self.tile_m, self.head_dim))
            tOcO = thr_mma_pv.partition_C(cO)

            for r in range(num_m_rows):
                inv_l = 1.0 / row_sum[r]
                for idx in range(cute.size(tOrO, mode=[2])):
                    row_coord = tOcO[(0, r, idx)][0]
                    if bidx * self.tile_m + row_coord < seq_len:
                        tOgO[(0, r, idx)] = (tOrO[(0, r, idx)] * inv_l).to(self.dtype)

            # Store logsumexp L_i = m_i + log(l_i) (only lane_id % 4 == 0 per row)
            if consumer_tidx % 4 == 0:
                for r in range(num_m_rows):
                    row_coord = tOcO[(0, r, 0)][0]
                    global_row = bidx * self.tile_m + row_coord
                    if global_row < seq_len:
                        lse = row_max[r] / LOG2_E + cute.math.log(row_sum[r])
                        mL[(batch_idx, head_idx, global_row)] = lse

    @cute.jit
    def _online_softmax(
            self,
            tSrS: cute.Tensor,
            tScS: cute.Tensor,
            tOrP: cute.Tensor,
            row_max: cute.Tensor,
            row_sum: cute.Tensor,
            tOrO: cute.Tensor,
            softmax_scale_log2: cutlass.Float32,
            q_row_base: cutlass.Int32,
            k_col_base: cutlass.Int32,
            seq_len: cutlass.Int32,
            rescale_o: cutlass.Constexpr[bool],
    ) -> cute.Tensor:
        """Compute row-wise online softmax on S and pack P_tilde into tOrP."""
        num_m_rows = cute.size(row_max)
        scale_o = cute.make_fragment(num_m_rows, cutlass.Float32)

        for r in range(num_m_rows):
            m_old = row_max[r]
            local_max = -cutlass.Float32.inf

            # Apply boundary / causal mask and compute local row max
            for idx in range(cute.size(tSrS, mode=[2])):
                coord = tScS[(0, r, idx)]
                global_row = q_row_base + coord[0]
                global_col = k_col_base + coord[1]
                valid = global_col < seq_len
                if cutlass.const_expr(self.is_causal):
                    valid = valid and (global_col <= global_row)
                val = tSrS[(0, r, idx)] * softmax_scale_log2 if valid else -cutlass.Float32.inf
                tSrS[(0, r, idx)] = val
                local_max = cute.arch.fmax(local_max, val)

            m_new = cute.arch.fmax(m_old, _warp_reduce_max_4(local_max))
            alpha = cute.math.exp2(m_old - m_new, fastmath=True) if m_old > -cutlass.Float32.inf else 0.0
            scale_o[r] = alpha
            row_max[r] = m_new

            local_sum = cutlass.Float32(0.0)
            for idx in range(cute.size(tSrS, mode=[2])):
                p_val = cute.math.exp2(tSrS[(0, r, idx)] - m_new, fastmath=True)
                tOrP[(0, r, idx)] = p_val.to(self.dtype)
                local_sum = local_sum + p_val

            row_sum[r] = row_sum[r] * alpha + _warp_reduce_sum_4(local_sum)

        if cutlass.const_expr(rescale_o):
            self._rescale_output(tOrO, scale_o)

        return scale_o

    @cute.jit
    def _rescale_output(self, tOrO: cute.Tensor, scale_o: cute.Tensor):
        """Rescale FP32 accumulator O_i by exp2(m_old - m_new) per row."""
        for r in range(cute.size(scale_o)):
            alpha = scale_o[r]
            for idx in range(cute.size(tOrO, mode=[2])):
                tOrO[(0, r, idx)] = tOrO[(0, r, idx)] * alpha


def flash_attention_3(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Launch FlashAttention-3 forward pass via CuTe DSL.

    Args:
        q, k, v: Tensors of shape (batch, heads, seq_len, head_dim) on CUDA,
                 dtype float16 or bfloat16, contiguous in head_dim.
        is_causal: Whether to apply causal masking.
        softmax_scale: Optional scaling factor (defaults to 1 / sqrt(head_dim)).

    Returns:
        (out, lse): Output tensor (B, H, N, D) and log-sum-exp (B, H, N).
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.shape == k.shape == v.shape
    batch, heads, seq_len, head_dim = q.shape
    assert head_dim in (64, 128, 256), f"Unsupported head_dim: {head_dim}"

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    cutlass_dtype = cutlass.Float16 if q.dtype == torch.float16 else cutlass.BFloat16
    out = torch.empty_like(q)
    lse = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)

    fa3 = FlashAttention3Forward(
        dtype=cutlass_dtype,
        head_dim=head_dim,
        tile_m=64,
        tile_n=64,
        num_stages=2,
        is_causal=is_causal,
    )

    compiled = cute.compile(
        fa3,
        from_dlpack(q, assumed_align=16),
        from_dlpack(k, assumed_align=16),
        from_dlpack(v, assumed_align=16),
        from_dlpack(out, assumed_align=16),
        from_dlpack(lse, assumed_align=16),
        softmax_scale,
    )
    compiled(
        from_dlpack(q, assumed_align=16),
        from_dlpack(k, assumed_align=16),
        from_dlpack(v, assumed_align=16),
        from_dlpack(out, assumed_align=16),
        from_dlpack(lse, assumed_align=16),
        softmax_scale,
    )
    return out, lse


if __name__ == "__main__":
    assert torch.cuda.is_available(), "FlashAttention-3 CuTe DSL requires an H100 (SM90) CUDA GPU."
    torch.manual_seed(0)

    B, H, N, D = 2, 8, 2048, 128
    dtype = torch.bfloat16
    q = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, N, D, device="cuda", dtype=dtype)

    for causal in (False, True):
        out, _ = flash_attention_3(q, k, v, is_causal=causal)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        print(f"FlashAttention-3 CuTe DSL (is_causal={causal}) passed!")

    # Simple latency benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        flash_attention_3(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - t0) * 1000.0 / 20.0
    print(f"Average runtime (B={B}, H={H}, N={N}, D={D}, causal=True): {avg_ms:.3f} ms")
