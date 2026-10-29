# Lecture 12: Tensor Cores

## Evolution of ML-specific hardware Nvidia GPU Architectures (Consumer and Data-Center)

| Year launched | Nvidia architecture name (consumer) | Architecture name (data center) | Distinct feature (Machine Learning & Deep Learning) |
| :--- | :--- | :--- | :--- |
| **2024** | **Blackwell** | **Blackwell** | 2nd-generation Transformer Engine with micro-tensor scaling and native 4-bit floating-point (NVFP4) math for massive generative AI models. |
| **2022** | **Ada Lovelace** | **Hopper** | 1st-generation Transformer Engine dynamically switching between FP8 and FP16 to accelerate Large Language Model (LLM) training; 4th-gen Tensor Cores. |
| **2020** | **Ampere** | **Ampere** | 3rd-gen Tensor Cores with TensorFloat-32 (TF32), native BF16 support, and hardware acceleration for structural network sparsity. |
| **2018** | **Turing** | **Turing (T4)** | 2nd-gen Tensor Cores introducing dedicated integer precisions (INT8, INT4) specifically optimized for ultra-low latency AI inference. |
| **2017** | **Volta** *(Titan V / Prosumer)* | **Volta** | 1st-generation Tensor Cores introduced, bringing specialized mixed-precision matrix multiply-accumulate hardware for deep learning. |
| **2016** | **Pascal** | **Pascal** | Native hardware FP16 (half-precision) math at 2x throughput and 1st-gen NVLink interconnect for multi-GPU training clusters. |
| **2014** | **Maxwell** | **Maxwell** | Massive energy-efficiency leap for large-scale inferencing and introduction of native FP16 storage concepts. |
| **2012** | **Kepler** | **Kepler** | GPUDirect RDMA introduced, allowing direct GPU-to-GPU network communication bypassing the CPU for scaled distributed neural network training. |
| **2010** | **Fermi** | **Fermi** | Full double precision (FP64) and true cache hierarchy (L1/L2), providing the compute foundation for early breakthroughs like AlexNet. |
| **2006 / 2007** | **Tesla** | **Tesla** | Introduction of CUDA and unified shaders, giving birth to General-Purpose GPU (GPGPU) computing and enabling early neural network experiments. |

To check the whitepapers, check the root level [README.md](https://github.com/VizuaraAI/Kernel-Engg/blob/main/README.md).