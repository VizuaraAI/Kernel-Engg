# Makefile for Lecture 14: CUDA Performance Triage with Nsight Compute

NVCC ?= nvcc
ARCH ?= sm_90 # Default to sm_90 (Hopper); override via ARCH=sm_80, sm_89, etc.
NVCCFLAGS ?= -O3 -lineinfo -arch=$(ARCH)

TARGETS = kernel1_sabotaged kernel1_fixed \
          kernel2_sabotaged kernel2_fixed \
          kernel3_sabotaged kernel3_fixed

.PHONY: all clean profile-k1 profile-k2 profile-k3

all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.ncu-rep *.csv

# Profile Kernel 1 (Decision 2 -> Decision 5: Uncoalesced Global Memory Access)
profile-k1: kernel1_sabotaged kernel1_fixed
	ncu --set full --import-source yes --clock-control none --cache-control none \
		-f -o k1_sabotaged ./kernel1_sabotaged
	ncu --set full --import-source yes --clock-control none --cache-control none \
		-f -o k1_fixed ./kernel1_fixed

# Profile Kernel 2 (Decision 2 -> Decision 3/5: Shared Memory Bank Conflicts)
profile-k2: kernel2_sabotaged kernel2_fixed
	ncu --set full --import-source yes --clock-control none --cache-control none \
		-f -o k2_sabotaged ./kernel2_sabotaged
	ncu --set full --import-source yes --clock-control none --cache-control none \
		-f -o k2_fixed ./kernel2_fixed

# Profile Kernel 3 (Decision 1: Grid Starvation & Under-saturation)
profile-k3: kernel3_sabotaged kernel3_fixed
	ncu --set full --import-source yes --clock-control none --cache-control none \
		-f -o k3_sabotaged ./kernel3_sabotaged
	ncu --set full --import-source yes --clock-control none --cache-control none \
		-f -o k3_fixed ./kernel3_fixed
