# Turbo4 MMVQ Integration Plan

## Problem
Turbo4 weights (GGML_TYPE_TURBO4_0) currently route through a custom CUDA kernel
(turbo-matmul.cu) that achieves ~12.6 tok/s on RTX 3090. The custom kernel uses
1 warp, full dequant to float, no warp-level parallelism.

## Solution
Wire turbo4 into the MMVQ (mul_mat_vec_q) pipeline — ggml-cuda's most optimized
matmul path. MMVQ gives 4-warp cooperative processing, shared memory for
activations, and coalesced memory access.

## Key Insight
Turbo4 stores 128 values per block: `value[i] = norm * codebook_4bit[qs_nibble]`
in WHT-rotated space. The codebook is symmetric Lloyd-Max for N(0,1/sqrt(128)).
For MMVQ, activations get pre-rotated with forward WHT so the dot product
works in rotated space. The vec_dot needs per-element table lookup instead of
dp4a because codebook values are non-uniform floats.

## Files to Modify

### 1. `common.cuh` — Type traits
Add `ggml_cuda_type_traits<GGML_TYPE_TURBO4_0>`:
- qk = QK_TURBO4 = 128
- qr = 1 (no extra scaling per sub-block)
- qi = QK_TURBO4 / 32 = 4 (4 int32 chunks from qs per MMVQ iteration)?

Actually: for the MMVQ kernel, qi determines how many int32-sized reads per
block are needed. For Q4_0: QI4_0 = QK4_0 / (4 * QR4_0) = 32 / 8 = 4.  
For turbo4: QK_TURBO4 / (4 * QR) = 128 / (4 * 1) = 32.

Wait, but QI4_0 is actually 4 by the formula, which means 4 int32s from qs
= 16 bytes, which is the full qs size of a Q4_0 block. So for turbo4, we'd
need QI_TURBO4 = 32 to cover the full 64-byte qs.

But VDR and the kernel iteration use qi differently. Let me check:

In the kernel:
```
constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;
```
For Q4_0: 2 * 4 * 32 / 4 = 64 blocks per iteration.
For turbo4 with QI=32: vdr * 4 * 32 / 32 = vdr * 4 blocks.

And threads are assigned: `kbx = tid / (qi/vdr)`. For Q4_0: tid / (4/2) = tid/2.
So pairs of threads share a kbx but use different iqs.

For turbo4: we need qi/vdr to work out cleanly. If vdr = 4 and qi = 32:
qi/vdr = 8, so tid/8 threads share kbx. With 32 threads per warp, every 8th
thread processes the same kbx with different iqs. Each group of 8 gets
iqs = vdr * (tid % 8) = 4 * (0..7) = 0, 4, 8, 12, 16, 20, 24, 28.

But 8 groups of 4 int32s at offsets 0,4,8,...,28 covers 32 int32s = 128 bytes.
That's more than the 64 bytes of qs. So we'd only need 4 groups.

Hmm, I think the right approach is:
- QI_TURBO4 = 16 (half of the 32 int32s in qs — we need to figure out the right convention)

Actually, I'm overcomplicating this. Let me look at how other types with
qk != 32 work. Q8_0 has QK8_0=256, QI8_0=32. That's 4x the Q4_0 sizes.

Let me check: QI8_0 = QK8_0 / (4 * QR8_0) = 256 / (4 * 2) = 32. So QI=32.
VDR_Q8_0_Q8_1_MMVQ = 4.

With qi=32, vdr=4: qi/vdr = 8.
blocks_per_iter = 4 * 4 * 32 / 32 = 16.
kbx = tid / 8, kqs = 4 * (tid % 8).

So for Q8_0, each vec_dot processes 32/8 * VDR = 4 * 4 = 16 int32 values
from qs? Let me check the actual Q8_0 vec_dot code...

Actually, for Q8_0: qs has 256 int8_t values = 64 int32s. QI8_0=32 means
32 int32s are processed per call. With VDR=4.

Let me just read the Q8_0 vec_dot to understand the pattern.

Actually, I realize I should just follow the existing pattern. Let me use
the type that's most similar to turbo4 — a type with qk > 32. Let me read
Q8_0's vec_dot.

But actually, I think a much simpler approach would work: since turbo4 has
QK=128, we can treat it like 4 × Q4_0-like sub-blocks conceptually but with
table lookup. The norm covers all 128 values. We process 32 elements at a time
(the Q8_1 block size) in 4 iterations, using the same norm for all.

Let me define:
- QK_TURBO4 = 128, but for MMVQ treat it as: the vec_dot processes 32 elements
  per call, reading from 1 Q8_1 block and 1/4 of a turbo4 block
- Actually, set QI_TURBO4 = QK_TURBO4 / (4 * 1) = 32 (same as Q8_0)
- Set VDR_TURBO4_0_Q8_1_MMVQ = 4 (process 4 int32s per call, covering 32 q4 values)

Wait, I think I've been going in circles. Let me just write the simplest
correct implementation and iterate.

## Implementation Files (final plan)

### common.cuh
```
template<>
struct ggml_cuda_type_traits<GGML_TYPE_TURBO4_0> {
    static constexpr int qk = QK_TURBO4;  // 128
    static constexpr int qr = 1;
    static constexpr int qi = QK_TURBO4 / (4 * 1);  // 32
};
```

### vecdotq.cuh
- Define: `#define VDR_TURBO4_0_Q8_1_MMVQ 4`
- Define: `#define VDR_TURBO4_0_Q8_1_MMQ  8`
- Write `vec_dot_turbo4_0_q8_1_impl` (template on vdr, uses codebook table)
- Write `vec_dot_turbo4_0_q8_1`
- Write `vec_dot_turbo4_0_q8_1_mmq` if needed

### mmvq.cu
Add `case GGML_TYPE_TURBO4_0:` to all 6+ switch statements:
1. `get_vec_dot_q_cuda()` — return vec_dot_turbo4_0_q8_1
2. `get_vdr_mmvq()` — return VDR_TURBO4_0_Q8_1_MMVQ
3. `get_mmvq_mmid_max_batch_pascal_older()` — default is fine (MMVQ_MAX_BATCH_SIZE)
4. `get_mmvq_mmid_max_batch_turing_plus()` — default is fine
5. `calc_nwarps()` — RDNA4 whitelist? For sm_86: falls through to GENERIC which always returns 1 or 2
6. Maybe add TURBO4 to the RDNA4 nwarps whitelist for ncols_dst=1

### ggml-cuda.cu
- In the type dispatch, route TURBO4 through MMVQ when it's a weight tensor
- Add WHT pre-rotation of activations before MMVQ kernel launch

### Additional files for the full mmq dispatch
- mmq.cuh has similar dispatch for the batched MMQ path
- Need to check if we need to add there too

## GPU-side codebook constant
The CENTROIDS_4BIT table needs to be available in device code.
Define as a `__constant__` array in vecdotq.cuh:
```
static constexpr __device__ float TURBO4_CODEBOOK[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};
```

## Block structure reminder
```
block_turbo4_0:
  ggml_half norm;         // 2 bytes
  uint8_t qs[64];         // 64 bytes (128 × 4-bit, low nibble first)
Total: 66 bytes

block_q8_1:
  ggml_half d;            // 2 bytes: scale
  ggml_half s;            // 2 bytes: sum of abs values
  int8_t qs[32];          // 32 bytes
Total: 36 bytes
```
