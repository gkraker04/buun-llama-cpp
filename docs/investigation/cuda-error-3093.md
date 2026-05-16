# CUDA Error 3093 Investigation — DFlash + `--parallel 2` Crash

**Date**: 2026-05-16  
**Branch**: `fix/cuda-error-3093` off `master`  
**Model**: Qwen3.6-27B (qwen35 architecture) + CLIP vision encoder (mmproj-f16.gguf)  
**Drafter**: DFlash draft-3.6-Q4_K_S.gguf

---

## Current Status

**Two distinct bugs have been identified:**

1. ✅ **Bug A (GGML_ASSERT in `build_rs`) — FIXED: `s->nb[1]` → `states->nb[1]`**
   - **File**: `src/llama-graph.cpp` line 2621
   - **Root cause**: `build_rs()` creates a 2D view of the recurrent state tensor `s` using `s->nb[1]`, but `s` is a 1D tensor (`ggml_new_tensor_1d`). The 1D tensor's `nb[1]` equals the TOTAL bytes of the tensor, not the row stride. After `ggml_reshape_2d` into `states`, `states->nb[1]` has the correct row stride.
   - **Fix**: Use `states->nb[1]` instead of `s->nb[1]` in the `ggml_view_2d` call.
   - **Evidence**: DEBUG logging confirmed `s_ne[1]=1` (1D) with `s_nb[1]=491520` (= total bytes), while `states->nb[1]=122880` (= row stride of 30720 × 4).

2. ❌ **Bug B (CUDA illegal memory access) — STILL OPEN**
   - Surfaces as `CUDA error: an illegal memory access was encountered` at `ggml-cuda.cu:3109` (`cudaStreamSynchronize`)
   - Triggers under concurrent `--parallel 2` load, AFTER the `build_rs` GGML_ASSERT is fixed
   - The synchronize catches an async kernel error from an earlier launch — doesn't identify the faulty kernel
   - See section "Bug B: CUDA Illegal Memory Access" below

---

## Bug A: Tensor View Bounds Assert (`s->nb[1]` vs `states->nb[1]`)

### Symptoms
```
/home/zg/buun-llama-cpp/ggml/src/ggml.c:1788: GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src)) failed
```

### Backtrace
```
ggml_new_tensor_impl (view bounds check)  →  ggml.c:1788
build_rs(ggml_tensor*, ggml_tensor*, ggml_tensor*, int32_t, int32_t, uint32_t, uint32_t, uint32_t, int32_t)
build_rs(llm_graph_input_rs*, ggml_tensor*, int32_t, int32_t)
build_layer_attn_linear
llm_build_qwen35
llama_model::build_graph
llama_context::process_ubatch
llama_context::decode
llama_decode
server_context_impl::update_slots
server_queue::start_loop
```

### Root Cause

In `src/llama-graph.cpp:2591-2624`, `build_rs()`:

```cpp
ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, rs_size);
// ...
ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])
```

The recurrent state tensors (`r_l[i]`, `s_l[i]`) are created as 1D via `ggml_new_tensor_1d` in either:
- `llama_memory_recurrent_context::resize()` (line 477: `ggml_tensor * r = ggml_new_tensor_1d(...)`)
- Initial allocation in `llama_memory_recurrent::init()`

For a 1D tensor with shape `[N]`:
- `nb[0]` = element size (e.g., 4 for float32)
- `nb[1]` = `ggml_nbytes(s)` = `N * element_size` (the entire tensor)

After `ggml_reshape_2d(ctx0, s, state_size, rs_size)`:
- `states` has shape `[state_size, rs_size]`
- `states->nb[1]` = `state_size * element_size` (correct row stride)

The view at the original line 2621 uses `s->nb[1]` (= total tensor bytes) instead of `states->nb[1]` (= correct row stride). This causes the offset `(rs_head + n_seqs) * s->nb[1]` to be much larger than intended, potentially exceeding `ggml_nbytes(s)` and triggering the assert.

### Fix Applied

Commit: `66db033bd` → Changed `s->nb[1]` to `states->nb[1]` in the `ggml_view_2d` call.

### Verification
- Build with DEBUG logging confirmed:
  - Pre-fix, `n_seqs=2, n_rs=3, rs_head=0`: offset = `(0+2) × 491520 = 983040`, data_size = 122880, total = 1105920 > 491520 → **ASSERT FAILS**
  - Post-fix: offset = `(0+2) × 122880 = 245760`, data_size = 122880, total = 368640 ≤ 491520 → **OK**
- Test with 2-concurrent sequential requests PASSES
- Test with exact production pattern (Hermes long + WebUI quick): ~50% success rate

---

## Bug B: CUDA Illegal Memory Access (Open)

### Symptoms
```
E CUDA synchronize error on device 0: an illegal memory access was encountered
E CUDA error: an illegal memory access was encountered
E   current device: 0, in function ggml_backend_cuda_synchronize at ggml-cuda.cu:3109
```

### Crash Sequence (from log `05-56-00`)

```
[slot 0 gen]   verify ubatch: 8 tok, 115.9ms
[slot 0 gen]   spec cycle: 391.5ms
[slot 0 gen]   verify ubatch: 8 tok, 119.9ms
[slot 0 gen]   spec cycle: 203.4ms
[slot 1 alloc] get_availabl: id 1 | task -1
[slot 1 pref]  new prompt, n_ctx_slot = 252160, task.n_tokens = 24
[slot 1 pref]  verify ubatch: 28 tok, 201.8ms
[slot 1 pref]  spec cycle: 250.1ms
[slot 1 pref]  prompt processing done, batch.n_tokens = 12
[slot 1 gen]   created context checkpoint 1 of 4
[slot 1 gen]   verify ubatch: 12 tok, 179.8ms
[slot 1 gen]   spec cycle: 412.9ms
                                  ← 285ms silence (GPU computation)
CUDA synchronize error               ← surfaces at next sync point
```

The crash occurs 285ms AFTER the last spec cycle, during silent GPU computation. The `cudaStreamSynchronize` is called during the NEXT `update_slots()` cycle, which catches an async error from a kernel launched during the previous cycle.

### Analysis

- This is NOT the `build_rs` assert bug — that one printed `GGML_ASSERT(view_src...)` directly to stderr
- The CUDA error surfaces at a `sched_reserve()` → `cudaDeviceSynchronize()` → `cudaStreamSynchronize` call
- The faulty kernel may have been launched during:
  1. GDN (Gated Delta Net) forward in tree mode  
  2. Cross-ring interleave kernel
  3. Tree verify (48 layers × 12 tokens = 864 MB allocated)
- Memory was resized from 2→4 cells (2.18.929) ≈ 290ms before the error, implying the resize itself is not the trigger

### Likely Causes (ordered by probability)

1. **Draft verify batch overruns tree verify buffer**: 864 MB allocated for 48 layers × 12 tokens. With concurrent slots, one slot's tree verify might overlap with another's, exceeding the allocation.

2. **GPU ring buffer concurrent access**: `dflash_max_slots=1` means only slot 0 has DFlash. But when slot 1 enters prefill, the shared drafter context accesses GPU ring data. With `set_cross_data_gpu()` writing to the ring while slot 0's generation reads it, a race condition on the DFlash cross-data structure causes illegal memory access.

3. **Cross-attention context length mismatch**: When both slots have different context lengths, the shared cross-attention structure (`cross.n_enc`, `cross.v_embd_gpu`) may have stale dimensions from the previous slot's operation.

### Next Steps

1. ~~Test `s->nb[1]` fix~~ ✅ FIXED
2. **[ ] Test with `CUDA_LAUNCH_BLOCKING=1`** — Serializes CUDA operations to identify the exact faulty kernel
3. **[ ] Test with `dflash_max_slots=2`** — Enabling DFlash for both slots might resolve the asymmetry
4. **[ ] Check if `set_cross_data_gpu` uses proper synchronization** between shared drafter context and target model backend
5. **[ ] Monitor buun's SD-075/SD-078 branches** for upstream fixes to the multi-slot DFlash crash
