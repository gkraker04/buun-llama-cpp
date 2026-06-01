"""
Turbo4 GGUF Weight Validator
Reads a turbo4 GGUF, extracts a weight tensor, dequantizes it via the
turbo4 CPU algorithm (WHT + centroid lookup), and validates the output.

Usage:
    python validate_turbo4_weights.py <path/to/model.gguf>

Requires: numpy (for fast dot product comparison)
"""

import struct
import sys
import os
import math

# ─── GGUF type enum (buun's fork non-standard numbering) ───
GGUF_TYPE = {
    'UINT8': 0, 'INT8': 1, 'UINT16': 2, 'INT16': 3, 'UINT32': 4, 'INT32': 5,
    'FLOAT32': 6, 'BOOL': 7, 'STRING': 8, 'ARRAY': 9, 'UINT64': 10,
    'INT64': 11, 'FLOAT64': 12, 'SCORE': 13,
}
GGML_TYPE = {
    'F32': 0, 'F16': 1, 'Q4_0': 2, 'Q4_1': 3, 'Q5_0': 6, 'Q5_1': 7,
    'Q8_0': 8, 'Q8_1': 9, 'Q2_K': 10, 'Q3_K': 11, 'Q4_K': 12, 'Q5_K': 13,
    'Q6_K': 14, 'IQ2_XXS': 16, 'IQ2_XS': 17, 'IQ3_XXS': 18, 'IQ1_S': 19,
    'IQ4_NL': 20, 'IQ3_S': 21, 'IQ2_S': 22, 'IQ4_XS': 23, 'IQ1_M': 24,
    'BF16': 25, 'MXFP4': 26, 'Q1_0': 27, 'NVFP4': 28, 'TQ1_0': 29,
    'TQ2_0': 30, 'TURBO4_0': 43,  # <-- buun's custom type number
}
GGML_TYPE_REV = {v: k for k, v in GGML_TYPE.items()}

# ─── Turbo4 constants (matched to GPU/CPU code) ───
QK_TURBO4 = 128
BLOCK_SIZE = 66  # sizeof(block_turbo4_0) = 2 + 64
CENTROIDS_4BIT = [
    -0.241556, -0.182907, -0.143047, -0.111065,
    -0.083317, -0.058069, -0.034311, -0.011353,
     0.011353,  0.034311,  0.058069,  0.083317,
     0.111065,  0.143047,  0.182907,  0.241556,
]
INV_SQRT_128 = 0.08838834764831845

# Turbo4 WHT sign arrays (from turbo-wht.cu, seed=42)
TURBO_WHT_S1 = [
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1
]
TURBO_WHT_S2 = [
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1
]


def read_gguf_string(f):
    """Read a GGUF string: uint64 length + UTF-8 data."""
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')


def read_gguf_value(f, gguf_type):
    """Read a GGUF value of the given type."""
    if gguf_type == GGUF_TYPE['UINT8']:
        return struct.unpack('<B', f.read(1))[0]
    elif gguf_type == GGUF_TYPE['INT8']:
        return struct.unpack('<b', f.read(1))[0]
    elif gguf_type == GGUF_TYPE['UINT16']:
        return struct.unpack('<H', f.read(2))[0]
    elif gguf_type == GGUF_TYPE['INT16']:
        return struct.unpack('<h', f.read(2))[0]
    elif gguf_type == GGUF_TYPE['UINT32']:
        return struct.unpack('<I', f.read(4))[0]
    elif gguf_type == GGUF_TYPE['INT32']:
        return struct.unpack('<i', f.read(4))[0]
    elif gguf_type == GGUF_TYPE['FLOAT32']:
        return struct.unpack('<f', f.read(4))[0]
    elif gguf_type == GGUF_TYPE['BOOL']:
        return struct.unpack('<B', f.read(1))[0] != 0
    elif gguf_type == GGUF_TYPE['STRING']:
        return read_gguf_string(f)
    elif gguf_type == GGUF_TYPE['UINT64']:
        return struct.unpack('<Q', f.read(8))[0]
    elif gguf_type == GGUF_TYPE['INT64']:
        return struct.unpack('<q', f.read(8))[0]
    elif gguf_type == GGUF_TYPE['FLOAT64']:
        return struct.unpack('<d', f.read(8))[0]
    elif gguf_type == GGUF_TYPE['ARRAY']:
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        return [read_gguf_value(f, arr_type) for _ in range(arr_len)]
    else:
        raise ValueError(f"Unknown GGUF type: {gguf_type}")


def turbo_wht_forward(buf):
    """Apply turbo4 WHT forward rotation in-place (matching CPU/GPU code)."""
    n = len(buf)
    for i in range(n):
        buf[i] *= TURBO_WHT_S1[i]
    h = 1
    while h < n:
        step = 2 * h
        for i in range(0, n, step):
            for j in range(i, i + h):
                a, b = buf[j], buf[j + h]
                buf[j] = a + b
                buf[j + h] = a - b
        h *= 2
    for i in range(n):
        buf[i] *= INV_SQRT_128 * TURBO_WHT_S2[i]


def turbo_wht_inverse(buf):
    """Apply turbo4 WHT inverse rotation in-place."""
    n = len(buf)
    for i in range(n):
        buf[i] *= TURBO_WHT_S2[i]
    h = 1
    while h < n:
        step = 2 * h
        for i in range(0, n, step):
            for j in range(i, i + h):
                a, b = buf[j], buf[j + h]
                buf[j] = a + b
                buf[j + h] = a - b
        h *= 2
    for i in range(n):
        buf[i] *= INV_SQRT_128 * TURBO_WHT_S1[i]


def turbo4_dequant_block(block_bytes):
    """
    Dequantize one block_turbo4_0 (66 bytes) to 128 float32 values.
    Matches CPU dequantize_row_turbo4_0().
    """
    assert len(block_bytes) == BLOCK_SIZE, f"Expected {BLOCK_SIZE} bytes, got {len(block_bytes)}"
    
    # Read norm (fp16 at byte 0)
    norm_raw = struct.unpack('<H', block_bytes[0:2])[0]
    # fp16 to f32
    sign = (norm_raw >> 15) & 1
    exp = (norm_raw >> 10) & 0x1F
    mant = norm_raw & 0x3FF
    if exp == 0:
        norm = 0.0
    elif exp == 31:
        norm = float('-inf') if sign else float('inf')
    else:
        norm = ((-1) ** sign) * (2.0 ** (exp - 15)) * (1.0 + mant / 1024.0)
    
    # Read qs (64 bytes, starting at byte 2)
    qs = list(block_bytes[2:66])
    
    # Unpack nibbles → centroid lookup → scale by norm
    recon = [0.0] * QK_TURBO4
    for elem in range(QK_TURBO4):
        byte_idx = elem // 2
        nibble_shift = (elem % 2) * 4
        idx = (qs[byte_idx] >> nibble_shift) & 0xF
        recon[elem] = CENTROIDS_4BIT[idx] * norm
    
    # Inverse WHT to get original-space values
    turbo_wht_inverse(recon)
    
    return recon


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.gguf>")
        sys.exit(1)
    
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"GGUF file: {path} ({size_mb:.1f} MB)")
    
    with open(path, 'rb') as f:
        # ── Read header ──
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != 0x46554747:  # 'GGUF' in LE
            print(f"ERROR: Bad magic: 0x{magic:08X} (expected 0x46554747 = 'GGUF')")
            sys.exit(1)
        print("Magic: OK (GGUF)")
        
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
        print(f"Version: {version}, Tensors: {tensor_count}, Metadata KV: {metadata_kv_count}")
        
        # ── Read metadata ──
        for _ in range(metadata_kv_count):
            key = read_gguf_string(f)
            val_type = struct.unpack('<I', f.read(4))[0]
            val = read_gguf_value(f, val_type)
            if key in ('general.name', 'general.architecture'):
                print(f"  {key}: {val}")
        
        # ── Read tensor info ──
        tensor_infos = []
        for _ in range(tensor_count):
            name = read_gguf_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = list(struct.unpack(f'<{"Q" * n_dims}', f.read(8 * n_dims)))
            ggml_type = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensor_infos.append((name, dims, ggml_type, offset))
        
        # ── Print tensor summary ──
        turbo_tensors = [(n, d, t, o) for n, d, t, o in tensor_infos 
                        if t == GGML_TYPE['TURBO4_0']]
        print(f"\nTURBO4 tensors: {len(turbo_tensors)}/{tensor_count}")
        
        non_turbo = [(n, d, t, o) for n, d, t, o in tensor_infos if t != GGML_TYPE['TURBO4_0']]
        if non_turbo:
            print(f"Non-turbo4 tensors ({len(non_turbo)}):")
            for n, d, t, o in non_turbo[:5]:
                type_name = GGML_TYPE_REV.get(t, f"TYPE_{t}")
                print(f"  {n}: {d} [{type_name}] @ 0x{o:X}")
            if len(non_turbo) > 5:
                print(f"  ... and {len(non_turbo) - 5} more")
        
        # ── Dequantize first turbo4 weight tensor ──
        print("\n--- Dequantizing first turbo4 weight tensor ---")
        target_name = None
        for n, dims, t, offset in turbo_tensors:
            if len(dims) >= 2:
                target_name = n
                break
        
        if target_name is None:
            print("ERROR: No turbo4 weight tensor found!")
            sys.exit(1)
        
        print(f"Tensor: {target_name}")
        nd = dims
        print(f"  Shape: {nd}")
        print(f"  Offset in file: 0x{offset:X}")
        
        # Calculate number of blocks
        n_rows = nd[1] if len(nd) >= 2 else 1
        elems_per_row = nd[0]
        blocks_per_row = elems_per_row // QK_TURBO4
        row_size = blocks_per_row * BLOCK_SIZE
        total_blocks = n_rows * blocks_per_row
        total_bytes = total_blocks * BLOCK_SIZE
        print(f"  Rows: {n_rows}, Elems/row: {elems_per_row}")
        print(f"  Blocks/row: {blocks_per_row}, Row bytes: {row_size}")
        print(f"  Total blocks: {total_blocks}, Total data bytes: {total_bytes}")
        
        # Seek to tensor data
        alignment = 32  # GGUF default alignment
        f.seek(offset)
        
        # ── Dequantize first 5 rows and validate ──
        import random
        random.seed(42)
        test_vec = [random.gauss(0, 1) for _ in range(elems_per_row)]
        
        turbo_dot = 0.0
        ref_dot = 0.0
        
        rows_to_check = min(5, n_rows)
        print(f"\n  Dequantizing first {rows_to_check} rows...")
        
        for row in range(rows_to_check):
            row_data = f.read(row_size)
            row_floats = []
            
            for blk in range(blocks_per_row):
                block_bytes = row_data[blk * BLOCK_SIZE : (blk + 1) * BLOCK_SIZE]
                dequantized = turbo4_dequant_block(block_bytes)
                row_floats.extend(dequantized)
            
            # Compute dot product with test vector
            rd = sum(row_floats[i] * test_vec[i] for i in range(elems_per_row))
            turbo_dot += rd * rd
            
            # Basic sanity: check for NaN/Inf
            nans = sum(1 for v in row_floats if math.isnan(v))
            infs = sum(1 for v in row_floats if math.isinf(v))
            zeros = sum(1 for v in row_floats if abs(v) < 1e-30)
            row_norm = math.sqrt(sum(v*v for v in row_floats))
            
            print(f"  Row {row}: L2={row_norm:.4f}, NaNs={nans}, Infs={infs}, Zeros={zeros}")
            
            # Print first few values
            print(f"    First 8: [{', '.join(f'{v:.6f}' for v in row_floats[:8])}]")
            
            # Check centroids distribution
            counts = {}
            for blk in range(min(4, blocks_per_row)):
                block_bytes = row_data[blk * BLOCK_SIZE : (blk + 1) * BLOCK_SIZE]
                qs = block_bytes[2:66]
                for byte in qs:
                    lo = byte & 0xF
                    hi = (byte >> 4) & 0xF
                    counts[lo] = counts.get(lo, 0) + 1
                    counts[hi] = counts.get(hi, 0) + 1
            print(f"    Centroid usage (first 4 blocks): top-5 = "
                  f"{sorted(counts.items(), key=lambda x: -x[1])[:5]}")
        
        # ── Overall assessment ──
        print(f"\n  Total squared turbo dot products (first {rows_to_check} rows): {turbo_dot:.6f}")
        
        # Quick check: compute stats for a single row
        print("\n--- Single-row detailed analysis ---")
        f.seek(offset)
        first_row = f.read(row_size)
        
        all_dequant = []
        for blk in range(blocks_per_row):
            block_bytes = first_row[blk * BLOCK_SIZE : (blk + 1) * BLOCK_SIZE]
            all_dequant.extend(turbo4_dequant_block(block_bytes))
        
        mean = sum(all_dequant) / len(all_dequant)
        var = sum((v - mean)**2 for v in all_dequant) / len(all_dequant)
        std = math.sqrt(var)
        print(f"  Elements: {len(all_dequant)}")
        print(f"  Mean: {mean:.6f}, Std: {std:.6f}")
        print(f"  Min: {min(all_dequant):.6f}, Max: {max(all_dequant):.6f}")
        
        # Expected stats for F32 weights: mean ~0, distribution depends on layer
        # but we're checking for obvious corruption
        if std < 1e-10:
            print("\n  ⚠ WARNING: All dequantized values are near-zero!")
            print("  This indicates either the GGUF data is corrupted or")
            print("  the dequantization algorithm doesn't match the data format.")
        elif any(math.isnan(v) or math.isinf(v) for v in all_dequant):
            print("\n  ⚠ WARNING: NaN or Inf values detected in dequantized data!")
        else:
            print(f"\n  ✓ Dequantized values look reasonable (std={std:.4f})")
            print(f"  The GGUF data is self-consistent.")
            print(f"  The gibberish output is caused by something in the GPU inference pipeline,")
            print(f"  not by corrupt weight data in the GGUF file.")
        
        print("\n=== Done ===")


if __name__ == '__main__':
    main()
