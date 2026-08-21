#!/usr/bin/env python3
# Odd-moment (skew) codec design study for turbo3.
# Mechanism: flip each rotated coordinate by the sign of its (train-split) skew so all
# coordinates are positively skewed, then quantize with ONE shared ASYMMETRIC 8-level
# Lloyd-Max book. Decode = book lookup * flip sign (rides the s2 sign-vector mechanism).
#
# Questions answered here, on real 27B rows:
#   1. How much MSE does asymmetry collect vs stock and vs the symmetric-retrain floor?
#   2. Are skew signs consistent enough for a GLOBAL 128-wide sign vector (per side),
#      or do we need per-layer / per-(head,coord) tables?
#   3. Per-side (K vs V) split.
import numpy as np, sys, glob, os

D = 128
INV = 1.0 / np.sqrt(D)
S1 = np.array([-1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
	-1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
	-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
	-1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1], dtype=np.float64)
S2 = np.array([1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
	1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
	1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
	1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1], dtype=np.float64)
CENT = np.array([-0.190685,-0.117832,-0.065717,-0.021460,0.021460,0.065717,0.117832,0.190685])

def hadamard(n):
	H = np.array([[1.0]])
	while H.shape[0] < n:
		H = np.block([[H, H], [H, -H]])
	return H

ROT = (np.diag(S1) @ (hadamard(D) * INV) @ np.diag(S2)).T

def rot_frame(raw, mu):
	g = raw.reshape(-1, 8, D) - mu.reshape(8, D)
	gn = np.linalg.norm(g, axis=2, keepdims=True)
	return (g / np.where(gn > 1e-10, gn, 1.0)) @ ROT, gn

def lloyd8(x, symmetric, iters=80):
	c = np.quantile(x, np.linspace(0.0625, 0.9375, 8))
	for _ in range(iters):
		mid = 0.5 * (c[:-1] + c[1:])
		idx = np.searchsorted(mid, x)
		newc = np.array([x[idx == i].mean() if np.any(idx == i) else c[i] for i in range(8)])
		if symmetric:
			newc = 0.5 * (newc - newc[::-1])
		if np.max(np.abs(newc - c)) < 1e-8:
			c = newc; break
		c = newc
	return np.sort(c)

def codec_mse(y, cent, flips, x_raw, mu, gn):
	# y: rotated unit rows (N,8,128); flips broadcastable to y; quantize flipped coords,
	# unflip, exact-L2 norm correction, raw-domain MSE.
	ya = y * flips
	mid = 0.5 * (cent[:-1] + cent[1:])
	q = cent[np.searchsorted(mid, ya)] * flips
	rn = np.linalg.norm(q, axis=2, keepdims=True)
	cn = np.where(rn > 1e-10, gn / rn, gn)
	rec = (q @ ROT.T) * cn + mu.reshape(8, D)
	return float(((rec - x_raw.reshape(-1, 8, D)) ** 2).mean())

def main(dump_dir):
	data = {}
	for side in ("k", "v"):
		for path in sorted(glob.glob(os.path.join(dump_dir, f"{side}_l*_w1024.f32"))):
			layer = int(os.path.basename(path).split("_l")[1][:3])
			raw = np.fromfile(path, dtype=np.float32).reshape(-1, 1024).astype(np.float64)
			n = raw.shape[0]
			mu = raw[: n // 2].mean(axis=0)
			ytr, _ = rot_frame(raw[: n // 2], mu)
			m2 = (ytr ** 2).mean(axis=0)
			sk = (ytr ** 3).mean(axis=0) / m2 ** 1.5      # (8,128) train skew
			data[(side, layer)] = (raw, mu, sk)

	# global sign tables per side
	for side in ("k", "v"):
		sks = np.stack([sk for (s, l), (_, _, sk) in data.items() if s == side])  # (L,8,128)
		gs128 = np.sign(np.median(sks, axis=(0, 1)))       # (128,) pooled layers+heads
		gs1024 = np.sign(np.median(sks, axis=0))           # (8,128) pooled layers only
		cons128 = float((np.sign(sks) == gs128).mean())
		cons1024 = float((np.sign(sks) == gs1024[None]).mean())
		print(f"[{side}] sign consistency: vs global-128 {cons128:.3f}, vs global-(head,coord) {cons1024:.3f}")

		# pooled aligned samples for global books (subsample for speed)
		rng = np.random.default_rng(3)
		pool_al128, pool_al1024, pool_plain = [], [], []
		for (s, l), (raw, mu, sk) in data.items():
			if s != side: continue
			ytr, _ = rot_frame(raw[: raw.shape[0] // 2], mu)
			sel = rng.choice(ytr.shape[0], min(512, ytr.shape[0]), replace=False)
			pool_al128.append((ytr[sel] * gs128).ravel())
			pool_al1024.append((ytr[sel] * gs1024[None]).ravel())
			pool_plain.append(ytr[sel].ravel())
		cb_sym  = lloyd8(np.concatenate(pool_plain), symmetric=True)
		cb_a128 = lloyd8(np.concatenate(pool_al128), symmetric=False)
		cb_a1024= lloyd8(np.concatenate(pool_al1024), symmetric=False)
		print(f"[{side}] global asym book (128-sign):  {np.array2string(cb_a128, precision=4)}")
		print(f"[{side}] global asym book (1024-sign): {np.array2string(cb_a1024, precision=4)}")

		print(f"{'side':4} {'L':>3} {'stock':>8} {'symLl':>7} {'g128':>7} {'g1024':>7} {'perL1024':>8}")
		rs = {k: [] for k in ("sym", "g128", "g1024", "perL")}
		for (s, l), (raw, mu, sk) in sorted(data.items()):
			if s != side: continue
			n = raw.shape[0]
			ev = raw[n // 2 :]
			yev, gn = rot_frame(ev, mu)
			base = codec_mse(yev, CENT, np.ones((1, 1, D)), ev, mu, gn)
			r_sym  = codec_mse(yev, cb_sym,  np.ones((1, 1, D)), ev, mu, gn) / base
			r_128  = codec_mse(yev, cb_a128, gs128[None, None, :], ev, mu, gn) / base
			r_1024 = codec_mse(yev, cb_a1024, gs1024[None], ev, mu, gn) / base
			# per-layer signs (train skew of THIS layer) + per-layer asym book
			flL = np.sign(sk); flL[flL == 0] = 1.0
			ytr, _ = rot_frame(raw[: n // 2], mu)
			cbL = lloyd8((ytr * flL[None]).ravel()[:: max(1, ytr.size // 400000)], symmetric=False)
			r_L = codec_mse(yev, cbL, flL[None], ev, mu, gn) / base
			for k, v in (("sym", r_sym), ("g128", r_128), ("g1024", r_1024), ("perL", r_L)):
				rs[k].append(v)
			print(f"{side:4} {l:3d} {base:8.6f} {r_sym:7.4f} {r_128:7.4f} {r_1024:7.4f} {r_L:8.4f}", flush=True)
		gm = lambda a: float(np.exp(np.mean(np.log(a))))
		print(f"GEOMEAN {side}: symLloyd {gm(rs['sym']):.4f}  asym-g128 {gm(rs['g128']):.4f}  "
		      f"asym-g1024 {gm(rs['g1024']):.4f}  asym-perL {gm(rs['perL']):.4f}  (vs stock, <1 better)")

if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "/root/cert_dump_27b")
