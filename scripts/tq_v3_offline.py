#!/usr/bin/env python3
# TorQuant V3 delta instruments on real 27B KV rows (/root/cert_dump_27b).
# Rows are raw pre-FWHT (post-RoPE) fp32, w=1024 (8 heads x 128), 4096 rows per (layer, K/V).
#
# Instruments per (layer, side):
#   - post-FWHT per-coordinate variance spread + Lemma 2.1 gauge-prize ratio (uniform-w proxy)
#   - Thm 3.5 Gaussian gap: excess kurtosis of post-FWHT coords
#   - Def 6.4 kappa amplitude coupling on rotated coord pairs
#   - faithful turbo3_0 codec sim MSE: base / tap / tap+S2std / tap+S2q999 / S2std
#     (S2 = per-coordinate dilation in the post-FWHT frame, undone at decode; norm
#      correction keeps exact L2 like the kernel, so only centroid hit pattern changes)
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
MID  = np.array([-0.154259,-0.091775,-0.043589,0.0,0.043589,0.091775,0.154259])

def hadamard(n):
	H = np.array([[1.0]])
	while H.shape[0] < n:
		H = np.block([[H, H], [H, -H]])
	return H

H128 = hadamard(D) * INV
ROT = (np.diag(S1) @ H128 @ np.diag(S2)).T  # y = (x*S1) H * S2 = x @ ROT (row-vector conv)

def rot_fwd(x):   # x: (..., 128)
	return x @ ROT

def rot_inv(y):
	return y @ ROT.T

def quantize3(y):
	# nearest of 8 centroids via midpoints
	idx = np.searchsorted(MID, y)
	return CENT[idx]

def codec(x, mu=None, d=None):
	# x: (N,128) raw group rows. Returns reconstruction in raw domain.
	xc = x - mu if mu is not None else x
	gn = np.linalg.norm(xc, axis=1, keepdims=True)
	gn_safe = np.where(gn > 1e-10, gn, 1.0)
	y = rot_fwd(xc / gn_safe)
	if d is not None:
		y = y * d
		# renormalize (absorbed exactly by the stored norm correction)
		g2 = np.linalg.norm(y, axis=1, keepdims=True)
		y = y / np.where(g2 > 1e-10, g2, 1.0)
	q = quantize3(y)
	if d is not None:
		q = q / d
	rn = np.linalg.norm(q, axis=1, keepdims=True)
	cn = np.where(rn > 1e-10, gn / rn, gn)   # corrected norm: exact L2 restore
	rec = rot_inv(q) * cn
	if mu is not None:
		rec = rec + mu
	return rec

def analyze(path, side, layer):
	raw = np.fromfile(path, dtype=np.float32).reshape(-1, 1024).astype(np.float64)
	n = raw.shape[0]
	tr, ev = raw[: n // 2], raw[n // 2 :]
	res = {}

	# calibration (train half): raw-domain mu per channel (the tap), per (head,coord)
	mu = tr.mean(axis=0)

	# post-FWHT frame stats on the eval half, tap applied, unit-normalized (codec frame)
	def rot_frame(block, use_mu):
		g = block.reshape(-1, 8, D) - (mu.reshape(8, D) if use_mu else 0.0)
		gn = np.linalg.norm(g, axis=2, keepdims=True)
		g = g / np.where(gn > 1e-10, gn, 1.0)
		return rot_fwd(g)   # (N,8,128)

	ytr = rot_frame(tr, True)
	yev = rot_frame(ev, True)

	M = (ytr ** 2).mean(axis=0)              # (8,128) per-coord second moment
	res["spread_maxmean"] = float((M.max(axis=1) / M.mean(axis=1)).mean())  # Lemma 2.1 prize ratio, uniform w
	res["std_p99_p1"] = float(np.mean(np.percentile(np.sqrt(M), 99, axis=1) / np.percentile(np.sqrt(M), 1, axis=1)))

	# Thm 3.5 Gaussian gap: excess kurtosis per rotated coord (eval half)
	m2 = (yev ** 2).mean(axis=0); m4 = (yev ** 4).mean(axis=0)
	kurt = m4 / (m2 ** 2) - 3.0
	res["kurt_med"] = float(np.median(kurt)); res["kurt_p95"] = float(np.percentile(kurt, 95)); res["kurt_max"] = float(kurt.max())

	# Def 6.4 kappa on 512 random in-head pairs (eval half)
	rng = np.random.default_rng(7)
	ks = []
	for _ in range(512):
		h = rng.integers(0, 8); i, j = rng.choice(D, 2, replace=False)
		a, b = yev[:, h, i], yev[:, h, j]
		mii, mjj = (a * a).mean(), (b * b).mean()
		rho = (a * b).mean() / np.sqrt(mii * mjj)
		m22 = ((a * a) * (b * b)).mean()
		ks.append(m22 / (mii * mjj) - 1.0 - 2.0 * rho * rho)
	ks = np.array(ks)
	res["kappa_med"] = float(np.median(ks)); res["kappa_p95"] = float(np.percentile(ks, 95)); res["kappa_max"] = float(ks.max())

	# S2 calibrations from train half (tap frame)
	s_std = np.sqrt(M)                                   # (8,128)
	d_std = (s_std.mean(axis=1, keepdims=True) / s_std)  # flatten std to head mean
	q999 = np.quantile(np.abs(ytr), 0.999, axis=0)
	d_q = (q999.mean(axis=1, keepdims=True) / q999)

	# codec sim on eval half, raw-domain MSE per variant
	x = ev.reshape(-1, 8, D)
	muh = mu.reshape(8, D)
	def mse(rec):
		return float(((rec - x) ** 2).mean())
	def run(use_mu, d):
		g = x.reshape(-1, D)
		m = np.repeat(muh[None], x.shape[0], axis=0).reshape(-1, D) if use_mu else None
		dd = None
		if d is not None:
			dd = np.repeat(d[None], x.shape[0], axis=0).reshape(-1, D)
		return mse(codec(g, m, dd).reshape(-1, 8, D))
	base = run(False, None)
	res["mse_base"] = base
	res["r_tap"] = run(True, None) / base
	res["r_tap_s2std"] = run(True, d_std) / base
	res["r_tap_s2q"] = run(True, d_q) / base
	res["r_s2std"] = run(False, d_std) / base
	return res

def main(dump_dir):
	print(f"{'side':4} {'L':>3} {'prize':>6} {'p99/p1':>6} {'kurt_med':>8} {'kurt_p95':>8} {'kap_med':>8} {'kap_p95':>7} "
	      f"{'tap':>6} {'tap+S2s':>7} {'tap+S2q':>7} {'S2s':>6}")
	agg = {}
	for side in ("k", "v"):
		for path in sorted(glob.glob(os.path.join(dump_dir, f"{side}_l*_w1024.f32"))):
			layer = int(os.path.basename(path).split("_l")[1][:3])
			r = analyze(path, side, layer)
			print(f"{side:4} {layer:3d} {r['spread_maxmean']:6.2f} {r['std_p99_p1']:6.2f} {r['kurt_med']:8.3f} {r['kurt_p95']:8.3f} "
			      f"{r['kappa_med']:8.3f} {r['kappa_p95']:7.2f} {r['r_tap']:6.4f} {r['r_tap_s2std']:7.4f} {r['r_tap_s2q']:7.4f} {r['r_s2std']:6.4f}",
			      flush=True)
			agg.setdefault(side, []).append(r)
	for side, rs in agg.items():
		def gm(key):
			return float(np.exp(np.mean(np.log([r[key] for r in rs]))))
		print(f"GEOMEAN {side}: tap {gm('r_tap'):.4f}  tap+S2std {gm('r_tap_s2std'):.4f}  "
		      f"tap+S2q {gm('r_tap_s2q'):.4f}  S2std {gm('r_s2std'):.4f}  (MSE ratio vs base, <1 better)")

if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "/root/cert_dump_27b")
