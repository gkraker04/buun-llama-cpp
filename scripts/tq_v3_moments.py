#!/usr/bin/env python3
# Higher-moment tower on the post-FWHT codec frame (follow-up: C4-null does not imply
# C6/C8-null; check the whole ladder against BOTH nulls, Gaussian and sphere-marginal).
# Plus the cash test: does an 8-level Lloyd-Max retrained on the layer's true pooled
# marginal beat the stock N(0,1/128)-designed centroids in faithful codec-sim MSE?
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
	return (g / np.where(gn > 1e-10, gn, 1.0)) @ ROT

def lloyd8(x, iters=60):
	# symmetric 8-level Lloyd-Max on pooled samples
	c = np.quantile(x, np.linspace(0.0625, 0.9375, 8))
	for _ in range(iters):
		mid = 0.5 * (c[:-1] + c[1:])
		idx = np.searchsorted(mid, x)
		newc = np.array([x[idx == i].mean() if np.any(idx == i) else c[i] for i in range(8)])
		newc = 0.5 * (newc - newc[::-1])   # enforce odd symmetry
		if np.max(np.abs(newc - c)) < 1e-7:
			c = newc; break
		c = newc
	return c

def codec_mse(y, cent, x_raw, mu, gn):
	# y: rotated unit rows (N,8,128); quantize with cent, exact-L2 norm correction, raw MSE
	mid = 0.5 * (cent[:-1] + cent[1:])
	q = cent[np.searchsorted(mid, y)]
	rn = np.linalg.norm(q, axis=2, keepdims=True)
	cn = np.where(rn > 1e-10, gn / rn, gn)
	rec = (q @ ROT.T) * cn + mu.reshape(8, D)
	return float(((rec - x_raw.reshape(-1, 8, D)) ** 2).mean())

# nulls: normalized even moments m_p/m2^{p/2}
GAUSS = {4: 3.0, 6: 15.0, 8: 105.0}
SPH   = {4: 3.0 * D / (D + 2),
         6: 15.0 * D * D / ((D + 2) * (D + 4)),
         8: 105.0 * D ** 3 / ((D + 2) * (D + 4) * (D + 6))}

def main(dump_dir):
	print(f"sphere nulls: K4={SPH[4]:.4f} K6={SPH[6]:.4f} K8={SPH[8]:.4f}  (gauss 3/15/105)")
	print(f"{'side':4} {'L':>3} {'skew|med|':>9} {'K4med':>7} {'K6med':>7} {'K8med':>8} "
	      f"{'K4s_med':>7} {'K6s_med':>7} {'K8s_med':>8} {'K6s_p95':>8} {'K8s_p95':>9} {'lloydMSE':>8}")
	for side in ("k", "v"):
		r6 = []; r8 = []; rl = []
		for path in sorted(glob.glob(os.path.join(dump_dir, f"{side}_l*_w1024.f32"))):
			layer = int(os.path.basename(path).split("_l")[1][:3])
			raw = np.fromfile(path, dtype=np.float32).reshape(-1, 1024).astype(np.float64)
			n = raw.shape[0]
			mu = raw[: n // 2].mean(axis=0)
			y = rot_frame(raw, mu)                     # full set for moment estimation
			m2 = (y ** 2).mean(axis=0)
			nrm = {p: (y ** p).mean(axis=0) / m2 ** (p / 2) for p in (3, 4, 5, 6, 8)}
			skew = np.abs(nrm[3])
			exG = {p: nrm[p] - GAUSS[p] for p in (4, 6, 8)}
			exS = {p: nrm[p] - SPH[p] for p in (4, 6, 8)}
			# cash test on eval half with train-half Lloyd
			ytr = rot_frame(raw[: n // 2], mu); yev = rot_frame(raw[n // 2 :], mu)
			cst = lloyd8(ytr.ravel())
			ev_raw = raw[n // 2 :]
			g = ev_raw.reshape(-1, 8, D) - mu.reshape(8, D)
			gn = np.linalg.norm(g, axis=2, keepdims=True)
			m_stock = codec_mse(yev, CENT, ev_raw, mu, gn)
			m_lloyd = codec_mse(yev, cst, ev_raw, mu, gn)
			ratio = m_lloyd / m_stock
			r6.append(np.median(exS[6])); r8.append(np.median(exS[8])); rl.append(ratio)
			print(f"{side:4} {layer:3d} {np.median(skew):9.4f} {np.median(exG[4]):7.3f} {np.median(exG[6]):7.2f} {np.median(exG[8]):8.1f} "
			      f"{np.median(exS[4]):7.3f} {np.median(exS[6]):7.2f} {np.median(exS[8]):8.1f} "
			      f"{np.percentile(exS[6],95):8.2f} {np.percentile(exS[8],95):9.1f} {ratio:8.4f}", flush=True)
		print(f"SUMMARY {side}: median K6_sphere-excess {np.median(r6):+.2f}, K8 {np.median(r8):+.1f}, "
		      f"lloyd-retrain MSE geomean {np.exp(np.mean(np.log(rl))):.4f} (<1 = marginal-shape money exists)")

if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "/root/cert_dump_27b")
