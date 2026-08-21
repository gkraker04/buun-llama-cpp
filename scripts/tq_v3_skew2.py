#!/usr/bin/env python3
# Odd-moment follow-up: (1) per-(head,coord) ASYMMETRIC-book ceiling (max collectable MSE),
# (2) stock-book per-coordinate reconstruction BIAS — the KLD-relevant quantity MSE can't
# see. K bias is softmax-absorbed (free); V bias folds into the V add-back table, so its
# raw-domain magnitude vs the existing mu_V correction sizes the implementable prize.
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
MID = 0.5 * (CENT[:-1] + CENT[1:])

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

def lloyd_percoord(x, iters=25):
	# x: (N, C) -> books (C, 8), asymmetric, vectorized across coords
	c = np.quantile(x, np.linspace(0.0625, 0.9375, 8), axis=0).T   # (C,8)
	for _ in range(iters):
		mid = 0.5 * (c[:, :-1] + c[:, 1:])                          # (C,7)
		idx = np.zeros(x.shape, dtype=np.int8)
		for m in range(7):
			idx += (x > mid[None if mid.ndim == 1 else slice(None), :][:, m][None, :] if False else (x > mid[:, m][None, :])).astype(np.int8)
		newc = c.copy()
		for lv in range(8):
			msk = idx == lv
			cnt = msk.sum(axis=0)
			s = np.where(msk, x, 0.0).sum(axis=0)
			upd = cnt > 0
			newc[upd, lv] = s[upd] / cnt[upd]
		newc = np.sort(newc, axis=1)
		if np.max(np.abs(newc - c)) < 1e-8:
			c = newc; break
		c = newc
	return c

def quant_percoord(x, books):
	mid = 0.5 * (books[:, :-1] + books[:, 1:])   # (C,7)
	idx = np.zeros(x.shape, dtype=np.int8)
	for m in range(7):
		idx += (x > mid[:, m][None, :]).astype(np.int8)
	return np.take_along_axis(np.broadcast_to(books[None], (x.shape[0],) + books.shape), idx[..., None].astype(np.int64), axis=2)[..., 0]

def main(dump_dir):
	print(f"{'side':4} {'L':>3} {'stockMSE':>9} {'ceil1024':>8} {'|bias|/std med':>14} {'p95':>6} {'Vbias/mu':>8} {'Vbias/row':>9}")
	for side in ("k", "v"):
		ceil_r, bias_rel = [], []
		for path in sorted(glob.glob(os.path.join(dump_dir, f"{side}_l*_w1024.f32"))):
			layer = int(os.path.basename(path).split("_l")[1][:3])
			raw = np.fromfile(path, dtype=np.float32).reshape(-1, 1024).astype(np.float64)
			n = raw.shape[0]
			mu = raw[: n // 2].mean(axis=0)
			ytr, _ = rot_frame(raw[: n // 2], mu)
			yev, gn = rot_frame(raw[n // 2 :], mu)
			ev = raw[n // 2 :]

			flat_tr = ytr.reshape(-1, 8 * D)
			flat_ev = yev.reshape(-1, 8 * D)

			# stock codec (per-coord shared book) baseline in raw domain
			def to_raw(qflat):
				q = qflat.reshape(-1, 8, D)
				rn = np.linalg.norm(q, axis=2, keepdims=True)
				cn = np.where(rn > 1e-10, gn / rn, gn)
				return (q @ ROT.T) * cn + mu.reshape(8, D)
			q_stock = CENT[np.searchsorted(MID, flat_ev)]
			mse_stock = float(((to_raw(q_stock) - ev.reshape(-1, 8, D)) ** 2).mean())

			# per-(head,coord) asymmetric Lloyd ceiling (train->eval)
			books = lloyd_percoord(flat_tr)
			q_ceil = quant_percoord(flat_ev, books)
			mse_ceil = float(((to_raw(q_ceil) - ev.reshape(-1, 8, D)) ** 2).mean())
			ceil_r.append(mse_ceil / mse_stock)

			# stock-book bias per coordinate, rotated frame (train est) + relative size
			q_tr = CENT[np.searchsorted(MID, flat_tr)]
			bias = (q_tr - flat_tr).mean(axis=0)                     # (1024,)
			stds = flat_tr.std(axis=0)
			rel = np.abs(bias) / np.where(stds > 0, stds, 1)
			bias_rel.append(np.median(rel))

			# raw-domain V-bias fold candidate: inv-rot(bias * mean corrected norm) per head
			mean_cn = gn.mean()
			b_rot = bias.reshape(8, D) * mean_cn
			b_raw = b_rot @ ROT.T                                    # (8,128) raw-domain bias
			mu_n = np.linalg.norm(mu.reshape(8, D), axis=1).mean()
			row_n = np.linalg.norm(ev.reshape(-1, 8, D), axis=2).mean()
			bn = np.linalg.norm(b_raw, axis=1).mean()
			print(f"{side:4} {layer:3d} {mse_stock:9.6f} {mse_ceil/mse_stock:8.4f} {np.median(rel):14.4f} "
			      f"{np.percentile(rel,95):6.3f} {bn/max(mu_n,1e-9):8.4f} {bn/row_n:9.5f}", flush=True)
		gm = lambda a: float(np.exp(np.mean(np.log(a))))
		print(f"SUMMARY {side}: per-coord asym ceiling geomean {gm(ceil_r):.4f}, median |bias|/std {np.median(bias_rel):.4f}")

if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "/root/cert_dump_27b")
