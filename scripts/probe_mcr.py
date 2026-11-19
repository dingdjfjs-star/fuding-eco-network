# -*- coding: utf-8 -*-
"""探针：诊断 MCR 距离场/前驱链为什么全断。只解 1 个源，打印关键量。"""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import rasterio
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

PRI = Path(r"E:\zcode\探索前沿\fuding-eco-priority")
NET = Path(r"E:\zcode\探索前沿\fuding-eco-network")

def rd(p):
    with rasterio.open(p) as s:
        return s.read(1)

wc30 = rd(PRI / "results/rasters/wc30_mode.tif")
src30 = rd(NET / "results/rasters/sources30.tif").astype(np.int32)
res = rd(NET / "results/rasters/res30_base.tif").astype(np.float32)
H, W = wc30.shape
analysis = wc30 > 0
node_id = np.full((H, W), -1, np.int32)
node_id[analysis] = np.arange(int(analysis.sum()))
N = int(analysis.sum()); V = N
rows, cols = np.where(analysis)
print(f"节点 {N}")

# 连通域：analysis 像元 8 邻接是否一体？
st = np.ones((3, 3), bool)
_, ncomp = connected_components(csr_matrix((np.ones(int(analysis.sum())),
                                            (node_id[analysis], node_id[analysis])), shape=(N, N)), directed=False)
# 上面构造的是全对角，不算数——用 ndimage 标记
import scipy.ndimage as ndimage
lab, nc = ndimage.label(analysis, st)
sizes = np.bincount(lab.ravel())[1:]
print(f"analysis 连通域 {nc} 个，最大 {sizes.max()} 像元")
big = np.argmax(sizes) + 1
in_big = lab == big
print(f"analysis 中不在最大域的像元 {int((analysis & ~in_big).sum())}")

src_ids = [int(i) for i in np.unique(src30) if i > 0]
src_px = {sid: node_id[src30 == sid] for sid in src_ids}
for sid in src_ids:
    labs = np.unique(lab[src30 == sid])
    print(f"源 {sid}: {len(src_px[sid])} 像元, 所在analysis域 {labs.tolist()}")

# 邻接（8邻，仅最大域内？不——按 analysis 全建，看 V 到各源是否可达）
OFFS = [(-1, 0, 30.0), (1, 0, 30.0), (0, -1, 30.0), (0, 1, 30.0),
        (-1, -1, 45.0), (-1, 1, 45.0), (1, -1, 45.0), (1, 1, 45.0)]
r0s, c0s, r1s, c1s, lens = [], [], [], [], []
for dy, dx, L in OFFS:
    rr0 = rows[(rows + dy >= 0) & (rows + dy < H) & (cols + dx >= 0) & (cols + dx < W)]
    cc0 = cols[(rows + dy >= 0) & (rows + dy < H) & (cols + dx >= 0) & (cols + dx < W)]
    ok = analysis[rr0 + dy, cc0 + dx]
    r0s.append(rr0[ok]); c0s.append(cc0[ok]); r1s.append(rr0[ok] + dy); c1s.append(cc0[ok] + dx)
    lens.append(np.full(int(ok.sum()), L, np.float32))
r0 = np.concatenate(r0s); c0 = np.concatenate(c0s)
r1 = np.concatenate(r1s); c1 = np.concatenate(c1s)
L = np.concatenate(lens)
n0 = node_id[r0, c0]; n1 = node_id[r1, c1]
print(f"边 {len(n0)}")

w = 0.5 * (res[r0, c0] + res[r1, c1]) * L
base = csr_matrix((np.concatenate([w, w]), (np.concatenate([n0, n1]), np.concatenate([n1, n0]))),
                  shape=(N + 1, N + 1), dtype=np.float32)
sid_s = src_ids[0]
px = src_px[sid_s]
vrow = np.concatenate([np.full(len(px), V), px])
vcol = np.concatenate([px, np.full(len(px), V)])
adj = base + csr_matrix((np.zeros(len(vrow), np.float32), (vrow, vcol)), shape=(N + 1, N + 1), dtype=np.float32)
print(f"adj nnz={adj.nnz}, 显式0个数={int((adj.data == 0).sum())}")
d, p = dijkstra(adj, directed=False, indices=V, return_predecessors=True)
print(f"d: min={d.min():.1f} max={d.max():.1f} inf个数={int(np.isinf(d).sum())}")
tgt = src_ids[1]
cand = src_px[tgt]
dv = d[cand]
print(f"源{sid_s}->源{tgt}: d min={dv.min():.1f} inf占比={float(np.isinf(dv).mean()):.2f}")
t = int(cand[np.argmin(dv)])
chain, node, guard = [], t, 0
while guard < 50:
    q = int(p[node])
    chain.append(q)
    if q < 0 or q == V or q == node:
        break
    node = q; guard += 1
print(f"前驱链(前50步): {chain}")
