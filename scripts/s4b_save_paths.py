# -*- coding: utf-8 -*-
"""
阶段4b：重解 base 阻力场的逐对廊道路径并落盘（npz）。
s4_mcr 只保存了廊道栅格（计数/掩膜），没保存"哪对源的路径经过哪些像元"；
阶段5 障碍点重算需要"窗口 -> 受影响廊道对"的映射，这里补齐。
输出：results/tables/corridor_paths.npz（paths/ends/pair_ids + 行列坐标）
"""
import json, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import rasterio
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

PRI = Path(r"E:\zcode\探索前沿\fuding-eco-priority")
NET = Path(r"E:\zcode\探索前沿\fuding-eco-network")
OUT_T = NET / "results" / "tables"

grid = json.loads((PRI / "results/tables/grid_spec.json").read_text(encoding="utf-8"))
H, W = grid["height"], grid["width"]

def rd(p):
    with rasterio.open(p) as s:
        return s.read(1)

wc30 = rd(PRI / "results/rasters/wc30_mode.tif")
src30 = rd(NET / "results/rasters/sources30.tif").astype(np.int32)
res = rd(NET / "results/rasters/res30_base.tif").astype(np.float32)
analysis = wc30 > 0

node_id = np.full((H, W), -1, np.int32)
node_id[analysis] = np.arange(int(analysis.sum()))
N = int(analysis.sum()); V = N
rows, cols = np.where(analysis)
OFFS = [(-1, 0, 30.0), (1, 0, 30.0), (0, -1, 30.0), (0, 1, 30.0),
        (-1, -1, 30 * 2 ** .5), (-1, 1, 30 * 2 ** .5), (1, -1, 30 * 2 ** .5), (1, 1, 30 * 2 ** .5)]
r0s, c0s, r1s, c1s, lens = [], [], [], [], []
for dy, dx, L in OFFS:
    rr0 = rows[(rows + dy >= 0) & (rows + dy < H) & (cols + dx >= 0) & (cols + dx < W)]
    cc0 = cols[(rows + dy >= 0) & (rows + dy < H) & (cols + dx >= 0) & (cols + dx < W)]
    ok = analysis[rr0 + dy, cc0 + dx]
    r0s.append(rr0[ok]); c0s.append(cc0[ok])
    r1s.append(rr0[ok] + dy); c1s.append(cc0[ok] + dx)
    lens.append(np.full(int(ok.sum()), L, np.float32))
r0 = np.concatenate(r0s); c0 = np.concatenate(c0s)
r1 = np.concatenate(r1s); c1 = np.concatenate(c1s)
L = np.concatenate(lens)
n0 = node_id[r0, c0]; n1 = node_id[r1, c1]
both = (n0 >= 0) & (n1 >= 0)
n0, n1, L = n0[both], n1[both], L[both]

src_ids = [int(i) for i in np.unique(src30) if i > 0]
src_px = {sid: node_id[src30 == sid] for sid in src_ids}

w = 0.5 * (res[r0, c0] + res[r1, c1]) * L
base = csr_matrix((np.concatenate([w, w]),
                   (np.concatenate([n0, n1]), np.concatenate([n1, n0]))),
                  shape=(N + 1, N + 1), dtype=np.float32)
print(f"[OK] 图就绪：节点 {N}，边 {len(n0)}")

fields = {}
t0 = time.time()
for sid in src_ids:
    px = src_px[sid]                               # 虚拟源只接当前源（与 s4 修复一致）
    vrow = np.concatenate([np.full(len(px), V), px])
    vcol = np.concatenate([px, np.full(len(px), V)])
    # ε 权重：scipy CSR 加法削显式 0 权边（见 s4 修复注释）
    adj = base + csr_matrix((np.full(len(vrow), 1e-3, np.float32), (vrow, vcol)),
                            shape=(N + 1, N + 1), dtype=np.float32)
    d, p = dijkstra(adj, directed=False, indices=V, return_predecessors=True)
    fields[sid] = (d, p)
print(f"[OK] base {len(src_ids)} 源 Dijkstra {time.time()-t0:.0f}s")

paths, ends, pairs = [], [], []
for a in range(len(src_ids)):
    for b in range(a + 1, len(src_ids)):
        sa, sb = src_ids[a], src_ids[b]
        d, p = fields[sa]
        cand = src_px[sb]
        t = int(cand[np.argmin(d[cand])])
        path, node = [t], t
        while True:
            q = int(p[node])
            if q < 0 or q == V or q == node:
                break
            path.append(q); node = q
        if len(path) < 2:
            continue
        paths.append(np.array(path, np.int32))
        ends.append((node_id[rows[path[-1]], cols[path[-1]]],
                     node_id[rows[path[0]], cols[path[0]]]))  # (起点侧像元, 目标侧像元)
        pairs.append([sa, sb])
paths_arr = np.concatenate(paths)
splits = np.cumsum([len(x) for x in paths])[:-1]
np.savez_compressed(OUT_T / "corridor_paths.npz",
                    node_idx=paths_arr, splits=splits,
                    pairs=np.array(pairs, np.int32),
                    end_start=np.array([e[0] for e in ends], np.int32),
                    end_end=np.array([e[1] for e in ends], np.int32),
                    rows=rows.astype(np.int32), cols=cols.astype(np.int32))
print(f"[DONE] s4b: {len(pairs)} 对路径已落盘 corridor_paths.npz")
