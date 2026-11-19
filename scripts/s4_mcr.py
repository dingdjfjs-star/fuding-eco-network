# -*- coding: utf-8 -*-
"""
阶段4 MCR 生态廊道提取（最小累积阻力路径）。
方法：8 邻接稀疏图 + scipy.csgraph.dijkstra，边权 = mean(阻力a,阻力b)×步长(m)；
每个源斑块接 0 成本虚拟超级源 -> 一次单源 Dijkstra 得到"该源到全场"的最小累积阻力场；
源地两两走廊由 predecessor 回溯重构（13 源 → 每方案 13 次求解 → C(13,2)=78 对走廊）。
敏感性：base/low/high 三套阻力面各跑一遍，90m 缓冲 Jaccard 同对重合率。
输出：corridor_count30.tif / corridor_mask30.tif / stage4_mcr.json / fig_corridors.png
"""
import json, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import rasterio
import scipy.ndimage as ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0, str(Path(__file__).parent))
from carto import extent_of, add_north, add_scalebar, add_note
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRI = Path(r"E:\zcode\探索前沿\fuding-eco-priority")
NET = Path(r"E:\zcode\探索前沿\fuding-eco-network")
OUT_R, OUT_T, OUT_F = NET / "results" / "rasters", NET / "results" / "tables", NET / "results" / "figures"

grid = json.loads((PRI / "results/tables/grid_spec.json").read_text(encoding="utf-8"))
H, W, TR30 = grid["height"], grid["width"], tuple(grid["transform"])

def rd(p):
    with rasterio.open(p) as s:
        return s.read(1)

wc30 = rd(PRI / "results/rasters/wc30_mode.tif")
src30 = rd(NET / "results/rasters/sources30.tif").astype(np.int32)
analysis = wc30 > 0

# ---- 图结构（只建一次）：节点=分析范围内像元，边=8 邻接 ------------------------
node_id = np.full((H, W), -1, np.int32)
node_id[analysis] = np.arange(int(analysis.sum()))
N = int(analysis.sum()); V = N                    # V=虚拟超级源节点
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
print(f"[OK] 节点 {N} + 虚拟源1，无向边 {len(n0)}")

src_ids = [int(i) for i in np.unique(src30) if i > 0]
src_px = {sid: node_id[src30 == sid] for sid in src_ids}
print(f"[OK] 源地 {len(src_ids)} 个: {src_ids}")

def build_base(res):
    """拓扑固定的基础邻接矩阵（不含虚拟源），权重随阻力面变。"""
    w = 0.5 * (res[r0, c0] + res[r1, c1]) * L      # MCR 半边权：两端均值×步长
    return csr_matrix((np.concatenate([w, w]),
                       (np.concatenate([n0, n1]), np.concatenate([n1, n0]))),
                      shape=(N + 1, N + 1), dtype=np.float32)

def solve(res, name):
    """阻力面 -> 逐源 Dijkstra -> {源id: (dist,pred)}。
    关键修复：虚拟超级源每次只接"当前源"的像元。此前接了全部源地，
    等于所有源共用一张全场图——目标像元自身 0 成本挂在 V 上，回溯一步就断，
    120 对廊道全退化成 1 像元（已实测失败留痕）。"""
    base = build_base(res)
    fields = {}
    t0 = time.time()
    for sid in src_ids:
        px = src_px[sid]
        vrow = np.concatenate([np.full(len(px), V), px])
        vcol = np.concatenate([px, np.full(len(px), V)])
        # 权重用 ε=0.001 而非 0：scipy CSR 加法会"规范化"削掉显式 0 权边（探针实测
        # adj nnz 不增、V 变孤岛、全场 inf），ε 成本对廊道量级(数千)可忽略
        adj = base + csr_matrix((np.full(len(vrow), 1e-3, np.float32), (vrow, vcol)),
                                shape=(N + 1, N + 1), dtype=np.float32)
        d, p = dijkstra(adj, directed=False, indices=V, return_predecessors=True)
        fields[sid] = (d, p)
    print(f"[OK] {name}: {len(src_ids)} 次全图 Dijkstra，用时 {time.time()-t0:.0f}s")
    return fields

def backtrack(fields, sid_s, sid_t):
    """目标斑块中 MCR 最小像元 -> 回溯到源斑块（起点侧）像元，返回像元索引列表。"""
    d, p = fields[sid_s]
    cand = src_px[sid_t]
    t = int(cand[np.argmin(d[cand])])
    path, node, mcr = [t], t, float(d[t])
    while True:
        q = int(p[node])
        if q < 0 or q == V or q == node:
            break
        path.append(q); node = q
    return path, mcr

def path_len_m(path):
    rr, cc = rows[path], cols[path]
    d = np.abs(np.diff(rr)) + np.abs(np.diff(cc))
    return float(np.sum(np.where(d == 2, 30 * 2 ** .5, 30.0)))

def paths_of(res, name):
    fields = solve(res, name)
    paths = {}
    for a in range(len(src_ids)):
        for b in range(a + 1, len(src_ids)):
            p, m = backtrack(fields, src_ids[a], src_ids[b])
            if len(p) >= 2:
                paths[(src_ids[a], src_ids[b])] = (p, m)
    return paths

LUC = {10: "林地", 20: "灌木", 30: "草地", 40: "耕地/茶园", 50: "建成区", 60: "裸地", 80: "水体", 90: "湿地"}
res_base = rd(NET / "results/rasters/res30_base.tif").astype(np.float32)
res_low = rd(NET / "results/rasters/res30_low.tif").astype(np.float32)
res_high = rd(NET / "results/rasters/res30_high.tif").astype(np.float32)

PB = paths_of(res_base, "base")
PL = paths_of(res_low, "low")
PH = paths_of(res_high, "high")

# ---- base 廊道栅格与逐对统计 ---------------------------------------------------
cnt = np.zeros((H, W), np.int16); mask = np.zeros((H, W), bool)
recs = []
for (a, b), (p, m) in sorted(PB.items(), key=lambda kv: -kv[1][1]):
    rr, cc = rows[p], cols[p]
    cnt[rr, cc] += 1; mask[rr, cc] = True
    vals, counts = np.unique(wc30[rr, cc], return_counts=True)
    lu = {LUC.get(int(v), str(int(v))): round(float(c) / len(p) * 100, 1)
          for v, c in zip(vals, counts) if c / len(p) >= 0.02}
    recs.append(dict(a=a, b=b, mcr=round(m, 0), length_m=round(path_len_m(p), 0),
                     px=len(p), landuse=lu))
recs.sort(key=lambda r: -r["mcr"])

meta = dict(driver="GTiff", height=H, width=W, count=1, dtype="int16",
            crs=grid["crs"], transform=TR30, nodata=0, compress="deflate")
with rasterio.open(OUT_R / "corridor_count30.tif", "w", **meta) as d:
    d.write(cnt, 1)
with rasterio.open(OUT_R / "corridor_mask30.tif", "w", **meta) as d:
    d.write(mask.astype(np.int16), 1)

# ---- 敏感性：90m 缓冲 Jaccard + 长度变化 --------------------------------------
def mask_of(path):
    m = np.zeros((H, W), bool); m[rows[path], cols[path]] = True
    return ndimage.binary_dilation(m, np.ones((3, 3), bool))   # ±1 像元 ≈ 90m 缓冲带

sens = []
for k, (p_b, m_b) in PB.items():
    if k not in PL or k not in PH:
        continue
    A, B, C = mask_of(p_b), mask_of(PL[k][0]), mask_of(PH[k][0])
    l_b, l_l, l_h = path_len_m(p_b), path_len_m(PL[k][0]), path_len_m(PH[k][0])
    sens.append(dict(pair=[k[0], k[1]],
                     jaccard_low=round(float((A & B).sum()) / max((A | B).sum(), 1), 3),
                     jaccard_high=round(float((A & C).sum()) / max((A | C).sum(), 1), 3),
                     len_change_low_pct=round((l_l - l_b) / max(l_b, 1) * 100, 1),
                     len_change_high_pct=round((l_h - l_b) / max(l_b, 1) * 100, 1)))

def agg_lu(recs_):
    agg = {}
    for r in recs_:
        for kk, v in r["landuse"].items():
            agg[kk] = agg.get(kk, 0) + v * r["px"]
    tot = sum(agg.values())
    return {kk: round(v / tot * 100, 1) for kk, v in sorted(agg.items(), key=lambda x: -x[1])}

lens_base = [r["length_m"] for r in recs]
out = dict(
    method="8邻接稀疏图Dijkstra；边权=mean(阻力)*步长m；虚拟超级源多起点；predecessor回溯；MCR单位=阻力·m",
    sources=src_ids, n_pairs=len(recs),
    corridors_base=recs, landuse_all_base=agg_lu(recs),
    length_stats=dict(total_km=round(sum(lens_base) / 1000, 1),
                      mean_km=round(float(np.mean(lens_base)) / 1000, 2),
                      max_km=round(max(lens_base) / 1000, 2)),
    sensitivity=dict(design="道路+居民点阻力×0.5(low)/×2.0(high)；90m缓冲同对Jaccard",
                     n=len(sens),
                     jaccard_low_mean=round(float(np.mean([s["jaccard_low"] for s in sens])), 3),
                     jaccard_high_mean=round(float(np.mean([s["jaccard_high"] for s in sens])), 3),
                     len_change_low_mean_pct=round(float(np.mean([s["len_change_low_pct"] for s in sens])), 1),
                     len_change_high_mean_pct=round(float(np.mean([s["len_change_high_pct"] for s in sens])), 1),
                     per_pair=sens),
    note="走廊两端点取自源地内最小累积阻力像元；穿过大斑块内部为低成本段属正常")
(OUT_T / "stage4_mcr.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- 出图 ----------------------------------------------------------------------
from rasterio.transform import Affine
ext = extent_of(Affine(*TR30[:6]), W, H)
bg = np.full((H, W, 3), (246, 246, 242), np.uint8)
bg[wc30 == 80] = (205, 227, 242); bg[wc30 == 0] = (228, 228, 228)
fig, ax = plt.subplots(figsize=(9.5, 7.4), dpi=150)
ax.imshow(bg, extent=ext, interpolation="nearest")
show = np.where(mask, 1.0, np.nan)
import matplotlib.colors as mcolors
cmap = mcolors.ListedColormap(["#e29a3d"])
ax.imshow(np.ma.masked_invalid(show), extent=ext, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
show2 = np.ma.masked_where(src30 == 0, src30)
ax.imshow(show2, extent=ext, cmap="Greens", interpolation="nearest", alpha=0.9)
road = rd(NET / "results/rasters/road_level30.tif")
ax.contour(np.where(road == 3, 1.0, 0.0), levels=[0.5], colors="red", linewidths=0.7, extent=ext)
ax.set_title(f"福鼎市 MCR 生态廊道（n={len(recs)} 对源地廊道，红线=高速/主干道路）", fontsize=12)
ax.legend([plt.Rectangle((0, 0), 1, 1, fc="#e29a3d"), plt.Rectangle((0, 0), 1, 1, fc="green"),
           plt.Line2D([], [], color="red")],
          ["MCR廊道", "生态源地", "3级道路"], loc="lower right", fontsize=8, framealpha=0.9)
add_north(ax); add_scalebar(ax, ext, km=10)
add_note(ax, "坐标系: EPSG:32651 (UTM 51N) · 廊道=MCR最小累积阻力路径", "bl")
ax.set_xlabel("东向 UTM (m)"); ax.set_ylabel("北向 UTM (m)")
fig.tight_layout(); fig.savefig(OUT_F / "fig_corridors.png"); plt.close(fig)
print(f"[DONE] stage4: {len(recs)} 对廊道, 总长 {out['length_stats']['total_km']} km, 平均 {out['length_stats']['mean_km']} km; "
      f"敏感性 Jaccard low/high = {out['sensitivity']['jaccard_low_mean']}/{out['sensitivity']['jaccard_high_mean']}")
