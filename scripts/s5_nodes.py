# -*- coding: utf-8 -*-
"""
阶段5 关键节点识别（电路理论简化口径 + 恢复增量法）。

【夹点 Pinch】廊道网络上的电路模拟（Circuitscape 思路，简化口径）：
  网络节点 = 78 条 MCR 廊道像元并集；源地 = 超节点（高电导接到该源全部廊道入点，
  模拟"源地内部畅通"）；逐源注入 1A、其余源地接地(V=0)——电流沿"并联的多条路径"
  分配，这是电路理论与 MCR 的本质区别：MCR 只认一条最省路，电路把所有可能路径
  都算电流。电流大的像元 = 生态流汇聚处 = 夹点。
【障碍点 Barrier】恢复增量法：廊道穿越耕地/建成区/裸地的窗口，把窗口阻力"恢复"
  为林地(1)，重算该廊道 MCR；成本下降最多的窗口 = 修复收益最大的障碍点。
简化声明（报告引用）：电路跑在 MCR 廊道网络上而非全景观；障碍点为 top15 候选窗口
增量重算，非全网格扫描。

输出：pinch_current30.tif / nodes.geojson / stage5_nodes.json / fig_nodes.png
"""
import json, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import rasterio
import scipy.ndimage as ndimage
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import spsolve
from pyproj import Transformer
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
res = rd(NET / "results/rasters/res30_base.tif").astype(np.float32)
npz = np.load(OUT_T / "corridor_paths.npz")
pairs = npz["pairs"]; node_idx = npz["node_idx"]; splits = npz["splits"]
end_start = npz["end_start"]; end_end = npz["end_end"]
rows_all, cols_all = npz["rows"], npz["cols"]
paths = np.split(node_idx, splits)
src_ids = [int(i) for i in np.unique(src30) if i > 0]
print(f"[OK] 路径 {len(paths)} 条，源地 {len(src_ids)} 个")

# ---- 1) 廊道网络：节点=路径像元并集，边=8邻接（4方向覆盖无向对）---------------
net_nodes = np.unique(node_idx)
M = len(net_nodes)
nid_grid = np.full((H, W), -1, np.int32)
nid_grid[rows_all[net_nodes], cols_all[net_nodes]] = np.arange(M)
rn, cn = rows_all[net_nodes], cols_all[net_nodes]
edge_u, edge_v, edge_l = [], [], []
for dy, dx, L in [(0, 1, 30.0), (1, 0, 30.0), (1, 1, 30 * 2 ** .5), (1, -1, 30 * 2 ** .5)]:
    sh = np.full((H, W), -1, np.int32)
    ys = slice(max(dy, 0), H + min(dy, 0)); xs = slice(max(dx, 0), W + min(dx, 0))
    yd = slice(max(-dy, 0), H + min(-dy, 0)); xd = slice(max(-dx, 0), W + min(-dx, 0))
    sh[yd, xd] = nid_grid[ys, xs]
    m = (nid_grid >= 0) & (sh >= 0)
    edge_u.append(nid_grid[m]); edge_v.append(sh[m]); edge_l.append(np.full(int(m.sum()), L, np.float32))
eu = np.concatenate(edge_u); ev = np.concatenate(edge_v); el = np.concatenate(edge_l)
r_e = 0.5 * (res[rn[eu], cn[eu]] + res[rn[ev], cn[ev]]) * el
c_e = 1.0 / r_e
print(f"[OK] 网络节点 {M}，边 {len(eu)}")

# ---- 2) 源地超节点：电导 100 接到该源全部廊道入点 ------------------------------
sup_entries = {sid: set() for sid in src_ids}
for pi, (a, b) in enumerate(pairs):
    sup_entries[int(a)].add(int(paths[pi][-1]))    # 路径末端 = 起点侧源地入点
    sup_entries[int(b)].add(int(paths[pi][0]))     # 路径首端 = 目标侧源地入点
sup_of = {}; su, sv, sc = [], [], []
for j, sid in enumerate([s for s in src_ids if sup_entries[s]]):
    sk = M + j; sup_of[sid] = sk
    for g in sup_entries[sid]:
        k = int(np.where(net_nodes == g)[0][0])
        su += [sk, k]; sv += [k, sk]; sc += [100.0, 100.0]
K = len(sup_of)
print(f"[OK] 超节点 {K} 个")

n_tot = M + K
def laplacian():
    """全尺寸拉普拉斯：网络边 + 超节点边。"""
    U = np.concatenate([eu, su]); Vv = np.concatenate([ev, sv]); C = np.concatenate([c_e, sc])
    off = coo_matrix((np.concatenate([C, C]),
                      (np.concatenate([U, Vv]), np.concatenate([Vv, U]))),
                     shape=(n_tot, n_tot), dtype=np.float64)
    deg = np.asarray(off.sum(axis=1)).ravel()
    return (off + coo_matrix((deg, (np.arange(n_tot), np.arange(n_tot))), shape=(n_tot, n_tot))).tocsr()

G = laplacian()
ncomp, comp = connected_components(G, directed=False)
print(f"[OK] 网络+超节点连通域 {ncomp} 个")
inj_comp = comp[sup_of[src_ids[0]]]
grounded_all = [sk for sk in sup_of.values() if comp[sk] == inj_comp]
print(f"[OK] 接地超节点（同连通域）{len(grounded_all)} 个")

# ---- 3) 逐源电流模拟：注入 1A，其余源地接地（V=0，行替换钉地）------------------
I_acc = np.zeros(M, np.float64)
t0 = time.time()
G_lil = G.tolil()
for sid, sk in sup_of.items():
    if comp[sk] != inj_comp:                       # 不在主连通域的源跳过（不会发生，防呆）
        continue
    b = np.zeros(n_tot)
    b[sk] = 1.0                                    # 注入 1A
    for g in grounded_all:
        if g == sk:
            continue
        G_lil.rows[g] = [g]                        # 行替换：V_g = 0（Dirichlet 接地）
        G_lil.data[g] = [1.0]
        b[g] = 0.0
    V = spsolve(G_lil.tocsr(), b)
    Ie = np.abs(c_e * (V[eu] - V[ev]))             # 景观边电流
    np.add.at(I_acc, eu, Ie)
    np.add.at(I_acc, ev, Ie)
print(f"[OK] {len(sup_of)} 次电路求解用时 {time.time()-t0:.0f}s")

cur_raster = np.zeros((H, W), np.float32)
cur_raster[rn, cn] = I_acc.astype(np.float32)
with rasterio.open(OUT_R / "pinch_current30.tif", "w", driver="GTiff", height=H, width=W,
                   count=1, dtype="float32", crs=grid["crs"], transform=TR30,
                   nodata=0, compress="deflate") as d:
    d.write(cur_raster, 1)

# ---- 4) 夹点聚类与分级 ---------------------------------------------------------
pos = I_acc[I_acc > 0]
thr = np.percentile(pos, 95) if pos.size else 0.0
pinch_mask = I_acc >= thr
pm = np.zeros((H, W), bool); pm[rn[pinch_mask], cn[pinch_mask]] = True
pl_lbl, _ = ndimage.label(ndimage.binary_dilation(pm, np.ones((3, 3), bool)))
pinch_nodes = []
for i, sl in enumerate(ndimage.find_objects(pl_lbl), 1):
    m = pl_lbl[sl] == i
    ys, xs = np.where(m)
    iy = ys + sl[0].start; ix = xs + sl[1].start
    ks = nid_grid[iy, ix]; ks = ks[ks >= 0]
    pinch_nodes.append(dict(row=int(iy.mean()), col=int(ix.mean()),
                            mean_I=float(I_acc[ks].mean()), px=int(len(iy))))
if pinch_nodes:
    q80, q50 = np.percentile([n["mean_I"] for n in pinch_nodes], [80, 50])
    for n in pinch_nodes:
        n["grade"] = "一级" if n["mean_I"] >= q80 else ("二级" if n["mean_I"] >= q50 else "一般")
        n["x"] = round(float(TR30[2] + TR30[0] * n["col"]), 1)
        n["y"] = round(float(TR30[5] + TR30[4] * n["row"]), 1)
print(f"[OK] 夹点聚类 {len(pinch_nodes)} 个（电流≥P95={thr:.3f}）")

# ---- 5) 障碍点：恢复增量法（top15 窗口，两次全网格 Dijkstra/窗口）--------------
cand = (nid_grid >= 0) & np.isin(wc30, [40, 50, 60])   # 廊道上的耕地/建成区/裸地像元
cl_lbl, _ = ndimage.label(ndimage.binary_dilation(cand, np.ones((3, 3), bool)))
cands = []
for i, sl in enumerate(ndimage.find_objects(cl_lbl), 1):
    m = cl_lbl[sl] == i
    ys, xs = np.where(m)
    iy = ys + sl[0].start; ix = xs + sl[1].start
    fp = np.zeros((H, W), bool); fp[iy, ix] = True
    fp = ndimage.binary_dilation(fp, np.ones((3, 3), bool), iterations=2)  # 外扩60m 恢复斑
    cands.append(dict(foot=fp, row=int(iy.mean()), col=int(ix.mean()),
                      prescore=float((res[fp & (res > 0)] - 1).sum()), npx=int(fp.sum())))
cands.sort(key=lambda c: -c["prescore"])
aff = {}
for pi, pth in enumerate(paths):
    hits = int(cand[rows_all[pth], cols_all[pth]].sum())
    if hits > 0:
        aff.setdefault((int(pairs[pi][0]), int(pairs[pi][1])), []).append((pi, hits))

analysis = wc30 > 0
node_id30 = np.full((H, W), -1, np.int32)
node_id30[analysis] = np.arange(int(analysis.sum()))
N30 = int(analysis.sum()); V30 = N30
rows30, cols30 = np.where(analysis)
r0s, c0s, r1s, c1s, lens = [], [], [], [], []
for dy, dx, L in [(0, 1, 30.0), (1, 0, 30.0), (1, 1, 30 * 2 ** .5), (1, -1, 30 * 2 ** .5)]:
    rr0 = rows30[(rows30 + dy >= 0) & (rows30 + dy < H) & (cols30 + dx >= 0) & (cols30 + dx < W)]
    cc0 = cols30[(rows30 + dy >= 0) & (rows30 + dy < H) & (cols30 + dx >= 0) & (cols30 + dx < W)]
    ok = analysis[rr0 + dy, cc0 + dx]
    r0s.append(rr0[ok]); c0s.append(cc0[ok]); r1s.append(rr0[ok] + dy); c1s.append(cc0[ok] + dx)
    lens.append(np.full(int(ok.sum()), L, np.float32))
R0 = np.concatenate(r0s); C0 = np.concatenate(c0s)
R1 = np.concatenate(r1s); C1 = np.concatenate(c1s)
Lall = np.concatenate(lens)
n0 = node_id30[R0, C0]; n1 = node_id30[R1, C1]
kp = (n0 >= 0) & (n1 >= 0)
n0, n1, Lall = n0[kp], n1[kp], Lall[kp]
src_px30 = {sid: node_id30[src30 == sid] for sid in src_ids}

def mcr_at(res_arr, sid_s, tgt_global):
    """单源(虚拟源法) Dijkstra，返回目标全局节点处的 MCR。"""
    px = src_px30[sid_s]
    vrow = np.concatenate([np.full(len(px), V30), px])
    vcol = np.concatenate([px, np.full(len(px), V30)])
    w = 0.5 * (res_arr[R0, C0] + res_arr[R1, C1]) * Lall
    adj = csr_matrix((np.concatenate([w, w]), (np.concatenate([n0, n1]), np.concatenate([n1, n0]))),
                     shape=(N30 + 1, N30 + 1), dtype=np.float32)
    adj = adj + csr_matrix((np.full(len(vrow), 1e-3, np.float32), (vrow, vcol)),
                           shape=(N30 + 1, N30 + 1), dtype=np.float32)
    d = dijkstra(adj, directed=False, indices=V30)   # 不带 return_predecessors 只返回距离数组
    return float(d[tgt_global])

barrier_nodes = []
K_TOP = 15
for c in cands[:K_TOP]:
    # 关键修复：选廊道对必须看"该对路径在【本窗口】内的像元数"——
    # 此前用全廊道候选命中数，所有窗口都错选同一条对，恢复落在别的廊道上增益恒 0
    best = None
    for (a, b), hits in aff.items():
        for pi, _ in hits:
            pth = paths[pi]
            inw = int(c["foot"][rows_all[pth], cols_all[pth]].sum())
            if inw > 0 and (best is None or inw > best[2]):
                best = (a, b, inw, int(end_end[pi]))
    if best is None:
        continue
    a, b, inw, tgt = best
    mcr_base = mcr_at(res, a, tgt)
    res2 = res.copy()
    res2[c["foot"] & analysis] = 1.0               # 恢复：窗口阻力=林地级1（重造林设想）
    mcr_new = mcr_at(res2, a, tgt)
    saving = mcr_base - mcr_new                    # 绝对节省（阻力·m）：排序主指标
    gain = saving / max(mcr_base, 1e-9) * 100
    barrier_nodes.append(dict(row=c["row"], col=c["col"], pair=[a, b],
                              mcr_base=round(mcr_base, 0), mcr_restored=round(mcr_new, 0),
                              saving=round(saving, 0), gain_pct=round(gain, 2), px=c["npx"]))
    print(f"  [窗口] ({c['row']},{c['col']}) 对({a},{b}) 窗内路径像元{inw} "
          f"MCR {mcr_base:.0f}->{mcr_new:.0f} 节省 {saving:.0f} ({gain:.2f}%)")
barrier_nodes.sort(key=lambda n: -n["saving"])
for i, n in enumerate(barrier_nodes):
    n["grade"] = "一级" if i < 5 else ("二级" if n["saving"] > 0 else "一般")
    n["x"] = round(float(TR30[2] + TR30[0] * n["col"]), 1)
    n["y"] = round(float(TR30[5] + TR30[4] * n["row"]), 1)
print(f"[OK] 障碍点 {len(barrier_nodes)} 个（按绝对节省排序，top5=一级）")

# ---- 6) 输出 -------------------------------------------------------------------
tf = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
feats = []
for kind, lst in [("pinch", pinch_nodes), ("barrier", barrier_nodes)]:
    for i, n in enumerate(lst):
        lon, lat = tf.transform(n["x"], n["y"])
        props = {k: v for k, v in n.items() if k not in ("row", "col", "x", "y")}
        props.update(type=kind, id=f"{kind[0].upper()}{i+1:02d}")
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]}})
gj = {"type": "FeatureCollection",
      "crs": {"type": "name", "properties": {"name": "EPSG:4326 (GeoJSON WGS84, 由 EPSG:32651 转换)"}},
      "features": feats}
(OUT_T / "nodes.geojson").write_text(json.dumps(gj, ensure_ascii=False, indent=2), encoding="utf-8")
out = dict(
    method_pinch="廊道网络电路模拟：逐源1A注入/其余源地接地(V=0)/超节点电导100；I=|电流|逐源累计",
    method_barrier="恢复增量法：耕地/建成区/裸地窗口(外扩60m)阻力恢复为1，重算MCR，成本降幅=修复收益%",
    simplified=["电路跑在MCR廊道网络上而非全景观（Circuitscape口径差异）",
                "障碍点为top15候选窗口增量重算，非全网格扫描",
                "接地为Dirichlet V=0，注入源单点1A"],
    pinch=dict(threshold_pct=95, n=len(pinch_nodes), nodes=pinch_nodes),
    barrier=dict(n=len(barrier_nodes), nodes=barrier_nodes))
(OUT_T / "stage5_nodes.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

from rasterio.transform import Affine
ext = extent_of(Affine(*TR30[:6]), W, H)
bg = np.full((H, W, 3), (246, 246, 242), np.uint8)
bg[wc30 == 80] = (205, 227, 242); bg[wc30 == 0] = (228, 228, 228)
fig, ax = plt.subplots(figsize=(9.5, 7.4), dpi=150)
ax.imshow(bg, extent=ext, interpolation="nearest")
cur_show = np.ma.masked_where(cur_raster <= 0, cur_raster)
# hot_r 的 0 端是白色，低电流值贴米色背景"隐形"（色带端点坑第3次出现）；改 inferno（0=黑）
ax.imshow(cur_show, extent=ext, cmap="inferno", interpolation="nearest", alpha=0.85)
ax.scatter([n["x"] for n in pinch_nodes], [n["y"] for n in pinch_nodes], s=70,
           facecolors="none", edgecolors="blue", linewidths=1.8, label=f"夹点节点 n={len(pinch_nodes)}")
ax.scatter([n["x"] for n in barrier_nodes], [n["y"] for n in barrier_nodes], s=70, marker="s",
           facecolors="none", edgecolors="red", linewidths=1.8, label=f"障碍点 n={len(barrier_nodes)}")
ax.set_title("福鼎市生态廊道电流密度与关键节点（电路模拟·简化口径）", fontsize=12)
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
add_north(ax); add_scalebar(ax, ext, km=10)
add_note(ax, "坐标系: EPSG:32651 (UTM 51N) · 电流密度=逐源注入累计|I|·P95以上聚类为节点", "bl")
ax.set_xlabel("东向 UTM (m)"); ax.set_ylabel("北向 UTM (m)")
fig.tight_layout(); fig.savefig(OUT_F / "fig_nodes.png"); plt.close(fig)
print("[DONE] stage5: nodes.geojson / stage5_nodes.json / pinch_current30.tif / fig_nodes.png")
