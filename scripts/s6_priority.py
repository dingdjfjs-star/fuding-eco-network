# -*- coding: utf-8 -*-
"""
阶段6 修复优先级与布局：关键节点 × 2点项目优先区/退化度 × 保护地 × 道路 × 村庄。
规则化打分（每个字段来源可溯源）+ 措施建议决策表 + 局限声明。
输出：stage6_priority.json / priority_table.json / fig_priority.png
"""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import rasterio
import scipy.ndimage as ndimage
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
prio = rd(PRI / "results/rasters/priority_class30.tif").astype(np.int32)   # 2点项目: 1一级 2二级 3三级, -9999=nodata
deg = rd(PRI / "results/rasters/dim_degradation30.tif").astype(np.float32) # 2点项目: 0~1
resv = rd(NET / "results/rasters/reserve30.tif")
road = rd(NET / "results/rasters/road_level30.tif")
vill = rd(NET / "results/rasters/village30.tif")
corr = rd(NET / "results/rasters/corridor_mask30.tif")
analysis = wc30 > 0

nodes = json.loads((OUT_T / "nodes.geojson").read_text(encoding="utf-8"))["features"]
prio[prio < 0] = 0
deg[~analysis] = 0

# 常备距离场（一次算好，所有节点共用）
d_resv = ndimage.distance_transform_edt(resv == 0) * 30        # 到保护地距离 m
d_vill = ndimage.distance_transform_edt(vill == 0) * 30
d_road3 = ndimage.distance_transform_edt(road != 3) * 30       # 到3级道路距离
d_corr = ndimage.distance_transform_edt(corr == 0) * 30        # 到廊道距离

WIN = 10    # 节点邻域窗口半径 10 像元=300m：措施落地的现实尺度
R = WIN

def win_stats(row, col):
    y0, y1 = max(row - R, 0), min(row + R + 1, H)
    x0, x1 = max(col - R, 0), min(col + R + 1, W)
    sl = np.s_[y0:y1, x0:x1]
    pv = prio[sl]; pv = pv[pv > 0]
    dv = deg[sl][deg[sl] > 0]
    return dict(
        prio_mode=int(np.bincount(pv).argmax()) if pv.size else 0,      # 窗口内2点项目优先区众数
        deg_mean=float(dv.mean()) if dv.size else 0.0,                  # 窗口退化度均值
        lu=dict(zip(*[a.tolist() for a in np.unique(wc30[sl], return_counts=True)])))

rows_all, cols_all = np.where(analysis)
def to_rc(x, y):
    col = int(round((x - TR30[2]) / TR30[0])); row = int(round((y - TR30[5]) / TR30[4]))
    return min(max(row, 0), H - 1), min(max(col, 0), W - 1)

LUC = {10: "林地", 20: "灌木", 30: "草地", 40: "耕地/茶园", 50: "建成区", 60: "裸地", 80: "水体", 90: "湿地"}

def measure(n, ws):
    """措施决策表：看节点周边 300m 的地类构成与压力要素。返回 1-2 条主措施。"""
    m = []
    lu = ws["lu"]
    dom = max(lu, key=lambda k: lu[k]) if lu else 0
    if d_road3[n["row"], n["col"]] < 120:      # 节点距3级道路<120m → 过路设施问题
        m.append("过路设施改造（动物通道/涵洞加宽/光屏蔽，需交通部门协同）")
    if dom == 40:
        m.append("农地/茶园缓冲带改造（留养绿篱、行间补植、减药减肥）")
    if dom == 50 or d_vill[n["row"], n["col"]] < 300:
        m.append("村庄周边林带补植+社区共管（家犬管控/垃圾清运）")
    if dom == 60:
        m.append("裸地复绿（矿山/采石迹地客土造林）")
    if dom in (80, 90) or (lu.get(80, 0) + lu.get(90, 0)) / max(sum(lu.values()), 1) > 0.3:
        m.append("滨海/河口湿地恢复（退养还湿、潮汐连通，注意养殖权属）")
    if dom in (10, 20, 30) and len(m) == 0:
        m.append("封育管护+近自然补植（目标：廊道两侧30m连续郁闭）")
    if d_resv[n["row"], n["col"]] < 500:
        m.append("保护地周边协同（缓冲带管护口径与保护区管理机构对接）")
    return m[:2] if m else ["封育管护"]

from pyproj import Transformer
_TF2 = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
table = []
for f in nodes:
    p = f["properties"]; lon, lat = f["geometry"]["coordinates"]
    # geojson 只有 WGS84 经纬度（s5 导出时不带 x/y），逆变换回 UTM 取窗口
    x, y = _TF2.transform(lon, lat)
    row, col = to_rc(x, y)
    ws = win_stats(row, col)
    ws["row"], ws["col"] = row, col
    g_score = {"一级": 3, "二级": 2, "一般": 1}[p["grade"]]
    prio_score = {1: 2, 2: 1, 3: 0}.get(ws["prio_mode"], 0)
    gain = float(p.get("gain_pct", 0) or 0)
    deg_s = ws["deg_mean"] * 2
    resv_b = 0.5 if d_resv[row, col] < 500 else 0.0
    score = round(g_score + prio_score + gain / 10 + deg_s + resv_b, 2)
    table.append(dict(id=p["id"], type=p["type"], grade=p["grade"], x=round(float(x), 1), y=round(float(y), 1),
                      lon=lon, lat=lat, prio_mode=ws["prio_mode"], deg_mean=round(ws["deg_mean"], 3),
                      reserve_dist_m=round(float(d_resv[row, col]), 0),
                      road3_dist_m=round(float(d_road3[row, col]), 0),
                      village_dist_m=round(float(d_vill[row, col]), 0),
                      corridor_dist_m=round(float(d_corr[row, col]), 0),
                      gain_pct=p.get("gain_pct"), mean_I=p.get("mean_I"),
                      window_lu=ws["lu"], score=score, measures=measure(dict(row=row, col=col), ws)))

# 优先排序：先按等级（一级节点优先），同级按 score
order = {"一级": 0, "二级": 1, "一般": 2}
table.sort(key=lambda t: (order[t["grade"]], -t["score"]))
for i, t in enumerate(table, 1):
    t["rank"] = i
(OUT_T / "priority_table.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")

# 统计
from collections import Counter
cnt = Counter((t["type"], t["grade"]) for t in table)
prio_hit = sum(1 for t in table if t["prio_mode"] == 1)
resv_hit = sum(1 for t in table if t["reserve_dist_m"] < 500)
road_hit = sum(1 for t in table if t["road3_dist_m"] < 120)
s4 = json.loads((OUT_T / "stage4_mcr.json").read_text(encoding="utf-8"))
out = dict(
    scoring="等级分(一级3/二级2/一般1)+2点项目一级优先区(+2)/二级(+1)+障碍增益/10+退化度×2+保护地500m(+0.5)，满分约8.5",
    counts={f"{k[0]}-{k[1]}": v for k, v in sorted(cnt.items())},
    # dict 关键字不能以数字开头（"3级道路..."会 SyntaxError），改用带引号的字典键
    overlay={"节点落在2点项目一级优先区": f"{prio_hit}/{len(table)}",
             "保护地500m内": resv_hit, "3级道路120m内": road_hit},
    corridor_total_km=s4["length_stats"]["total_km"],
    sensitivity=s4["sensitivity"],
    top10=[{k: t[k] for k in ("rank", "id", "type", "grade", "score", "measures")} for t in table[:10]],
    limitations=["连通性视角≠工程可行性：未考虑用地权属/资金/地形可达性/施工窗口",
                 "电路为廊道网络简化口径；障碍点为top15窗口增量重算非全网格扫描",
                 "阻力赋值多为经验值档位，敏感性分析见 stage4_mcr.json",
                 "生态红线未取得，保护地为OSM弱参考口径",
                 "滩涂源地受WorldCover口径限制未进源地（讨论见报告）"])
(OUT_T / "stage6_priority.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- 图：优先级布局 ------------------------------------------------------------
from rasterio.transform import Affine
ext = extent_of(Affine(*TR30[:6]), W, H)
bg = np.full((H, W, 3), (246, 246, 242), np.uint8)
bg[wc30 == 80] = (205, 227, 242); bg[wc30 == 0] = (228, 228, 228)
pz = np.zeros((H, W, 3), np.uint8); pz[...] = bg
pz[prio == 1] = (240, 170, 160); pz[prio == 2] = (248, 214, 190); pz[prio == 3] = (252, 238, 214)
pz[wc30 == 80] = (205, 227, 242); pz[~analysis] = (228, 228, 228)
fig, ax = plt.subplots(figsize=(9.5, 7.4), dpi=150)
ax.imshow(pz, extent=ext, interpolation="nearest")
import matplotlib.colors as mcolors
cmask = np.ma.masked_where(corr == 0, corr)
ax.imshow(cmask, extent=ext, cmap=mcolors.ListedColormap(["#6ea8dc"]), interpolation="nearest", alpha=0.7)
gcol = {"一级": "#d62728", "二级": "#ff7f0e", "一般": "#8c564b"}
for g, mk in [("一级", "o"), ("二级", "^"), ("一般", "s")]:
    xs = [t["x"] for t in table if t["type"] == "pinch" and t["grade"] == g]
    ys = [t["y"] for t in table if t["type"] == "pinch" and t["grade"] == g]
    if xs:
        ax.scatter(xs, ys, marker=mk, s=90, c=gcol[g], edgecolors="k", linewidths=0.8,
                   label=f"夹点-{g}", zorder=5)
    xs = [t["x"] for t in table if t["type"] == "barrier" and t["grade"] == g]
    ys = [t["y"] for t in table if t["type"] == "barrier" and t["grade"] == g]
    if xs:
        ax.scatter(xs, ys, marker=mk, s=90, facecolors="none", edgecolors=gcol[g],
                   linewidths=1.8, label=f"障碍-{g}", zorder=5)
ax.set_title("福鼎市修复关键节点优先级布局（底色=2点项目修复优先区分级）", fontsize=12)
ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9, ncol=2)
add_north(ax); add_scalebar(ax, ext, km=10)
add_note(ax, "坐标系: EPSG:32651 (UTM 51N) · 节点=电路夹点(圆)/障碍点(方块) 分级着色", "bl")
ax.set_xlabel("东向 UTM (m)"); ax.set_ylabel("北向 UTM (m)")
fig.tight_layout(); fig.savefig(OUT_F / "fig_priority.png"); plt.close(fig)
print(f"[DONE] stage6: {len(table)} 节点排序完成；一级优先区命中 {prio_hit}，保护地邻近 {resv_hit}，3级道路邻近 {road_hit}")
