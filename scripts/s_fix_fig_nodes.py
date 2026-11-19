# -*- coding: utf-8 -*-
"""重出 fig_nodes：电流密度改 inferno 色带（0=黑，可见），廊道折线灰底做参照。"""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import rasterio
sys.path.insert(0, str(Path(__file__).parent))
from carto import extent_of, add_north, add_scalebar, add_note
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rasterio.transform import Affine

PRI = Path(r"E:\zcode\探索前沿\fuding-eco-priority")
NET = Path(r"E:\zcode\探索前沿\fuding-eco-network")
OUT_R, OUT_T, OUT_F = NET / "results" / "rasters", NET / "results" / "tables", NET / "results" / "figures"
grid = json.loads((PRI / "results/tables/grid_spec.json").read_text(encoding="utf-8"))
H, W, TR30 = grid["height"], grid["width"], tuple(grid["transform"])

def rd(p):
    with rasterio.open(p) as s:
        return s.read(1)

wc30 = rd(PRI / "results/rasters/wc30_mode.tif")
cur = rd(NET / "results/rasters/pinch_current30.tif")
npz = np.load(OUT_T / "corridor_paths.npz")
paths = np.split(npz["node_idx"], npz["splits"])
rows_all, cols_all = npz["rows"], npz["cols"]
nodes = json.loads((OUT_T / "nodes.geojson").read_text(encoding="utf-8"))["features"]
from pyproj import Transformer
tf = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
X = 194010.0 + 30.0 * (cols_all + 0.5)
Y = 3040170.0 - 30.0 * (rows_all + 0.5)
ext = extent_of(Affine(*TR30[:6]), W, H)
bg = np.full((H, W, 3), (246, 246, 242), np.uint8)
bg[wc30 == 80] = (205, 227, 242); bg[wc30 == 0] = (228, 228, 228)

fig, ax = plt.subplots(figsize=(9.5, 7.4), dpi=150)
ax.imshow(bg, extent=ext, interpolation="nearest")
for pth in paths:                                    # 廊道浅灰线做参照
    ax.plot(X[pth], Y[pth], color="#dcdcdc", lw=0.7, zorder=2)
# 电流只显示上半段（低于P40的隐藏）：inferno 低值端近黑，全显会读成"黑色蛛网"
thr_show = np.percentile(cur[cur > 0], 40) if (cur > 0).any() else 0.0
cs = np.ma.masked_where(cur <= max(thr_show, 0), cur)
im = ax.imshow(cs, extent=ext, cmap="inferno", interpolation="nearest", alpha=0.9, zorder=3)
plt.colorbar(im, ax=ax, shrink=0.6, label="累计电流 |I|（逐源注入）")
pn = [f for f in nodes if f["properties"]["type"] == "pinch"]
bn = [f for f in nodes if f["properties"]["type"] == "barrier"]
for f in pn:
    lon, lat = f["geometry"]["coordinates"]; x, y = tf.transform(lon, lat)
    g = f["properties"]["grade"]
    ax.scatter([x], [y], s=100, facecolors="none", edgecolors={"一级": "#1f5fbf", "二级": "#4d9de0", "一般": "#9ecae1"}[g],
               linewidths=2.0, zorder=5)
for f in bn:
    lon, lat = f["geometry"]["coordinates"]; x, y = tf.transform(lon, lat)
    g = f["properties"]["grade"]
    ax.scatter([x], [y], s=100, marker="s", facecolors="none", edgecolors={"一级": "#c1121f", "二级": "#e85d04", "一般": "#f4a261"}[g],
               linewidths=2.0, zorder=5)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", color="w", markeredgecolor="#1f5fbf", markeredgewidth=2, markersize=10, label="夹点(蓝系=等级)"),
                   Line2D([], [], marker="s", color="w", markeredgecolor="#c1121f", markeredgewidth=2, markersize=10, label="障碍点(红系=等级)")],
          loc="lower right", fontsize=8, framealpha=0.95)
ax.set_title("福鼎市生态廊道电流密度与关键节点（电路模拟·简化口径）", fontsize=12)
add_north(ax); add_scalebar(ax, ext, km=10)
add_note(ax, "坐标系: EPSG:32651 (UTM 51N) · 电流=逐源1A注入累计|I|，P95以上聚类为夹点", "bl")
ax.set_xlabel("东向 UTM (m)"); ax.set_ylabel("北向 UTM (m)")
fig.tight_layout(); fig.savefig(OUT_F / "fig_nodes.png"); plt.close(fig)
print("[DONE] fig_nodes.png 重出（inferno 电流+分级节点+廊道灰线）")
