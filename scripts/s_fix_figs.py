# -*- coding: utf-8 -*-
"""
渲染修复：重出 fig_corridors / fig_sources（数据栅格不动，只修图）。
修的两个渲染 bug：
①源地按 ID 归一化到 Greens -> ID=2/11 的巨型源被映射到色带最浅端，近白不可见
②廊道以 1 像元 imshow 且被后画图层覆盖 -> 改为按 npz 路径画折线（与道路同级清晰）
"""
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
src30 = rd(NET / "results/rasters/sources30.tif")
road = rd(NET / "results/rasters/road_level30.tif")
res_arr = rd(NET / "results/rasters/reserve30.tif")
npz = np.load(OUT_T / "corridor_paths.npz")
paths = np.split(npz["node_idx"], npz["splits"])
rows_all, cols_all = npz["rows"], npz["cols"]
s2 = json.loads((OUT_T / "stage2_mspa.json").read_text(encoding="utf-8"))
n_src = s2["core_patches"]["source_count"]

X = 194010.0 + 30.0 * (cols_all + 0.5)          # 像元中心 UTM
Y = 3040170.0 - 30.0 * (rows_all + 0.5)
ext = extent_of(Affine(*TR30[:6]), W, H)

bg = np.full((H, W, 3), (246, 246, 242), np.uint8)
bg[wc30 == 80] = (205, 227, 242)
bg[wc30 == 0] = (228, 228, 228)

# ---- fig_corridors -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 7.4), dpi=150)
ax.imshow(bg, extent=ext, interpolation="nearest")
sm = np.ma.masked_where(src30 == 0, np.ones_like(src30, float))   # 修复①：固定色，不按ID归一化
SRC_C = "#2ca05a"                                  # ListedColormap 要 0-1 浮点/十六进制，整数元组会错色
ax.imshow(sm, extent=ext, cmap=matplotlib.colors.ListedColormap([SRC_C]),
          interpolation="nearest", alpha=0.85)
for pi, pth in enumerate(paths):                                 # 修复②：廊道画折线
    rr = rows_all[pth]; cc = cols_all[pth]
    ax.plot(X[pth], Y[pth], color="#e29a3d", lw=1.1, solid_capstyle="round", zorder=4)
ax.contour(np.where(road == 3, 1.0, 0.0), levels=[0.5], colors="red", linewidths=0.7, extent=ext)
ax.plot([], [], color="#e29a3d", lw=2, label=f"MCR廊道 n={len(paths)}")
ax.plot([], [], color=(44/255, 160/255, 78/255), lw=6, label=f"生态源地 n={n_src}")
ax.plot([], [], color="red", lw=1.2, label="3级道路(高速/主干)")
ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
ax.set_title(f"福鼎市 MCR 生态廊道（{len(paths)} 对源地廊道，红线=高速/主干道路）", fontsize=12)
add_north(ax); add_scalebar(ax, ext, km=10)
add_note(ax, "坐标系: EPSG:32651 (UTM 51N) · 廊道=MCR最小累积阻力路径", "bl")
ax.set_xlabel("东向 UTM (m)"); ax.set_ylabel("北向 UTM (m)")
fig.tight_layout(); fig.savefig(OUT_F / "fig_corridors.png"); plt.close(fig)

# ---- fig_sources ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 7.4), dpi=150)
ax.imshow(bg, extent=ext, interpolation="nearest")
ax.imshow(sm, extent=ext, cmap=matplotlib.colors.ListedColormap([SRC_C]),
          interpolation="nearest", alpha=0.9)
ax.contour(res_arr.astype(float), levels=[0.5], colors="red", linewidths=1.2, extent=ext)
ax.plot([], [], color=SRC_C, lw=6, label=f"生态源地 n={n_src}")
ax.plot([], [], color="red", lw=1.2, label="OSM保护地(弱参考)")
ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
ax.set_title(f"福鼎市生态源地分布（MSPA核心区≥100ha 且 NDVI≥0.40，n={n_src}）", fontsize=12)
add_north(ax); add_scalebar(ax, ext, km=10)
add_note(ax, "坐标系: EPSG:32651 (UTM 51N) · 生态红线未取得，用 OSM 保护地替代", "bl")
ax.set_xlabel("东向 UTM (m)"); ax.set_ylabel("北向 UTM (m)")
fig.tight_layout(); fig.savefig(OUT_F / "fig_sources.png"); plt.close(fig)
print("[DONE] fig_corridors.png / fig_sources.png 已按修复后的渲染重出")
