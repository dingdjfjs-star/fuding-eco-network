# -*- coding: utf-8 -*-
"""
阶段2 MSPA v2（10m 原生分辨率）：形态学七分类 + 生态源地筛选 + 降采样到 30m 网格。
v1（30m-mode）诊断：核心占前景 88%、桥接/环道=0 —— mode 重采样把窄谷糊掉，失去分辨力，
v1 产物保留作口径对比证据（mspa30.tif / stage2_mspa.json 被本版覆盖前已摘录进 JSON）。
与 Guidos Toolbox 的差异逐条写在输出 JSON 与 docs 里，报告如实引用。
输出：results/rasters/mspa10.tif / sources30.tif（供 MCR 用）
      results/tables/stage2_mspa.json（含 10m vs 30m 对比）
      results/figures/fig_mspa.png / fig_sources.png
"""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).parent))
from carto import extent_of, add_north, add_scalebar, add_note
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRI = Path(r"E:\zcode\探索前沿\fuding-eco-priority")
NET = Path(r"E:\zcode\探索前沿\fuding-eco-network")
OUT_R, OUT_T, OUT_F = NET / "results" / "rasters", NET / "results" / "tables", NET / "results" / "figures"

S8 = np.ones((3, 3), int)                      # 前景 8 邻接（MSPA 规范）
S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], int)  # 背景互补 4 邻接

# ---- 1) 10m 前景 -------------------------------------------------------------
src10 = rasterio.open(PRI / "results/rasters/fuding_worldcover2021_10m.tif")
wc10 = src10.read(1)
H10, W10 = wc10.shape
TR10 = src10.transform; CRS10 = src10.crs
PX_HA10 = 0.01                                  # 10m 像元 = 100 m² = 0.01 ha
analysis10 = wc10 > 0
F = np.isin(wc10, [10, 20, 30, 90]) & analysis10  # 前景：林地10+灌木20+草地30+湿地90（茶园=耕地40 不进）
print(f"[OK] 10m 前景像元 {int(F.sum())} = {F.sum()*PX_HA10/100:.1f} km²（占分析范围 {F.sum()/analysis10.sum()*100:.1f}%）")

# ---- 2) 简化 MSPA 七分类（口径与 v1 相同，分辨率 10m）------------------------
REUSE = "--reuse" in sys.argv          # 形态学结果已落盘时跳过重算（调门槛用）
if REUSE and (OUT_R / "mspa10.tif").exists():
    with rasterio.open(OUT_R / "mspa10.tif") as s:
        mspa10 = s.read(1)
    core = mspa10 == 1
    print("[OK] --reuse：形态学结果从 mspa10.tif 读回，跳过重算")
else:
    core = ndimage.binary_erosion(F, S8) & F
    lblF, nF = ndimage.label(F, S8)
    core_lbl = np.where(core, lblF, 0).astype(np.int32)
    has_core = np.unique(core_lbl[core_lbl > 0])
    islet = F & ~np.isin(lblF, has_core)

    lblB4, _ = ndimage.label(~F, S4)
    border_lbls = set(lblB4[0, :]) | set(lblB4[-1, :]) | set(lblB4[:, 0]) | set(lblB4[:, -1])
    holes = np.isin(lblB4, [l for l in np.unique(lblB4) if l > 0 and l not in border_lbls])
    rim = ndimage.binary_dilation(core, S8) & F & ~core
    perforation = rim & ndimage.binary_dilation(holes, S8)
    edge = rim & ~perforation
    conn = F & ~core & ~rim
    lblC, nC = ndimage.label(conn, S8)
    print(f"[OK] 连接体组件数 {nC}（包围盒裁剪逐个判连通函数）")
    bridge = np.zeros(F.shape, bool); loop = np.zeros(F.shape, bool); branch = np.zeros(F.shape, bool)
    for i, sl in enumerate(ndimage.find_objects(lblC), 1):
        if sl is None:
            continue
        r0 = max(sl[0].start - 2, 0); r1 = min(sl[0].stop + 2, H10)
        c0 = max(sl[1].start - 2, 0); c1 = min(sl[1].stop + 2, W10)
        m = lblC[r0:r1, c0:c1] == i
        touch = ndimage.binary_dilation(m, S8)
        cl_arr = core_lbl[r0:r1, c0:c1]
        cl = np.unique(cl_arr[touch & (cl_arr > 0)])
        if cl.size >= 2:
            bridge[r0:r1, c0:c1] |= m
        elif cl.size == 1:
            tp = touch & (cl_arr == cl[0])
            _, ntp = ndimage.label(tp, S4)
            if ntp >= 2:
                loop[r0:r1, c0:c1] |= m
            else:
                branch[r0:r1, c0:c1] |= m
        else:
            branch[r0:r1, c0:c1] |= m
    print(f"[OK] MSPA 分型：核心 {int(core.sum())} 边缘 {int(edge.sum())} 孔隙 {int(perforation.sum())} "
          f"桥接 {int(bridge.sum())} 环道 {int(loop.sum())} 支线 {int(branch.sum())} 孤岛 {int(islet.sum())} 像元(10m)")
    mspa10 = np.zeros(F.shape, np.uint8)            # 1核心 2边缘 3孔隙 4桥接 5环道 6支线 7孤岛
    mspa10[core] = 1; mspa10[edge] = 2; mspa10[perforation] = 3
    mspa10[bridge] = 4; mspa10[loop] = 5; mspa10[branch] = 6; mspa10[islet] = 7
    meta10 = dict(driver="GTiff", height=H10, width=W10, count=1, dtype="uint8",
                  crs=CRS10, transform=TR10, nodata=0, compress="deflate")
    with rasterio.open(OUT_R / "mspa10.tif", "w", **meta10) as d:
        d.write(mspa10, 1)

mspa10 = mspa10  # 已在上方（计算或读回）得到

# ---- 3) 源地筛选（≥100 ha）+ NDVI 门槛（在 30m NDVI 上算，避免 10m NDVI 缺失）--
SRC_MIN_HA = 100       # 需我确认：县级尺度文献常用 50~100 ha 起筛
core_lbl2, nCore = ndimage.label(core, S8)
sizes = np.bincount(core_lbl2.ravel())
keep_ids = [pid for pid in range(1, nCore + 1) if sizes[pid] * PX_HA10 >= SRC_MIN_HA]
src10_map = np.zeros(F.shape, np.int32)
for pid in keep_ids:
    src10_map[core_lbl2 == pid] = pid
# 10m -> 30m 众数重采样（源地是面状对象，众数保 ID 且不被单个 10m 像元噪声牵着走）
grid = json.loads((PRI / "results/tables/grid_spec.json").read_text(encoding="utf-8"))
H30, W30, TR30 = grid["height"], grid["width"], tuple(grid["transform"])
src30_mode_full = np.zeros((H30, W30), np.int32)     # 全量候选的 30m 掩膜（算 NDVI 用）
reproject(src10_map, src30_mode_full, src_transform=TR10, src_crs=CRS10,
          dst_transform=TR30, dst_crs=grid["crs"], resampling=Resampling.mode)

with rasterio.open(PRI / "results/rasters/ndvi2025_30m.tif") as s:
    ndvi30 = s.read(1).astype("float32")
res_arr = rasterio.open(NET / "results/rasters/reserve30.tif").read(1)
SRC_MIN_NDVI = 0.40    # 需我确认：2025 生长季 NDVI 均值 <0.40 视为退化严重不作源地
patches = []
for pid in keep_ids:
    m = src30_mode_full == pid
    nd = ndvi30[m]; nd = nd[(nd > -1) & (nd < 1)]
    ndm = float(nd.mean()) if nd.size else -1.0
    patches.append(dict(id=pid, area_ha=round(sizes[pid] * PX_HA10, 1), ndvi=round(ndm, 3),
                        reserve=(bool((m & (res_arr > 0)).any()))))
patches.sort(key=lambda p: -p["area_ha"])
sources = [p for p in patches if p["ndvi"] >= SRC_MIN_NDVI]
# 关键修复：NDVI 门槛要落实到栅格——只保留通过门槛的斑块 ID（此前只过滤了清单，
# 栅格里残留退化斑块，下游 MCR 会多连出不该有的廊道）
src30 = np.zeros((H30, W30), np.int32)
for p in sources:
    src30[src30_mode_full == p["id"]] = p["id"]
with rasterio.open(OUT_R / "sources30.tif", "w", driver="GTiff", height=H30, width=W30,
                   count=1, dtype="int32", crs=grid["crs"], transform=TR30,
                   nodata=0, compress="deflate") as d:
    d.write(src30, 1)
print(f"[OK] 核心斑块 ≥{SRC_MIN_HA}ha 共 {len(patches)} 个；NDVI≥{SRC_MIN_NDVI} 过滤后源地 {len(sources)} 个（已落实到栅格）")

# ---- 4) 统计表（含 v1/v2 口径对比）-------------------------------------------
CLS = {1: "核心区", 2: "边缘", 3: "孔隙", 4: "桥接区", 5: "环道", 6: "支线", 7: "孤岛"}
tot10 = int(analysis10.sum())
cls_stat = {CLS[k]: dict(px=int((mspa10 == k).sum()), ha=round((mspa10 == k).sum() * PX_HA10, 1),
                         pct=round((mspa10 == k).sum() / tot10 * 100, 2)) for k in CLS}
out = dict(
    version="v2 MSPA@10m 原生分辨率（前景=WC10/20/30/90）；v1@30m-mode 诊断保留作对比",
    v1_30m_mode_diagnosis=dict(core_pct_of_fg=88.0, bridge_px=0, loop_px=0,
        note="mode 重采样把 30~100m 宽的窄谷/细廊道吸收进林地，全图近乎连成一块，MSPA 失去分辨能力 -> 弃用"),
    foreground10=dict(px=int(F.sum()), km2=round(F.sum() * PX_HA10 / 100, 1),
                      classes="WC10林地+20灌木+30草地+90湿地；茶园(40)/建成区(50)/水体(80)不进前景"),
    mspa10=cls_stat,
    core_patches=dict(all_count=int(nCore), ge100ha_count=len(patches),
                      ge100ha_area_ha=round(sum(p["area_ha"] for p in patches), 1),
                      src_min_ha=SRC_MIN_HA, src_min_ndvi=SRC_MIN_NDVI,
                      sources=sources, source_count=len(sources),
                      source_area_ha=round(sum(p["area_ha"] for p in sources), 1)),
    simplified_vs_guidos=[
        "结构元固定3x3/8邻接，边缘宽度1像元(10m)，无第三维与自定义宽度",
        "bridge/loop/branch 用连通函数近似（接触核心数/接触点聚类）替代形态学收窄精确口径",
        "与核心直接相接的1像元过渡带计入边缘，Guidos 会把细廊道整体归桥接",
        "源地只用核心区类——该类口径保守、与 Guidos 一致性最高，对后续廊道无系统性偏差"],
    caveat=["WorldCover 湿地类(90)极少，滩涂源地系统性缺失【受限口径·需我确认】",
            "草地(30)含园地草被混分，作前景偏宽——由 NDVI 门槛与面积阈值兜底",
            "源地筛选阈值 100ha/NDVI0.40 为默认方案【需我确认】"])
(OUT_T / "stage2_mspa.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- 5) 出图 -----------------------------------------------------------------
ext10 = extent_of(TR10, W10, H10)
colors = {1: (13, 107, 58), 2: (186, 219, 167), 3: (168, 112, 184),
          4: (230, 152, 61), 5: (250, 210, 88), 6: (196, 156, 118), 7: (128, 128, 128)}
rgb = np.full((H10, W10, 3), 255, np.uint8)
for k, c in colors.items():
    rgb[mspa10 == k] = c
rgb[~analysis10] = (238, 238, 238)
rgb = rgb[::4, ::4]                              # 显示降采样（出图用，不影响数据）
ext10d = ext10
fig, ax = plt.subplots(figsize=(9, 7.4), dpi=150)
ax.imshow(rgb, extent=ext10d, interpolation="nearest")
ax.set_title("福鼎市 MSPA 景观类型图（10m 原生分辨率，前景=林地/灌木/草地/湿地）", fontsize=12)
handles = [plt.Rectangle((0, 0), 1, 1, fc=np.array(c) / 255) for c in colors.values()]
ax.legend(handles, [f"{k}-{CLS[k]}" for k in colors], loc="lower right", fontsize=8, framealpha=0.9)
add_north(ax); add_scalebar(ax, ext10d, km=10); add_note(ax, "坐标系: EPSG:32651 (UTM 51N) · 底图 ESA WorldCover 2021 10m", "bl")
ax.set_xlabel("东向 UTM (m)"); ax.set_ylabel("北向 UTM (m)")
fig.tight_layout(); fig.savefig(OUT_F / "fig_mspa.png"); plt.close(fig)

gridjson = json.loads((PRI / "results/tables/stage0_boundary_stats.json").read_text(encoding="utf-8"))
from rasterio.transform import Affine
TR30_aff = Affine(*TR30[:6])                     # carto.extent_of 需要 Affine 对象，网格JSON里是6元组
ext30 = extent_of(TR30_aff, W30, H30)
bg = np.full((H30, W30, 3), (245, 245, 240), np.uint8)
with rasterio.open(PRI / "results/rasters/wc30_mode.tif") as s:
    wc30 = s.read(1)
bg[wc30 == 80] = (205, 227, 242)
bg[wc30 == 0] = (228, 228, 228)
fig, ax = plt.subplots(figsize=(9, 7.2), dpi=150)
ax.imshow(bg, extent=ext30, interpolation="nearest")
show = np.ma.masked_where(src30 == 0, src30)
ax.imshow(show, extent=ext30, cmap="Greens", interpolation="nearest")
ax.contour(res_arr.astype(float), levels=[0.5], colors="red", linewidths=1.2, extent=ext30)
ax.set_title(f"福鼎市生态源地分布（MSPA核心区≥{SRC_MIN_HA}ha 且 NDVI≥{SRC_MIN_NDVI}，n={len(sources)}）", fontsize=12)
ax.legend([plt.Rectangle((0, 0), 1, 1, fc="green"), plt.Line2D([], [], color="red")],
          ["生态源地(MSPA核心区)", "OSM保护地(弱参考)"], loc="lower right", fontsize=8, framealpha=0.9)
add_north(ax); add_scalebar(ax, ext30, km=10); add_note(ax, "坐标系: EPSG:32651 (UTM 51N) · 生态红线未取得，用 OSM 保护地替代", "bl")
ax.set_xlabel("东向 UTM (m)"); ax.set_ylabel("北向 UTM (m)")
fig.tight_layout(); fig.savefig(OUT_F / "fig_sources.png"); plt.close(fig)
print("[DONE] stage2 v2 完成：mspa10/sources30/两图/stage2_mspa.json")
