# -*- coding: utf-8 -*-
"""
阶段3 阻力面构建：土地利用/坡度/道路/居民点/水系 五因子，max 融合；
并生成低/高两组扰动阻力面供阶段4 敏感性分析。
赋值依据逐项写进输出 JSON：有文献区间标[文献]，纯判断标[经验值]。
输出：results/rasters/res30_base/low/high.tif + results/tables/stage3_resistance.json
"""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import rasterio
from scipy import ndimage

PRI = Path(r"E:\zcode\探索前沿\fuding-eco-priority")
NET = Path(r"E:\zcode\探索前沿\fuding-eco-network")
OUT_R, OUT_T = NET / "results" / "rasters", NET / "results" / "tables"

grid = json.loads((PRI / "results" / "tables" / "grid_spec.json").read_text(encoding="utf-8"))
TRANSFORM = tuple(grid["transform"]); W, H = grid["width"], grid["height"]
RES = 30.0  # 像元边长 m

def rd(name):
    with rasterio.open(name) as s:
        return s.read(1)

wc = rd(PRI / "results/rasters/wc30_mode.tif")
slope = rd(PRI / "results/rasters/slope30.tif").astype("float32")
road = rd(NET / "results/rasters/road_level30.tif")
vill = rd(NET / "results/rasters/village30.tif")
stream = rd(NET / "results/rasters/stream30.tif")
analysis = wc > 0

# ---- 因子1 土地利用阻力（WC30 类查表）--------------------------------------
# [文献区间+经验值]：林地1、灌木3、草地10、耕地30（茶园在耕地类，条垦常绿取中低档）、
# 建成区100、裸地80、水体100、湿地20。文献常见区间 林地1~建成区100+，取整档便于解释。
LU_RES = {10: 1, 20: 3, 30: 10, 40: 30, 50: 100, 60: 80, 80: 100, 90: 20}
lu = np.full((H, W), 100.0, np.float32)          # 未列入类的兜底=最高阻力（安全默认）
for k, v in LU_RES.items():
    lu[wc == k] = v
lu[~analysis] = 0.0                               # 分析范围外 nodata

# ---- 因子2 坡度阻力（slope30 分段）------------------------------------------
# [经验值] <2°:1 → >25°:32 等比翻倍；福鼎53%坡地>15°（2点项目 stage2_frag），山地县坡度是硬约束
slope_res = np.full((H, W), 32.0, np.float32)
for hi, v in [(2, 1), (5, 2), (8, 4), (15, 8), (25, 16)]:
    slope_res[(slope > 0) & (slope < hi)] = v
slope_res[~analysis] = 0.0

# ---- 因子3 道路阻力（等级峰值 + 指数距离衰减，EDT 单位是像元×30m）-----------
# [文献+经验值] 高速/国道峰值400 λ300m；省道200/200m；县乡道100/100m。
# res = 1 + (peak-1)·exp(-d/λ)：路面上=峰值，随距离指数衰减到 1（不超调、单调、平滑）
def road_factor(peak, lam_px, road_mask):
    d_px = ndimage.distance_transform_edt(road_mask == 0)   # 到最近该级道路的像元距离
    return 1.0 + (peak - 1.0) * np.exp(-d_px / lam_px)

road_res = np.maximum.reduce([
    road_factor(400, 300 / RES, road == 3),
    road_factor(200, 200 / RES, road == 2),
    road_factor(100, 100 / RES, road == 1)])
road_res[~analysis] = 0.0

# ---- 因子4 居民点阻力（184 村点 + 指数衰减）---------------------------------
# [经验值] 村点峰值300 λ250m：村庄/家犬/采集干扰半径≈250m；建成区在因子1已有100兜底
vill_res = 1.0 + (300 - 1.0) * np.exp(-ndimage.distance_transform_edt(vill == 0) / (250 / RES))
vill_res[~analysis] = 0.0

# ---- 因子5 水系阻力（溪流像元抬升，窄线状不加衰减）--------------------------
# [经验值] 溪流像元 60：跨溪有成本但非完全阻断；大河/海面在因子1 水体100 兜底
stream_res = np.maximum(lu, 1.0)
stream_res[stream > 0] = np.maximum(stream_res[stream > 0], 60.0)

# ---- 融合：max（保守口径：最强障碍主导通行选择）-----------------------------
base = np.maximum.reduce([lu, slope_res, road_res, vill_res, stream_res])
base = np.clip(base, 1.0, 100.0) * analysis        # 分析范围内 [1,100]，范围外 0(nodata)

# ---- 敏感性方案：道路+居民点因子 ×0.5 / ×2（基础设施主导的扰动）-------------
road_low = 1.0 + (road_res - 1.0) * 0.5
vill_low = 1.0 + (vill_res - 1.0) * 0.5
low = np.clip(np.maximum.reduce([lu, slope_res, road_low, vill_low, stream_res]), 1, 100) * analysis
road_hi = np.clip(1.0 + (road_res - 1.0) * 2.0, 1, 200)
vill_hi = np.clip(1.0 + (vill_res - 1.0) * 2.0, 1, 200)
high = np.clip(np.maximum.reduce([lu, slope_res, road_hi, vill_hi, stream_res]), 1, 200) * analysis

meta = dict(driver="GTiff", height=H, width=W, count=1, dtype="float32",
            crs=grid["crs"], transform=TRANSFORM, nodata=0, compress="deflate")
for nm, arr in [("res30_base", base), ("res30_low", low), ("res30_high", high)]:
    with rasterio.open(OUT_R / f"{nm}.tif", "w", **meta) as d:
        d.write(arr.astype("float32"), 1)

# ---- 统计与赋值依据表 --------------------------------------------------------
va = base[base > 0]
out = dict(
    combine="max(五因子)，范围[1,100]，分析范围外nodata=0",
    land_use={str(k): v for k, v in LU_RES.items()},
    land_use_basis="林地1/建成区100为文献通用锚点；草地10、耕地30(含茶园)、裸地80、湿地20、水体100为[经验值]档位，需我确认",
    slope_basis="分段1/2/4/8/16/32 [经验值]，与2点项目坡度>15°占53.2%呼应",
    road_basis="3级峰值400λ300m、2级200λ200m、1级100λ100m，指数衰减 [文献区间+经验值]",
    village_basis="峰值300 λ250m [经验值]",
    stream_basis="溪流像元60 [经验值]",
    sensitivity=dict(design="道路+居民点阻力因子 ×0.5(low)/×2.0(high)，其余不动",
                     base=dict(p50=float(np.median(va)), p95=float(np.percentile(va, 95)), max=float(va.max())),
                     low_p50=float(np.median(low[low > 0])), high_p50=float(np.median(high[high > 0])),
                     mean_ratio_low_high=float(np.mean(high[high > 0]) / max(np.mean(low[low > 0]), 1e-9))))
(OUT_T / "stage3_resistance.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] 阻力面 base: p50={out['sensitivity']['base']['p50']:.1f} p95={out['sensitivity']['base']['p95']:.1f} "
      f"max={out['sensitivity']['base']['max']:.1f}; low/high p50={out['sensitivity']['low_p50']:.1f}/{out['sensitivity']['high_p50']:.1f}")
print("[DONE] stage3 阻力面三方案已写出")
