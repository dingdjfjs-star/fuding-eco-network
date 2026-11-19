# -*- coding: utf-8 -*-
"""
阶段1 数据准备：把复用数据统一到 30m 分析网格，矢量烧录为栅格。
输入全部只读自 fuding-eco-priority（绝不写入对方目录）。
输出：results/rasters/*.tif + results/tables/stage1_prep.json
"""
import json, sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import rasterize
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

# 矢量数据全是 WGS84 经纬度，网格是 UTM51N —— rasterize 不会自动重投影，
# 必须先把几何坐标从 4326 变到 32651，否则全部落在网格外烧出 0 像元
_TF = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
def to_utm(features):
    """把 GeoJSON features 的几何重投影到 EPSG:32651，返回 shapely 几何列表。"""
    geoms = []
    for f in features:
        g = shape(f["geometry"])
        geoms.append(shp_transform(lambda x, y, z=None: _TF.transform(x, y), g))
    return geoms

PRI = Path(r"E:\zcode\探索前沿\fuding-eco-priority")          # 姊妹项目根（只读）
OUT_R = Path(r"E:\zcode\探索前沿\fuding-eco-network\results\rasters")
OUT_T = Path(r"E:\zcode\探索前沿\fuding-eco-network\results\tables")

# ---- 0) 读网格规格（与 2 点项目逐字一致）--------------------------------
grid = json.loads((PRI / "results" / "tables" / "grid_spec.json").read_text(encoding="utf-8"))
CRS = grid["crs"]                       # EPSG:32651，中央经线120°E过福鼎，变形最小
TRANSFORM = tuple(grid["transform"])    # (30,0,194010, 0,-30,3040170)
W, H = grid["width"], grid["height"]    # 2748 x 2074
RES = grid["res"]                       # 30 m

def grid_template(dtype="uint8", nodata=0):
    """统一网格模板：所有输出栅格都从这里生成，杜绝错位。"""
    return dict(driver="GTiff", height=H, width=W, count=1, dtype=dtype,
                crs=CRS, transform=TRANSFORM, nodata=nodata, compress="deflate")

# ---- 1) 一致性断言：复用栅格必须与本网格严丝合缝 -------------------------
checked = {}
for name in ["wc30_mode", "slope30", "flowacc30", "fuding_dem30_utm51n",
             "ndvi2025_30m", "dim_degradation30", "priority_class30"]:
    p = PRI / "results" / "rasters" / f"{name}.tif"
    with rasterio.open(p) as s:
        assert s.crs.to_string() == CRS, f"{name}: CRS不一致 {s.crs}"
        assert (s.width, s.height) == (W, H), f"{name}: 尺寸不一致 {s.width}x{s.height}"
        assert tuple(s.transform)[:6] == TRANSFORM, f"{name}: transform不一致"
        checked[name] = dict(dtype=s.dtypes[0], nodata=s.nodata)
print("[OK] 7 个复用栅格与本网格完全对齐")

# ---- 2) 矢量 -> 栅格（先重投影 4326->32651 再烧录）-----------------------
# 烧录参数 all_touched=False：像元中心落在要素内才算，避免道路"膨胀"
def burn(features, out_name, default_value, dtype="uint8", nodata=0):
    """重投影 GeoJSON 要素并烧到统一网格，返回像元计数做非空断言。"""
    geoms = to_utm(features)
    arr = rasterize(
        [(g, default_value) for g in geoms],
        out_shape=(H, W), transform=TRANSFORM,
        fill=nodata, all_touched=False, dtype=dtype)
    meta = grid_template(dtype=dtype, nodata=nodata)
    with rasterio.open(OUT_R / out_name, "w", **meta) as d:
        d.write(arr, 1)
    n = int((arr != nodata).sum())
    assert n > 0, f"{out_name} 烧出 0 像元——先查投影再查几何！"
    return n

# 2a) 道路分级：motorway/trunk/primary=3(高阻力) secondary=2 tertiary=1
roads = json.loads((PRI / "data" / "osm" / "roads.geojson").read_text(encoding="utf-8"))["features"]
LV = {"motorway": 3, "trunk": 3, "primary": 3, "secondary": 2, "tertiary": 1}
f_by_lv = {1: [], 2: [], 3: []}
n_skip = 0
for f in roads:
    lv = LV.get(f["properties"].get("highway", ""))
    if lv is None:
        n_skip += 1; continue           # 其他等级(footway等)不参与阻力
    f_by_lv[lv].append(f)
# 分级从低到高烧录，高等级覆盖低等级（同像元取更严的限制）
road_arr = np.zeros((H, W), dtype="uint8")
for lv in (1, 2, 3):
    if f_by_lv[lv]:
        g_arr = rasterize(to_utm(f_by_lv[lv]), out_shape=(H, W), transform=TRANSFORM,
                          fill=0, all_touched=False, dtype="uint8")
        road_arr = np.maximum(road_arr, g_arr * lv)
with rasterio.open(OUT_R / "road_level30.tif", "w", **grid_template()) as d:
    d.write(road_arr, 1)
# 长度合理性检查：UTM 下的真实公里数（OSM 口径，含 2km 缓冲带内的过境道路）
len_km = {lv: sum(g.length for g in to_utm(f_by_lv[lv])) / 1000 for lv in (1, 2, 3)}
print(f"[OK] road_level30: 3级={int((road_arr==3).sum())}px/{len_km[3]:.0f}km "
      f"2级={int((road_arr==2).sum())}px/{len_km[2]:.0f}km "
      f"1级={int((road_arr==1).sum())}px/{len_km[1]:.0f}km, 跳过其他等级 {n_skip} 条")

# 2b) 居民点：184 个村/镇点（阻力阶段用 EDT 距离场扩散）
vill = json.loads((PRI / "data" / "osm" / "villages.geojson").read_text(encoding="utf-8"))["features"]
n_v = burn(vill, "village30.tif", 1)
print(f"[OK] village30: {n_v} 像元（{len(vill)} 个点）")

# 2c) 保护地：8 处（太姥山/海洋保护区等），面状烧录
resv = json.loads((PRI / "data" / "osm" / "reserves.geojson").read_text(encoding="utf-8"))["features"]
n_r = burn(resv, "reserve30.tif", 1)
print(f"[OK] reserve30: {n_r} 像元 ≈ {n_r*0.09:.1f} ha（{len(resv)} 处）")

# ---- 3) 水系：flowacc 阈值法（与 DEM 自洽，不用 realrivers.js 的局部坐标）--
with rasterio.open(PRI / "results" / "rasters" / "flowacc30.tif") as s:
    fa = s.read(1).astype("float32")
valid = fa[fa > 0]
print(f"[OK] flowacc 范围 {valid.min():.0f}~{valid.max():.0f}, 中位 {np.median(valid):.0f}")
TH = 1000   # 汇流累积≥1000个上游像元=汇水0.9km² 才算溪流：兼顾"成网"与"不过密"
stream = (fa >= TH).astype("uint8")
with rasterio.open(OUT_R / "stream30.tif", "w", **grid_template()) as d:
    d.write(stream, 1)
print(f"[OK] stream30(阈值{TH}): {int(stream.sum())} 像元 ≈ {int(stream.sum())*0.0009:.1f} km 长度量级")

# ---- 4) 质量统计：NDVI/坡度/优先区值域确认（按元数据 nodata 掩膜）---------
stats = {}
for name, key in [("ndvi2025_30m", "ndvi2025"), ("slope30", "slope"),
                  ("priority_class30", "priority"), ("dim_degradation30", "degradation")]:
    with rasterio.open(PRI / "results" / "rasters" / f"{name}.tif") as s:
        a = s.read(1).astype("float32")
        nd = s.nodata if s.nodata is not None else 0.0
    v = a[(a != nd) & (a != 0) & (a > -9900)]   # 排除 nodata 与边界外 0 值
    stats[key] = dict(min=float(v.min()), p50=float(np.median(v)),
                      max=float(v.max()), valid_px=int(v.size))
    print(f"[OK] {name}: min={stats[key]['min']:.3f} p50={stats[key]['p50']:.3f} "
          f"max={stats[key]['max']:.3f} valid={stats[key]['valid_px']}")

# ---- 5) 数据清单（溯源表）-------------------------------------------------
manifest = dict(
    grid=dict(crs=CRS, res=RES, transform=list(TRANSFORM), width=W, height=H),
    reused={k: str(PRI / "results" / "rasters" / f"{k}.tif") for k in checked},
    derived={"road_level30": "OSM roads 5级->3档(3=高速/主干/一级,2=二级,1=三级)",
             "village30": "OSM 184 居民点", "reserve30": "OSM 8 保护地",
             "stream30": f"flowacc30>={TH}(汇水0.9km²)溪流网"},
    qa=dict(ndvi=stats["ndvi2025"], slope=stats["slope"],
            priority=stats["priority"], degradation=stats["degradation"],
            road_px=dict(lv3=int((road_arr==3).sum()), lv2=int((road_arr==2).sum()),
                         lv1=int((road_arr==1).sum())),
            stream_px=int(stream.sum()), reserve_px=n_r,
            skipped_roads=n_skip))
(OUT_T / "stage1_prep.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("[DONE] stage1_prep.json 已写出")
