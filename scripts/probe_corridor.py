# -*- coding: utf-8 -*-
"""探针：廊道/源地栅格内容核验（渲染bug定位用）。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import rasterio, numpy as np
NET = r"E:\zcode\探索前沿\fuding-eco-network\results\rasters"
m = rasterio.open(NET + r"\corridor_mask30.tif").read(1)
s = rasterio.open(NET + r"\sources30.tif").read(1)
c = rasterio.open(NET + r"\corridor_count30.tif").read(1)
print("corridor px:", int((m > 0).sum()), " max overlap count:", int(c.max()))
print("corridor px inside sources:", int(((m > 0) & (s > 0)).sum()),
      " outside sources:", int(((m > 0) & (s == 0)).sum()))
print("source px total:", int((s > 0).sum()))
vals, cnts = np.unique(s[s > 0], return_counts=True)
print("source ids/px:", dict(zip(vals.tolist(), cnts.tolist())))
