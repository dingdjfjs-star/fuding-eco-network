# -*- coding: utf-8 -*-
"""制图公共件：指北针/比例尺/坐标系注记/统一出图。每张图自动带齐制图要素。"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

def extent_of(transform, W, H):
    """栅格仿射 -> matplotlib extent (left,right,bottom,top)，单位 m。"""
    left, top = transform.c, transform.f
    right = left + transform.a * W
    bottom = top + transform.e * H
    return left, right, bottom, top

def add_north(ax):
    ax.annotate("N", xy=(0.955, 0.955), xytext=(0.955, 0.885), xycoords="axes fraction",
                ha="center", va="center", fontsize=13, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6), zorder=10)

def add_scalebar(ax, extent, km=10, loc_y=0.045):
    """extent=(l,r,b,t) 米制 -> 在轴下方画 km 比例尺。"""
    l, r, b, t = extent
    km_deg_m = km * 1000.0
    x0 = l + (r - l) * 0.06
    y0 = b + (t - b) * loc_y
    ax.plot([x0, x0 + km_deg_m], [y0, y0], "k-", lw=3.5, zorder=10)
    for xx in (x0, x0 + km_deg_m):
        ax.plot([xx, xx], [y0 - (t - b) * 0.008, y0 + (t - b) * 0.008], "k-", lw=1.4, zorder=10)
    ax.plot([x0, x0 + km_deg_m / 2], [y0, y0], "w-", lw=1.6, zorder=10)
    ax.text(x0 + km_deg_m / 2, y0 + (t - b) * 0.02, f"{km} km", ha="center",
            fontsize=9, zorder=10, backgroundcolor="none")

def add_note(ax, text, loc="bl"):
    x, y = (0.01, 0.01) if loc == "bl" else (0.99, 0.01)
    ha = "left" if loc == "bl" else "right"
    ax.text(x, y, text, transform=ax.transAxes, fontsize=7, color="#333333",
            ha=ha, va="bottom", zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999", alpha=0.85))

def km_ticks(ax, extent, step_km=10):
    l, r, b, t = extent
    x0 = math.ceil(l / (step_km * 1000)) * step_km * 1000
    y0 = math.ceil(b / (step_km * 1000)) * step_km * 1000
    ax.set_xticks(np.arange(x0, r, step_km * 1000))
    ax.set_yticks(np.arange(y0, t, step_km * 1000))
    ax.set_xticklabels([f"{int(v/1000)}" for v in ax.get_xticks()], fontsize=7)
    ax.set_yticklabels([f"{int(v/1000)}" for v in ax.get_yticks()], fontsize=7)
    ax.set_xlabel("UTM 51N 东向 (km)", fontsize=8)
    ax.set_ylabel("北向 (km)", fontsize=8)

def imshow_grid(ax, arr, transform, cmap="viridis", vmin=None, vmax=None,
                nodata=-9999.0, discrete=None):
    a = np.ma.masked_where((arr == nodata) | ~np.isfinite(arr), arr)
    if discrete is not None:
        cmap = matplotlib.colors.ListedColormap(discrete["colors"])
        cmap.set_bad("white", 0)
        norm = matplotlib.colors.BoundaryNorm(
            discrete["boundaries"], cmap.N)
        im = ax.imshow(a, cmap=cmap, norm=norm, extent=extent_of(transform, arr.shape[1], arr.shape[0]),
                       origin="upper", interpolation="nearest")
    else:
        im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent_of(transform, arr.shape[1], arr.shape[0]),
                       origin="upper", interpolation="nearest")
    return im

def legend_classes(ax, items, title="图例", loc="lower right"):
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="#666666", label=t) for t, c in items]
    leg = ax.legend(handles=handles, title=title, loc=loc, fontsize=7.5,
                    title_fontsize=8, framealpha=0.92)
    return leg

def fig_with_map(arr, transform, out_png, title, cmap="viridis", vmin=None, vmax=None,
                 discrete=None, legend_items=None, note="坐标系: EPSG:32651 (WGS84/UTM 51N)",
                 figsize=(7.2, 8.0), dpi=150, step_km=10):
    l, r, b, t = extent_of(transform, arr.shape[1], arr.shape[0])
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = imshow_grid(ax, arr, transform, cmap=cmap, vmin=vmin, vmax=vmax,
                     discrete=discrete)
    if legend_items:
        legend_classes(ax, legend_items)
    else:
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.8)
    km_ticks(ax, (l, r, b, t), step_km)
    add_north(ax)
    add_scalebar(ax, (l, r, b, t))
    add_note(ax, note)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png
