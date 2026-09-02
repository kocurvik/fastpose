# -*- coding: utf-8 -*-
"""Render the social card (og.png, 1200x630) from the benchmark data.

    python src/make_card.py

Draws the accuracy-time frontier the page leads with, in the page's own
palette, beside the headline and the measured speed-up ranges.
"""
import io
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# the page's dark palette
PAGE, SURFACE = '#0b0e13', '#141920'
INK, INK2, MUTED = '#eef2f7', '#aeb7c4', '#7d8794'
GRID, AXIS = '#222932', '#38414d'
S1, S2, S3, BASE = '#3987e5', '#d95926', '#199e70', '#6d747f'

DISPLAY = ['Archivo', 'Segoe UI', 'DejaVu Sans']
BODY = ['Source Sans Pro', 'Segoe UI', 'DejaVu Sans']
MONO = ['IBM Plex Mono', 'Consolas', 'DejaVu Sans Mono']

# the panel the page opens with
PANEL = dict(matcher='RoMa', n=4096, dataset='scannetpp', problem='kf_reposed')
CURVES = [('poselib_1c', 'PoseLib, 1 core', BASE),
          ('fastpose_1c', 'fastpose, 1 core', S1),
          ('fastpose_4c', 'fastpose, 4 cores', S2),
          ('fastpose_gpu', 'fastpose, GPU', S3)]
STATS = [('fastpose_1c', '1 core', S1),
         ('fastpose_4c', '4 cores', S2),
         ('fastpose_gpu', 'GPU', S3)]


def load_rows():
    with io.open(os.path.join(HERE, 'bench.json'), encoding='utf-8') as f:
        return json.load(f)['rows']


def speed_range(rows, device):
    """min-max speed-up over PoseLib across every solver and dataset."""
    ref, got = {}, {}
    for r in rows:
        if r['matcher'] != PANEL['matcher'] or r['n'] != PANEL['n'] or r['iters'] != 5000:
            continue
        key = (r['dataset'], r['problem'])
        if r['device'] == 'poselib_1c':
            ref[key] = r['mean_runtime']
        elif r['device'] == device:
            got[key] = r['mean_runtime']
    ratios = [ref[k] / got[k] for k in got if k in ref]
    return min(ratios), max(ratios)


def fmt(v):
    return '%.0f' % v if v >= 10 else '%.1f' % v


def build():
    rows = load_rows()
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=PAGE)

    # ------------------------------------------------ left column: the claim
    fig.text(0.052, 0.885, 'F A S T P O S E', color=S1, fontsize=14.5,
             fontweight='bold', family=DISPLAY)

    for i, line in enumerate(['PoseLib accuracy,', 'a fraction of', 'the runtime.']):
        fig.text(0.052, 0.775 - i * 0.093, line,
                 color=S1 if i else INK, fontsize=34, fontweight='bold',
                 family=DISPLAY)

    fig.text(0.052, 0.475, 'A numba-compiled LO-RANSAC engine', color=INK2,
             fontsize=14, family=BODY)
    fig.text(0.052, 0.425, 'for robust relative pose.', color=INK2,
             fontsize=14, family=BODY)

    y = 0.315
    for device, label, color in STATS:
        lo, hi = speed_range(rows, device)
        fig.text(0.052, y, '●', color=color, fontsize=10, family=BODY, va='center')
        fig.text(0.072, y, label, color=INK2, fontsize=14.5, family=BODY, va='center')
        fig.text(0.335, y, '%s–%s×' % (fmt(lo), fmt(hi)), color=INK,
                 fontsize=17, fontweight='bold', family=MONO, va='center', ha='right')
        y -= 0.077

    fig.text(0.052, 0.065, 'github.com/kocurvik/fastpose', color=MUTED,
             fontsize=13, family=MONO)
    fig.text(0.948, 0.065, 'ETH3D  ·  ScanNet++  ·  PhotoTourism',
             color=MUTED, fontsize=13, family=BODY, ha='right')

    # ------------------------------------------------ right column: the frontier
    ax = fig.add_axes([0.45, 0.185, 0.50, 0.615])
    ax.set_facecolor(SURFACE)

    lows, highs = [], []
    for device, label, color in CURVES:
        pts = sorted((r['mean_runtime'], r['pose_mAA_10'])
                     for r in rows
                     if all(r[k] == v for k, v in PANEL.items())
                     and r['device'] == device)
        if not pts:
            continue
        xs, ys = zip(*pts)
        lows.append(min(ys))
        highs.append(max(ys))
        ax.plot(xs, ys, color=color, lw=2.4, marker='o', ms=6.5, mec=SURFACE,
                mew=1.8, zorder=3, label=label, solid_capstyle='round')

    # keep the accuracy axis honest: a window wide enough to show the curves
    # are flat, rather than one zoomed until noise looks like a trend
    mid = (min(lows) + max(highs)) / 2.0
    ax.set_ylim(mid - 1.25, mid + 1.25)

    ax.set_xscale('log')
    ax.set_xlabel('mean runtime per pair (ms, log)', color=MUTED, fontsize=12,
                  family=BODY, labelpad=8)
    ax.set_ylabel('pose mAA@10°', color=MUTED, fontsize=12, family=BODY, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=11, length=0)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_family(MONO)
    ax.grid(True, which='major', color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)

    leg = ax.legend(loc='lower left', frameon=True, fontsize=11.5,
                    facecolor=SURFACE, edgecolor=AXIS, labelcolor=INK2,
                    handlelength=1.7, borderpad=0.65, labelspacing=0.45)
    leg.get_frame().set_linewidth(0.8)
    for text in leg.get_texts():
        text.set_family(BODY)

    ax.set_title('ScanNet++ · calibrated RePoseD 3-point · RoMa v2 4096',
                 color=MUTED, fontsize=11.5, family=BODY, pad=11, loc='left')

    out = os.path.join(ROOT, 'og.png')
    fig.savefig(out, facecolor=PAGE, dpi=100)
    plt.close(fig)

    # social scrapers are happier without an alpha channel
    Image.open(out).convert('RGB').save(out)
    print('wrote %s (%.0f KB)' % (out, os.path.getsize(out) / 1024.0))


if __name__ == '__main__':
    build()
