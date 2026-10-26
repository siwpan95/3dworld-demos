# -*- coding: utf-8 -*-
"""Wardfall 按 ward 规模的 benchmark 基线。

两种信号分开测，因为决定帧率的是 GPU、决定"开城耗时"的是 CPU 几何：

* **几何基线（无显示也能跑）**：直接 import ``build_city_mesh`` 计时，得到
  tris / vertices / vram_bytes / buildings / towers / 几何 build_ms。顶点数与
  显存占用随 ward 数线性增长，是 ward 规模有意义的静态信号（单 draw call，
  GPU 顶点数差异对帧率影响很小）。vram_bytes 按每顶点 13 个 float 估
  （pos3 + nrm3 + uv2 + col4 + kind1 = 52 字节/顶点）。
* **帧率基线（需要真实 GL 窗口）**：对每个规模跑 ``wardfall.py --bench``，
  解析 ``[bench] fps mean=…`` 行。本沙箱没有显示，跑这个会卡住，所以
  默认只出几何基线；在有 GPU 的机器上加 ``--bench-run`` 才会真的测帧率。

用法::

    python scripts/bench_wardfall.py                只出几何基线（无显示可跑）
    python scripts/bench_wardfall.py --bench-run    再补每段 --bench 帧率实测
    python scripts/bench_wardfall.py --bench-sec 8  帧率采样秒数（默认 8）

产出 ``benchmarks/wardfall.json``：每段含 wards / tris / 计数 / build_ms（实测）
与 fps_mean / fps_median / fps_p1 / worst_frame_ms / jank_pct（仅 --bench-run 填）。
"""
import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, 'examples')
OUT = os.path.join(ROOT, 'benchmarks', 'wardfall.json')

# (label, --flag 值或 None, ward 数)
SIZES = [
    ('small',   'small', 24),
    ('default', None,    40),
    ('big',     'big',   64),
]

BENCH_RE = re.compile(
    r'fps mean=(?P<mean>[\d.]+) median=(?P<median>[\d.]+) '
    r'p1=(?P<p1>[\d.]+) worst_frame_ms=(?P<worst>[\d.]+) '
    r'jank_gt20ms=(?P<jank>[\d.]+)%')
WARD_RE = re.compile(
    r'wards=(?P<wards>\d+) buildings=(?P<buildings>\d+) towers=(?P<towers>\d+) '
    r'forest=(?P<forest>\d+) longhouse=(?P<longhouse>\d+) rune=(?P<rune>\d+) '
    r'gate=(?P<gate>\d+) tris=(?P<tris>\d+)')


def _py():
    p = os.path.join(ROOT, '.venv', 'Scripts', 'python.exe')
    return p if os.path.exists(p) else sys.executable


def geom_baseline(n_wards, seed=20261024):
    """无显示：import wardfall 里的 build_city_mesh 计时（纯几何，不碰 GL）。"""
    sys.path.insert(0, EXAMPLES)
    wf = importlib.import_module('wardfall')
    t0 = time.time()
    mb, mbn, town, nb, nt, info = wf.build_city_mesh(seed, n_wards)
    build_ms = (time.time() - t0) * 1000.0
    tris = (mb.vertex_count + mbn.vertex_count) // 3
    vertices = mb.vertex_count + mbn.vertex_count
    # 每顶点 13 个 float：pos3 + nrm3 + uv2 + col4 + kind1 = 52 字节
    vram_bytes = vertices * 52
    return {
        'tris': tris,
        'vertices': vertices,
        'vram_bytes': vram_bytes,
        'buildings': nb,
        'towers': nt,
        'forest': info['forest'],
        'longhouse': info['longhouse'],
        'rune': info['rune'],
        'gate': info['gate'],
        'streets': len(town['streets']),
        'build_ms': round(build_ms, 1),
    }


def bench_run(label, flag, bench_sec):
    """有显示：跑 --bench 解析帧率。失败返回 None（例如本沙箱无显示）。"""
    cmd = [_py(), os.path.join(EXAMPLES, 'wardfall.py'), '--bench', str(bench_sec)]
    if flag:
        cmd += ['--flag', flag]
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                             timeout=bench_sec + 60)
    except subprocess.TimeoutExpired:
        return {'error': 'timeout (no display?)'}
    if out.returncode != 0:
        return {'error': (out.stderr or out.stdout)[-400:]}
    fps = BENCH_RE.search(out.stdout)
    ward = WARD_RE.search(out.stdout)
    if not fps:
        return {'error': 'no [bench] line; stderr=' + (out.stderr or '')[-300:]}
    return {
        'fps_mean': float(fps['mean']),
        'fps_median': float(fps['median']),
        'fps_p1': float(fps['p1']),
        'worst_frame_ms': float(fps['worst']),
        'jank_pct': float(fps['jank']),
        'wards_observed': int(ward['wards']) if ward else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench-run', action='store_true',
                    help='额外跑每段 --bench 实测帧率（需要真实 GL 窗口）')
    ap.add_argument('--bench-sec', type=float, default=8.0,
                    help='--bench 采样秒数（默认 8）')
    args = ap.parse_args()

    rows = []
    for label, flag, n in SIZES:
        g = geom_baseline(n)
        row = {'label': label, 'flag': flag, 'wards': n, **g}
        print(f'[{label}] wards={n} tris={g["tris"]} vtx={g["vertices"]} '
              f'vram={g["vram_bytes"]/1024:.0f}KB build_ms={g["build_ms"]}',
              end='')
        if args.bench_run:
            b = bench_run(label, flag, args.bench_sec)
            if b and 'error' not in b:
                row.update(b)
                print(f'  fps_mean={b["fps_mean"]} worst={b["worst_frame_ms"]}ms')
            else:
                row['fps_mean'] = None
                print(f'  BENCH FAILED: {b.get("error", "unknown")[:120]}')
        else:
            row['fps_mean'] = None
            row['fps_median'] = None
            row['fps_p1'] = None
            row['worst_frame_ms'] = None
            row['jank_pct'] = None
            print('  (fps 留待 --bench-run 在有显示机器上补)')
        rows.append(row)

    doc = {
        'meta': {
            'generated_by': 'scripts/bench_wardfall.py',
            'engine': 'world3d (editable install)',
            'seed': 20261024,
            'note': ('tris/vertices/vram_bytes/build_ms/counts 为本机实测几何基线；'
                     'fps_* 需在有显示的 GPU 机器上跑 '
                     '`python scripts/bench_wardfall.py --bench-run` 填充。'),
            'command_full': 'python scripts/bench_wardfall.py --bench-run',
        },
        'sizes': rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f'\n[ok] wrote {OUT}')


if __name__ == '__main__':
    main()
