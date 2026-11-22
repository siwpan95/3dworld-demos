# -*- coding: utf-8 -*-
"""幕间拼接 —— 把各幕 MP4 用 `xfade` 串成一条时间线。

为什么在只有两幕的时候就要做
--------------------------
现在拼一次，成本是"一条 88 秒的片子"；等 12 幕都做完再拼，色彩跳变 / 节奏断裂
会**一次暴露 11 处**，而那时每条幕都要重渲（十几分钟起）。拼接器本身不难，
难的是"发现问题的时间点"。

实现要点
--------
* **同尺寸**：`xfade` 要求两路输入分辨率一致，而各幕可能在不同 `--size` 下渲过
  （第 I 幕 1080p、第 II 幕 540p）。所以每路先 `scale` 到统一尺寸再进 xfade。
* **offset 是累计量**：第 k 次过渡的 offset = 前 k 段时长之和 − (k+1)×xfade，
  不是"上一段时长"。写错会让过渡整体后移、末尾拖黑。
* **时长可预测**：成片 = Σ幕时长 − (n−1)×xfade。这条要当成断言来验。

用法::

    python film/render/concat.py                      # 已有成片的幕按顺序拼
    python film/render/concat.py --out out/v1.mp4 --crf 18
    python film/render/concat.py --dry-run            # 只打印计划
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILM_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(FILM_DIR)

#: 参与拼接的幕状态（built = 完整；partial = 画面成立但缺能力，先拼进来）
CONCAT_STATUS = ('built', 'partial')


def load_project():
    with open(os.path.join(FILM_DIR, 'project.json'), encoding='utf-8') as f:
        return json.load(f)


def ffmpeg_exe():
    cands = []
    if os.environ.get('WORLD3D_FFMPEG'):
        cands.append(os.environ['WORLD3D_FFMPEG'])
    w = shutil.which('ffmpeg')
    if w:
        cands.append(w)
    for exe in cands:
        try:
            subprocess.run([exe, '-version'], capture_output=True, timeout=20, check=True)
            return exe
        except Exception:
            continue
    raise SystemExit('找不到可用的 ffmpeg')


def duration_of(path):
    """用 `ffmpeg -i` 读时长（本机没有 ffprobe，`E:\\ffmpeg\\bin` 只有 ffmpeg.exe）。"""
    r = subprocess.run([ffmpeg_exe(), '-hide_banner', '-i', path],
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace')
    m = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.\d+)', r.stderr or '')
    if not m:
        raise RuntimeError(f'读不到时长: {path}\n{(r.stderr or "")[-400:]}')
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def collect(project):
    """按 acts 顺序收集已出片的幕。"""
    out = []
    for a in project['acts']:
        if a.get('status') not in CONCAT_STATUS:
            continue
        p = os.path.join(FILM_DIR, 'out', f'{a["id"]}.mp4')
        if not os.path.exists(p):
            print(f'  [skip] {a["id"]}: 没有 {p}')
            continue
        out.append(dict(id=a['id'], title=a['title'], path=p,
                        duration=a['duration'], real=duration_of(p)))
    return out


def build_filter(items, xfade, size):
    """每路先 scale 到统一尺寸，再链式 xfade；返回 (filter_str, 输出标签)。"""
    w, h = size
    parts = []
    labels = []
    for i in range(len(items)):
        lab = f'[s{i}]'
        parts.append(f'[{i}:v]scale={w}:{h}:flags=lanczos,setsar=1{lab}')
        labels.append(lab)
    prev = labels[0]
    acc = 0.0
    for i in range(1, len(items)):
        acc += items[i - 1]['real'] - xfade
        lab = f'[vx{i}]'
        parts.append(f'{prev}{labels[i]}xfade=transition=fade:'
                     f'duration={xfade}:offset={acc:.3f}{lab}')
        prev = lab
    return ';'.join(parts), prev, acc + items[-1]['real']


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--crf', type=int, default=18)
    ap.add_argument('--size', default=None, help='统一尺寸，默认取 project.stage')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)

    project = load_project()
    xfade = float(project.get('xfade', 1.2))
    size = tuple(project['stage']) if not args.size else tuple(
        int(v) for v in args.size.lower().split('x'))
    items = collect(project)
    if len(items) < 2:
        raise SystemExit(f'可拼接的幕不足 2 个（当前 {len(items)} 个）')

    print(f'拼接 {len(items)} 幕 → {size[0]}x{size[1]}，xfade {xfade}s')
    for it in items:
        print(f'  {it["id"]:12} {it["real"]:7.2f}s  {it["path"]}')
    filter_str, last, predicted = build_filter(items, xfade, size)
    expect = sum(i['real'] for i in items) - xfade * (len(items) - 1)
    print(f'预计时长 {expect:.2f}s（Σ幕 − {len(items) - 1}×{xfade}）')

    out_path = args.out or os.path.join(FILM_DIR, 'out', 'worldwright_v1.mp4')
    out_path = out_path if os.path.isabs(out_path) else os.path.join(ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd = [ffmpeg_exe(), '-y', '-hide_banner', '-loglevel', 'error']
    for it in items:
        cmd += ['-i', it['path']]
    cmd += ['-filter_complex', filter_str, '-map', last,
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(args.crf),
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out_path]
    if args.dry_run:
        print('[dry-run]', ' '.join(cmd[:6]), '…')
        return 0

    subprocess.run(cmd, check=True)
    got = duration_of(out_path)
    size_mb = os.path.getsize(out_path) / 1048576
    print(f'[ok] {out_path}  {got:.2f}s（预期 {expect:.2f}s，差 {got - expect:+.2f}s）  {size_mb:.1f}MB')

    # 解码检查：零错误才算过（拼接最容易出的是"末尾几帧读不出"）
    chk = subprocess.run([ffmpeg_exe(), '-v', 'error', '-i', out_path, '-f', 'null', '-'],
                         capture_output=True, text=True, encoding='utf-8', errors='replace')
    if chk.stderr.strip():
        print('[warn] 解码有错误输出：\n' + chk.stderr.strip()[:500])
        print('CONCAT_FAILED')
        return 1
    if abs(got - expect) > 0.10:
        print('CONCAT_FAILED（时长偏差 > 0.1s）')
        return 1
    print('CONCAT_OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
