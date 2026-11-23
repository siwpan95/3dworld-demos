# -*- coding: utf-8 -*-
"""批量重渲 —— 按统一分辨率重出所有幕。

为什么需要它
-----------
各幕是分期做的：第 I 幕按 1080p 出、其余按 540p 迭代（快 3.5 倍）。拼接器
会把 540p 上采样到 1080p，成片里第 I 幕明显更锐 —— 这叫"混分辨率"，
必须在定稿前统一。

用法::

    python film/render/render_all.py --size 1920x1080 --crf 20
    python film/render/render_all.py --only 05_crystal 06_cave
    python film/render/render_all.py --skip 01_void          # 已是 1080p
    python film/render/render_all.py --dry-run

设计要点
--------
* **逐幕独立**：某幕失败不影响其它幕，最后汇总失败清单并以非零退出码结束；
* **可续跑**：已存在的成片默认跳过（`--force` 覆盖），中断后重跑不会从头再来；
* **不静默**：每幕打印耗时与体积，失败时把 render_film 的末尾几行贴出来。
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FILM_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(FILM_DIR)

ALL_ACTS = ('01_void', '02_terrain', '03_city', '04_machine', '05_crystal',
            '06_cave', '07_swarm', '08_body', '09_field', '10_void',
            '11_loop', '12_finale')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', default='1920x1080')
    ap.add_argument('--crf', default='20')
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--skip', nargs='*', default=None)
    ap.add_argument('--force', action='store_true', help='已存在也重渲')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)

    acts = list(args.only) if args.only else list(ALL_ACTS)
    skip = set(args.skip or ())
    acts = [a for a in acts if a not in skip]

    out_dir = os.path.join(FILM_DIR, 'out')
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env.setdefault('WORLD3D_PROJECT_ROOT', ROOT)
    env.setdefault('WORLD3D_CONFIGS', os.path.join(ROOT, 'configs'))

    print(f'批量重渲：{len(acts)} 幕 @ {args.size} crf{args.crf}')
    ok, failed, skipped = [], [], []
    t_all = time.time()
    for i, act in enumerate(acts, 1):
        target = os.path.join(out_dir, f'{act}.mp4')
        if os.path.exists(target) and not args.force:
            skipped.append(act)
            print(f'[{i}/{len(acts)}] {act:12} 跳过（已存在；--force 覆盖）')
            continue
        cmd = [sys.executable, os.path.join(HERE, 'render_film.py'),
               '--act', act, '--size', args.size, '--crf', args.crf]
        if args.dry_run:
            print(f'[{i}/{len(acts)}] {" ".join(cmd[-6:])}')
            continue
        t0 = time.time()
        print(f'[{i}/{len(acts)}] {act:12} 开始…', flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', env=env)
        dt = time.time() - t0
        if r.returncode == 0:
            mb = os.path.getsize(target) / 1048576 if os.path.exists(target) else 0.0
            ok.append(act)
            print(f'[{i}/{len(acts)}] {act:12} 完成 {dt:.0f}s / {mb:.1f}MB', flush=True)
        else:
            failed.append(act)
            tail = '\n'.join((r.stderr or r.stdout or '').strip().splitlines()[-6:])
            print(f'[{i}/{len(acts)}] {act:12} **失败**（{dt:.0f}s）\n{tail}', flush=True)

    if args.dry_run:
        print('== DRY_RUN_OK ==')
        return 0
    total = time.time() - t_all
    print('-' * 60)
    print(f'完成 {len(ok)} / 跳过 {len(skipped)} / 失败 {len(failed)}，总耗时 {total / 60:.1f} 分钟')
    if failed:
        print('失败清单:', ' '.join(failed))
        print('== RENDER_ALL_PARTIAL ==')
        return 1
    print('== RENDER_ALL_OK ==')
    return 0


if __name__ == '__main__':
    sys.exit(main())
