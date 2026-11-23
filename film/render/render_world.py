# -*- coding: utf-8 -*-
"""渲染"世界层" —— 一条 777 秒的连续相机路径穿过 12 个区域。

与 render_film 的区别：那个按 `project.json` 的 act 逐个渲染；这个读
`film/world.py` 产出的 `world.scene.json` / `world.shot.json`，一次跑完整条路径。

用法::

    python film/render/render_world.py --size 1920x1080 --crf 20
    python film/render/render_world.py --frames 0,300,700        # 只出几帧（验证用）
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FILM_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(FILM_DIR)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault('WORLD3D_PROJECT_ROOT', ROOT)
os.environ.setdefault('WORLD3D_CONFIGS', os.path.join(ROOT, 'configs'))

import pyglet                                                    # noqa: E402
from passes import ActRenderer                                    # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', default='1920x1080')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--crf', default='20')
    ap.add_argument('--frames', default=None, help='只出这些帧号（逗号分隔）')
    args = ap.parse_args(argv)

    w, h = (int(x) for x in args.size.lower().split('x'))
    shot_path = os.path.join(FILM_DIR, 'shots', 'world.shot.json')
    with open(shot_path, encoding='utf-8') as f:
        shot = json.load(f)
    duration = float(shot['duration'])
    act = {'id': 'world', 'duration': duration,
           'scene': 'scenes/world.scene.json',
           'shot': 'shots/world.shot.json'}

    win = pyglet.window.Window(w, h, visible=False)
    win.switch_to()
    t0 = time.time()
    r = ActRenderer(win, FILM_DIR, act, fps=args.fps)
    print(f'[world] 场景就绪 {time.time() - t0:.1f}s', flush=True)

    total = int(round(duration * args.fps))
    out_dir = os.path.join(FILM_DIR, 'out')
    os.makedirs(out_dir, exist_ok=True)

    if args.frames:
        from pyglet.image import get_buffer_manager
        for fidx in [int(x) for x in args.frames.split(',')]:
            r.draw(fidx / float(args.fps))
            path = os.path.join(out_dir, f'world_f{fidx:05d}.png')
            get_buffer_manager().get_color_buffer().get_image_data().save(path)
            print(f'[world] frame {fidx} -> {path}')
        win.close()
        print('WORLD_FRAMES_OK')
        return 0

    import subprocess
    out = os.path.join(out_dir, 'world.mp4')
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
           '-r', str(args.fps), '-i', '-',
           '-c:v', 'libx264', '-preset', 'medium', '-crf', args.crf,
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    from pyglet.image import get_buffer_manager
    tt0 = time.time()
    for i in range(total):
        r.draw(i / float(args.fps))
        data = get_buffer_manager().get_color_buffer().get_image_data().get_data(
            'RGB', (w * 3 + 3) & ~3)
        proc.stdin.write(data)
        if i % 300 == 0:
            el = time.time() - tt0
            fps = (i + 1) / max(el, 0.1)
            eta = (total - i - 1) / max(fps, 0.1)
            print(f'  [world] {i}/{total} ({fps:.1f} fps, ETA {eta / 60:.1f} min)',
                  flush=True)
    proc.stdin.close()
    proc.wait()
    win.close()
    mb = os.path.getsize(out) / 1048576 if os.path.exists(out) else 0
    print(f'[ok] {out}  {total} 帧 / {duration:.1f}s / {mb:.1f}MB')
    print('WORLD_RENDER_OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
