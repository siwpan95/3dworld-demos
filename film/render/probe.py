# -*- coding: utf-8 -*-
"""单帧探针 —— 给一个 ``t``，渲染一张 PNG。

为什么先有探针再有成片
--------------------
12 幕 21600 帧，如果每改一次镜头参数就整幕重渲，迭代周期会从"几秒"变成"几分钟"，
最后人就不改了 —— 成片质量停在"能跑"的水平。探针把反馈压到单帧，
镜头与材质才调得动。

用法::

    python film/render/probe.py --act 01_void --t 16 --out .baseline/probe_16.png
    python film/render/probe.py --act 01_void --t 0,8,16,24 --out-dir .baseline/void

``--t`` 支持逗号并列，一次出多帧（幕内关键帧的"胶片条"）。
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILM_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(FILM_DIR)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import json                                                    # noqa: E402

import pyglet                                                  # noqa: E402

from passes import ActRenderer                                 # noqa: E402
from world3d.platform.app import capture                        # noqa: E402


def load_project(film_dir):
    with open(os.path.join(film_dir, 'project.json'), encoding='utf-8') as f:
        return json.load(f)


def find_act(project, act_id):
    for a in project['acts']:
        if a['id'] == act_id:
            return a
    raise SystemExit(f'project.json 里没有幕 {act_id!r}（有：'
                     f'{", ".join(a["id"] for a in project["acts"])}）')


def parse_size(text):
    w, h = str(text).lower().split('x')
    return int(w), int(h)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--act', required=True, help='幕 id，如 01_void')
    ap.add_argument('--t', default='0', help='幕内时刻（秒），逗号并列可出多帧')
    ap.add_argument('--out', default=None, help='单帧输出路径（--t 只有一个时有效）')
    ap.add_argument('--out-dir', default=None, help='多帧输出目录')
    ap.add_argument('--size', default=None, help='如 1280x720；默认取 project.json')
    ap.add_argument('--particles', type=int, default=None, help='覆盖粒子数（调试用）')
    args = ap.parse_args(argv)

    project = load_project(FILM_DIR)
    act = find_act(project, args.act)
    w, h = parse_size(args.size) if args.size else tuple(project['stage'])
    times = [float(x) for x in str(args.t).split(',') if x.strip()]

    out_dir = args.out_dir or os.path.join(ROOT, '.baseline')
    os.makedirs(out_dir, exist_ok=True)

    # visible=True：GL 上下文需要真实窗口；离屏 FBO 会绕过 Renderer 的默认帧缓冲路径
    window = pyglet.window.Window(w, h, caption=f'worldwright · {act["id"]}', visible=True)
    window.switch_to()
    try:
        kwargs = {}
        if args.particles is not None:
            kwargs['particle_count'] = args.particles
        renderer = ActRenderer(window, FILM_DIR, act, fps=project['fps'], **kwargs)
        for t in times:
            energy, applied = renderer.draw(t)
            if args.out and len(times) == 1:
                path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
            else:
                path = os.path.join(out_dir, f'{act["id"]}_t{t:05.2f}.png')
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            capture(window, path)
            print(f'[probe] t={t:6.2f}s energy={energy:.3f} '
                  f'channels={applied}/{len(renderer.shot.channels)} -> {path}', flush=True)
    finally:
        window.close()
    print('PROBE_OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
