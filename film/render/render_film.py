# -*- coding: utf-8 -*-
"""逐帧渲染一幕 → MP4（确定性 ``dt = 1/fps``，不吃墙钟）。

为什么不用实时录屏
----------------
实时录屏的帧率会漂：同一段镜头在慢机器上少走几步、快机器上多走几步，
"同一秒"永远不是同一个画面。本脚本手动推进时间轴、每帧 ``glReadPixels``
喂 ffmpeg 的 rawvideo 管道 —— 帧序号与动画时间严格一一对应。

⚠️ **粒子相位是累加的**（``ParticleSystem.advance`` 只改变推进速率、不改已积累相位），
所以渲染**必须从 t=0 顺序推进**，不能跳帧。``--check-determinism`` 因此是
"两次都从头跑到第 N 帧再比字节"，而不是"同一个 t 渲两次"——后者只能证明
着色器是纯函数，证明不了整条链路。

用法::

    python film/render/render_film.py --act 01_void --size 960x540
    python film/render/render_film.py --act 01_void --size 1920x1080 --crf 18
    python film/render/render_film.py --act 01_void --check-determinism 120
"""
import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FILM_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(FILM_DIR)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import pyglet                                                       # noqa: E402
from pyglet.gl import GL_RGBA, GL_UNSIGNED_BYTE, glFinish, glReadPixels  # noqa: E402

from passes import ActRenderer                                      # noqa: E402


def load_project():
    with open(os.path.join(FILM_DIR, 'project.json'), encoding='utf-8') as f:
        return json.load(f)


def find_act(project, act_id):
    for a in project['acts']:
        if a['id'] == act_id:
            return a
    raise SystemExit(f'project.json 里没有幕 {act_id!r}')


def parse_size(text):
    w, h = str(text).lower().split('x')
    return int(w), int(h)


def ffmpeg_exe():
    """``WORLD3D_FFMPEG`` → PATH，逐一校验 ``-version`` 能跑（PATH 里可能有坏壳）。"""
    cands = []
    if os.environ.get('WORLD3D_FFMPEG'):
        cands.append(os.environ['WORLD3D_FFMPEG'])
    which = shutil.which('ffmpeg')
    if which:
        cands.append(which)
    for exe in cands:
        try:
            subprocess.run([exe, '-version'], capture_output=True, timeout=20, check=True)
            return exe
        except Exception:
            continue
    raise SystemExit('找不到可用的 ffmpeg：设 WORLD3D_FFMPEG 或让它进 PATH')


def _grab(window, w, h):
    """读回一帧 RGBA（GL 原点在左下，编码时由 ffmpeg 的 vflip 翻回来）。"""
    buf = (ctypes.c_ubyte * (w * h * 4))()
    glFinish()
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf)
    return bytes(buf)


def render_act(window, act, fps, out_path, crf, size, stop_frame=None, quiet=False):
    """渲染一幕；返回 (帧数, 耗时)。``stop_frame`` 用于确定性检查（提前停）。"""
    w, h = size
    renderer = ActRenderer(window, FILM_DIR, act, fps=fps)
    total = int(round(float(act['duration']) * fps))
    n = total if stop_frame is None else min(total, int(stop_frame))
    cmd = [ffmpeg_exe(), '-y', '-hide_banner', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{w}x{h}',
           '-r', str(fps), '-i', '-', '-vf', 'vflip',
           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(crf),
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    try:
        for i in range(n):
            renderer.draw(i / float(fps))          # 幕内时间；必须顺序推进（粒子相位累加）
            proc.stdin.write(_grab(window, w, h))
            if not quiet and (i + 1) % (fps * 5) == 0:
                rate = (i + 1) / max(1e-6, time.time() - t0)
                print(f'  [{act["id"]}] {i + 1}/{n} 帧 ({rate:.1f} fps)', flush=True)
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f'ffmpeg 退出码 {proc.returncode}')
    return n, time.time() - t0


def check_determinism(window, act, fps, size, frame):
    """两次都从 t=0 跑到第 ``frame`` 帧，比较像素 sha256。"""
    w, h = size
    hashes = []
    for k in range(2):
        renderer = ActRenderer(window, FILM_DIR, act, fps=fps)
        pix = None
        for i in range(int(frame) + 1):
            renderer.draw(i / float(fps))
            if i == int(frame):
                pix = _grab(window, w, h)
        hashes.append(hashlib.sha256(pix).hexdigest())
        print(f'  run{k + 1}: {hashes[0][:16]}… ({len(pix)} bytes)', flush=True)
    same = hashes[0] == hashes[1]
    print(f'[determinism] frame {frame}: {"SAME" if same else "**DIFF**"}')
    return same


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--act', required=True)
    ap.add_argument('--size', default=None, help='如 960x540；默认取 project.json')
    ap.add_argument('--fps', type=int, default=None)
    ap.add_argument('--crf', type=int, default=20)
    ap.add_argument('--out', default=None)
    ap.add_argument('--check-determinism', type=int, default=None, metavar='FRAME',
                    help='两次从头跑到第 FRAME 帧，比较像素 sha256')
    args = ap.parse_args(argv)

    project = load_project()
    act = find_act(project, args.act)
    fps = int(args.fps or project['fps'])
    size = parse_size(args.size) if args.size else tuple(project['stage'])
    w, h = size

    out_path = args.out or os.path.join(ROOT, 'film', 'out', f'{act["id"]}.mp4')
    out_path = out_path if os.path.isabs(out_path) else os.path.join(ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    window = pyglet.window.Window(w, h, caption=f'worldwright · {act["id"]}', visible=True)
    window.switch_to()
    try:
        if args.check_determinism is not None:
            ok = check_determinism(window, act, fps, size, args.check_determinism)
            print('DETERMINISM_OK' if ok else 'DETERMINISM_FAILED')
            return 0 if ok else 1
        n, secs = render_act(window, act, fps, out_path, args.crf, size)
    finally:
        window.close()

    size_mb = os.path.getsize(out_path) / 1048576
    print(f'[ok] {act["id"]}: {n} 帧 / {n / fps:.1f}s / {w}x{h} / '
          f'{size_mb:.1f}MB / 渲染 {secs:.0f}s -> {out_path}')
    print('RENDER_OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
