# -*- coding: utf-8 -*-
"""《造界》配乐 —— 用 numpy 合成 12 幕的氛围铺底。

为什么是"铺底"而不是"配乐"
------------------------
真正的配乐要有旋律、动机发展、与画面的点对点同步。这一版只做**和声垫**
（pad）+ 低频脉动 + 空气噪声，作用是让 706 秒不再是静音，并给每幕一个
不同的调性 —— 换幕时耳朵能先于眼睛察觉"到这里了"。

设计约束
--------
* **与画面同长**（706.6s）、**与 xfade 对齐**（每幕交叠 2.4s）；
* **确定性**：噪声用固定种子，同一版本重复生成必须逐字节一致；
* **不削波**：总增益在合成后按峰值归一化到 -3dBFS；
* 只依赖 numpy + 标准库 `wave`（不引第三方音频库）。

用法::

    python film/audio/score.py                 # -> film/out/score.wav
    python film/audio/score.py --sr 44100
"""
import argparse
import math
import os
import struct
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILM_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(FILM_DIR)

#: 每幕的 (时长, 和弦频率)。和弦决定这一幕的"调性色彩"。
#: 12 收束刻意与 01 同调（A 小调），呼应首尾闭环。
ACTS = (
    ('01_void',    24.0, (110.00, 164.81, 220.00)),   # A 小调
    ('02_terrain', 66.0, (73.42, 110.00, 174.61)),    # D 小调
    ('03_city',    80.0, (98.00, 146.83, 246.94)),    # G 大调
    ('04_machine', 74.0, (65.41, 98.00, 155.56)),     # C 小调
    ('05_crystal', 64.0, (92.50, 138.59, 233.08)),    # F# 大调
    ('06_cave',    64.0, (61.74, 92.50, 123.47)),     # B 小调
    ('07_swarm',   58.0, (82.41, 123.47, 196.00)),    # E 小调
    ('08_body',    58.0, (65.41, 98.00, 130.81)),     # C 大调
    ('09_field',   54.0, (110.00, 164.81, 277.18)),   # A 大调
    ('10_void',    80.0, (73.42, 110.00, 185.00)),    # D 大调
    ('11_loop',    64.0, (98.00, 146.83, 233.08)),    # G 小调
    ('12_finale',  47.0, (110.00, 164.81, 220.00)),   # 回到 A 小调
)

XFADE = 2.4
#: 世界层的段间飞行过渡（与世界层 film/world.py 的 BLEND 一致）
WORLD_BLEND = 6.0
SR_DEFAULT = 44100


def _env(n, attack, release):
    """对称的线性淡入/淡出包络（样本数）。"""
    e = np.ones(n, dtype=np.float64)
    a = min(int(attack), n // 2)
    r = min(int(release), n // 2)
    if a > 0:
        e[:a] = np.linspace(0.0, 1.0, a)
    if r > 0:
        e[n - r:] = np.linspace(1.0, 0.0, r)
    return e


def _pad(freqs, n, sr, seed, shimmer=0.0):
    """一层和声垫：基频 + 五度 + 八度，加轻微失谐与慢速颤动。"""
    t = np.arange(n, dtype=np.float64) / sr
    out = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for i, f in enumerate(freqs):
        # 每个音略微失谐（±0.15%），避免纯正弦的"电子管风琴"感
        det = 1.0 + rng.uniform(-0.0015, 0.0015)
        phase = rng.uniform(0, 2 * math.pi)
        vib = 1.0 + 0.0008 * np.sin(2 * math.pi * (0.07 + 0.02 * i) * t)
        out += np.sin(2 * math.pi * f * det * t * vib + phase) / (i + 1.6)
        # 二次谐波（很轻）给一点"身体"
        out += 0.16 * np.sin(4 * math.pi * f * det * t + phase) / (i + 1.6)
    if shimmer > 0.0:
        # 高频泛音层：晶体/虚空用，营造"亮"
        for k, f in enumerate(freqs):
            out += shimmer * np.sin(2 * math.pi * f * 4.01 * t + k) / (k + 3.0)
    return out


def _pulse(n, sr, hz=0.5, depth=0.18):
    """低频脉动：一个很慢的振幅调制，让静态画面"有呼吸"。"""
    t = np.arange(n, dtype=np.float64) / sr
    return 1.0 - depth * (0.5 + 0.5 * np.sin(2 * math.pi * hz * t))


def _air(n, sr, seed, level=0.012):
    """空气噪声：低通后的白噪，给"空间感"。"""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, n)
    # 一阶低通（简单指数平滑）
    a = 0.0025
    out = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc += a * (noise[i] - acc)
        out[i] = acc
    return out * (level / (np.abs(out).max() or 1.0))


def _reverb(x, sr, delay=0.13, decay=0.30, taps=6):
    """简易混响：几个衰减延迟抽头（够用了，不追求真实空间）。"""
    out = x.copy()
    d = int(delay * sr)
    for k in range(1, taps + 1):
        shift = d * k
        if shift >= len(x):
            break
        out[shift:] += x[:-shift] * (decay ** k)
    return out


def build(sr=SR_DEFAULT, out_path=None):
    # 世界层：每段后接 6s 飞行过渡（不再用 2.4s xfade）
    total = sum(a[1] for a in ACTS) + WORLD_BLEND * (len(ACTS) - 1)
    n_total = int(round(total * sr))
    track = np.zeros(n_total, dtype=np.float64)
    # 幕起点（累计，含交叠）
    starts, acc = [], 0.0
    for _, dur, _ in ACTS:
        starts.append(int(round(acc * sr)))
        acc += dur + WORLD_BLEND

    for idx, (name, dur, freqs) in enumerate(ACTS):
        n = int(round(dur * sr))
        shimmer = 0.10 if name in ('05_crystal', '09_field', '10_void') else 0.0
        seg = _pad(freqs, n, sr, seed=1000 + idx, shimmer=shimmer)
        seg *= _pulse(n, sr, hz=0.37 + 0.05 * (idx % 4))
        seg += _air(n, sr, seed=2000 + idx, level=0.012)
        # 收束幕：整段渐弱到几乎无声（与画面的 sun_intensity 曲线呼应）
        if name == '12_finale':
            seg *= np.linspace(1.0, 0.02, n) ** 1.2
        else:
            seg *= _env(n, attack=min(2.5 * sr, n * 0.25),
                        release=min(3.5 * sr, n * 0.3))
        seg = _reverb(seg, sr)
        s0 = starts[idx]
        s1 = min(n_total, s0 + n)
        track[s0:s1] += seg[:s1 - s0]

    # 归一化到 -3dBFS，并做 DC 阻断
    track -= track.mean()
    peak = np.abs(track).max() or 1.0
    track *= (10 ** (-3.0 / 20.0)) / peak
    stereo = np.stack([track, np.roll(track, int(0.0009 * sr))], axis=1)

    out_path = out_path or os.path.join(FILM_DIR, 'out', 'score.wav')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pcm = np.clip(stereo, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype('<i2')
    with wave.open(out_path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return total, out_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--sr', type=int, default=SR_DEFAULT)
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    total, path = build(sr=args.sr, out_path=args.out)
    mb = os.path.getsize(path) / 1048576
    print(f'[score] {total:.2f}s / {args.sr}Hz / {mb:.1f}MB -> {path}')
    print('SCORE_OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
