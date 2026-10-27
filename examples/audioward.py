# -*- coding: utf-8 -*-
"""《振响 / Audioward》—— 音频反应可视化（creative coding 缺口之三：看得到的声音）。

一条**程序合成的 24s 循环乐段**（A 小调 120 BPM，引擎 `engine.audio.Track` 离线
渲染——与 kkrieger OST 同一条技术路线）被"翻译"成视觉：

1. **离线 FFT**（numpy rfft，帧 1024 / hop 512）把整段音频预计算成
   64 个对数频桶的能量包络 + bass/mid/treble 三频带——启动时一次算完，运行时
   按 playhead 查表，**零运行时依赖**（无声卡/无麦克风也能跑，视觉只吃 playhead）；
2. **乐谱即时间轴**：kick 时刻直接来自合成时的节拍常量（"编译期对齐"，不是
   onset 检测）——beat 冲击环与鼓点严格同帧；
3. 视觉：64 柱对数频谱环（GL_LINES 发光柱 + additive 混合）+ 中央核心随 bass
   脉冲 + kick 冲击波环扩散；相机慢速环绕（可拖拽接管）。

**确定性**：同 playhead 同画面（能量表是纯函数），--shot 抓帧可复现。

用法::

    python examples/audioward.py                 正常运行（含合成 OST 播放）
    python examples/audioward.py --smoke 2       冒烟
    python examples/audioward.py --shot          抓帧到 docs/ 后关闭

按键：拖拽/滚轮 环绕缩放   SPACE 暂停   M 静音   ESC 退出
"""
import math
import os
import sys
import wave

import numpy as np
from pyglet.gl import (GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
                       GL_DEPTH_TEST, GL_LINES, GL_ONE, GL_SRC_ALPHA,
                       GL_PROGRAM_POINT_SIZE, glBlendFunc, glClear, glClearColor,
                       glDisable, glEnable)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key

from world3d.engine.audio import Track, render_wav                    # noqa: E402
from world3d.engine.camera import OrbitCamera                         # noqa: E402
from world3d.logic.camera import look_at_matrix                       # noqa: E402
from world3d.platform.app import DemoApp, orbit_camera_input, run     # noqa: E402

LOOP = 24.0                # 乐段时长（秒）
RATE = 32000               # 合成采样率
BARS = 64                  # 频谱柱数
BPM = 120
BEAT = 60.0 / BPM          # 0.5s

BANDS = ((20.0, 160.0), (160.0, 2000.0), (2000.0, 8000.0))   # bass/mid/treble


# ------------------------------------------------------------ 乐谱合成 --
def build_loop(path):
    """合成 24s 循环乐段（A 小调 120BPM）。返回 (path, 时长)。"""
    tr = Track(LOOP, RATE)
    chords = ((55.00, (0, 3, 7)), (43.65, (0, 4, 7)),
              (65.41, (0, 4, 7)), (49.00, (0, 4, 7)))          # Am F C G
    penta = (220.0, 261.63, 293.66, 329.63, 392.0, 440.0)
    nbars = int(LOOP / (BEAT * 4))
    kicks = []
    for bar in range(nbars):
        t0 = bar * BEAT * 4
        root, semi = chords[bar % 4]
        # pad 和弦（每小节）
        for k, s in enumerate(semi):
            f = root * 2 ** (s / 12) * 2.0
            tr.tone(t0, BEAT * 4 + 0.3, f, wave='saw', amp=0.045,
                    attack=0.3, release=0.5, lp=0.06)
        # kick：四分（每拍）
        for b in range(4):
            tk = t0 + b * BEAT
            tr.tone(tk, 0.13, 150.0, 42.0, wave='sine', amp=0.5,
                    attack=0.001, release=0.05)
            tr.noise(tk, 0.02, amp=0.045, decay=40, lp=0.6, seed=bar * 8 + b)
            kicks.append(tk)
        # bass：八分 [根,根,高八度,根]
        for b in range(8):
            octv = 12.0 if b % 4 == 2 else 0.0
            tr.tone(t0 + b * BEAT / 2, 0.2, root * 2 ** (octv / 12),
                    wave='square', amp=0.13, duty=0.35, attack=0.003,
                    release=0.04, lp=0.22)
        # arp：十六分（每 8 个留一拍呼吸）
        for b in range(16):
            if b % 8 == 7:
                continue
            f = penta[(bar * 16 + b) * 3 % 6] * 2.0
            tr.tone(t0 + b * BEAT / 4, 0.10, f, wave='square', amp=0.07,
                    duty=0.25, attack=0.002, release=0.03, lp=0.35)
    path2, dur = render_wav(tr, path)
    return path2, dur, tuple(kicks)


def ensure_loop(path='assets/audioward_loop_24s.wav'):
    if os.path.exists(path):
        return path, LOOP, _KICKS_CACHE[0] if _KICKS_CACHE else ()
    path2, dur, kicks = build_loop(path)
    _KICKS_CACHE.append(kicks)
    return path2, dur, kicks


_KICKS_CACHE = []


# ------------------------------------------------------------ FFT 包络 --
def band_envelopes(wav_path, n_bars=BARS):
    """整段音频 → (bars[n_frames, 64], bands[n_frames, 3], n_frames)。"""
    with wave.open(wav_path, 'rb') as w:
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    n_fft, hop = 1024, 512
    frames = max(1, (len(x) - n_fft) // hop)
    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / rate)
    # 对数分箱：40Hz ~ 8kHz 均匀取 n_bars+1 个边界
    edges = np.geomspace(40.0, 8000.0, n_bars + 1)
    bars = np.zeros((frames, n_bars), np.float32)
    bands = np.zeros((frames, 3), np.float32)
    for i in range(frames):
        seg = x[i * hop:i * hop + n_fft] * win
        mag = np.abs(np.fft.rfft(seg)) / n_fft
        for b in range(n_bars):
            lo, hi = edges[b], edges[b + 1]
            m = (freqs >= lo) & (freqs < hi)
            bars[i, b] = float(mag[m].mean()) if m.any() else 0.0
        for b, (lo, hi) in enumerate(BANDS):
            m = (freqs >= lo) & (freqs < hi)
            bands[i, b] = float(np.sqrt((mag[m] ** 2).mean())) if m.any() else 0.0
    # 双向指数平滑：前向保因果、反向消抖动
    def smooth(a, k=0.72):
        out = a.copy()
        for i in range(1, len(out)):
            out[i] = out[i] * (1 - k) + out[i - 1] * k
        for i in range(len(out) - 2, -1, -1):
            out[i] = out[i] * (1 - 0.35) + out[i + 1] * 0.35
        return out
    bars, bands = smooth(bars), smooth(bands)
    # 95 分位归一化（把包络抬到 0~1 观感区间）
    for b in range(n_bars):
        q = np.quantile(bars[:, b], 0.95)
        bars[:, b] = bars[:, b] / q if q > 1e-6 else bars[:, b]
    for b in range(3):
        q = np.quantile(bands[:, b], 0.95)
        bands[:, b] = bands[:, b] / q if q > 1e-6 else bands[:, b]
    return np.clip(bars, 0.0, 1.6), np.clip(bands, 0.0, 1.6), frames, rate, hop


# ------------------------------------------------------------ 着色器 --
LINE_VERT = '''#version 330 core
in vec3 position;
in vec4 colors;
uniform WindowBlock { mat4 projection; mat4 view; } window;
uniform mat4 u_model;
out vec4 v_col;
void main() {
    v_col = colors;
    gl_Position = window.projection * window.view * u_model * vec4(position, 1.0);
}'''

LINE_FRAG = '''#version 330 core
in vec4 v_col;
out vec4 final_colors;
void main() {
    final_colors = v_col;
}'''


class Audioward(DemoApp):
    TITLE = 'AUDIoward - audio-reactive spectrum (procedural OST + offline FFT)'
    HELP = ('drag/scroll orbit  SPACE pause  M mute  ESC')
    DURATION = LOOP
    SHOTS = ((6.3, 'docs/audioward_01.png'), (17.7, 'docs/audioward_02.png'))
    FOV = 55.0

    def setup(self):
        glEnable(GL_DEPTH_TEST)
        self.cam = OrbitCamera(target=Vec3(0, 1.2, 0), yaw=35.0, pitch=14.0,
                               dist=11.0, pitch_range=(-5.0, 60.0),
                               dist_range=(5.0, 40.0))
        orbit_camera_input(self, self.cam)
        self.prog = ShaderProgram(Shader(LINE_VERT, 'vertex'),
                                  Shader(LINE_FRAG, 'fragment'))
        # ---- 音频：合成 + FFT 预计算（一次）----
        wav, dur, kicks = ensure_loop()
        self.kicks = tuple(t % LOOP for t in kicks)
        self.bars, self.bands, nframes, rate, hop = band_envelopes(wav)
        self.fps_env = rate / float(hop)
        # ---- 频谱柱几何：64 柱 × 12 边（GL_LINES），底在 y=0 ----
        self._bar_base = np.zeros((BARS, 16, 3), np.float32)
        for b in range(BARS):
            self._bar_base[b] = _bar_lines(b, BARS)
        self.bar_vl = self.prog.vertex_list(
            BARS * 16, GL_LINES,
            position=('f', self._bar_base.ravel()),
            colors=('f', np.full((BARS * 16, 4), 0.5, 'f').ravel()))
        self._heights = np.zeros(BARS, np.float32)
        # ---- 核心线框（立方体）----
        self.core_vl = self.prog.vertex_list(
            24, GL_LINES,
            position=('f', _cube_lines().ravel()),
            colors=('f', np.tile((1.0, 0.62, 0.2, 1.0), 24)))
        # ---- kick 冲击环（3 个活跃槽，每环 64 段线）----
        self.ring_vl = self.prog.vertex_list(
            3 * 64 * 2, GL_LINES,
            position=('f', np.zeros(3 * 64 * 2 * 3, 'f')),
            colors=('f', np.zeros(3 * 64 * 2 * 4, 'f')))
        self._ring_t = (-99.0, -99.0, -99.0)     # 各槽的起爆时刻
        self._ring_i = 0
        # ---- 音频播放（无声卡环境静默降级：视觉只吃 playhead）----
        self.audio = None
        try:
            from world3d.engine.audio import TimelineAudio
            self.audio = TimelineAudio(wav, loop=True, volume=0.8)
            if self.audio.ok:
                self.audio.play()
            else:
                print('[audio] unavailable (silent driver / no device)')
        except Exception as e:
            print(f'[audio] init failed: {e}')
            self.audio = None

    # ------------------------------------------------------------ 主循环 --
    def update(self, dt):
        if self.paused:
            return
        if self.audio is not None:
            self.audio.sync(self.time % LOOP, True)
        # 频带能量（按 playhead 查表 + 相邻帧插值）
        fi = self.time % LOOP * self.fps_env
        i0 = int(fi) % len(self.bands)
        i1 = (i0 + 1) % len(self.bands)
        u = fi - int(fi)
        bands = self.bands[i0] * (1 - u) + self.bands[i1] * u
        bars = self.bars[i0] * (1 - u) + self.bars[i1] * u
        self._bands_now = bands
        # 柱高平滑（快攻慢放）
        target = 0.12 + bars * 3.0
        self._heights = np.where(target > self._heights,
                                 self._heights + (target - self._heights) * 0.55,
                                 self._heights + (target - self._heights) * 0.18)
        self._upload_bars()
        # kick 冲击环：越过 kick 时刻就起爆一个槽
        t = self.time % LOOP
        for k in self.kicks:
            if self._crossed(k, dt):
                self._ring_t[self._ring_i] = t
                self._ring_i = (self._ring_i + 1) % 3
        self._upload_rings(t)
        # 核心脉冲
        s = 0.5 + bands[0] * 0.55
        self.core_vl.position[:] = _cube_lines(s, y=1.2).astype('f').ravel()

    def _crossed(self, k, dt):
        """playhead 本帧是否越过时刻 k（处理循环回绕）。"""
        t0 = (self.time - dt) % LOOP
        t1 = self.time % LOOP
        if t0 <= t1:
            return t0 < k <= t1
        return k > t0 or k <= t1              # 回绕帧

    def _upload_bars(self):
        pos = np.empty((BARS, 16, 3), np.float32)
        for b in range(BARS):
            pos[b] = _bar_lines(b, BARS, height=float(self._heights[b]))
        self.bar_vl.position[:] = pos.ravel()
        bands = getattr(self, '_bands_now', np.zeros(3))
        col = np.empty((BARS, 16, 4), np.float32)
        for b in range(BARS):
            h = min(1.0, float(self._heights[b]) / 3.0)
            # 低能量蓝绿 → 高能量金橙（additive 下叠亮）
            col[b] = (0.30 + h * 0.95, 0.52 + h * 0.30, 0.28 + h * 0.05, 0.9)
        self.bar_vl.colors[:] = col.ravel()

    def _upload_rings(self, t):
        pos_arr = np.zeros((3 * 64 * 2, 3), np.float32)
        col_arr = np.zeros((3 * 64 * 2, 4), np.float32)
        for s in range(3):
            age = t - self._ring_t[s]
            if not (0.0 <= age <= 1.4):
                age = -1.0
            r = 1.2 + age * 6.0
            a = max(0.0, 1.0 - age / 1.4)
            base = s * 64 * 2
            for k in range(64):
                a0 = 2 * math.pi * k / 64
                a1 = 2 * math.pi * (k + 1) / 64
                pos_arr[base + k * 2] = (math.cos(a0) * r, 0.03, math.sin(a0) * r)
                pos_arr[base + k * 2 + 1] = (math.cos(a1) * r, 0.03, math.sin(a1) * r)
                c = (0.9 * a, 0.5 * a, 0.2 * a, a)
                col_arr[base + k * 2] = c
                col_arr[base + k * 2 + 1] = c
        self.ring_vl.position[:] = pos_arr.astype('f').ravel()
        self.ring_vl.colors[:] = col_arr.astype('f').ravel()

    # ------------------------------------------------------------ 绘制 --
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.02, 0.022, 0.03, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)     # additive：发光体叠亮
        self.window.view = self.cam.view_matrix()   # 引擎 OrbitCamera 官方视图矩阵
        self.prog.use()
        self.prog['u_model'] = Mat4()
        self.bar_vl.draw(GL_LINES)
        self.prog['u_model'] = Mat4.from_translation(Vec3(0, 1.2, 0))
        self.core_vl.draw(GL_LINES)
        self.prog['u_model'] = Mat4()
        self.ring_vl.draw(GL_LINES)
        glDisable(GL_BLEND)

    def status_text(self):
        b = getattr(self, '_bands_now', (0, 0, 0))
        audio = '-'
        if self.audio is not None and self.audio.ok:
            audio = 'MUTED' if getattr(self.audio, 'muted', False) else 'ON'
        return (f't={self.time % LOOP:5.2f}s / {LOOP:.0f}s   '
                f'bass {b[0]:.2f}  mid {b[1]:.2f}  treb {b[2]:.2f}   audio {audio}')

    def on_key(self, symbol, modifiers):
        if symbol == key.M and self.audio is not None:
            self.audio.toggle_mute()

    def close(self):
        if getattr(self, 'audio', None) is not None:
            self.audio.close()
        super().close()


# ------------------------------------------------------------ 几何工具 --
def _bar_lines(i, n, height=0.1, radius=4.2):
    """第 i 根频谱柱的 8 条棱（16 个线端点），绕环排布，底在 y=0。"""
    a = 2 * math.pi * i / n
    hw = 0.0225                            # 柱半宽
    x0, z0 = math.cos(a) * radius, math.sin(a) * radius
    px, pz = -math.sin(a) * hw, math.cos(a) * hw   # 切向半宽
    bl = (x0 - px, 0.0, z0 - pz)           # bottom-left
    br = (x0 + px, 0.0, z0 + pz)           # bottom-right
    tl = (x0 - px, height, z0 - pz)        # top-left
    tr = (x0 + px, height, z0 + pz)        # top-right
    edges = [(bl, tl), (br, tr),           # 两根竖棱
             (bl, br), (tl, tr),           # 底横 / 顶横
             (bl, tr),                     # 斜撑
             ((x0 - px, height * 0.4, z0 - pz), (x0 + px, height * 0.4, z0 + pz)),
             ((x0 - px, height * 0.7, z0 - pz), (x0 + px, height * 0.7, z0 + pz)),
             (tl, tr)]                     # 顶横重复补位到 8 条
    out = []
    for p, q in edges:
        out += [p, q]
    return np.array(out, np.float32).reshape(-1, 3)[:16]


def _cube_lines(size=1.0, y=0.0):
    s = size / 2
    c = [(-s, -s, -s), (s, -s, -s), (s, -s, s), (-s, -s, s),
         (-s, s, -s), (s, s, -s), (s, s, s), (-s, s, s)]
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
    out = []
    for a, b in e:
        out += [c[a], c[b]]
    return np.array(out, np.float32)


if __name__ == '__main__':
    sys.exit(run(Audioward))
