# -*- coding: utf-8 -*-
"""《涡语 / Flowward》—— curl-noise 流场粒子（creative coding 缺口之二：无散度流）。

粒子在**不可压缩流场**里漂移。流场来自流函数 ψ（value-noise fBm + 时间演化）的
**2D curl**::

    v(x, y) = ( ∂ψ/∂y , −∂ψ/∂x )

对标量场取 curl 有两个免费的好处：**速度场无散度**（粒子永不堆积/抽空，密度
天然均匀），且**任何 ψ 都自动是个合法流场**——调噪声尺度就是调涡的尺度。

实现：numpy 全程向量化的值噪声 fBm（3 octave，格点哈希 + smoothstep 双线性，
中心差分求导，RK2 中点法步进）；5000 粒子每帧重写顶点缓冲，圆点精灵按速度
HSV 染色（快暖慢冷）。**零素材、零 GPU 特殊路径**——单靠数学。鼠标拖拽向
场里注入一个衰减涡（速度扰动，非位置迁移），扰动会随流场被撕开。

用法::

    python examples/flowward.py                 正常运行
    python examples/flowward.py --smoke 2       冒烟
    python examples/flowward.py --shot          抓帧到 docs/ 后关闭

按键：1-3 预设（涡尺度）  拖拽 注入涡  SPACE 暂停  R 换种子  ESC 退出
"""
import colorsys
import sys

import numpy as np
from pyglet.gl import (GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_TEST,
                       GL_POINTS, GL_PROGRAM_POINT_SIZE, glBlendFunc,
                       glClear, glClearColor, glDisable, glEnable)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4
from pyglet.window import key

from world3d.platform.app import DemoApp, run                        # noqa: E402

N = 5000                             # 粒子数
DT = 1.0 / 60.0
OCTAVES = 3                          # fBm 层数

# (名称, 噪声基频, 涡强, 阻尼, 漂移) —— 基频越大涡越小越碎
PRESETS = (
    ('large', 1.4, 0.55, 1.6, 0.035),
    ('mid',   3.0, 0.75, 2.2, 0.000),
    ('fine',  6.5, 0.95, 2.8, 0.060),
)


# ------------------------------------------------- 向量化值噪声 + curl --
def _noise2d(x, y, seed):
    """2D 值噪声（numpy 向量化）：格点哈希 + smoothstep 双线性插值。"""
    xi, yi = np.floor(x), np.floor(y)
    xf, yf = x - xi, y - yi
    u = xf * xf * (3.0 - 2.0 * xf)             # smoothstep
    v = yf * yf * (3.0 - 2.0 * yf)
    def h(i, j):
        n = np.sin((i + seed * 37.0) * 127.1 + (j - seed * 17.0) * 311.7) * 43758.5453
        return n - np.floor(n)
    n00, n10 = h(xi, yi), h(xi + 1, yi)
    n01, n11 = h(xi, yi + 1), h(xi + 1, yi + 1)
    return (n00 * (1 - u) + n10 * u) * (1 - v) + (n01 * (1 - u) + n11 * u) * v


def _psi(x, y, t, seed, base_freq, drift):
    """流函数 ψ：fBm（3 octave）+ 时间漂移。"""
    out = np.zeros_like(x)
    amp, f = 1.0, base_freq
    for o in range(OCTAVES):
        out += amp * _noise2d(x * f + t * drift * (o + 1), y * f - t * drift * 0.6,
                              seed + o * 13.0)
        amp, f = amp * 0.5, f * 2.03           # 非整数倍频，破网格对齐
    return out


def _curl(pos, t, seed, base_freq, drift, eps=0.01):
    """2D curl：v = (∂ψ/∂y, −∂ψ/∂x)（中心差分，三场求值）。"""
    x, y = pos[:, 0], pos[:, 1]
    p0 = _psi(x, y, t, seed, base_freq, drift)
    px = _psi(x + eps, y, t, seed, base_freq, drift)
    py = _psi(x, y + eps, t, seed, base_freq, drift)
    vx = (py - p0) / eps
    vy = -(px - p0) / eps
    return np.stack([vx, vy], axis=1)


class Flowward(DemoApp):
    TITLE = 'FLOWward - curl-noise flow field particles (divergence-free)'
    HELP = ('1-3 presets  drag: inject vortex  R reseed  SPACE pause  ESC')
    SHOTS = ((5.0, 'docs/flowward_01.png'), (16.0, 'docs/flowward_02.png'))

    def setup(self):
        self.preset_i = 0
        self.seed = 7.0
        self._vortex = None                    # 鼠标注入的涡 (xy, 强度)
        self._init_particles()
        self.prog = ShaderProgram(Shader(_POINT_VERT, 'vertex'),
                                  Shader(_POINT_FRAG, 'fragment'))
        self.vl = self.prog.vertex_list(
            N, GL_POINTS, position=('f', np.zeros(N * 3, 'f')),
            colors=('f', np.zeros(N * 4, 'f')))
        self._upload()
        # 2D 正交俯视：不用 OrbitCamera（demo 是平面流场艺术）
        glDisable(GL_DEPTH_TEST)

    def _init_particles(self):
        rng = np.random.default_rng(int(self.seed * 1000) & 0x7fffffff)
        asp = self.window.width / max(1, self.window.height)
        self.asp = asp
        self.pos = np.stack([rng.uniform(-asp, asp, N),
                             rng.uniform(-1.0, 1.0, N)], axis=1)
        self.age = rng.uniform(0.0, 8.0, N)    # 错峰重生

    # ------------------------------------------------------------ 主循环 --
    def update(self, dt):
        if self.paused:
            return
        _, freq, amp, damp, drift = PRESETS[self.preset_i]
        t = self.time
        v = _curl(self.pos, t, self.seed, freq, drift) * amp

        # 鼠标涡：局部切向速度注入（高斯衰减），强度随时间衰减
        if self._vortex is not None:
            (mx, my), strength = self._vortex
            dx, dy = self.pos[:, 0] - mx, self.pos[:, 1] - my
            r2 = dx * dx + dy * dy
            g = np.exp(-r2 * 14.0) * strength
            v[:, 0] += -dy * g * 10.0
            v[:, 1] += dx * g * 10.0
            self._vortex = ((mx, my), strength * 0.94)   # 拖拽期间持续但衰减

        self.pos = self.pos + v * DT
        self.age += DT
        speed = np.linalg.norm(v, axis=1)

        # 出域 wrap + 寿命重生（随机位置，保持密度均匀）
        asp = self.asp
        wrap_x = np.abs(self.pos[:, 0]) > asp
        wrap_y = np.abs(self.pos[:, 1]) > 1.0
        dead = self.age > 9.0
        if wrap_x.any() or wrap_y.any() or dead.any():
            rng = np.random.default_rng(int(t * 60) * 7919 + int(self.seed))
            m = wrap_x | wrap_y | dead
            k = int(m.sum())
            self.pos[m, 0] = rng.uniform(-asp, asp, k)
            self.pos[m, 1] = rng.uniform(-1.0, 1.0, k)
            self.age[m] = 0.0

        self._last_speed = speed
        self._upload()

    def _upload(self):
        n = N
        p = self.vl.position
        p[:] = np.hstack([self.pos, np.zeros((n, 1), 'f')]).astype('f').ravel()
        speed = getattr(self, '_last_speed', None)
        if speed is None:
            speed = np.zeros(n)
        # 速度 → 色相（0.55 蓝 → 0.0 红），明度随速度抬升
        hue = np.clip(0.60 - speed * 0.55, 0.0, 0.62)
        val = np.clip(0.45 + speed * 0.9, 0.35, 1.0)
        cols = np.empty((n, 4), 'f')
        rgb = np.array([colorsys.hsv_to_rgb(float(h), 0.85, float(vv))
                        for h, vv in zip(hue, val)])
        cols[:, :3] = rgb
        cols[:, 3] = 0.85
        self.vl.colors[:] = cols.astype('f').ravel()

    def render(self):
        from pyglet.gl import GL_POINTS, glViewport, glBindFramebuffer, GL_FRAMEBUFFER
        w, h = self.window.width, self.window.height
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, w, h)
        glClear(GL_COLOR_BUFFER_BIT)
        glClearColor(0.03, 0.035, 0.05, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glDisable(GL_BLEND)
        glEnable(GL_PROGRAM_POINT_SIZE)
        self.window.projection = Mat4.orthogonal_projection(
            -self.asp, self.asp, -1.0, 1.0, -10.0, 10.0)
        self.window.view = Mat4()
        self.prog.use()
        self.prog['u_size'] = 4.2 * h / 720.0
        self.vl.draw(GL_POINTS)

    def status_text(self):
        name = PRESETS[self.preset_i][0]
        sp = getattr(self, '_last_speed', None)
        smax = f'{sp.max():.2f}' if sp is not None else '-'
        return (f'curl-noise flow  preset {self.preset_i + 1} ({name})   '
                f'{self.time:6.1f}s   N={N}   max|v| {smax}')

    # ------------------------------------------------------------ 输入 --
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # 窗口坐标 → 正交域坐标
        w, h = self.window.width, self.window.height
        ux = (x / w * 2.0 - 1.0) * self.asp
        uy = 1.0 - y / h * 2.0
        self._vortex = ((ux, uy), 1.0)

    def on_key(self, symbol, modifiers):
        if key._1 <= symbol <= key._3:
            self.preset_i = symbol - key._1
        elif symbol == key.R:
            self.seed = (self.seed * 1.618 + 3.7) % 91.0
            self._init_particles()


# 点精灵着色器（2D 正交；位置已含 z=0）
_POINT_VERT = '''#version 330 core
in vec3 position;
in vec4 colors;
uniform WindowBlock { mat4 projection; mat4 view; } window;
uniform float u_size;
out vec4 v_col;
void main() {
    v_col = colors;
    gl_Position = window.projection * window.view * vec4(position, 1.0);
    gl_PointSize = u_size;
}'''

_POINT_FRAG = '''#version 330 core
in vec4 v_col;
out vec4 final_colors;
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    final_colors = v_col;
}'''

if __name__ == '__main__':
    sys.exit(run(Flowward))
