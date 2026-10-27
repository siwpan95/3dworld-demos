# -*- coding: utf-8 -*-
"""《曲径 / Curveward》—— 参数曲线的弧长匀速巡游（logic.curves 首秀）。

**为什么值得单独一个 demo**
----------------------------
``world3d.logic.curves`` 提供两套曲线原语：Catmull-Rom 基数样条（必过全部控制
点、可闭合）与任意阶 Bézier（De Casteljau 递推）。此前演示仓**零个 demo**消费过
它。本 demo 把"曲线到底是什么、怎么沿它走"做成看得见的：

* 同一条控制多边形，分别用 ``catmull_rom_spline``（默认闭合，必过控制点）与
  ``bezier_curve``（控制多边形只是"牵引框"）生成两条对照曲线，画成管线折线段。
* 若干个标记点（圆锥）沿曲线以**恒定速度**行进——用 ``curve_length`` 量出的
  总弧长做弧长参数化（不是朴素的参数 t，否则在曲率大的地方会"忽快忽慢"），
  用 ``catmull_rom_tangent_at`` / ``bezier_tangent`` 取出切线朝向来给圆锥定向。
* 轨道相机自动绕场景旋转，便于观察两条曲线的形状差异。

展示的引擎 API（world3d.logic.curves）
--------------------------------------
* ``catmull_rom_spline(control, segments_per_span, closed)`` → 采样点折线
* ``bezier_curve(control, segments)`` → 采样点折线
* ``catmull_rom_tangent_at(control, i, closed)`` → 第 i 个控制点处切线方向
* ``bezier_tangent(control, t, eps)`` → 参数 t 处切线方向（中心差分）
* ``bezier_point(control, t)`` → 参数 t 处曲线点（用于弧长↔参数的反查）
* ``curve_length(points)`` → 折线总长（弧长参数化的基准）

HUD（``status_text`` 读什么）
-----------------------------
实时显示：**曲线类型**（Catmull-Rom / Bézier）、**采样点数**（折线的顶点数）、
**曲线总长**（``curve_length`` 量出的弧长）、**当前行进参数 t**（0 号标记点的归一化
弧长进度，0→1 循环）。

控制键
------
C 切换 Catmull-Rom / Bézier 两种曲线   R 换一组控制点（固定种子，--shot 可复现）
SPACE 暂停   拖拽/滚轮 转视角缩放   ESC 退出

用法::

    python examples/curveward.py                正常运行
    python examples/curveward.py --smoke 3      冒烟
    python examples/curveward.py --shot         抓剧照到 docs/curveward_01/02.png
"""
import bisect
import math
import random

from pyglet.gl import (GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
                       GL_DEPTH_TEST, GL_FILL, GL_FRONT_AND_BACK, GL_LINES,
                       GL_TRIANGLES, glClear, glClearColor, glEnable,
                       glPolygonMode)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key

# ⚠️ 不要再加 sys.path 引导：引擎由 `pip install 3dworld` 提供（见 requirements.txt）。
from world3d.logic.camera import OrbitCamera                          # noqa: E402
from world3d.logic.curves import (bezier_curve, bezier_point,          # noqa: E402
                                  bezier_tangent, catmull_rom_spline,
                                  catmull_rom_tangent_at, curve_length)
from world3d.platform.app import DemoApp, orbit_camera_input, run     # noqa: E402

SPAN = 24          # Catmull-Rom：每段离散的段数
BEZ_SEG = 180      # Bézier：离散段数（采样点 = BEZ_SEG + 1）
N_CTRL = 6         # 控制点个数
MARKERS = 6        # 沿曲线行进的标记点数量
SPEED = 7.0        # 标记点沿弧长行进的速度（世界单位 / 秒）
SEED0 = 20261026   # 固定种子，保证 --shot 可复现

# 标记点配色（与曲线、控制点区分开），4 元组（含 alpha）
MARK_COLS = ((0.95, 0.45, 0.30, 1.0), (0.35, 0.78, 0.55, 1.0),
             (0.45, 0.62, 0.95, 1.0), (0.92, 0.82, 0.36, 1.0),
             (0.86, 0.46, 0.92, 1.0), (0.40, 0.86, 0.86, 1.0))
CURVE_COL = (0.55, 0.78, 0.95, 1.0)
CTRL_COL = (0.96, 0.62, 0.40, 1.0)
GROUND_COL = (0.20, 0.21, 0.25, 1.0)


# ------------------------------------------------------------- 向量小工具 --
def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    L = math.sqrt(sum(x * x for x in a))
    if L < 1e-9:
        return (0.0, 1.0, 0.0)
    return tuple(x / L for x in a)


# --------------------------------------------------------------- 几何构建 --
def cone_geom(radius=0.6, height=1.7, seg=14):
    """一个指向 +Z 的圆锥（apex 在 +Z，底面在 z=0）。

    标记点用它表现"切线朝向"；用 flat 着色，法线无意义，给 0。
    """
    pos, nrm = [], []
    for j in range(seg):
        a0 = 2.0 * math.pi * j / seg
        a1 = 2.0 * math.pi * (j + 1) / seg
        x0, y0 = radius * math.cos(a0), radius * math.sin(a0)
        x1, y1 = radius * math.cos(a1), radius * math.sin(a1)
        # 侧面三角形：apex(0,0,H) → p0 → p1
        for p in ((0.0, 0.0, height), (x0, y0, 0.0), (x1, y1, 0.0)):
            pos.extend(p)
            nrm.extend((0.0, 0.0, 0.0))
        # 底面三角形：原点 → p1 → p0
        for p in ((0.0, 0.0, 0.0), (x1, y1, 0.0), (x0, y0, 0.0)):
            pos.extend(p)
            nrm.extend((0.0, 0.0, 0.0))
    return pos, nrm


def sphere_geom(r, seg=10, ring=6):
    """一个以原点为中心、半径 r 的球（位置/法线）。"""
    pos, nrm = [], []
    for i in range(ring):
        t0, t1 = math.pi * i / ring, math.pi * (i + 1) / ring
        y0, y1 = math.cos(t0), math.cos(t1)
        r0, r1 = math.sin(t0), math.sin(t1)
        for j in range(seg):
            p0, p1 = 2 * math.pi * j / seg, 2 * math.pi * (j + 1) / seg
            a = (r0 * math.cos(p0), y0, r0 * math.sin(p0))
            b = (r0 * math.cos(p1), y0, r0 * math.sin(p1))
            c = (r1 * math.cos(p1), y1, r1 * math.sin(p1))
            d = (r1 * math.cos(p0), y1, r1 * math.sin(p0))
            for t in (a, b, c, a, c, d):
                pos.extend((t[0] * r, t[1] * r, t[2] * r))
                nrm.extend(t)
    return pos, nrm


def align_z(dir3):
    """返回一个 Mat4：把本地 +Z 轴对齐到单位方向 dir3（标记圆锥朝向）。"""
    d = _norm(dir3)
    up = (0.0, 1.0, 0.0)
    right = _cross(up, d)
    if math.sqrt(sum(x * x for x in right)) < 1e-6:
        right = (1.0, 0.0, 0.0)
    else:
        right = _norm(right)
    real_up = _cross(d, right)
    return Mat4(
        right[0], real_up[0], d[0], 0.0,
        right[1], real_up[1], d[1], 0.0,
        right[2], real_up[2], d[2], 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


# ----------------------------------------------------------------- 着色 --
VERT = '''#version 330 core
in vec3 position;
in vec3 normals;
in vec4 colors;

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;
uniform mat4 u_model;

out vec3 v_nrm;
out vec4 v_col;
out vec3 v_world;

void main() {
    vec4 world = u_model * vec4(position, 1.0);
    v_world = world.xyz;
    v_nrm = normalize(normals);
    v_col = colors;
    gl_Position = window.projection * window.view * world;
}'''

FRAG = '''#version 330 core
in vec3 v_nrm;
in vec4 v_col;
in vec3 v_world;
out vec4 final_colors;

uniform vec3 u_sun;
uniform vec3 u_fog;
uniform float u_flat;      // 1 = 纯色（曲线 / 标记点 / 控制点）

void main() {
    if (u_flat > 0.5) {
        final_colors = vec4(v_col.rgb, 1.0);
        return;
    }
    vec3 n = normalize(v_nrm);
    float lam = max(dot(n, u_sun), 0.0);
    float hemi = 0.5 + 0.5 * n.y;
    vec3 amb = mix(vec3(0.12, 0.13, 0.16), vec3(0.28, 0.32, 0.40), hemi);
    vec3 col = v_col.rgb * (amb + lam * vec3(0.95, 0.92, 0.84));
    float fog = clamp(length(v_world.xz) * 0.010, 0.0, 0.55);
    col = mix(col, u_fog, fog);
    final_colors = vec4(col, 1.0);
}'''


# ------------------------------------------------------------------- app --
class Curveward(DemoApp):
    TITLE = 'CURVEward - arc-length paced walk on curves (logic/curves)'
    HELP = ('C curve  R reseed  SPACE pause  drag/scroll  ESC')
    DURATION = None
    SHOTS = ((6.0, 'docs/curveward_01.png'), (16.0, 'docs/curveward_02.png'))
    FOV = 55.0

    def setup(self):
        glEnable(GL_DEPTH_TEST)
        self.prog = ShaderProgram(Shader(VERT, 'vertex'), Shader(FRAG, 'fragment'))
        self.kind = 'catmull'
        self.closed = True
        self.seed = SEED0
        self.rng = random.Random(self.seed)
        self.cam = OrbitCamera(target=Vec3(0, 4, 0), yaw=34.0, pitch=42.0,
                               dist=46.0, pitch_range=(4.0, 89.0),
                               dist_range=(12.0, 160.0))
        orbit_camera_input(self, self.cam)
        self.model = Mat4()
        # 几何模板（只建一次）
        self.cone_pos, self.cone_nrm = cone_geom()
        self.n_cone = len(self.cone_pos) // 3
        g_pos, g_nrm = sphere_geom(0.55)
        self.ctrl_pos, self.ctrl_nrm = g_pos, g_nrm
        self.n_ctrl = len(g_pos) // 3
        # 地面（一块大四边形，受光）
        self.ground = self._ground_vl()
        self._build()

    # ------------------------------------------------------------ 构建 --
    def _make_control(self):
        """固定种子生成一组控制点（3D，y 抬高一点更立体）。"""
        rng = random.Random(self.seed)
        pts = []
        for _ in range(N_CTRL):
            pts.append((rng.uniform(-15.0, 15.0),
                        rng.uniform(1.0, 9.0),
                        rng.uniform(-15.0, 15.0)))
        return pts

    def _build(self):
        """根据 self.kind / self.seed（重新）生成曲线与所有几何。"""
        self.control = self._make_control()
        if self.kind == 'catmull':
            self.curve = catmull_rom_spline(self.control,
                                            segments_per_span=SPAN,
                                            closed=self.closed)
            self.is_bezier = False
        else:
            self.curve = bezier_curve(self.control, segments=BEZ_SEG)
            self.is_bezier = True

        # 弧长表：用 curve_length 取总长，并自己累积一份供反查
        self.total = curve_length(self.curve)
        self.cum = [0.0]
        for a, b in zip(self.curve, self.curve[1:]):
            self.cum.append(self.cum[-1] + math.sqrt(
                sum((x - y) ** 2 for x, y in zip(a, b))))

        # 曲线折线（管线）
        self.curve_vl = self._line_vl(self.curve, CURVE_COL)

        # 控制点小球（位置烘进顶点，固定）
        cpos, cnrm, ccls = [], [], []
        for p in self.control:
            for k in range(self.n_ctrl):
                x = self.ctrl_pos[3 * k] + p[0]
                y = self.ctrl_pos[3 * k + 1] + p[1]
                z = self.ctrl_pos[3 * k + 2] + p[2]
                cpos.extend((x, y, z))
                cnrm.extend(self.ctrl_nrm[3 * k:3 * k + 3])
                ccls.extend(CTRL_COL)
        self.ctrl_vl = self.prog.vertex_list(
            len(cpos) // 3, GL_TRIANGLES,
            position=('f', cpos), normals=('f', cnrm), colors=('f', ccls))

        # 标记点：沿弧长均匀分布初始进度
        self.marker_d = [self.total * i / MARKERS for i in range(MARKERS)]
        self._cone_cache = {}
        print(f'[curveward] kind={self.kind} closed={self.closed} '
              f'ctrl={len(self.control)} samples={len(self.curve)} '
              f'len={self.total:.3f}')

    def _ground_vl(self):
        hw = 60.0
        pos = (-hw, -0.4, -hw, hw, -0.4, -hw, hw, -0.4, hw,
               -hw, -0.4, -hw, hw, -0.4, hw, -hw, -0.4, hw)
        nrm = (0.0, 1.0, 0.0) * 6
        cls = GROUND_COL * 6
        return self.prog.vertex_list(
            6, GL_TRIANGLES, position=('f', pos), normals=('f', nrm),
            colors=('f', cls))

    def _line_vl(self, points, col):
        """把折线点串成 GL_LINES（相邻点成对）。"""
        pos, nrm, cls = [], [], []
        for a, b in zip(points, points[1:]):
            pos.extend(a)
            pos.extend(b)
            nrm.extend((0.0, 0.0, 0.0))
            nrm.extend((0.0, 0.0, 0.0))
            cls.extend(col)
            cls.extend(col)
        n = len(pos) // 3
        return self.prog.vertex_list(
            n, GL_LINES, position=('f', pos), normals=('f', nrm),
            colors=('f', cls))

    # ------------------------------------------------- 弧长参数化查询 --
    def _span_t(self, dist):
        """dist（弧长）→ 该曲线类型对应的 (span, t)，用于查切线。"""
        dist = dist % self.total if self.total > 0 else 0.0
        k = bisect.bisect_right(self.cum, dist) - 1
        if k < 0:
            k = 0
        if k >= len(self.curve) - 1:
            k = len(self.curve) - 2
        denom = self.cum[k + 1] - self.cum[k]
        local = (dist - self.cum[k]) / denom if denom > 1e-12 else 0.0
        if self.is_bezier:
            t = (k + local) / BEZ_SEG
            span = None
        else:
            t = (k + local) / SPAN
            span = k // SPAN
        return span, t

    def pos_at(self, dist):
        """弧长 → 曲线上的世界坐标。

        Bézier 走 ``bezier_point(control, t)``（精确）；Catmull-Rom 无 point(t)
        API，故在采样折线上线性插值（弧长表反查到段内局部参数）。
        """
        dist = dist % self.total if self.total > 0 else 0.0
        k = bisect.bisect_right(self.cum, dist) - 1
        if k < 0:
            k = 0
        if k >= len(self.curve) - 1:
            k = len(self.curve) - 2
        denom = self.cum[k + 1] - self.cum[k]
        f = (dist - self.cum[k]) / denom if denom > 1e-12 else 0.0
        if self.is_bezier:
            t = (k + f) / BEZ_SEG
            return tuple(bezier_point(self.control, t))
        a, b = self.curve[k], self.curve[k + 1]
        return tuple(a[i] + (b[i] - a[i]) * f for i in range(3))

    def tangent_at(self, dist):
        """弧长 → 切线方向（归一化），真正用到引擎的 *_tangent* API。"""
        span, t = self._span_t(dist)
        if self.is_bezier:
            d = bezier_tangent(self.control, t)
        else:
            # 该弧长落在第 span 段（控制点 span 与 span+1 之间）
            d = catmull_rom_tangent_at(self.control, span, closed=self.closed)
        return _norm(d)

    # ------------------------------------------------------------ 交互 --
    def on_key(self, symbol, modifiers):
        if symbol == key.C:
            self.kind = 'bezier' if self.kind == 'catmull' else 'catmull'
            self._build()
        elif symbol == key.R:
            self.seed = (self.seed * 1103515245 + 12345) & 0x7FFFFFFF
            self._build()

    # ------------------------------------------------------------ 更新 --
    def update(self, dt):
        if self.paused or self.total <= 0:
            return
        for i in range(MARKERS):
            self.marker_d[i] = (self.marker_d[i] + SPEED * dt) % self.total
        self.cam.yaw += dt * 0.35     # 轨道相机缓慢绕场景

    # ------------------------------------------------------------ 渲染 --
    def _cone_vl(self, col):
        key_c = col
        cache = self._cone_cache
        if key_c not in cache:
            n = self.n_cone
            cache[key_c] = self.prog.vertex_list(
                n, GL_TRIANGLES,
                position=('f', self.cone_pos), normals=('f', self.cone_nrm),
                colors=('f', tuple(col) * n))
        return cache[key_c]

    def render(self):
        w, h = self.window.width, self.window.height
        glClearColor(0.06, 0.07, 0.10, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.window.projection = Mat4.perspective_projection(
            w / max(1, h), z_near=0.5, z_far=800.0, fov=self.FOV)
        self.window.view = self.cam.view_matrix()

        prog = self.prog
        prog.use()
        prog['u_model'] = self.model
        prog['u_sun'] = (0.42, 0.82, 0.42)
        prog['u_fog'] = (0.06, 0.07, 0.10)

        # 地面（受光）
        prog['u_flat'] = 0.0
        self.ground.draw(GL_TRIANGLES)

        # 曲线折线（纯色）
        prog['u_flat'] = 1.0
        self.curve_vl.draw(GL_LINES)

        # 控制点小球（纯色）
        self.ctrl_vl.draw(GL_TRIANGLES)

        # 标记点：圆锥沿切线朝向
        for i in range(MARKERS):
            p = self.pos_at(self.marker_d[i])
            d = self.tangent_at(self.marker_d[i])
            m = (Mat4.from_translation(Vec3(*p))
                 @ align_z(d)
                 @ Mat4.from_scale(Vec3(1.0, 1.0, 1.0)))
            prog['u_model'] = m
            self._cone_vl(MARK_COLS[i % len(MARK_COLS)]).draw(GL_TRIANGLES)

        prog['u_flat'] = 0.0
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.window.projection = self.perspective

    # ------------------------------------------------------------ 状态 --
    def status_text(self):
        t0 = (self.marker_d[0] / self.total) if self.total > 0 else 0.0
        return (f'{self.kind}{" (closed)" if (self.closed and not self.is_bezier) else ""}'
                f'  samples {len(self.curve)}  len {self.total:.3f}'
                f'  t0 {t0:.3f}  {self.fps:5.1f} fps'
                f'{"  [PAUSED]" if self.paused else ""}')


if __name__ == '__main__':
    run(Curveward)
