# -*- coding: utf-8 -*-
"""《机枢 / FSMward》—— 有限状态机（engine.fsm: ``Machine`` / ``machine_from_keys``）。

**为什么值得单独一个 demo**
----------------------------
``world3d.engine.fsm`` 是引擎的确定性有限状态机模块，此前**零个 demo 消费过它**。
本 demo 把它接成一条直观可见的 AI 链路：一堆简单实体（"守卫"）各自跑**同一台**
有限状态机，颜色就是它此刻的状态，HUD 实时给出各状态的实体分布 ——
"它到底在干什么"不再是黑盒。

状态机怎么建的（核心知识点）
----------------------------
每个实体持有一台**独立的** ``Machine``（同一结构、各自 blackboard/状态）。
结构由 ``Agent.build_fsm`` 声明：

* 5 个状态 ``idle / patrol / chase / attack / flee``，每个都挂 ``on_enter`` /
  ``on_exit`` 钩子（进入时记录进入时刻、写 HUD 事件流；进入 ``patrol`` 时
  顺手挑一个随机巡游方向）；
* 事件驱动转移（``fire(event)``）——
  ``spawn``（→idle）、``wander``（idle→patrol）、``see``（idle/patrol→chase）、
  ``reach``（chase→attack）、``lost``（chase/attack→patrol）、
  ``danger``（任意→flee，最高优先级）、``safe``（flee→patrol）；
* 转移条件由**实体间距离 / 计时器**驱动：看到"猎物"进入视野半径→chase，
  贴到攻击半径→attack，丢失目标→回 patrol；"威胁"进入危险半径→flee，
  离开后→回 patrol；idle 停够 ``idle_time`` 秒后自动 wander。

此外，全局"导演"状态（``calm / alert / panic``，随时间改变危险半径）用
**另一套 API** ``machine_from_keys`` 建出——它是**时间轴驱动**的
（关键帧 ``[(t, state), ...]`` + ``at(t)``），正好与实体的事件驱动形成对照，
把模块的两个真实用法都演示到了。

HUD 读了什么
------------
``status_text`` 显示：当前导演相位 ``phase`` 与危险半径 ``danger_r``、实体总数、
各状态实体分布 ``[id3 pa5 ch4 at2 fl1]``（顺序 idle/patrol/chase/attack/flee）、
累计帧号 ``frame``、最近一次状态切换 ``last``、帧率、暂停标记。

控制键
------
``SPACE`` 暂停/继续   ``R`` 重置所有实体   ``1`` 切换"鼠标点选猎物目标点"
``W`` 线框   drag/scroll 转视角/缩放   ``ESC`` 退出。

约定：从 ``world3d...`` 导入；GLSL 330；``colors`` 顶点属性为 ``vec4``；
WindowBlock 名为 ``window``（含 ``mat4 projection; mat4 view;``）；固定种子
``20261026`` 保证可复现。
"""
import math
import random

from pyglet.gl import (GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
                       GL_DEPTH_TEST, GL_FILL, GL_FRONT_AND_BACK, GL_LINE,
                       GL_LINES, GL_LINE_STRIP, GL_POLYGON_MODE, GL_TRIANGLES,
                       glClear, glClearColor, glEnable, glPolygonMode)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key

from world3d.engine.fsm import Machine, machine_from_keys            # noqa: E402
from world3d.logic.camera import OrbitCamera                          # noqa: E402
from world3d.platform.app import DemoApp, orbit_camera_input, run     # noqa: E402

SEED = 20261026

# 状态顺序（同时决定 HUD 分布展示顺序与配色索引）
STATE_ORDER = ('idle', 'patrol', 'chase', 'attack', 'flee')

# 颜色 = 当前状态（rgb + alpha，alpha 恒为 1）
STATE_COLS = {
    'idle':   (0.50, 0.54, 0.62, 1.0),
    'patrol': (0.30, 0.80, 0.42, 1.0),
    'chase':  (0.96, 0.82, 0.26, 1.0),
    'attack': (0.96, 0.30, 0.24, 1.0),
    'flee':   (0.82, 0.36, 0.96, 1.0),
}


# ------------------------------------------------------------- 几何构建 --
def _normal(a, b, c):
    """三角形面法线（单位向量）。"""
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    L = math.hypot(nx, ny, nz) or 1.0
    return (nx / L, ny / L, nz / L)


def marker_geom(H=1.6, R=0.7):
    """一个"宝石"标记（上下双锥 = 8 三角面），底在 y=0、尖在 y=H。"""
    top = (0.0, H, 0.0)
    bot = (0.0, 0.0, 0.0)
    ring = ((R, 0.0, 0.0), (0.0, 0.0, R), (-R, 0.0, 0.0), (0.0, 0.0, -R))
    pos, nrm = [], []
    for i in range(4):
        a = ring[i]
        b = ring[(i + 1) % 4]
        for tri in ((top, a, b), (bot, b, a)):
            n = _normal(*tri)
            for p in tri:
                pos.extend(p)
                nrm.extend(n)
    return pos, nrm


def circle_geom(seg=72):
    """XZ 平面、半径 1 的单位圆折线（危险半径可视化用）。"""
    pos, nrm, cls = [], [], []
    col = (0.96, 0.55, 0.18, 1.0)
    for i in range(seg + 1):
        a = 2.0 * math.pi * i / seg
        p = (math.cos(a), 0.0, math.sin(a))
        pos.extend(p)
        nrm.extend((0.0, 1.0, 0.0))
        cls.extend(col)
    return pos, nrm, cls


def quad(out, p0, p1, p2, p3, nrm, col):
    """一个四边形 = 两个三角形（位置/法线/颜色 vec4）。"""
    pos, nms, cls = out
    for p in (p0, p1, p2, p0, p2, p3):
        pos.extend(p)
        nms.extend(nrm)
        cls.extend(col)


# --------------------------------------------------------------- 实体 --
class Agent:
    """一个守卫：私有 ``Machine`` + 私有位置/速度。"""

    __slots__ = ('world', 'id', 'state', 'enter_t', 'x', 'z', 'hx', 'hz',
                 'vx', 'vz', 'machine')

    def __init__(self, world, id_):
        self.world = world
        self.id = id_
        self.state = 'idle'
        self.enter_t = 0.0
        self.x = 0.0
        self.z = 0.0
        self.hx = 1.0              # 巡游方向（单位向量 x/z）
        self.hz = 1.0
        self.vx = 0.0
        self.vz = 0.0
        self.machine = None

    # ----------------------------------------------------- 建状态机 --
    def build_fsm(self):
        """声明一台独立 ``Machine``：状态 + on_enter/on_exit + 事件转移。"""
        w = self.world

        def on_enter(src, dst, t):
            self.state = dst
            self.enter_t = t
            if dst == 'patrol':                 # 进入巡逻顺手挑个随机方向
                ang = w.rng.uniform(0.0, 2.0 * math.pi)
                self.hx, self.hz = math.cos(ang), math.sin(ang)
            w.push_event(self.id, src, dst)

        def on_exit(src, dst, t):
            pass

        m = Machine('agent%d' % self.id)
        for s in STATE_ORDER:
            m.state(s, on_enter=on_enter, on_exit=on_exit)
        # 事件驱动转移：src 可为状态名 / 元组 / '*'
        m.transition('spawn', '*', 'idle')
        m.transition('wander', 'idle', 'patrol')
        m.transition('see', ('idle', 'patrol'), 'chase')
        m.transition('reach', 'chase', 'attack')
        m.transition('lost', ('chase', 'attack'), 'patrol')
        m.transition('danger', '*', 'flee')
        m.transition('safe', 'flee', 'patrol')
        m.fire('spawn', 0.0)                    # 落点：idle
        self.machine = m

    # ----------------------------------------------------- 感知决策 --
    def think(self, world, dt, t):
        """按距离/计时器 fire 事件，再据当前状态算期望速度。"""
        m = self.machine
        cur = m.current
        d_t = math.hypot(self.x - world.target[0], self.z - world.target[1])
        d_h = math.hypot(self.x - world.threat[0], self.z - world.threat[1])

        # 1) 威胁最高优先级：进入危险半径 → flee；离开 → 回 patrol
        if d_h < world.danger_r:
            if cur != 'flee':
                m.fire('danger', t)
        else:
            if cur == 'flee':
                m.fire('safe', t)
        cur = m.current

        # 2) idle 计时器：停够 idle_time 秒 → wander
        if cur == 'idle' and (t - self.enter_t) > world.idle_time:
            m.fire('wander', t)
            cur = m.current

        # 3) 猎物感知
        if cur in ('idle', 'patrol'):
            if d_t < world.sight_r:
                m.fire('see', t)
        elif cur == 'chase':
            if d_t < world.attack_r:
                m.fire('reach', t)
            elif d_t > world.sight_r:
                m.fire('lost', t)
        elif cur == 'attack':
            if d_t > world.attack_r * 1.35:
                m.fire('lost', t)
        cur = m.current

        # 4) 据状态求速度
        sp = world.speeds
        if cur == 'idle':
            self.vx = self.vz = 0.0
        elif cur == 'patrol':
            self.vx, self.vz = self.hx * sp['patrol'], self.hz * sp['patrol']
        elif cur == 'chase':
            dx, dz = world.target[0] - self.x, world.target[1] - self.z
            L = math.hypot(dx, dz) or 1.0
            self.vx, self.vz = dx / L * sp['chase'], dz / L * sp['chase']
        elif cur == 'attack':
            dx, dz = world.target[0] - self.x, world.target[1] - self.z
            L = math.hypot(dx, dz) or 1.0
            self.vx, self.vz = dx / L * sp['attack'], dz / L * sp['attack']
        elif cur == 'flee':
            dx, dz = self.x - world.threat[0], self.z - world.threat[1]
            L = math.hypot(dx, dz) or 1.0
            self.vx, self.vz = dx / L * sp['flee'], dz / L * sp['flee']

    # ----------------------------------------------------- 积分 --
    def update(self, world, dt, t):
        self.think(world, dt, t)
        self.x += self.vx * dt
        self.z += self.vz * dt
        # 边界反弹（仅影响巡游方向符号）
        A = world.A - 0.8
        if self.x > A:
            self.x, self.hx = A, -abs(self.hx)
        elif self.x < -A:
            self.x, self.hx = -A, abs(self.hx)
        if self.z > A:
            self.z, self.hz = A, -abs(self.hz)
        elif self.z < -A:
            self.z, self.hz = -A, abs(self.hz)


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
uniform float u_flat;

out vec3 v_nrm;
out vec3 v_col;
out vec3 v_world;

void main() {
    vec4 world = u_model * vec4(position, 1.0);
    v_world = world.xyz;
    v_nrm = normalize(normals);
    v_col = colors.rgb;
    gl_Position = window.projection * window.view * world;
}'''

FRAG = '''#version 330 core
in vec3 v_nrm;
in vec3 v_col;
in vec3 v_world;
out vec4 final_colors;

uniform vec3 u_sun;
uniform vec3 u_fog;
uniform float u_flat;

void main() {
    vec3 col = v_col;
    if (u_flat < 0.5) {
        vec3 n = normalize(v_nrm);
        float lam = max(dot(n, u_sun), 0.0);
        float hemi = 0.5 + 0.5 * n.y;
        vec3 amb = mix(vec3(0.12, 0.12, 0.14), vec3(0.28, 0.32, 0.40), hemi);
        col = v_col * (amb + lam * vec3(0.95, 0.92, 0.84));
        col = mix(col, u_fog, clamp(length(v_world.xz) * 0.010, 0.0, 0.5));
    }
    final_colors = vec4(col, 1.0);
}'''


# ------------------------------------------------------------------ app --
class FsmWard(DemoApp):
    TITLE = 'FSMward - finite state machines (engine.fsm: Machine / machine_from_keys)'
    HELP = ('SPACE pause  R reset  1 toggle click-target  W wire  '
            'drag/scroll  ESC')
    SHOTS = ((5.0, 'docs/fsmward_01.png'), (15.0, 'docs/fsmward_02.png'))
    DURATION = None
    FOV = 55.0

    def setup(self):
        glEnable(GL_DEPTH_TEST)
        self.prog = ShaderProgram(Shader(VERT, 'vertex'),
                                  Shader(FRAG, 'fragment'))
        self.A = 18.0
        self.n = 28
        self.idle_time = 2.5
        self.sight_r = 7.0
        self.attack_r = 1.6
        self.speeds = {'patrol': 2.2, 'chase': 4.5, 'attack': 3.0, 'flee': 5.0}
        self.click_mode = False
        self.wire = 'wire' in self.args.flags      # --flag wire：线框对照（也用于验证 GL_LINE 路径）
        self.frame = 0
        self.recent = []
        self.phase = 'calm'
        self.rng = random.Random(SEED)
        self.cam = OrbitCamera(target=Vec3(0, 0, 0), yaw=40.0, pitch=50.0,
                               dist=46.0, pitch_range=(6.0, 89.0),
                               dist_range=(15.0, 160.0))
        orbit_camera_input(self, self.cam)
        self.model = Mat4()
        # 鼠标点选猎物目标点（仅 click_mode 下生效）
        self.window.set_handler('on_mouse_press', self._on_click)

        # 标记几何 + 每状态一份缓存 vertex_list（颜色 = 状态）
        self.marker_pos, self.marker_nrm = marker_geom(1.6, 0.7)
        self.marker_n = len(self.marker_pos) // 3
        self.state_vl = {
            s: self.prog.vertex_list(
                self.marker_n, GL_TRIANGLES,
                position=('f', self.marker_pos),
                normals=('f', self.marker_nrm),
                colors=('f', tuple(STATE_COLS[s]) * self.marker_n))
            for s in STATE_ORDER}
        self.target_vl = self.prog.vertex_list(
            self.marker_n, GL_TRIANGLES,
            position=('f', self.marker_pos), normals=('f', self.marker_nrm),
            colors=('f', (0.20, 0.90, 0.95, 1.0) * self.marker_n))
        self.threat_vl = self.prog.vertex_list(
            self.marker_n, GL_TRIANGLES,
            position=('f', self.marker_pos), normals=('f', self.marker_nrm),
            colors=('f', (0.96, 0.55, 0.18, 1.0) * self.marker_n))

        # 危险半径圆环
        rp, rn, rc = circle_geom(72)
        self.ring = self.prog.vertex_list(
            len(rp) // 3, GL_LINE_STRIP,
            position=('f', rp), normals=('f', rn), colors=('f', rc))

        self._build_static()

        # 导演 FSM：用 machine_from_keys 时间轴驱动（与实体的事件驱动对照）
        KEYS = [(0.0, 'calm'), (10.0, 'alert'), (20.0, 'panic'),
                (30.0, 'calm'), (40.0, 'alert'), (50.0, 'panic')]
        self.director = machine_from_keys(KEYS, name='director')
        self.phase_r = {'calm': 4.0, 'alert': 9.0, 'panic': 14.0}
        self.danger_r = 4.0

        self._reset()

    # ------------------------------------------------------------ 世界 --
    def _build_static(self):
        A = self.A
        pos, nrm, cls = [], [], []
        out = (pos, nrm, cls)
        # 地面
        quad(out, (-A, 0, -A), (A, 0, -A), (A, 0, A), (-A, 0, A),
              (0.0, 1.0, 0.0), (0.16, 0.17, 0.20, 1.0))
        # 网格
        gpos, gnrm, gcls = [], [], []
        gcol = (0.22, 0.24, 0.28, 1.0)
        for i in range(-int(A), int(A) + 1, 4):
            for a, b in (((-A, i), (A, i)), ((i, -A), (i, A))):
                gpos.extend((a[0], 0.01, a[1], b[0], 0.01, b[1]))
                gnrm.extend((0.0, 1.0, 0.0, 0.0, 1.0, 0.0))
                gcls.extend(gcol + gcol)
        self.ground = self.prog.vertex_list(
            len(pos) // 3, GL_TRIANGLES,
            position=('f', pos), normals=('f', nrm), colors=('f', cls))
        self.grid = self.prog.vertex_list(
            len(gpos) // 3, GL_LINES,
            position=('f', gpos), normals=('f', gnrm), colors=('f', gcls))

    def _reset(self):
        """固定种子重建：位置与 rng 序列可复现。"""
        self.rng = random.Random(SEED)
        self.agents = []
        for i in range(self.n):
            a = Agent(self, i)
            a.x = self.rng.uniform(-self.A + 2, self.A - 2)
            a.z = self.rng.uniform(-self.A + 2, self.A - 2)
            a.hx, a.hz = self.rng.choice((-1, 1)), self.rng.choice((-1, 1))
            a.build_fsm()
            self.agents.append(a)
        self.target = [0.0, 0.0]
        self.target_goal = [self.rng.uniform(-self.A + 3, self.A - 3),
                            self.rng.uniform(-self.A + 3, self.A - 3)]
        self.threat = [6.0, -6.0]
        self.threat_goal = [self.rng.uniform(-self.A + 3, self.A - 3),
                            self.rng.uniform(-self.A + 3, self.A - 3)]
        self.recent = []
        self.frame = 0
        print('[fsmward] reset agents=%d arena=%.0f' % (self.n, self.A))

    def push_event(self, id_, src, dst):
        """把状态切换记进 HUD 事件流（最近 8 条）。"""
        s = src if src else '·'
        self.recent.append('%d:%s>%s' % (id_, s, dst))
        if len(self.recent) > 8:
            self.recent.pop(0)

    # ------------------------------------------------------------ 交互 --
    def on_key(self, symbol, modifiers):
        k = key
        if symbol in (k.R,):
            self._reset()
        elif symbol in (k._1, k.NUM_1):
            self.click_mode = not self.click_mode
            print('[fsmward] click-target =', self.click_mode)
        elif symbol in (k.W,):
            self.wire = not self.wire

    def _on_click(self, x, y, button, modifiers):
        """click_mode 下：把鼠标点选反投影到地面，设为猎物目标点。"""
        if not self.click_mode:
            return
        g = self._screen_to_ground(x, y)
        if g is not None:
            self.target[0], self.target[1] = g
            print('[fsmward] target -> (%.2f, %.2f)' % (g[0], g[1]))

    def _screen_to_ground(self, sx, sy):
        """屏幕坐标 → 地面 (y=0) 世界坐标（逆投影射线与地平面求交）。"""
        w, h = self.window.width, self.window.height
        x_ndc = (sx / w) * 2.0 - 1.0
        y_ndc = (sy / h) * 2.0 - 1.0
        pv = self.window.projection * self.window.view
        inv = pv.inverse()
        near = inv @ Vec3(x_ndc, y_ndc, -1.0)
        far = inv @ Vec3(x_ndc, y_ndc, 1.0)
        if abs(far.y - near.y) < 1e-6:
            return None
        t = -near.y / (far.y - near.y)
        if t < 0.0 or t > 1.0:
            return None
        return (near.x + (far.x - near.x) * t, near.z + (far.z - near.z) * t)

    # ------------------------------------------------------------ 更新 --
    def _roam(self, pos, goal, dt, speed):
        dx, dz = goal[0] - pos[0], goal[1] - pos[1]
        L = math.hypot(dx, dz)
        if L < 1.0:                       # 到点了，挑新目标
            goal[0] = self.rng.uniform(-self.A + 3, self.A - 3)
            goal[1] = self.rng.uniform(-self.A + 3, self.A - 3)
            return
        pos[0] += dx / L * speed * dt
        pos[1] += dz / L * speed * dt

    def update(self, dt):
        if self.paused:
            return
        self.frame += 1
        t = self.time
        # 导演相位（machine_from_keys 时间轴驱动）
        self.director.at(t % 60.0)
        self.phase = self.director.current
        self.danger_r = self.phase_r[self.phase]
        # 猎物 / 威胁 漫游
        self._roam(self.target, self.target_goal, dt, 3.0)
        self._roam(self.threat, self.threat_goal, dt, 2.4)
        for a in self.agents:
            a.update(self, dt, t)
        self._separate()

    def _separate(self):
        """轻量 O(n²) 分离，避免实体叠在一起（不影响状态机语义）。"""
        ags, n = self.agents, len(self.agents)
        for i in range(n):
            a = ags[i]
            for j in range(i + 1, n):
                b = ags[j]
                dx, dz = b.x - a.x, b.z - a.z
                d2 = dx * dx + dz * dz
                if 0.0 < d2 < 0.81:
                    d = math.sqrt(d2)
                    push = (0.9 - d) * 0.5
                    ux, uz = dx / d, dz / d
                    a.x -= ux * push
                    a.z -= uz * push
                    b.x += ux * push
                    b.z += uz * push

    # ------------------------------------------------------------ 渲染 --
    def render(self):
        glClearColor(0.06, 0.07, 0.09, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.window.projection = self.perspective
        self.window.view = self.cam.view_matrix()

        if self.wire:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        prog = self.prog
        prog.use()
        prog['u_model'] = self.model
        prog['u_sun'] = (0.45, 0.80, 0.45)
        prog['u_fog'] = (0.06, 0.07, 0.09)
        prog['u_flat'] = 0.0
        self.ground.draw(GL_TRIANGLES)

        prog['u_flat'] = 1.0
        self.grid.draw(GL_LINES)

        # 危险半径圆环（贴威胁点、按 danger_r 缩放）
        prog['u_model'] = (Mat4.from_translation(
            Vec3(self.threat[0], 0.02, self.threat[1]))
            @ Mat4.from_scale(Vec3(self.danger_r, 1.0, self.danger_r)))
        self.ring.draw(GL_LINE_STRIP)

        prog['u_flat'] = 0.0
        for a in self.agents:
            prog['u_model'] = Mat4.from_translation(Vec3(a.x, 0.0, a.z))
            self.state_vl[a.state].draw(GL_TRIANGLES)

        # 猎物（青）/ 威胁（橙）
        prog['u_model'] = (Mat4.from_translation(
            Vec3(self.target[0], 0.0, self.target[1]))
            @ Mat4.from_scale(Vec3(1.3, 1.3, 1.3)))
        self.target_vl.draw(GL_TRIANGLES)
        prog['u_model'] = (Mat4.from_translation(
            Vec3(self.threat[0], 0.0, self.threat[1]))
            @ Mat4.from_scale(Vec3(1.2, 1.2, 1.2)))
        self.threat_vl.draw(GL_TRIANGLES)

        if self.wire:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.window.projection = self.perspective

    # ------------------------------------------------------------ 状态 --
    def status_text(self):
        counts = {s: 0 for s in STATE_ORDER}
        for a in self.agents:
            counts[a.state] = counts.get(a.state, 0) + 1
        dist = ' '.join('%s%d' % (s[:2], counts[s]) for s in STATE_ORDER)
        last = self.recent[-1] if self.recent else '-'
        return (f'phase {self.phase} danger {self.danger_r:.1f}  n={len(self.agents)}'
                f'  [{dist}]  frame {self.frame}  last:{last}'
                f'  {self.fps:5.1f} fps'
                f'{"  [PAUSED]" if self.paused else ""}')


if __name__ == '__main__':
    run(FsmWard)
