# -*- coding: utf-8 -*-
"""《事枢 / Eventward》—— 事件总线驱动的实体协同（logic/events 首秀）。

**为什么值得单独一个 demo**
----------------------------
``world3d.logic.events.EventBus`` 是整个引擎里"最被低估"的一块：它把
``pyglet.event.EventDispatcher`` 包了一层，预注册了引擎级事件
（``on_frame`` / ``on_view_changed`` / ``on_timeline_end``），但演示仓此前
**零个 demo 消费过它**——大家都直接写死调用，从没"松耦合地发事件"。本 demo
把"一个实体发事件、一群订阅者响应"做成**看得见**的：

    ball 撞墙 ——emit on_bounce——▶ 监听器①：把该 ball 闪白（高亮）
                                ▶ 监听器②（日志）：记一条 HUD 滚动日志
    定时器  ——emit on_spawn ——▶ 监听器：在场上生成一圈扩散涟漪（新物体）
    鼠标点击——emit on_click ——▶ 监听器③（合唱）：所有 ball 集体跳一下+闪白
                                ▶ 监听器：在点击落点也生成一圈涟漪

于是"事件流"在画面里是流动的：被触发的实体高亮、地面涟漪一圈圈散开、
左上角 HUD 滚动列出最近若干条事件（类型 + 序号 + 落点）。

展示了 logic.events 的什么 API
------------------------------
* **EventBus**：本 demo 把它子类化并隔离 ``event_types``，再用
  ``DemoBus.register_event_type('on_bounce'|'on_spawn'|'on_click')`` 登记
  三种自定义事件（不污染基类）。
* **订阅（subscribe）**：``bus.push_handlers(on_bounce=h)`` 给每个监听器
  推一个独立栈帧——同名事件因此可以挂多个订阅者（日志 + 高亮同时收到
  ``on_bounce``）。直接用 ``weakref`` 包装的是方法，精确 ``remove_handler``
  会因相等性比较失效，所以本 demo 用 **LIFO 的 ``pop_handlers``** 来"退订"
  那个可切换的监听器，演示订阅/退订对画面（球是否闪白）的直观影响。
* **发射（emit/dispatch）**：``bus.dispatch_event(type, *args)`` 同步派发
  （payload 直接传对象/坐标元组，本 demo 没造独立 Event 类）；
  ``post_event`` 走主线程队列用于跨线程，这里用不到。
* **EventDispatcher 与 EventBus 的区别**：EventDispatcher 是泛型基类，
  发任何事件前都要先 ``register_event_type``；EventBus 是它的子类，已预注册
  引擎级事件类型，业务代码继承它即可。二者订阅/发射契约完全一致。

HUD 读了什么（status_text + 左上滚动日志）
----------------------------------------
* 已发射事件**总数**（``emit`` 里自增，与是否有订阅者无关）；
* 各事件类型**计数**（on_bounce / on_spawn / on_click）；
* **订阅者数**：活跃监听器个数，外加取自 ``bus._event_stack`` 的真实
  handler 绑定数（按顶层帧下推，订阅/退订时这个数会跟着变）；
* 左上滚动日志：最近若干条事件的「类型 #序号 落点」，``1`` 键可隐藏。

控制键：SPACE 暂停   R 重置   1 切换事件日志   C 切换"高亮监听器"的订阅
       鼠标左键 emit 一个 on_click（全场响应）   drag/scroll 转/缩放   ESC

用法::

    python examples/eventward.py                正常运行
    python examples/eventward.py --smoke 3      冒烟
    python examples/eventward.py --shot         抓剧照到 docs/eventward_01/02.png
"""
import math
import random

from pyglet.gl import (GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
                       GL_DEPTH_TEST, GL_FRONT_AND_BACK, GL_FILL,
                       GL_LINE_LOOP, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA,
                       GL_TRIANGLES, glBlendFunc, glClear, glClearColor,
                       glDisable, glEnable, glPolygonMode)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key
from pyglet import text

# ⚠️ 不要再加 sys.path 引导：引擎由 `pip install 3dworld` 提供
# （见 requirements.txt 与 docs/依赖与版本.md）。
from world3d.engine.fonts import resolve_cjk_font                       # noqa: E402
from world3d.logic.camera import OrbitCamera                            # noqa: E402
from world3d.logic.events import EventBus                               # noqa: E402
from world3d.platform.app import DemoApp, orbit_camera_input, run       # noqa: E402

# ----------------------------------------------------------------- 常量 --
ARENA_R = 16.0                       # 圆形场地半径（球碰壁反弹）
N_BALLS = 22                         # 场上实体数
BALL_R = 0.7                         # 球半径
FLASH_TIME = 0.55                    # 高亮（闪白）持续时间
HOP_V = 7.0                          # 点击时集体起跳初速度
SPAWN_INTERVAL = 1.4                 # 发射器周期（秒）
RIPPLE_LIFE = 1.2                    # 涟漪寿命（秒）
MAX_RIPPLES = 48
SEED = 20261026                      # 固定种子，保证可复现
LOG_LINES = 12                       # HUD 滚动日志保留行数

# 自定义事件总线：隔离 event_types，不污染基类里预注册的引擎级事件。
# 三种都是"业务自定义事件"，必须先 register_event_type 才能 dispatch。
class DemoBus(EventBus):
    event_types = []


DemoBus.register_event_type('on_bounce')     # payload: Ball 对象（撞墙的那个）
DemoBus.register_event_type('on_spawn')      # payload: (x, z) 生成位置
DemoBus.register_event_type('on_click')      # payload: (x, z) 鼠标落点


# ------------------------------------------------------------- 几何体工具 --
def sphere_geom(r=1.0, seg=10, ring=6):
    """单位球（半径 r）的三角汤：返回 (pos, nrm)，法线为单位方向。"""
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


def unit_circle(seg=64, y=0.0):
    """XZ 平面上的单位圆（用于边界环 / 涟漪），返回 (pos,)。"""
    pos = []
    for j in range(seg):
        a = 2.0 * math.pi * j / seg
        pos.extend((math.cos(a), y, math.sin(a)))
    return pos


def hsv_to_rgb(h, s, v):
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    tcol = v * (1.0 - (1.0 - f) * s)
    cols = ((v, tcol, p), (q, v, p), (p, v, tcol),
            (p, q, v), (tcol, p, v), (v, p, q))
    return cols[i]


# --------------------------------------------------------------- 实体 --
class Ball:
    __slots__ = ('id', 'x', 'z', 'y', 'vx', 'vz', 'hop_v', 'flash', 'col',
                 'vl', 'nv')

    def __init__(self, bid, x, z, vx, vz, col):
        self.id = bid
        self.x, self.z, self.y = x, z, 0.0
        self.vx, self.vz = vx, vz
        self.hop_v = 0.0
        self.flash = 0.0
        self.col = col
        self.vl = None
        self.nv = 0

    def update(self, dt, app):
        # 平移 + 撞墙反弹（反弹时 emit on_bounce）
        self.x += self.vx * dt
        self.z += self.vz * dt
        d = math.hypot(self.x, self.z)
        lim = ARENA_R - BALL_R
        if d > lim and d > 1e-6:
            nx, nz = self.x / d, self.z / d
            self.x, self.z = nx * lim, nz * lim      # 拉回边界内，避免连发
            dot = self.vx * nx + self.vz * nz
            self.vx -= 2.0 * dot * nx                # 关于法线反射
            self.vz -= 2.0 * dot * nz
            app.emit('on_bounce', self)              # ← 发射事件（payload=自己）
        # 起跳（竖向运动 + 重力）
        if self.hop_v != 0.0 or self.y > 0.0:
            self.y += self.hop_v * dt
            self.hop_v -= 18.0 * dt
            if self.y <= 0.0:
                self.y, self.hop_v = 0.0, 0.0
        # 高亮衰减
        if self.flash > 0.0:
            self.flash = max(0.0, self.flash - dt)


# ----------------------------------------------------------- 监听器 --
# 每个监听器都是「订阅者」，通过 bus.push_handlers 挂到独立栈帧上。
class Logger:
    """订阅全部事件：只负责把事件写进 HUD 滚动日志（永远在线）。"""

    def __init__(self, app):
        self.app = app

    def on_bounce(self, ball):
        self.app._log_event('on_bounce', f'ball{ball.id}')

    def on_spawn(self, x, z):
        self.app._log_event('on_spawn', f'@({x:.1f},{z:.1f})')

    def on_click(self, x, z):
        self.app._log_event('on_click', f'@({x:.1f},{z:.1f})')


class Responder:
    """订阅 on_bounce：把「撞墙的那个球」闪白高亮（可用 C 退订演示效果）。"""

    def __init__(self, app):
        self.app = app

    def on_bounce(self, ball):
        ball.flash = FLASH_TIME


class Spawner:
    """订阅 on_spawn / on_click：在落点生成一圈扩散涟漪（事件驱动地造新物体）。"""

    def __init__(self, app):
        self.app = app

    def on_spawn(self, x, z):
        self.app.add_ripple(x, z, (0.20, 0.90, 1.00))

    def on_click(self, x, z):
        self.app.add_ripple(x, z, (1.00, 0.85, 0.20))


class Chorus:
    """订阅 on_click：让全场所有球集体起跳 + 闪白（一个事件，一群订阅者）。"""

    def __init__(self, app):
        self.app = app

    def on_click(self, x, z):
        for b in self.app.balls:
            b.hop_v = HOP_V
            b.flash = FLASH_TIME


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
uniform float u_flat;      // 1 = 纯色（地面/边界/涟漪）

void main() {
    if (u_flat > 0.5) {
        final_colors = v_col;
        return;
    }
    vec3 n = normalize(v_nrm);
    float lam = max(dot(n, u_sun), 0.0);
    float hemi = 0.5 + 0.5 * n.y;
    vec3 amb = mix(vec3(0.10, 0.09, 0.10), vec3(0.26, 0.30, 0.38), hemi);
    vec3 col = v_col.rgb * (amb + lam * vec3(0.95, 0.90, 0.80));
    float fog = clamp(length(v_world.xz) * 0.012, 0.0, 0.5);
    col = mix(col, u_fog, fog);
    final_colors = vec4(col, 1.0);
}'''


# ------------------------------------------------------------------ app --
class Eventward(DemoApp):
    TITLE = 'EVENTward - event-driven entities on an EventBus (logic/events)'
    HELP = ('SPACE pause  R reset  1 log  C subscribe/responder  '
            'click=emit  drag/scroll  ESC')
    DURATION = None
    SHOTS = ((6.0, 'docs/eventward_01.png'), (16.0, 'docs/eventward_02.png'))
    FOV = 55.0

    def setup(self):
        glEnable(GL_DEPTH_TEST)
        self.prog = ShaderProgram(Shader(VERT, 'vertex'), Shader(FRAG, 'fragment'))
        self.rng = random.Random(SEED)
        self.cam = OrbitCamera(target=Vec3(0, 0, 0), yaw=38.0, pitch=44.0,
                               dist=46.0, pitch_range=(6.0, 89.0),
                               dist_range=(14.0, 160.0))
        orbit_camera_input(self, self.cam)

        # ---- 事件总线 + 监听器（订阅） ----
        self.bus = DemoBus()
        self.logger = Logger(self)
        self.responder = Responder(self)
        self.spawner = Spawner(self)
        self.chorus = Chorus(self)
        # 固定订阅者：各自一个独立栈帧（同名事件可多订阅者）
        self.bus.push_handlers(on_bounce=self.logger.on_bounce,
                               on_spawn=self.logger.on_spawn,
                               on_click=self.logger.on_click)
        self.bus.push_handlers(on_spawn=self.spawner.on_spawn,
                               on_click=self.spawner.on_click)
        self.bus.push_handlers(on_click=self.chorus.on_click)
        # 可切换的「高亮监听器」推到最顶层，便于 pop_handlers 精确退订
        self.resp_enabled = True
        self.bus.push_handlers(on_bounce=self.responder.on_bounce)

        # ---- 计数 / 日志状态 ----
        self.total_emitted = 0
        self.counts = {'on_bounce': 0, 'on_spawn': 0, 'on_click': 0}
        self.event_log = []
        self.show_log = True
        self.ripples = []
        self.spawn_timer = 0.0

        # ---- 几何 ----
        self._build_static()
        self._build_balls()
        self._build_log_pool()

        # 鼠标点击 → emit on_click（基类没转发 on_mouse_press，自己挂一个）
        self.window.push_handlers(on_mouse_press=self._on_mouse_press)
        print(f'[eventward] bus={self.bus.__class__.__name__} balls={N_BALLS} '
              f'arena_r={ARENA_R} seed={SEED}')

    # -------------------------------------------------------- 发射 / 订阅 --
    def emit(self, etype, *args):
        """统一的「发射」入口：记账后同步派发（dispatch_event）。"""
        self.total_emitted += 1
        self.counts[etype] += 1
        # dispatch_event 同步调用所有订阅者（顶层帧向下）。
        self.bus.dispatch_event(etype, *args)

    def _toggle_responder(self):
        """演示订阅/退订：用 LIFO 的 pop_handlers 拿掉最顶层的「高亮监听器」。

        坑位：push_handlers 对方法会用 weakref.WeakMethod 存储，而
        remove_handler 用相等性比较（WeakMethod != 绑定方法），精确退订单个
        监听者会静默失败；改用 pop_handlers（只退顶层帧）最稳妥。
        """
        if self.resp_enabled:
            self.bus.pop_handlers()          # 退订顶层帧（= Responder）
            self.resp_enabled = False
        else:
            self.bus.push_handlers(on_bounce=self.responder.on_bounce)
            self.resp_enabled = True

    # --------------------------------------------------------- 几何构建 --
    def _build_static(self):
        # 地面（大四边形）
        hw = ARENA_R * 1.7
        gpos = (-hw, 0, -hw, hw, 0, -hw, hw, 0, hw,
                -hw, 0, -hw, hw, 0, hw, -hw, 0, hw)
        self.ground = self.prog.vertex_list(
            6, GL_TRIANGLES,
            position=('f', gpos), normals=('f', (0.0, 1.0, 0.0) * 6),
            colors=('f', (0.16, 0.18, 0.22, 1.0) * 6))
        # 边界环（单位圆放大到 ARENA_R）
        cpos = unit_circle(72, 0.02)
        n = len(cpos) // 3
        self.boundary = self.prog.vertex_list(
            n, GL_LINE_LOOP, position=('f', cpos),
            normals=('f', (0.0, 1.0, 0.0) * n),
            colors=('f', (0.30, 0.36, 0.46, 1.0) * n))
        # 共享的单位圆（涟漪复用，逐帧改色/缩放）
        self.circle_n = 64
        c0 = unit_circle(self.circle_n, 0.0)
        self.circle_vl = self.prog.vertex_list(
            self.circle_n, GL_LINE_LOOP, position=('f', c0),
            normals=('f', (0.0, 1.0, 0.0) * self.circle_n),
            colors=('f', (1.0, 1.0, 1.0, 1.0) * self.circle_n))

    def _build_balls(self):
        pos0, nrm0 = sphere_geom(BALL_R, seg=10, ring=6)
        nv = len(pos0) // 3
        self.balls = []
        for i in range(N_BALLS):
            ang = self.rng.uniform(0, math.tau)
            rad = self.rng.uniform(0, ARENA_R - BALL_R - 1.0)
            x = math.cos(ang) * rad
            z = math.sin(ang) * rad
            sp = self.rng.uniform(3.0, 6.0)
            sa = self.rng.uniform(0, math.tau)
            col = hsv_to_rgb(i / float(N_BALLS), 0.55, 0.95)
            b = Ball(i, x, z, math.cos(sa) * sp, math.sin(sa) * sp, col)
            b.nv = nv
            b.vl = self.prog.vertex_list(
                nv, GL_TRIANGLES,
                position=('f', pos0), normals=('f', nrm0),
                colors=('f', (tuple(col) + (1.0,)) * nv))
            self.balls.append(b)

    def _build_log_pool(self):
        _cjk = resolve_cjk_font()
        kw = {'font_name': _cjk} if _cjk else {}
        self.log_pool = [
            text.Label('', font_size=12, color=(180, 230, 255, 255),
                       x=14, y=0, anchor_x='left', anchor_y='center', **kw)
            for _ in range(LOG_LINES)]

    # ----------------------------------------------------------- 交互 --
    def on_key(self, symbol, modifiers):
        k = key
        if symbol == k.R:
            self.reset()
        elif symbol == k._1:
            self.show_log = not self.show_log
        elif symbol == k.C:
            self._toggle_responder()

    def _on_mouse_press(self, x, y, button, modifiers):
        gx, gz = self.screen_to_ground(x, y)
        self.emit('on_click', gx, gz)           # 点击 = 广播一个事件

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        return super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        return super().on_mouse_scroll(x, y, scroll_x, scroll_y)

    # 屏幕坐标 → 地面 (y=0) 交点：用相机 eye/forward 构造射线，不依赖矩阵求逆。
    def screen_to_ground(self, px, py):
        w, h = self.window.width, self.window.height
        ndc_x = (px / w) * 2.0 - 1.0
        ndc_y = (py / h) * 2.0 - 1.0
        eye = self.cam.eye
        tgt = self.cam.target
        fwd = Vec3(tgt.x - eye.x, tgt.y - eye.y, tgt.z - eye.z).normalize()
        up = Vec3(0.0, 1.0, 0.0)
        right = fwd.cross(up).normalize()
        cam_up = right.cross(fwd).normalize()
        aspect = w / max(1, h)
        tan_h = math.tan(math.radians(self.FOV) * 0.5)
        dirv = (fwd + right * (ndc_x * tan_h * aspect)
                + cam_up * (ndc_y * tan_h)).normalize()
        if abs(dirv.y) < 1e-5:
            dirv = Vec3(dirv.x, -1e-5 if dirv.y >= 0 else 1e-5, dirv.z)
        t = -eye.y / dirv.y
        return (eye.x + dirv.x * t, eye.z + dirv.z * t)

    # ----------------------------------------------------------- 涟漪 --
    def add_ripple(self, x, z, col):
        if len(self.ripples) >= MAX_RIPPLES:
            self.ripples.pop(0)
        self.ripples.append({'x': x, 'z': z, 'color': col, 't': 0.0,
                             'life': RIPPLE_LIFE})

    # ----------------------------------------------------------- 日志 --
    def _log_event(self, etype, where):
        """etype 传事件类型全名（'on_bounce'）；日志显示时剥掉 on_ 前缀。"""
        n = self.counts[etype]
        short = etype[3:] if etype.startswith('on_') else etype
        self.event_log.append(f'{short:6s} #{n:03d} {where}')

    # ----------------------------------------------------------- 重置 --
    def reset(self):
        self.total_emitted = 0
        self.counts = {'on_bounce': 0, 'on_spawn': 0, 'on_click': 0}
        self.event_log = []
        self.ripples = []
        self.spawn_timer = 0.0
        self.rng = random.Random(SEED)
        for b in self.balls:
            ang = self.rng.uniform(0, math.tau)
            rad = self.rng.uniform(0, ARENA_R - BALL_R - 1.0)
            b.x, b.z, b.y = math.cos(ang) * rad, math.sin(ang) * rad, 0.0
            sp = self.rng.uniform(3.0, 6.0)
            sa = self.rng.uniform(0, math.tau)
            b.vx, b.vz = math.cos(sa) * sp, math.sin(sa) * sp
            b.hop_v, b.flash = 0.0, 0.0
        # 重置回默认订阅状态（高亮监听器在线）
        if not self.resp_enabled:
            self._toggle_responder()
        print('[eventward] reset')

    # ----------------------------------------------------------- 更新 --
    def update(self, dt):
        if self.paused:
            return
        self.cam.yaw += dt * 5.0                  # 缓慢自转，截图有动感
        self.spawn_timer += dt
        if self.spawn_timer >= SPAWN_INTERVAL:
            self.spawn_timer -= SPAWN_INTERVAL
            ang = self.rng.uniform(0, math.tau)
            r = self.rng.uniform(0, ARENA_R * 0.8)
            self.emit('on_spawn', math.cos(ang) * r, math.sin(ang) * r)
        for b in self.balls:
            b.update(dt, self)
        for rp in self.ripples:
            rp['t'] += dt
        self.ripples = [rp for rp in self.ripples if rp['t'] < rp['life']]

    # ----------------------------------------------------------- 渲染 --
    def render(self):
        w, h = self.window.width, self.window.height
        glClearColor(0.06, 0.07, 0.10, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.window.projection = Mat4.perspective_projection(
            w / max(1, h), z_near=0.1, z_far=500.0, fov=self.FOV)
        self.window.view = self.cam.view_matrix()

        prog = self.prog
        prog.use()
        prog['u_model'] = Mat4()
        prog['u_sun'] = (0.42, 0.80, 0.42)
        prog['u_fog'] = (0.06, 0.07, 0.10)

        # 地面 + 边界（纯色）
        prog['u_flat'] = 1.0
        self.ground.draw(GL_TRIANGLES)
        prog['u_model'] = Mat4.from_translation(Vec3(0.0, 0.02, 0.0)) \
            @ Mat4.from_scale(Vec3(ARENA_R, 1.0, ARENA_R))
        self.boundary.draw(GL_LINE_LOOP)

        # 涟漪（半透明，需混合）
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        for rp in self.ripples:
            self._draw_ripple(prog, rp)
        glDisable(GL_BLEND)

        # 球（光照）
        prog['u_flat'] = 0.0
        for b in self.balls:
            self._draw_ball(prog, b)

        self.window.projection = self.perspective
        self._draw_log(w, h)

    def _draw_ball(self, prog, b):
        if b.flash > 0.0:
            f = b.flash / FLASH_TIME
            r = b.col[0] + (1.0 - b.col[0]) * f
            g = b.col[1] + (1.0 - b.col[1]) * f
            bl = b.col[2] + (1.0 - b.col[2]) * f
        else:
            r, g, bl = b.col
        b.vl.colors[:] = (r, g, bl, 1.0) * b.nv
        prog['u_model'] = Mat4.from_translation(
            Vec3(b.x, BALL_R + b.y, b.z))
        b.vl.draw(GL_TRIANGLES)

    def _draw_ripple(self, prog, rp):
        age = rp['t'] / rp['life']
        radius = 0.4 + age * 4.5
        alpha = (1.0 - age) * 0.9
        c = rp['color']
        self.circle_vl.colors[:] = (c[0], c[1], c[2], alpha) * self.circle_n
        prog['u_model'] = Mat4.from_translation(Vec3(rp['x'], 0.06, rp['z'])) \
            @ Mat4.from_scale(Vec3(radius, 1.0, radius))
        self.circle_vl.draw(GL_LINE_LOOP)

    def _draw_log(self, w, h):
        if not self.show_log:
            return
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)
        self.window.projection = Mat4.orthogonal_projection(
            0, w, 0, h, -255, 255)
        self.window.view = Mat4()
        recent = self.event_log[-LOG_LINES:]
        for i, lab in enumerate(self.log_pool):
            if i < len(recent):
                lab.text = recent[i]
                lab.position = (14, h - 60 - i * 16, 0)
                lab.visible = True
                lab.draw()
            else:
                lab.visible = False
        self.window.projection = self.perspective
        glEnable(GL_DEPTH_TEST)

    # ----------------------------------------------------------- 状态 --
    def status_text(self):
        binds = sum(len(f) for f in self.bus._event_stack)   # 真实 handler 绑定数
        subs = 3 + (1 if self.resp_enabled else 0)
        c = self.counts
        logflag = 'log ON ' if self.show_log else 'log OFF'
        resp = 'resp ON ' if self.resp_enabled else 'resp OFF'
        return (f'emit {self.total_emitted}  bounce {c["on_bounce"]} '
                f'spawn {c["on_spawn"]} click {c["on_click"]}  '
                f'subs {subs} ({binds} handlers)  {logflag}{resp}'
                f'  {self.fps:5.1f} fps'
                f'{"  [PAUSED]" if self.paused else ""}')


if __name__ == '__main__':
    run(Eventward)
