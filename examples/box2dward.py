# -*- coding: utf-8 -*-
"""《箱境 / Box2Dward》—— 2D 刚体物理侧视演示（world3d.physics.box2d_lite 首秀）。

**为什么值得单独一个 demo**
----------------------------
``world3d.physics.box2d_lite`` 是引擎的**纯 2D** 刚体求解器（与 rigidward 用到的
``physics.jolt_lite`` 是完全不同的另一套——后者是 3D GJK/EPA + 顺序冲量，前者是
Box2D-lite 风格的 2D 顺序冲量 + warm start + Baumgarte + Coulomb 摩擦）。演示仓里
**此前零个 demo 消费过它**，而它恰好是讲「2D 物理」最直接的一块能力。

本 demo 把 2D 仿真画成**侧视**：物理平面是 (x, y)，重力沿 -y；渲染时把它映射到
3D 世界 (x, y, 0)，方块沿 z 轴挤出厚度、圆形画成球、地面是一条板。相机从斜前方
看向 XY 平面（OrbitCamera 侧视角度），于是「2D 重力下落 + 堆叠 + 滚动」一目了然。

展示了 box2d_lite 的哪些 API
----------------------------
* ``World(gravity=...)``：建世界、装重力；``world.step(dt, iterations)`` 推进一帧。
* ``Body(shape, mass, pos, angle, vel, ang_vel, static, friction)``：刚体；
  位置在 ``body.pos``（2D numpy 向量），朝向 ``body.angle``（绕 z 轴弧度），
  速度 ``body.vel``，角速度 ``body.ang_vel``，``body.static`` / ``body.sleeping``
  标记是否与休眠；重力与积分由 ``World.step`` 负责，刚体只持有状态。
* ``Box((hx, hy))`` / ``Circle(radius)``：凸形状，挂到 ``Body(shape=...)`` 上；
  Box 半边长、Circle 半径即几何尺寸。
* ``Arbiter`` / ``World._arbiters``：每一对接触体一个 Arbiter，``arb.contacts``
  是接触点列表，每个点有 ``pos``（2D 接触位置）与穿透 ``pen``——HUD 的「接触对数」
  读的就是 ``len(world._arbiters)``，并额外统计接触点总数；演示还把每个接触点
  画成红点，让"碰撞"看得见。
* ``collide(a, b)`` 由 ``World.step`` 内部调用做窄相（圆-圆 / 圆-盒 / 盒-盒 SAT），
  本 demo 不直接调用它，只消费它产出的 Arbiter。

玩法（固定随机种子，可复现）
----------------------------
* 方块与圆形从顶部按固定种子不断掉落、堆叠、碰撞、滚动；数量到上限后把**最上层
  已休眠**的刚体回收到顶部继续落，形成持续"喷泉"且不删体（避免下方悬空）。
* 鼠标左键在光标处制造一次**爆裂冲量**（把附近刚体向外推开）；`R` 重置场景；
  `SPACE` 暂停；`W` 线框；`C` 切换接触点可视化；拖拽/滚轮转视角看厚度。

HUD（右上角 ``status_text``）实时读：
  刚体总数 ``bodies``、动态刚体数 ``dyn``、当前接触/碰撞对数 ``pairs``、
  接触点总数 ``pts``、帧率 ``fps``。

用法::

    python examples/box2dward.py                正常运行
    python examples/box2dward.py --smoke 2      冒烟（2 秒后自动关闭）
    python examples/box2dward.py --shot          抓剧照到 docs/box2dward_01/02.png

按键：左键 爆裂   R 重置   SPACE 暂停   W 线框   C 接触点   拖拽/滚轮 转视角   ESC 退出
"""
import math
import random

from pyglet.gl import (GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
                       GL_FILL, GL_FRONT_AND_BACK, GL_LINE, GL_TRIANGLES, glClear,
                       glClearColor, glEnable, glPolygonMode)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key

# ⚠️ 不要再加 sys.path 引导：引擎由 `pip install 3dworld` 提供
# （见 requirements.txt 与 docs/依赖与版本.md）。
from world3d.logic.camera import OrbitCamera                            # noqa: E402
from world3d.platform.app import DemoApp, orbit_camera_input, run       # noqa: E402

try:
    # 注意：**不能在 import 期 raise** —— verify_demos 批量 import 每个 demo
    # 读 SHOTS 时，这里一 raise 整个批量验收就跟着退出。
    from world3d.physics.box2d_lite import Body, Box, Circle, World     # noqa: E402
    HAS_BOX2D = True
except ModuleNotFoundError:                                            # noqa: E402
    HAS_BOX2D = False

SEED = 20261026
STEP = 1.0 / 60.0          # 固定步长，与渲染解耦
MAX_CATCHUP = 4            # 卡顿后最多追 4 步，避免死亡螺旋
ITERATIONS = 12            # 顺序冲量迭代次数
MAX_BODIES = 64            # 动态刚体上限（到达后回收最上层继续落）
SPAWN_DT = 0.25            # 落体节奏（仿真秒）
SPAWN_X = 12.0             # 落点半宽
SPAWN_Y = 20.0             # 出生高度
WALL_X = 16.0              # 左右墙位置
HALF_DEPTH = 0.9           # 3D 挤出厚度（z 方向半厚）

# 暖色给方块、冷色给圆，休眠时压暗——一眼区分形状与状态
PALETTE = ((0.86, 0.42, 0.30), (0.95, 0.72, 0.28), (0.36, 0.72, 0.62),
           (0.42, 0.58, 0.86), (0.78, 0.48, 0.78), (0.66, 0.78, 0.36),
           (0.88, 0.55, 0.40), (0.50, 0.70, 0.78))


# ------------------------------------------------------------- 网格构建 --
def box_mesh():
    """单位盒（半边长 1），逐面法线。"""
    faces = (((1, 0, 0), (0, 0, -1), (0, 1, 0)),    # +X
             ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),    # -X
             ((0, 1, 0), (1, 0, 0), (0, 0, 1)),     # +Y
             ((0, -1, 0), (1, 0, 0), (0, 0, -1)),   # -Y
             ((0, 0, 1), (1, 0, 0), (0, 1, 0)),     # +Z
             ((0, 0, -1), (-1, 0, 0), (0, 1, 0)))   # -Z
    pos, nrm = [], []
    for n, u, v in faces:
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1)):
            pos.extend(n[i] + u[i] * su + v[i] * sv for i in range(3))
            nrm.extend(n)
    return pos, nrm


def sphere_mesh(seg=10, ring=7):
    """单位球（半径 1）。法线即顶点位置。"""
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
            for tri in (a, b, c, a, c, d):
                pos.extend(tri)
                nrm.extend(tri)
    return pos, nrm


# --------------------------------------------------------------- 着色 --
VERTEX_SRC = '''#version 330 core
in vec3 position;
in vec3 normals;
in vec4 colors;

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;
uniform mat4 u_model;

out vec3 v_nrm;
out vec3 v_col;
out vec3 v_world;

void main() {
    vec4 world = u_model * vec4(position, 1.0);
    v_world = world.xyz;
    // 缩放只影响长度：归一化后轴向/旋转后法线方向仍正确
    v_nrm = normalize(mat3(u_model) * normals);
    v_col = colors.rgb;          // 顶点色属性（vec4）参与着色
    gl_Position = window.projection * window.view * world;
}'''

FRAGMENT_SRC = '''#version 330 core
in vec3 v_nrm;
in vec3 v_col;
in vec3 v_world;
out vec4 final_colors;

uniform vec3 u_tint;
uniform vec3 u_sun;
uniform vec3 u_fog;
uniform float u_flat;      // 1 = 纯色（接触点等）

void main() {
    if (u_flat > 0.5) {
        final_colors = vec4(v_col * u_tint, 1.0);
        return;
    }
    vec3 n = normalize(v_nrm);
    float lam = max(dot(n, u_sun), 0.0);
    float hemi = 0.5 + 0.5 * n.y;
    vec3 amb = mix(vec3(0.14, 0.13, 0.14), vec3(0.30, 0.34, 0.42), hemi);
    vec3 col = v_col * u_tint * (amb + lam * vec3(0.95, 0.92, 0.84));
    col = mix(col, u_fog, clamp(length(v_world.xz) * 0.012, 0.0, 0.55));
    final_colors = vec4(col, 1.0);
}'''


# ----------------------------------------------------------------- 小工具 --
def _norm(v):
    L = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return (v[0] / L, v[1] / L, v[2] / L)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


# ------------------------------------------------------------------- app --
class Box2Dward(DemoApp):
    TITLE = 'BOX2Dward - 2D rigid bodies, side view (physics/box2d_lite)'
    HELP = ('click explode  R reset  SPACE pause  W wire  C contacts  '
            'drag/scroll  ESC')
    DURATION = None
    SHOTS = ((6.0, 'docs/box2dward_01.png'), (16.0, 'docs/box2dward_02.png'))
    FOV = 50.0

    def setup(self):
        glEnable(GL_DEPTH_TEST)
        self.prog = ShaderProgram(Shader(VERTEX_SRC, 'vertex'),
                                  Shader(FRAGMENT_SRC, 'fragment'))
        self.meshes = {}
        for name, (pos, nrm) in (('box', box_mesh()),
                                 ('sphere', sphere_mesh())):
            n = len(pos) // 3
            white = (1.0, 1.0, 1.0, 1.0) * n
            self.meshes[name] = self.prog.vertex_list(
                n, GL_TRIANGLES,
                position=('f', pos), normals=('f', nrm), colors=('f', white))
        self.box_vl = self.meshes['box']
        self.sphere_vl = self.meshes['sphere']

        self.wire = 'wire' in self.args.flags      # --flag wire：线框对照（也用于验证 GL_LINE 路径）
        self.show_contacts = True
        self.acc = 0.0
        self.sim_t = 0.0
        self.spawn_acc = 0.0
        self.cam = OrbitCamera(target=Vec3(0, 7, 0), yaw=60.0, pitch=18.0,
                               dist=46.0, pitch_range=(2.0, 88.0),
                               dist_range=(12.0, 90.0))
        orbit_camera_input(self, self.cam)
        # 鼠标左键爆裂：base 未转发 on_mouse_press，这里单独挂一个
        self.window.push_handlers(on_mouse_press=self._on_mouse_press)
        self.rebuild()

    # ------------------------------------------------------------ 世界 --
    def rebuild(self):
        self.rng = random.Random(SEED)
        self.world = World(gravity=(0.0, -9.8))
        # 地面（顶面 y=0）+ 左右墙，围出落体井
        self.world.add(Body(Box((WALL_X + 2.0, 0.5)), mass=0.0,
                            pos=(0.0, -0.5), static=True, friction=0.85))
        for sx in (-WALL_X, WALL_X):
            self.world.add(Body(Box((0.5, 22.0)), mass=0.0,
                                pos=(sx, 11.0), static=True, friction=0.4))
        self.dynamic = []            # 仅动态刚体（用于爆裂/回收/计数）
        self._color_of = {}         # body.id -> 基色
        self.acc = 0.0
        self.sim_t = 0.0
        self.spawn_acc = 0.0
        print(f'[box2dward] world built  ground + 2 walls  gravity=-9.8')

    def _spawn(self):
        """从顶部掉一个方块或圆（固定种子，可复现）。"""
        if self.rng.random() < 0.5:
            hx = self.rng.uniform(0.55, 1.1)
            hy = self.rng.uniform(0.55, 1.1)
            shape = Box((hx, hy))
            sx, sy = hx, hy
        else:
            r = self.rng.uniform(0.55, 1.0)
            shape = Circle(r)
            sx = sy = r
        x = self.rng.uniform(-SPAWN_X, SPAWN_X)
        body = Body(shape, mass=1.0 if isinstance(shape, Box) else 0.8,
                    pos=(x, SPAWN_Y), angle=self.rng.uniform(-0.25, 0.25),
                    vel=(self.rng.uniform(-1.0, 1.0), 0.0),
                    ang_vel=self.rng.uniform(-1.0, 1.0), friction=0.4)
        self.world.add(body)
        self.dynamic.append(body)
        col = PALETTE[len(self.dynamic) % len(PALETTE)]
        if isinstance(shape, Box):
            col = (min(1.0, col[0] + 0.05), col[1], col[2])  # 方块偏暖
        else:
            col = (col[0], min(1.0, col[1] + 0.05), min(1.0, col[2] + 0.05))  # 圆偏冷
        self._color_of[body.id] = col

    def _recycle_top(self):
        """把最上层（pos.y 最大）已休眠的刚体送回顶部继续落。

        只回收最上层 => 它上方没有支撑体，移除不会让别的刚体悬空。
        不删除刚体，仅传送，故 world.bodies 数量恒定、无残留 Arbiter。
        """
        best = None
        for b in self.dynamic:
            if not b.sleeping:
                continue
            if best is None or b.pos[1] > best.pos[1]:
                best = b
        if best is None:
            return
        x = self.rng.uniform(-SPAWN_X, SPAWN_X)
        best.pos[0], best.pos[1] = x, SPAWN_Y
        best.angle = self.rng.uniform(-0.25, 0.25)
        best.vel[0], best.vel[1] = self.rng.uniform(-1.0, 1.0), 0.0
        best.ang_vel = self.rng.uniform(-1.0, 1.0)
        best.sleeping = False
        best._sleep_frames = 0

    def _step_sim(self, dt):
        """固定步长推进（最多追 MAX_CATCHUP 步），并在节奏上落体/回收。"""
        self.acc += dt
        steps = 0
        while self.acc >= STEP and steps < MAX_CATCHUP:
            self.world.step(STEP, ITERATIONS)
            self.acc -= STEP
            self.sim_t += STEP
            steps += 1
        if steps == 0:
            return
        self.spawn_acc += dt
        while self.spawn_acc >= SPAWN_DT:
            self.spawn_acc -= SPAWN_DT
            if len(self.dynamic) < MAX_BODIES:
                self._spawn()
            else:
                self._recycle_top()

    # ------------------------------------------------------------ 交互 --
    def on_key(self, symbol, modifiers):
        k = key
        if symbol == k.R:
            self.rebuild()
        elif symbol == k.W:
            self.wire = not self.wire
        elif symbol == k.C:
            self.show_contacts = not self.show_contacts

    def _on_mouse_press(self, x, y, button, modifiers):
        p = self._pick_world_xy(x, y)
        if p is not None:
            self._explode(p[0], p[1])

    def _pick_world_xy(self, sx, sy):
        """屏幕坐标 -> 物理平面 (x, y)（z=0 平面射线求交）。"""
        w, h = self.window.width, self.window.height
        eye = self.cam.eye
        tgt = self.cam.target
        fwd = _norm((tgt.x - eye.x, tgt.y - eye.y, tgt.z - eye.z))
        up = (0.0, 1.0, 0.0)
        right = _norm(_cross(fwd, up))
        cam_up = _norm(_cross(right, fwd))
        aspect = w / max(1, h)
        tan_v = math.tan(math.radians(self.FOV) * 0.5)
        nx = 2.0 * sx / w - 1.0
        ny = 2.0 * sy / h - 1.0
        dir = _norm((fwd[0] + nx * tan_v * aspect * right[0] + ny * tan_v * cam_up[0],
                     fwd[1] + nx * tan_v * aspect * right[1] + ny * tan_v * cam_up[1],
                     fwd[2] + nx * tan_v * aspect * right[2] + ny * tan_v * cam_up[2]))
        if abs(dir[2]) < 1e-6:
            return None
        t = -eye.z / dir[2]
        if t < 0:
            return None
        return (eye.x + t * dir[0], eye.y + t * dir[1])

    def _explode(self, px, py):
        """在 (px, py) 制造一次爆裂冲量：附近刚体被向外推开。"""
        for b in self.dynamic:
            dx = b.pos[0] - px
            dy = b.pos[1] - py
            d = math.hypot(dx, dy)
            if d > 14.0:
                continue
            if d < 1e-3:
                dx, dy = self.rng.uniform(-1, 1), self.rng.uniform(-1, 1)
                d = math.hypot(dx, dy) or 1.0
            mag = 14.0 * (1.0 - d / 14.0) + 2.0
            b.vel[0] += dx / d * mag
            b.vel[1] += dy / d * mag + 2.0     # 略微上抛，更"炸"
            b.ang_vel += self.rng.uniform(-6.0, 6.0)
            b.sleeping = False
            b._sleep_frames = 0

    # ------------------------------------------------------------ 更新 --
    def update(self, dt):
        if not self.paused:
            self._step_sim(dt)

    # ------------------------------------------------------------ 渲染 --
    def render(self):
        w, h = self.window.width, self.window.height
        glClearColor(0.08, 0.09, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.window.projection = Mat4.perspective_projection(
            w / max(1, h), z_near=0.5, z_far=600.0, fov=self.FOV)
        self.window.view = self.cam.view_matrix()

        prog = self.prog
        prog.use()
        prog['u_sun'] = (0.44, 0.80, 0.44)
        prog['u_fog'] = (0.08, 0.09, 0.12)
        prog['u_flat'] = 0.0
        if self.wire:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        for b in self.world.bodies:
            if isinstance(b.shape, Box):
                mesh = self.box_vl
                sx, sy = b.shape.half[0], b.shape.half[1]
                model = (Mat4.from_translation(Vec3(b.pos[0], b.pos[1], 0.0))
                         @ Mat4.from_rotation(b.angle, Vec3(0, 0, 1))
                         @ Mat4.from_scale(Vec3(sx, sy, HALF_DEPTH)))
            else:
                mesh = self.sphere_vl
                r = b.shape.radius
                model = (Mat4.from_translation(Vec3(b.pos[0], b.pos[1], 0.0))
                         @ Mat4.from_scale(Vec3(r, r, r)))
            prog['u_model'] = model
            if b.static:
                prog['u_tint'] = (0.38, 0.40, 0.46)
            else:
                col = self._color_of.get(b.id, (0.8, 0.8, 0.8))
                if b.sleeping:
                    col = tuple(c * 0.5 + 0.06 for c in col)
                prog['u_tint'] = col
            mesh.draw(GL_TRIANGLES)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        if self.show_contacts:
            self._draw_contacts(prog)

        prog['u_flat'] = 0.0
        self.window.projection = self.perspective

    def _draw_contacts(self, prog):
        """把每个接触点画成红点（读 Arbiter.contacts，碰撞看得见）。"""
        arbs = getattr(self.world, '_arbiters', {})
        if not arbs:
            return
        prog['u_flat'] = 1.0
        prog['u_tint'] = (1.0, 0.32, 0.26)
        for arb in arbs.values():
            for c in arb.contacts:
                x, y = float(c.pos[0]), float(c.pos[1])
                prog['u_model'] = (Mat4.from_translation(Vec3(x, y, 0.0))
                                   @ Mat4.from_scale(Vec3(0.18, 0.18, 0.18)))
                self.sphere_vl.draw(GL_TRIANGLES)
        prog['u_flat'] = 0.0

    # ------------------------------------------------------------ 状态 --
    def status_text(self):
        arbs = getattr(self.world, '_arbiters', {})
        pairs = len(arbs)
        pts = sum(len(a.contacts) for a in arbs.values())
        sleeping = sum(1 for b in self.dynamic if b.sleeping)
        return (f'box2d  bodies {len(self.world.bodies)}'
                f'  dyn {len(self.dynamic)}  sleeping {sleeping}'
                f'  pairs {pairs}  pts {pts}'
                f'  t {self.sim_t:5.1f}s  {self.fps:5.1f} fps'
                f'{"  [PAUSED]" if self.paused else ""}')


if __name__ == '__main__':
    if not HAS_BOX2D:
        raise SystemExit(
            '[box2dward] 需要 world3d.physics.box2d_lite：'
            '当前引擎安装缺少该模块。')
    run(Box2Dward)
