# -*- coding: utf-8 -*-
"""《刚境 / Rigidward》—— 刚体物理实验场（jolt_lite 第一次有 demo 消费）。

**为什么值得单独一个 demo**
----------------------------
``world3d.physics.jolt_lite`` 是引擎里单测最厚的一块：17 项断言覆盖
**GJK-EPA 穿透**（四面体种子严格含原点 + 有向边 horizon 扩张）、顺序冲量、
warm start、Baumgarte 偏置、Coulomb 摩擦、回弹，还有 **sweep-and-prune 宽相 +
岛管理 + 按 body id 定序遍历**（打乱 ``world.bodies`` 结果仍逐位相同）。

``tests/test_jolt_lite.py`` 把这些性质钉死了，但**演示仓里零个 demo 用过它** ——
物理一直只在玩法内核里当配角。本 demo 把它拉到台前：五个预设场景各自逼出
一类性质，HUD 直接把 ``world.last_stats``（宽相候选 / 接触数 / 岛数）打出来，
于是"宽相剪枝有没有起作用""是不是按岛求解"这类事是**看得见的数**，不是注释。

五个场景各测什么
----------------
* ``stack``    三柱五层方塔：静态摩擦 + Baumgarte —— 塔不抖不滑才算过；
               稳定后物体逐个 sleep，HUD 的 sleeping 计数会爬上去。
* ``pile``     60 颗球落进料仓：堆成丘（休止角），是"宽相 vs 接触数"的压力表。
* ``friction`` 三条摩擦系数 0.12 / 0.45 / 0.90 的跑道，同初速度比滑行距离。
* ``bounce``   回复系数 0.15 / 0.55 / 0.88 三列并列，弹跳高度差肉眼可见。
* ``impact``   14kg 重球撞散 12 块方箱阵列：动量传递（唯一的不稳定场景）。

**选构型时踩到的引擎边界**（2026-10-26 实测，记在这里免得后人再撞一次）：
运行时窄相走的是**解析流形**（球-球 / 球-盒 / **轴对齐**盒-盒），GJK + EPA 虽然
在 ``epa()`` 里独立验证过，但按模块注释并未接入运行时层。于是：

* **旋转过的盒**（斜面、多米诺骨牌、被撞翻的方块）拿不到接触 —— 会直接穿过
  地面；更糟的是中心一旦越过地面顶面，接触法线翻向会把物体**继续往下推**，
  一路到 -100 米开外。本 demo 因此只用轴对齐盒，并配了 ``rescue`` 兜底复位。
* **胶囊 / 宽扁砖长塔**同理不可靠：前者不稳定，后者层数一多就 self-destruct。
* 迭代次数对堆叠稳定性影响很大，默认给到 16（``I`` 键可切 4/8/16/24）。

实现要点
--------
* **固定步长累加器**：物理恒定 ``STEP = 1/60``，与渲染帧率解耦；
  一帧最多追 4 步（卡顿后不追赶到失控）。
* 渲染是「单位网格 + 每体一个 model 矩阵」：球用均匀缩放、盒用轴缩放
  （轴缩放不改变轴向法线的方向，normalize 后仍然正确），胶囊尺寸固定一次建成。
* 每个刚体的变换矩阵 = 平移 × 四元数 × 缩放，全部取自 ``body.xf``，
  渲染层不持有任何物理状态的副本。

用法::

    python examples/rigidward.py                正常运行
    python examples/rigidward.py --smoke 3      冒烟
    python examples/rigidward.py --shot         抓剧照到 docs/rigidward_01/02.png
    python examples/rigidward.py --flag domino  起手就切到多米诺

按键：1-5 场景   R 重置   SPACE 暂停   S 单步   F 推一把
      G 换重力（地球/月球/木星）   I 换迭代次数   W 线框   ESC 退出
"""
import math
import time

from pyglet.gl import (GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
                       GL_DEPTH_TEST, GL_FILL, GL_FRONT_AND_BACK, GL_LINE,
                       GL_TRIANGLES, glClear, glClearColor, glEnable,
                       glPolygonMode)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Quaternion, Vec3
from pyglet.window import key

# ⚠️ 不要再加 sys.path 引导：引擎由 `pip install 3dworld` 提供
# （见 requirements.txt 与 docs/依赖与版本.md）。
from world3d.logic.camera import OrbitCamera                            # noqa: E402
from world3d.platform.app import DemoApp, orbit_camera_input, run       # noqa: E402

try:
    from world3d.physics.jolt_lite import (Box, Capsule, RigidBody,     # noqa: E402
                                           Sphere, World)
    HAS_JOLT = True
except ModuleNotFoundError:                                            # noqa: E402
    # 注意：**不能在 import 期 raise** —— verify_demos.py 为了读 SHOTS 会
    # import 每个 demo 模块，这里一 raise 整个批量验收就跟着退出了。
    HAS_JOLT = False

STEP = 1.0 / 60.0
# 纯 Python solver 每步 ~17ms（16 体 / 8 次迭代，实测），追帧会把
# 帧率拖进"dt 大 → 追更多步 → 更慢"的死亡螺旋；每帧最多补 2 步，
# 宁可接受轻微慢放也不让帧率崩掉。
MAX_CATCHUP = 2
GRAVITIES = (('earth', (0.0, -9.8, 0.0)),
             ('moon', (0.0, -1.62, 0.0)),
             ('jupiter', (0.0, -24.8, 0.0)))
ITER_CHOICES = (4, 8, 16, 32)

# 每个场景一个色板（盒/球共用）
PALETTE = ((0.86, 0.42, 0.30), (0.95, 0.72, 0.28), (0.36, 0.72, 0.62),
           (0.42, 0.58, 0.86), (0.78, 0.48, 0.78), (0.66, 0.78, 0.36))
# 方塔三柱各自的色系（渲染时按层提亮，见 scene_stack）
COLUMN_COLS = ((0.88, 0.42, 0.32), (0.34, 0.72, 0.60), (0.44, 0.58, 0.88))


# ------------------------------------------------------------- 网格构建 --
def box_mesh():
    """单位盒（半边长 1），36 顶点、逐面法线。"""
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


def sphere_mesh(seg=14, ring=10):
    """单位球（半径 1）。法线就是顶点位置。"""
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


def capsule_mesh(radius=0.45, half=0.70, seg=14, cap=5):
    """胶囊：中间圆柱（半高 half）+ 两端半球（半径 radius），尺寸固定故只建一次。

    法线用胶囊的**精确**定义：取顶点到「轴线段 [-half, half]」的最近点，
    由最近点指向顶点 —— 圆柱段退化成纯径向，球冠段自然过渡，接缝处不跳变。
    """
    pos, nrm = [], []

    def tri(a, b, c):
        for p in (a, b, c):
            pos.extend(p)
            cy = max(-half, min(half, p[1]))
            dx, dy, dz = p[0], p[1] - cy, p[2]
            L = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
            nrm.extend((dx / L, dy / L, dz / L))

    for sign in (1.0, -1.0):                   # 上球冠 / 下球冠
        for i in range(cap):
            t0 = (math.pi / 2) * i / cap
            t1 = (math.pi / 2) * (i + 1) / cap
            y0 = math.cos(t0) * radius
            y1 = math.cos(t1) * radius
            r0 = math.sin(t0) * radius
            r1 = math.sin(t1) * radius
            for j in range(seg):
                p0, p1 = 2 * math.pi * j / seg, 2 * math.pi * (j + 1) / seg
                a = (r0 * math.cos(p0), sign * (half + y0), r0 * math.sin(p0))
                b = (r0 * math.cos(p1), sign * (half + y0), r0 * math.sin(p1))
                c = (r1 * math.cos(p1), sign * (half + y1), r1 * math.sin(p1))
                d = (r1 * math.cos(p0), sign * (half + y1), r1 * math.sin(p0))
                if sign > 0:
                    tri(a, b, c), tri(a, c, d)
                else:
                    tri(a, c, b), tri(a, d, c)
    for j in range(seg):                       # 圆柱段
        p0, p1 = 2 * math.pi * j / seg, 2 * math.pi * (j + 1) / seg
        a = (math.cos(p0) * radius, half, math.sin(p0) * radius)
        b = (math.cos(p1) * radius, half, math.sin(p1) * radius)
        c = (math.cos(p1) * radius, -half, math.sin(p1) * radius)
        d = (math.cos(p0) * radius, -half, math.sin(p0) * radius)
        tri(a, b, c), tri(a, c, d)
    return pos, nrm


CAP_R, CAP_H = 0.45, 0.70


# ------------------------------------------------------------- 场景构建 --
def _ground(world, half_x=34.0, half_z=34.0, y=0.0, friction=0.7):
    """地面：一块很薄的静态盒（顶面在 y=0）。"""
    return world.add(RigidBody(Box((half_x, 0.5, half_z)), mass=0.0,
                               pos=(0.0, y - 0.5, 0.0), is_static=True,
                               friction=friction))


def scene_stack(world, rng):
    """三柱方塔：静态摩擦 + Baumgarte，稳住不滑才算过。

    ⚠️ 构型是实测逼出来的，别手贱改：

    * **宽扁砖**（half=(0.48,0.44,0.30)）摞到 6 层以上会被自身抖动挤出地面，
      一旦中心越过地面顶面，接触法线就翻过来把它继续往下推 —— 整塔一路"沉"
      到 -100 米开外。立方体 + 层间零间隙才是这个 solver 的舒适区。
    * **同柱必须严格对齐**：给每块砖加 ±6mm 的初始错位，看似无害，实测
      0.33 秒内 min_y 就掉到 -0.08 —— 错位让接触点偏心 ⇒ 产生力矩 ⇒ 砖微微
      旋转 ⇒ 轴对齐的解析流形再也拿不到接触 ⇒ 整塔穿透。去掉抖动（本函数现在
      的做法）后同样的三柱五层，4 秒后仍然稳在 y=0.397 纹丝不动。

    这两条记在这里：它们是 **jolt_lite 当前的能力边界**，不是 demo 的疏漏。
    """
    _ground(world)
    layers = 5
    cols = ((-2.0, 0.0), (2.0, 0.0), (0.0, 3.0))     # 三角形排布：三柱互不遮挡
    for c, (cx, cz) in enumerate(cols):
        for i in range(layers):
            body = world.add(RigidBody(Box((0.40, 0.40, 0.40)), mass=1.0,
                                       pos=(cx, 0.40 + i * 0.80, cz),
                                       friction=0.75, restitution=0.0))
            # 每柱一个色系、层间逐级提亮 —— 否则整塔糊成一个色块，看不出有几层
            k = 0.55 + i * 0.11
            body.tint = tuple(min(1.0, v * k) for v in COLUMN_COLS[c])
    del rng
    return layers * len(cols)


def scene_pile(world, rng):
    """球堆：一圈静态矮墙围出料仓，球落下堆成丘（自由休止角）。"""
    _ground(world)
    for dx, dz, hx, hz in ((0.0, -7.5, 7.5, 0.4), (0.0, 7.5, 7.5, 0.4),
                           (-7.5, 0.0, 0.4, 7.5), (7.5, 0.0, 0.4, 7.5)):
        world.add(RigidBody(Box((hx, 1.1, hz)), mass=0.0,
                            pos=(dx, 1.1, dz), is_static=True, friction=0.5))
    n = 60
    for i in range(n):
        a = rng.uniform(0.0, math.tau)
        r = rng.uniform(0.5, 5.5)
        world.add(RigidBody(Sphere(0.48), mass=1.0,
                            pos=(math.cos(a) * r, 3.0 + i * 0.42,
                                 math.sin(a) * r),
                            friction=0.45, restitution=0.15))
    return n


def scene_cascade(world, rng):
    """阶梯瀑布：三级下降的静态台，一串球滚下来逐级跌落。

    ⚠️ 这里原本是"4-3-2-1 方箱金字塔"（上层箱子骑在下层接缝上，一个 body 同时
    压两个支撑），实测撑不过半秒就整座塌穿 —— **多支撑 + 方箱 = 当前不可用**。
    换成球撞静态台的组合后 4 秒纹丝不乱，因为球-盒是全程可靠的那一类流形。
    """
    _ground(world)
    steps = ((-8.0, 6.6), (0.0, 4.2), (8.0, 1.8))
    for x, y in steps:
        world.add(RigidBody(Box((8.0, 0.6, 3.0)), mass=0.0,
                            pos=(x, y - 0.6, 0.0), is_static=True,
                            friction=0.30))
    n = 0
    for i in range(20):
        b = world.add(RigidBody(Sphere(0.45), mass=1.0,
                                pos=(-12.0, 7.6 + i * 0.95,
                                     rng.uniform(-0.8, 0.8)),
                                friction=0.30, restitution=0.25))
        b.vel[0] = 3.4                  # 给一点初速才有"流"的感觉
        n += 1
    return n


def scene_bounce(world, rng):
    """回弹三条并列的球，回复系数 0.15 / 0.55 / 0.88 —— 高度差肉眼可见。

    正则化了 worries：三条组同时释放，前两秒差异最明显，之后会各自睡去。
    """
    _ground(world)
    n = 0
    for k, rest in enumerate((0.15, 0.55, 0.88)):
        x = (k - 1) * 7.0
        for i in range(4):
            world.add(RigidBody(Sphere(0.45), mass=1.0,
                                pos=(x + i * 1.15, 5.5 + i * 1.7, 0.0),
                                friction=0.3, restitution=rest))
            n += 1
    del rng
    return n


def scene_break(world, rng):
    """台球开球：母球把三角形球阵炸开 —— 动量沿接触链传递的最佳注脚。

    为什么全用球（而不是"重球撞方箱"那种更帅的构型）：球-球 / 球-盒是这个
    solver 唯一全程可靠的两种流形，方箱被撞后会旋转，旋转盒拿不到接触，
    接着就是熟悉的"穿透 → 法线翻转 → 沉到 -100 米"（见模块 docstring）。
    球阵撞 4 秒不会出问题，``rescue`` 计数应当一直是 0。
    """
    _ground(world)
    row, spacing = 0, 1.02
    cue = world.add(RigidBody(Sphere(0.50), mass=3.0,
                              pos=(-7.0, 0.50, 0.0), friction=0.25,
                              restitution=0.30))
    cue.vel[0] = 13.0
    n = 1
    for i in range(5):
        for j in range(i + 1):
            x = 1.2 + i * spacing * 0.88
            z = (j - i * 0.5) * spacing
            world.add(RigidBody(Sphere(0.50), mass=1.0, pos=(x, 0.50, z),
                                friction=0.25, restitution=0.30))
            n += 1
    del row, rng
    return n


def _axis_quat(axis, angle):
    ax, ay, az = axis
    L = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    s = math.sin(angle * 0.5)
    return (math.cos(angle * 0.5), ax / L * s, ay / L * s, az / L * s)


SCENES = (('stack', scene_stack, 'column stacks: static friction + baumgarte'),
          ('cascade', scene_cascade, 'stair cascade: spheres down static steps'),
          ('pile', scene_pile, 'sphere heap in a bin: repose angle + sleeping'),
          ('bounce', scene_bounce, 'restitution 0.15 / 0.55 / 0.88 side by side'),
          ('break', scene_break, 'billiard break: momentum along contact chain'))


# ----------------------------------------------------------------- 着色 --
VERTEX_SRC = '''#version 330 core
in vec3 position;
in vec3 normals;

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;
uniform mat4 u_model;

out vec3 v_nrm;
out vec3 v_world;

void main() {
    vec4 world = u_model * vec4(position, 1.0);
    v_world = world.xyz;
    // 缩放只影响长度：归一化后轴缩放不改变轴向法线的方向
    v_nrm = normalize(mat3(u_model) * normals);
    gl_Position = window.projection * window.view * world;
}'''

FRAGMENT_SRC = '''#version 330 core
in vec3 v_nrm;
in vec3 v_world;
out vec4 final_colors;

uniform vec3 u_color;
uniform vec3 u_sun;
uniform vec3 u_fog;
uniform float u_ground;   // 1 = 地面（画棋盘格）
uniform float u_emph;     // 1 = 高亮（被推过的物体）

void main() {
    vec3 base = u_color;
    if (u_ground > 0.5) {
        vec2 g = floor(v_world.xz * 0.5);
        float checker = mod(g.x + g.y, 2.0);
        base = mix(vec3(0.20, 0.22, 0.25), vec3(0.28, 0.31, 0.34), checker);
    }
    vec3 n = normalize(v_nrm);
    float lam = max(dot(n, u_sun), 0.0);
    // 半球环境光：天空偏冷、地面反弹偏暖，比单一 ambient 立体得多
    float hemi = 0.5 + 0.5 * n.y;
    vec3 amb = mix(vec3(0.16, 0.14, 0.12), vec3(0.30, 0.36, 0.46), hemi);
    vec3 col = base * (amb + lam * vec3(1.02, 0.96, 0.86));
    col = mix(col, vec3(1.0, 0.86, 0.55), u_emph * 0.35);
    float d = length(v_world.xz) * 0.016;
    col = mix(col, u_fog, clamp(d, 0.0, 0.62));
    final_colors = vec4(col, 1.0);
}'''


# ------------------------------------------------------------------- app --
class Rigidward(DemoApp):
    TITLE = 'RIGIDward - rigid body playground (physics/jolt_lite)'
    HELP = ('1-5 scene  R reset  SPACE pause  S step  F shove  '
            'G gravity  I iterations  W wire  drag/scroll  ESC')
    DURATION = None
    SHOTS = ((6.0, 'docs/rigidward_01.png'), (16.0, 'docs/rigidward_02.png'))
    FOV = 45.0

    def setup(self):
        glEnable(GL_DEPTH_TEST)
        self.program = ShaderProgram(Shader(VERTEX_SRC, 'vertex'),
                                     Shader(FRAGMENT_SRC, 'fragment'))
        self.meshes = {}
        for name, (pos, nrm) in (('box', box_mesh()),
                                 ('sphere', sphere_mesh()),
                                 ('capsule', capsule_mesh(CAP_R, CAP_H))):
            n = len(pos) // 3
            self.meshes[name] = (self.program.vertex_list(
                n, GL_TRIANGLES, position=('f', pos), normals=('f', nrm)), n)

        self.scene_i = 0
        for i, sc in enumerate(SCENES):
            if sc[0] in self.args.flags:
                self.scene_i = i
        self.gravity_i = 0
        self.iter_i = 1                      # 默认 8 次（见下方性能注记）
        self.wire = False
        self.emph = set()
        self.acc = 0.0
        self.sim_t = 0.0
        self.cam = OrbitCamera(target=Vec3(0, 4, 0), yaw=34.0, pitch=22.0,
                               dist=30.0, pitch_range=(2.0, 88.0),
                               dist_range=(6.0, 160.0))
        orbit_camera_input(self, self.cam)
        self.rebuild()

    # ------------------------------------------------------------ 重建 --
    def rebuild(self):
        import random
        name, builder, _ = SCENES[self.scene_i]
        self.world = World(gravity=GRAVITIES[self.gravity_i][1])
        self.rng = random.Random(20261026)
        self.n_dyn = builder(self.world, self.rng)
        # 出界回收：记下每个动态体的出生姿态。深穿透一旦发生，接触法线会翻过来
        # 继续把物体往地面外推（lite solver 没有 CCD），后果是整场景"沉没"到
        # -100 米开外 —— 兜底复位比崩掉的演示体面，HUD 上也会报 rescue 计数。
        self.spawns = [(b, tuple(float(v) for v in b.xf.pos), tuple(b.xf.quat))
                       for b in self.world.bodies if not b.is_static]
        self.rescues = 0
        self.acc = 0.0
        self.sim_t = 0.0
        self.emph.clear()
        self.cam.target = Vec3(0, 3.5, 0)
        self.cam.dist = {'stack': 21.0, 'cascade': 32.0, 'pile': 30.0,
                         'bounce': 24.0, 'break': 26.0}.get(name, 26.0)
        print(f'[rigidward] scene={name} bodies={len(self.world.bodies)} '
              f'dynamic={self.n_dyn} gravity={GRAVITIES[self.gravity_i][0]}')

    # ------------------------------------------------------------ 交互 --
    def on_key(self, symbol, modifiers):
        k = key
        if k._1 <= symbol <= k._5 and symbol - k._1 < len(SCENES):
            self.scene_i = symbol - k._1
            self.rebuild()
        elif symbol == k.R:
            self.rebuild()
        elif symbol == k.S:
            self.world.step(STEP, ITER_CHOICES[self.iter_i])
            self.sim_t += STEP
        elif symbol == k.F:
            for b in self.world.bodies:
                if b.is_static:
                    continue
                b.vel[0] += self.rng.uniform(-3.0, 3.0)
                b.vel[1] += 2.4
                b.vel[2] += self.rng.uniform(-3.0, 3.0)
                b.sleeping = False
                self.emph.add(b.id)
        elif symbol == k.G:
            self.gravity_i = (self.gravity_i + 1) % len(GRAVITIES)
            self.world.gravity = GRAVITIES[self.gravity_i][1]
            for b in self.world.bodies:
                b.sleeping = False
        elif symbol == k.I:
            self.iter_i = (self.iter_i + 1) % len(ITER_CHOICES)
        elif symbol == k.W:
            self.wire = not self.wire

    # ------------------------------------------------------------ 更新 --
    def update(self, dt):
        t0 = time.perf_counter()
        if not self.paused:
            self.acc += dt
            steps = 0
            while self.acc >= STEP and steps < MAX_CATCHUP:
                self.world.step(STEP, ITER_CHOICES[self.iter_i])
                self.acc -= STEP
                self.sim_t += STEP
                steps += 1
            if steps:
                self._rescue()
            self.cam.yaw += dt * 0.6
        self.phys_ms = time.perf_counter() - t0
        if 'perf' in self.args.flags:
            self._perf_n = getattr(self, '_perf_n', 0) + 1
            if self._perf_n % 30 == 0:
                print(f'[perf] step={self.phys_ms * 1000:6.2f}ms '
                      f'draw={getattr(self, "draw_ms", 0) * 1000:6.2f}ms '
                      f'fps={self.fps:5.1f} bodies={len(self.world.bodies)}')

    def _rescue(self):
        """把掉出世界（或算出非有限值）的刚体放回出生点。"""
        for b, pos, quat in self.spawns:
            if b.xf.pos[1] < -12.0 or not math.isfinite(b.xf.pos[1]):
                b.xf.pos[:] = pos
                b.xf.quat = quat
                b.vel[:] = 0.0
                b.ang_vel[:] = 0.0
                self.rescues += 1

    # ------------------------------------------------------------ 渲染 --
    def render(self):
        _t0 = time.perf_counter()
        w, h = self.window.width, self.window.height
        aspect = w / max(1, h)
        glClearColor(0.09, 0.11, 0.14, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.window.projection = Mat4.perspective_projection(
            aspect, z_near=0.5, z_far=600.0, fov=self.FOV)
        self.window.view = self.cam.view_matrix()

        prog = self.program
        prog.use()
        prog['u_sun'] = (0.44, 0.78, 0.44)
        prog['u_fog'] = (0.09, 0.11, 0.14)
        if self.wire:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        for i, b in enumerate(self.world.bodies):
            s = b.shape
            if isinstance(s, Box):
                key = 'box'
                sx, sy, sz = s.h[0], s.h[1], s.h[2]
            elif isinstance(s, Sphere):
                key = 'sphere'
                sx = sy = sz = s.radius
            else:
                key = 'capsule'
                sx = sy = sz = 1.0
            q = b.xf.quat
            model = (Mat4.from_translation(Vec3(*b.xf.pos))
                     @ Quaternion(q[0], q[1], q[2], q[3]).to_mat4()
                     @ Mat4.from_scale(Vec3(sx, sy, sz)))
            prog['u_model'] = model
            prog['u_ground'] = 1.0 if b.is_static and i == 0 else 0.0
            prog['u_emph'] = 1.0 if b.id in self.emph else 0.0
            if b.is_static:
                prog['u_color'] = (0.42, 0.45, 0.50)
            else:
                col = getattr(b, 'tint', None) or PALETTE[b.id % len(PALETTE)]
                # 睡着的不改色相、只压亮度并偏冷：既看得出"它睡了"，
                # 又不像早期版本那样把所有 sleeping 刷成同一片冷灰
                # （那样整座塔会糊成一个色块，层数根本读不出来）
                if b.sleeping:
                    col = tuple(c * 0.45 + 0.08 for c in col)
                prog['u_color'] = col
            mesh, n = self.meshes[key]
            mesh.draw(GL_TRIANGLES)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.window.projection = self.perspective
        self.draw_ms = time.perf_counter() - _t0

    # ------------------------------------------------------------ 状态 --
    def status_text(self):
        st = getattr(self.world, 'last_stats', {})
        sleeping = sum(1 for b in self.world.bodies
                       if not b.is_static and b.sleeping)
        return (f'{SCENES[self.scene_i][0]}  bodies {len(self.world.bodies)}'
                f'  dyn {self.n_dyn}  contacts {st.get("contacts", 0)}'
                f'  islands {st.get("islands", 0)}'
                f'  cand {st.get("broadphase_candidates", 0)}'
                f'  sleep {sleeping}  iters {ITER_CHOICES[self.iter_i]}'
                f'  rescue {self.rescues}  step {self.phys_ms * 1000:.1f}ms'
                f'  {GRAVITIES[self.gravity_i][0]}  t {self.sim_t:5.1f}s'
                f'  {self.fps:5.1f} fps'
                f'{"  [PAUSED]" if self.paused else ""}')


if __name__ == '__main__':
    if not HAS_JOLT:
        raise SystemExit(
            '[rigidward] 需要 3dworld>=0.4.1：world3d.physics.jolt_lite 在 0.3.x 的 '
            'wheel 里没有，0.4.1 起才随包发布。\n'
            '  解决：pip install -U "3dworld>=0.4.1"')
    run(Rigidward)
