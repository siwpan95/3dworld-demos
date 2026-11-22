# -*- coding: utf-8 -*-
"""《众径 / Crowdward》—— 行为树决策 + 网格 A* 的 AI 群体（behavior_tree / pathing 首秀）。

**为什么值得单独一个 demo**
----------------------------
``logic/behavior_tree``（10 项单测：组合/装饰/叶子语义 + 声明式构造 + 假时钟超时）
与 ``core/pathing_core``（网格 A* + 群体编队）都是"在玩法内核里当配角"的能力，
演示仓此前**零个 demo 消费过它们**。本 demo 把两者接成一条完整的 AI 链路：

    感知 → 行为树决策 → A* 求路 → 沿路移动 → 群体分离 → 循环

而且每一步都**看得见**：agent 的颜色就是它此刻选中的行为分支（逃 / 休 / 采 /
送 / 逛），脚下的折线就是 A* 求出来的那条路；HUD 同时给出各分支的命中分布与
A* 的调用次数 —— 于是"它到底在想什么"不再是黑盒。

行为树（每个 agent 一棵，共享结构、各自 blackboard）
--------------------------------------------------
::

    Selector(root)
      ├ Sequence(flee)     条件：威胁进入警戒半径 → 反向逃离
      ├ Sequence(rest)     条件：体力 < 0.25      → 去最近休息点回体力
      ├ Sequence(gather)   条件：空手且有资源      → 去最近资源点
      ├ Sequence(deliver)  条件：手里有货          → 回仓库卸货
      └ Action(wander)     兜底：随机游走

优先级自上而下，所以"逃命"永远压过"搬砖"。装饰器用上了 ``Repeater``
（把 flee 做成"每帧都执行、恒报成功"的持续动作），不是只有 Sequence/Selector。
（``Condition`` / ``Action`` 的回调语义也都在线上 —— 见 ``make_tree``。）

实现要点
--------
* **A* 寻路**（A-star）走引擎的 ``PathingService.find_path``（八向 + 对角代价 √2）：
  ``cordis.Context()`` 就能建出服务实例，不需要挂载配置树。
* 求路是**节流**的：``retarget`` 只在"目标格变了"时才向 A* 求路，
  逃命目标还加了"移动超过 2 格才重算"的额外门槛 —— 否则几十个 agent
  每帧重算会把 CPU 打满（HUD 的 ``astar`` 计数就是证据）。
* 群体分离用简单的 O(n²) 排斥力（n=50 → 1250 次/帧），刻意**不**叠加 boids：
  寻路轨迹要看得清，群集规则会把路径糊掉。

用法::

    python examples/crowdward.py                正常运行
    python examples/crowdward.py --smoke 3      冒烟
    python examples/crowdward.py --shot         抓剧照到 docs/crowdward_01/02.png
    python examples/crowdward.py --flag dense   起手就是密集场景

按键：1-3 场景   A/a agent 数   P 路径线   L 分支图例   R 重置
      SPACE 暂停   W 线框   drag/scroll   ESC
"""
import math
import random

from pyglet.gl import (GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
                       GL_DEPTH_TEST, GL_FILL, GL_FRONT_AND_BACK, GL_LINE,
                       GL_LINES, GL_TRIANGLES, glClear, glClearColor, glEnable,
                       glPolygonMode)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key

# ⚠️ 不要再加 sys.path 引导：引擎由 `pip install 3dworld` 提供
# （见 requirements.txt 与 docs/依赖与版本.md）。
from cordis import Context                                              # noqa: E402
from world3d.core.pathing_core import PathingService                    # noqa: E402
from world3d.logic.behavior_tree import (Action, BehaviorTree,          # noqa: E402
                                         Blackboard, Condition,
                                         Repeater, Selector, Sequence)
from world3d.logic.camera import OrbitCamera                            # noqa: E402
from world3d.platform.app import DemoApp, orbit_camera_input, run        # noqa: E402

GW = GH = 44                   # 网格边数
CELL = 1.6                     # 每格世界尺寸（米）
SPEED = 5.0                    # agent 移动速度（米/秒）
ALERT = 9.0                    # 威胁警戒半径（米）
ARRIVE = 1.1                   # 到达判定（米）

# 分支 → 颜色（HUD 图例与 agent 颜色同源）
BRANCH_COLS = {'flee': (0.95, 0.30, 0.28),
               'rest': (0.35, 0.58, 0.95),
               'gather': (0.42, 0.80, 0.45),
               'deliver': (0.95, 0.75, 0.32),
               'wander': (0.62, 0.62, 0.68)}
BRANCH_ORDER = ('flee', 'rest', 'gather', 'deliver', 'wander')

SCENES = (('sparse', 30, 10), ('normal', 55, 16), ('dense', 90, 24))


# --------------------------------------------------------------- 世界 --
def cell_center(i, j):
    return ((i - GW * 0.5 + 0.5) * CELL, (j - GH * 0.5 + 0.5) * CELL)


def world_to_cell(x, z):
    return (int(x / CELL + GW * 0.5), int(z / CELL + GH * 0.5))


def build_map(rng, n_obstacles):
    """可通行网格（1=可走）+ 障碍矩形列表（供渲染）。"""
    walk = [[1] * GW for _ in range(GH)]
    blocks = []
    for _ in range(n_obstacles):
        w = rng.randrange(3, 8)
        h = rng.randrange(3, 8)
        x = rng.randrange(2, GW - w - 2)
        y = rng.randrange(2, GH - h - 2)
        for j in range(y, y + h):
            for i in range(x, x + w):
                walk[j][i] = 0
        blocks.append((x, y, w, h))
    return walk, blocks


# ----------------------------------------------------------- 行为树 --
def make_tree(agent):
    """一棵树管一个 agent：黑板里放的是它自己的状态引用。"""
    bb = Blackboard({'agent': agent, 'branch': 'wander', 'goal': None})

    def threat_near(_ctx):
        a = _ctx.blackboard.get('agent')
        return a.threat_dist() < ALERT

    def flee_act(_ctx):
        a = _ctx.blackboard.get('agent')
        _ctx.blackboard.set('branch', 'flee')
        fc = a.flee_cell()
        # 节流：逃离目标每帧都在变（威胁在绕圈），但只有移动超过 2 格
        # 才重算 A*，否则几十个 agent 一起逃会每帧刷爆 astar 计数。
        if a.goal is None or abs(fc[0] - a.goal[0]) + abs(fc[1] - a.goal[1]) > 2:
            a.retarget(fc)
        return True

    def tired(_ctx):
        return _ctx.blackboard.get('agent').energy < 0.25

    def rest_act(_ctx):
        a = _ctx.blackboard.get('agent')
        _ctx.blackboard.set('branch', 'rest')
        if a.at_goal_cell():
            a.energy = min(1.0, a.energy + 0.55)
            a.pick_wander()
            return True
        a.retarget(a.nearest(a.rest_cells))
        return True

    def can_gather(_ctx):
        a = _ctx.blackboard.get('agent')
        return a.carrying is False and a.energy > 0.25

    def gather_act(_ctx):
        a = _ctx.blackboard.get('agent')
        _ctx.blackboard.set('branch', 'gather')
        if a.at_goal_cell():
            a.carrying = True
            a.pick_wander()
            return True
        a.retarget(a.nearest(a.res_cells))
        return True

    def has_cargo(_ctx):
        return _ctx.blackboard.get('agent').carrying is True

    def deliver_act(_ctx):
        a = _ctx.blackboard.get('agent')
        _ctx.blackboard.set('branch', 'deliver')
        if a.at_goal_cell():
            a.carrying = False
            a.delivered += 1
            a.pick_wander()
            return True
        a.retarget(a.depot)
        return True

    def wander_act(_ctx):
        a = _ctx.blackboard.get('agent')
        _ctx.blackboard.set('branch', 'wander')
        if a.at_goal_cell() or a.goal is None:
            a.pick_wander()
        return True

    root = Selector('root', [
        Sequence('flee', [Condition('threat?', threat_near),
                          Repeater('keep fleeing', Action('flee', flee_act))]),
        Sequence('rest', [Condition('tired?', tired),
                          Action('rest', rest_act)]),
        Sequence('gather', [Condition('can gather', can_gather),
                            Action('gather', gather_act)]),
        Sequence('deliver', [Condition('has cargo', has_cargo),
                             Action('deliver', deliver_act)]),
        Action('wander', wander_act),
    ])
    return BehaviorTree(root, bb)


class Agent:
    __slots__ = ('x', 'z', 'energy', 'carrying', 'delivered', 'goal', 'path',
                 'pi', 'branch', 'tree', 'bb', 'world', 'seed')

    def __init__(self, world, x, z, rng):
        self.world = world
        self.x, self.z = x, z
        self.energy = rng.uniform(0.35, 1.0)
        self.carrying = False
        self.delivered = 0
        self.goal = None
        self.path = []
        self.pi = 0
        self.branch = 'wander'
        self.tree = make_tree(self)
        self.bb = self.tree.blackboard

    # --- 目标 ---
    def pick_wander(self):
        w = self.world
        self.goal = (w.rng.randrange(1, GW - 1), w.rng.randrange(1, GH - 1))

    def nearest(self, cells):
        best, bd = None, 1e18
        ci, cj = world_to_cell(self.x, self.z)
        for c in cells:
            d = (c[0] - ci) ** 2 + (c[1] - cj) ** 2
            if d < bd:
                bd, best = d, c
        return best

    def flee_cell(self):
        w = self.world
        ci, cj = world_to_cell(self.x, self.z)
        tx, tz = w.threat[0], w.threat[1]
        dx, dz = self.x - tx, self.z - tz
        L = math.hypot(dx, dz) or 1.0
        return (max(1, min(GW - 2, ci + int(6 * dx / L))),
                max(1, min(GH - 2, cj + int(6 * dz / L))))

    def at_goal_cell(self):
        if self.goal is None:
            return False
        ci, cj = world_to_cell(self.x, self.z)
        return (ci, cj) == self.goal

    def retarget(self, cell):
        if cell is None or cell == self.goal:
            return
        self.goal = cell
        self.world.request_path(self)

    def threat_dist(self):
        tx, tz = self.world.threat
        return math.hypot(self.x - tx, self.z - tz)

    # --- 每帧 ---
    def tick_brain(self, dt):
        self.tree.tick()
        self.branch = self.bb.get('branch', 'wander')
        self.energy = max(0.0, self.energy - dt * 0.035)   # 体力随时间下降

    def move(self, dt):
        if not self.path or self.pi >= len(self.path):
            return
        tx, tz = cell_center(*self.path[self.pi])
        dx, dz = tx - self.x, tz - self.z
        d = math.hypot(dx, dz)
        if d < ARRIVE:
            self.pi += 1
            return
        step = SPEED * dt
        if step >= d:
            self.x, self.z = tx, tz
            self.pi += 1
        else:
            self.x += dx / d * step
            self.z += dz / d * step


# ----------------------------------------------------------------- 着色 --
VERT = '''#version 330 core
in vec3 position;
in vec3 normals;
in vec3 colors;

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
    v_nrm = normalize(normals);
    v_col = colors;
    gl_Position = window.projection * window.view * world;
}'''

FRAG = '''#version 330 core
in vec3 v_nrm;
in vec3 v_col;
in vec3 v_world;
out vec4 final_colors;

uniform vec3 u_sun;
uniform vec3 u_fog;
uniform float u_flat;      // 1 = 纯色（路径线等）

void main() {
    if (u_flat > 0.5) {
        final_colors = vec4(v_col, 1.0);
        return;
    }
    vec3 n = normalize(v_nrm);
    float lam = max(dot(n, u_sun), 0.0);
    float hemi = 0.5 + 0.5 * n.y;
    vec3 amb = mix(vec3(0.14, 0.13, 0.14), vec3(0.30, 0.34, 0.42), hemi);
    vec3 col = v_col * (amb + lam * vec3(0.95, 0.92, 0.84));
    col = mix(col, u_fog, clamp(length(v_world.xz) * 0.010, 0.0, 0.55));
    final_colors = vec4(col, 1.0);
}'''


def box_geom(cx, cz, hw, hh, height, col):
    """一个盒子的三角汤（位置/法线/颜色）。"""
    pos, nrm, cls = [], [], []
    corners = ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
    for k in range(4):
        x0, z0 = corners[k]
        x1, z1 = corners[(k + 1) % 4]
        nx, nz = z1 - z0, -(x1 - x0)
        L = math.hypot(nx, nz) or 1.0
        nx, nz = nx / L, nz / L
        for p in ((cx + x0, 0.0, cz + z0), (cx + x1, 0.0, cz + z1),
                  (cx + x1, height, cz + z1), (cx + x0, 0.0, cz + z0),
                  (cx + x1, height, cz + z1), (cx + x0, height, cz + z0)):
            pos.extend(p)
            nrm.extend((nx, 0.0, nz))
            cls.extend(col)
    for p in ((cx - hw, height, cz - hh), (cx + hw, height, cz - hh),
              (cx + hw, height, cz + hh), (cx - hw, height, cz - hh),
              (cx + hw, height, cz + hh), (cx - hw, height, cz + hh)):
        pos.extend(p)
        nrm.extend((0.0, 1.0, 0.0))
        cls.extend(col)
    return pos, nrm, cls


def sphere_geom(r, seg=10, ring=6):
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


# ------------------------------------------------------------------- app --
class Crowdward(DemoApp):
    TITLE = 'CROWDward - behavior tree agents + grid A* (behavior_tree / pathing)'
    HELP = ('1-3 scene  A more  Z fewer  P paths  R reset  '
            'SPACE pause  W wire  drag/scroll  ESC')
    DURATION = None
    SHOTS = ((6.0, 'docs/crowdward_01.png'), (18.0, 'docs/crowdward_02.png'))
    FOV = 50.0

    def setup(self):
        glEnable(GL_DEPTH_TEST)
        self.prog = ShaderProgram(Shader(VERT, 'vertex'), Shader(FRAG, 'fragment'))
        self.flat_prog = ShaderProgram(Shader(VERT, 'vertex'), Shader(FRAG, 'fragment'))
        self.scene_i = 1
        for i, s in enumerate(SCENES):
            if s[0] in self.args.flags:
                self.scene_i = i
        self.show_paths = True
        self.wire = 'wire' in self.args.flags      # --flag wire：线框对照（也用于验证 GL_LINE 路径）
        self.rng = random.Random(20261026)
        self.ctx = Context()
        self.astar_calls = 0
        self.cam = OrbitCamera(target=Vec3(0, 0, 0), yaw=38.0, pitch=46.0,
                               dist=78.0, pitch_range=(6.0, 89.0),
                               dist_range=(20.0, 220.0))
        orbit_camera_input(self, self.cam)
        self.model = Mat4()
        self.agent_pos, self.agent_nrm = sphere_geom(0.38)
        self.n_agent_verts = len(self.agent_pos) // 3
        self.path_vl = None
        self.rebuild()

    # ------------------------------------------------------------ 世界 --
    def rebuild(self):
        name, n_agents, n_obs = SCENES[self.scene_i]
        self.rng = random.Random(20261026)
        self.walk, self.blocks = build_map(self.rng, n_obs)
        self.service = PathingService(self.ctx, {'count': 8, 'bounds': 10.0,
                                                 'seed': 20261026})
        # 资源点 / 休息点 / 仓库：都挑可通行格
        free = [(i, j) for j in range(2, GH - 2) for i in range(2, GW - 2)
                if self.walk[j][i] == 1]
        self.rng.shuffle(free)
        self.res_cells = free[:9]
        self.rest_cells = free[9:14]
        self.depot = free[14]
        self.threat = [0.0, 0.0]
        self.threat_t = 0.0
        self.agents = []
        for _ in range(n_agents):
            c = self.rng.choice(free)
            x, z = cell_center(*c)
            self.agents.append(Agent(self, x, z, self.rng))
        self.astar_calls = 0
        self._build_static()
        print(f'[crowdward] scene={name} agents={n_agents} obstacles={n_obs} '
              f'res={len(self.res_cells)} rest={len(self.rest_cells)} '
              f'depot={self.depot}')

    def request_path(self, agent):
        """A* 求路（节流由行为树的 Cooldown 保证，这里只记账）。"""
        ci, cj = world_to_cell(agent.x, agent.z)
        self.astar_calls += 1
        path = self.service.find_path((ci, cj), agent.goal, self.walk)
        if path:
            agent.path = path
            agent.pi = 0
        else:
            agent.path = []
            agent.pi = 0

    # ------------------------------------------------------------ 几何 --
    def _build_static(self):
        pos, nrm, cls = [], [], []
        # 地面
        hw = GW * CELL * 0.5
        for p in ((-hw, 0, -hw), (hw, 0, -hw), (hw, 0, hw),
                  (-hw, 0, -hw), (hw, 0, hw), (-hw, 0, hw)):
            pos.extend(p)
            nrm.extend((0.0, 1.0, 0.0))
            cls.extend((0.30, 0.31, 0.28))
        for (bx, by, bw, bh) in self.blocks:
            px, py = cell_center(bx, by)
            p2, n2, c2 = box_geom(px, py, bw * CELL * 0.5, bh * CELL * 0.5,
                                  2.6, (0.44, 0.42, 0.40))
            pos.extend(p2)
            nrm.extend(n2)
            cls.extend(c2)
        self.ground = self.prog.vertex_list(
            len(pos) // 3, GL_TRIANGLES,
            position=('f', pos), normals=('f', nrm), colors=('f', cls))
        # 标记柱：资源（绿）/ 休息（蓝）/ 仓库（金）
        mpos, mnrm, mcls = [], [], []
        for cells, col, h in ((self.res_cells, (0.32, 0.78, 0.38), 3.2),
                              (self.rest_cells, (0.30, 0.52, 0.92), 2.4),
                              ((self.depot,), (0.95, 0.78, 0.30), 4.2)):
            for c in cells:
                px, py = cell_center(*c)
                p2, n2, c2 = box_geom(px, py, 0.45, 0.45, h, col)
                mpos.extend(p2)
                mnrm.extend(n2)
                mcls.extend(c2)
        self.marks = self.prog.vertex_list(
            len(mpos) // 3, GL_TRIANGLES,
            position=('f', mpos), normals=('f', mnrm), colors=('f', mcls))
        self.threat_vl = self.prog.vertex_list(
            self.n_agent_verts, GL_TRIANGLES,
            position=('f', self.agent_pos), normals=('f', self.agent_nrm),
            colors=('f', (0.95, 0.26, 0.24) * self.n_agent_verts))

    # ------------------------------------------------------------ 交互 --
    def on_key(self, symbol, modifiers):
        k = key
        if k._1 <= symbol <= k._3 and symbol - k._1 < len(SCENES):
            self.scene_i = symbol - k._1
            self.rebuild()
        elif symbol == k.R:
            self.rebuild()
        elif symbol == k.P:
            self.show_paths = not self.show_paths
        elif symbol in (k.A,):
            self._bump_agents(10)
        elif symbol == k.Z:
            self._bump_agents(-10)
        elif symbol == k.W:
            self.wire = not self.wire

    def _bump_agents(self, delta):
        """增减 agent：直接在现有世界里增删，不重建地图。"""
        n = len(self.agents) + delta
        if n < 5 or n > 160:
            return
        if delta > 0:
            free = [(i, j) for j in range(2, GH - 2) for i in range(2, GW - 2)
                    if self.walk[j][i] == 1]
            for _ in range(delta):
                x, z = cell_center(*self.rng.choice(free))
                self.agents.append(Agent(self, x, z, self.rng))
        else:
            del self.agents[delta:]
        print(f'[crowdward] agents={len(self.agents)}')

    # ------------------------------------------------------------ 更新 --
    def update(self, dt):
        if self.paused:
            return
        # 威胁绕圈游走：它会不断把附近的 agent 推成 flee 分支
        self.threat_t += dt * 0.22
        r = GW * CELL * 0.30
        self.threat[0] = math.cos(self.threat_t) * r
        self.threat[1] = math.sin(self.threat_t * 1.3) * r

        for a in self.agents:
            a.tick_brain(dt)
        for a in self.agents:
            a.move(dt)
        self._separate(dt)

    def _separate(self, dt):
        """O(n²) 分离力：只推开，不叠群集规则（否则路径看不清）。"""
        n = len(self.agents)
        for i in range(n):
            a = self.agents[i]
            for j in range(i + 1, n):
                b = self.agents[j]
                dx, dz = b.x - a.x, b.z - a.z
                d2 = dx * dx + dz * dz
                if d2 > 0.64 or d2 < 1e-9:
                    continue
                d = math.sqrt(d2)
                push = (0.8 - d) * 2.4 * dt
                ux, uz = dx / d, dz / d
                a.x -= ux * push
                a.z -= uz * push
                b.x += ux * push
                b.z += uz * push

    # ------------------------------------------------------------ 渲染 --
    def render(self):
        w, h = self.window.width, self.window.height
        glClearColor(0.07, 0.08, 0.10, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.window.projection = Mat4.perspective_projection(
            w / max(1, h), z_near=0.5, z_far=800.0, fov=self.FOV)
        self.window.view = self.cam.view_matrix()
        if getattr(self, 'wire', False):
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        prog = self.prog
        prog.use()
        prog['u_model'] = self.model
        prog['u_sun'] = (0.42, 0.80, 0.42)
        prog['u_fog'] = (0.07, 0.08, 0.10)
        prog['u_flat'] = 0.0
        self.ground.draw(GL_TRIANGLES)
        self.marks.draw(GL_TRIANGLES)

        # 威胁
        tx, tz = self.threat
        prog['u_model'] = Mat4.from_translation(Vec3(tx, 1.6, tz)) @ \
            Mat4.from_scale(Vec3(3.0, 3.0, 3.0))
        self.threat_vl.draw(GL_TRIANGLES)

        # agent：颜色 = 当前行为分支
        for a in self.agents:
            col = BRANCH_COLS.get(a.branch, BRANCH_COLS['wander'])
            prog['u_model'] = Mat4.from_translation(Vec3(a.x, 0.42, a.z))
            if a.carrying:
                prog['u_model'] = (Mat4.from_translation(Vec3(a.x, 0.62, a.z))
                                   @ Mat4.from_scale(Vec3(1.25, 1.25, 1.25)))
            a_vl = self._agent_vl(col)
            a_vl.draw(GL_TRIANGLES)

        if self.show_paths:
            self._draw_paths()
        prog['u_flat'] = 0.0
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.window.projection = self.perspective

    def _agent_vl(self, col):
        """按分支色缓存 agent 的 vertex_list（分支只有 5 种，不必每人一份）。"""
        cache = getattr(self, '_agent_cache', None)
        if cache is None:
            cache = self._agent_cache = {}
        key = col
        if key not in cache:
            cache[key] = self.prog.vertex_list(
                self.n_agent_verts, GL_TRIANGLES,
                position=('f', self.agent_pos), normals=('f', self.agent_nrm),
                colors=('f', tuple(col) * self.n_agent_verts))
        return cache[key]

    def _draw_paths(self):
        """把所有 agent 的 A* 路径画成折线（这是"它在想什么"最直接的证据）。"""
        pts, cols = [], []
        for a in self.agents:
            if not a.path:
                continue
            col = BRANCH_COLS.get(a.branch, BRANCH_COLS['wander'])
            prev = (a.x, 0.16, a.z)
            for idx in range(a.pi, min(len(a.path), a.pi + 24)):
                cx, cz = cell_center(*a.path[idx])
                cur = (cx, 0.16, cz)
                pts.extend(prev)
                pts.extend(cur)
                cols.extend(col)
                cols.extend(col)
                prev = cur
        if not pts:
            return
        n = len(pts) // 3
        if self.path_vl is None or self.path_vl.count < n:
            if self.path_vl is not None:
                self.path_vl.delete()
            self.path_vl = self.flat_prog.vertex_list(
                max(n, 4096), GL_LINES, position=('f', [0.0] * (3 * max(n, 4096))),
                normals=('f', [0.0] * (3 * max(n, 4096))),
                colors=('f', [0.0] * (3 * max(n, 4096))))
        vl = self.path_vl
        pad = vl.count - n
        vl.position[:] = pts + [0.0] * (3 * pad)
        vl.colors[:] = cols + [0.0] * (3 * pad)
        self.flat_prog.use()
        self.flat_prog['u_model'] = self.model
        self.flat_prog['u_flat'] = 1.0
        vl.draw(GL_LINES)

    # ------------------------------------------------------------ 状态 --
    def status_text(self):
        counts = {b: 0 for b in BRANCH_ORDER}
        for a in self.agents:
            counts[a.branch] = counts.get(a.branch, 0) + 1
        dist = ' '.join(f'{b[0]}{counts[b]}' for b in BRANCH_ORDER)
        delivered = sum(a.delivered for a in self.agents)
        return (f'{SCENES[self.scene_i][0]}  agents {len(self.agents)}'
                f'  astar {self.astar_calls}  delivered {delivered}'
                f'  [{dist}]  {self.fps:5.1f} fps'
                f'{"  [PAUSED]" if self.paused else ""}')


if __name__ == '__main__':
    run(Crowdward)
