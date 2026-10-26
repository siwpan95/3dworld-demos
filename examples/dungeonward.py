# -*- coding: utf-8 -*-
"""《窟境 / Dungeonward》—— 程序化地牢漫游（logic/dungeon 第一次有 demo 消费）。

**为什么值得单独一个 demo**
----------------------------
``world3d.logic.dungeon`` 有三套生成器：``generate_dungeon``（随机房间 + L 形走廊）、
``generate_dungeon_bsp``（BSP 切分后递归连通）、``generate_cave``（元胞自动机平滑
出的洞穴 + ``_label_regions`` 数连通块）。``tests/test_dungeon.py`` 6 项断言钉住的是
**连通性洪水填充验证 + 确定性 + 结构约束** —— 但这些一直是"看不见的正确"：
演示仓里零个 demo 用过它。

本 demo 把「连通性」这件事做成**看得见的**：HUD 实时显示 ``可达 / 总地板`` 两个
数（自己再跑一遍洪水填充，不信任生成器的自述），沿 BFS 最短路自动巡游的发光点
就是"这条路真的通"的实证；cave 模式下还会显示连通块数（元胞洞穴偶尔会切出孤岛，
``regions > 1`` 的时候画面上就会看到走不过去的角落）。

三套生成器 + 三种看法
----------------------
* ``rooms`` 房间 + L 形走廊：房间按归属染色，走廊是另一种色，结构最清晰。
* ``bsp``   BSP 切分：房间尺寸更规整，走廊严格走树边，适合看「递归连通」。
* ``cave``  元胞洞穴：有机形态，看的是平滑迭代与连通块。

实现要点
--------
* 墙体只生成**暴露面**（邻居是地板才生成侧面），1536 格的墙也不过几千个三角形；
  地板每格一块四边形，颜色来自房间归属 / 区域号。
* 巡游点是 BFS 最短路上的插值，不是硬编码路线 —— 生成器换种子它自己会跟着改。
* 相机两套：轨道俯视（默认）与第一人称（``F`` 切换，WASD 走、拖拽转头，
  带网格碰撞），第一人称下巡游点依然在跑，于是"跟着光点走"是最省事的观光方式。

用法::

    python examples/dungeonward.py                正常运行
    python examples/dungeonward.py --smoke 3      冒烟
    python examples/dungeonward.py --shot         抓剧照到 docs/dungeonward_01/02.png
    python examples/dungeonward.py --flag cave    起手切到元胞洞穴

按键：1-3 生成器   R 重播种   +/- 改尺寸   F 第一人称   C 跟随巡游点
      SPACE 暂停   W 线框   ESC 退出
"""
import math
from collections import deque

from pyglet.gl import (GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
                       GL_DEPTH_TEST, GL_FILL, GL_FRONT_AND_BACK, GL_LINE,
                       GL_TRIANGLES, glClear, glClearColor, glEnable,
                       glPolygonMode)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key

# ⚠️ 不要再加 sys.path 引导：引擎由 `pip install 3dworld` 提供
# （见 requirements.txt 与 docs/依赖与版本.md）。
from world3d.logic.camera import OrbitCamera                            # noqa: E402
from world3d.logic.dungeon import (generate_cave, generate_dungeon,     # noqa: E402
                                   generate_dungeon_bsp)
from world3d.platform.app import DemoApp, orbit_camera_input, run       # noqa: E402

CELL = 1.0
WALL_H = 2.4
EYE_H = 1.55
WALK_SPEED = 4.2

# 房间归属 → 地板色（灰蓝 → 暖沙的循环，保证相邻房间色差不撞车）
ROOM_COLS = ((0.30, 0.38, 0.50), (0.46, 0.40, 0.28), (0.28, 0.44, 0.40),
             (0.48, 0.32, 0.38), (0.36, 0.36, 0.30), (0.26, 0.34, 0.46))
CORRIDOR_COL = (0.22, 0.24, 0.28)
WALL_COL = (0.34, 0.33, 0.31)
WALL_TOP_COL = (0.46, 0.45, 0.42)

GENERATORS = (('rooms', 'random rooms + L corridors'),
              ('bsp', 'BSP split, recursive connect'),
              ('cave', 'cellular cave + region labels'))


# ------------------------------------------------------------- 几何构建 --
def _quad(out, p0, p1, p2, p3, nrm, col):
    """一个四边形 = 两个三角形（p0-p1-p2 / p0-p2-p3），顺带写颜色。"""
    pos, nms, cls = out
    for p in (p0, p1, p2, p0, p2, p3):
        pos.extend(p)
        nms.extend(nrm)
        cls.extend(col)


def build_world_mesh(grid, room_of):
    """地牢网格 → 一次 draw 的三角汤。

    ``room_of`` 是 (x, y) → 房间号 的映射（None = 走廊）；没有房间概念的
    cave 模式整张传 None，地板统一按区域号着色。
    """
    pos, nms, cls = [], [], []
    out = (pos, nms, cls)
    h = len(grid)
    w = len(grid[0])

    def cell_origin(x, y):
        return ((x - w * 0.5) * CELL, (y - h * 0.5) * CELL)

    def floor_of(x, y):
        return 0 <= y < h and 0 <= x < w and grid[y][x] == 1

    for y in range(h):
        for x in range(w):
            ox, oz = cell_origin(x, y)
            if grid[y][x] == 1:
                rid = room_of.get((x, y)) if room_of else None
                if rid is None:
                    col = CORRIDOR_COL
                else:
                    col = ROOM_COLS[rid % len(ROOM_COLS)]
                _quad(out,
                      (ox - 0.5 * CELL, 0.0, oz - 0.5 * CELL),
                      (ox + 0.5 * CELL, 0.0, oz - 0.5 * CELL),
                      (ox + 0.5 * CELL, 0.0, oz + 0.5 * CELL),
                      (ox - 0.5 * CELL, 0.0, oz + 0.5 * CELL),
                      (0.0, 1.0, 0.0), col)
                continue
            # --- 墙：顶面永远画，侧面只画"朝向地板"的那几面 ---
            _quad(out,
                  (ox - 0.5 * CELL, WALL_H, oz - 0.5 * CELL),
                  (ox + 0.5 * CELL, WALL_H, oz - 0.5 * CELL),
                  (ox + 0.5 * CELL, WALL_H, oz + 0.5 * CELL),
                  (ox - 0.5 * CELL, WALL_H, oz + 0.5 * CELL),
                  (0.0, 1.0, 0.0), WALL_TOP_COL)
            sides = ((-1, 0), (1, 0), (0, -1), (0, 1))
            for dx, dz in sides:
                if not floor_of(x + dx, y + dz):
                    continue
                if dx:
                    sx = ox + dx * 0.5 * CELL
                    a = (sx, 0.0, oz - 0.5 * CELL)
                    b = (sx, 0.0, oz + 0.5 * CELL)
                    c = (sx, WALL_H, oz + 0.5 * CELL)
                    d = (sx, WALL_H, oz - 0.5 * CELL)
                else:
                    sz = oz + dz * 0.5 * CELL
                    a = (ox - 0.5 * CELL, 0.0, sz)
                    b = (ox + 0.5 * CELL, 0.0, sz)
                    c = (ox + 0.5 * CELL, WALL_H, sz)
                    d = (ox - 0.5 * CELL, WALL_H, sz)
                if (dx, dz) in ((-1, 0), (0, 1)):
                    a, b, c, d = b, a, d, c
                _quad(out, a, b, c, d, (float(dx), 0.0, float(dz)), WALL_COL)
    return pos, nms, cls


# --------------------------------------------------------------- 图算法 --
def flood_reachable(grid):
    """从第一个地板格洪水填充：返回 (可达数, 总地板数)。"""
    h, w = len(grid), len(grid[0])
    start = None
    total = 0
    for y in range(h):
        for x in range(w):
            if grid[y][x] == 1:
                total += 1
                if start is None:
                    start = (x, y)
    if start is None:
        return 0, 0
    seen = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n = (x + dx, y + dz)
            if n in seen or not (0 <= n[0] < w and 0 <= n[1] < h):
                continue
            if grid[n[1]][n[0]] == 1:
                seen.add(n)
                q.append(n)
    return len(seen), total


def longest_path(grid, start):
    """两次 BFS 求最短路的端点对，再回溯出路径（沿路巡游用）。"""
    h, w = len(grid), len(grid[0])

    def bfs(src):
        prev = {src: None}
        q = deque([src])
        far = src
        while q:
            x, y = q.popleft()
            if (x - src[0]) ** 2 + (y - src[1]) ** 2 > \
               (far[0] - src[0]) ** 2 + (far[1] - src[1]) ** 2:
                far = (x, y)
            for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                n = (x + dx, y + dz)
                if n in prev or not (0 <= n[0] < w and 0 <= n[1] < h):
                    continue
                if grid[n[1]][n[0]] == 1:
                    prev[n] = (x, y)
                    q.append(n)
        return prev, far

    _, a = bfs(start)
    prev, b = bfs(a)
    path = []
    cur = b
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path


# ----------------------------------------------------------------- 着色 --
VERTEX_SRC = '''#version 330 core
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
    v_nrm = normals;
    v_col = colors;
    gl_Position = window.projection * window.view * world;
}'''

FRAGMENT_SRC = '''#version 330 core
in vec3 v_nrm;
in vec3 v_col;
in vec3 v_world;
out vec4 final_colors;

uniform vec3 u_sun;
uniform vec3 u_fog;
uniform vec3 u_torch;     // 巡游点（火把）位置
uniform float u_torch_on; // 0 = 关掉火把（俯视时太亮）

void main() {
    vec3 n = normalize(v_nrm);
    float lam = max(dot(n, u_sun), 0.0);
    float hemi = 0.5 + 0.5 * n.y;
    vec3 amb = mix(vec3(0.10, 0.09, 0.10), vec3(0.26, 0.30, 0.38), hemi);
    vec3 col = v_col * (amb + lam * vec3(0.95, 0.90, 0.80));

    // 火把：平方反比衰减的暖光，让"通路"自己亮起来
    vec3 d = u_torch - v_world;
    float dist = length(d);
    float atten = u_torch_on / (1.0 + 0.35 * dist * dist);
    col += v_col * atten * max(dot(n, normalize(d)), 0.0) * vec3(1.0, 0.72, 0.40);

    float fog = clamp(length(v_world.xz) * 0.012, 0.0, 0.55);
    col = mix(col, u_fog, fog);
    final_colors = vec4(col, 1.0);
}'''


# ------------------------------------------------------------------- app --
class Dungeonward(DemoApp):
    TITLE = 'DUNGEONward - procedural dungeon walk (logic/dungeon)'
    HELP = ('1-3 generator  R reseed  +/- size  F first-person  C follow  '
            'WASD walk  SPACE pause  W wire  drag/scroll  ESC')
    DURATION = None
    SHOTS = ((5.0, 'docs/dungeonward_01.png'), (16.0, 'docs/dungeonward_02.png'))
    FOV = 60.0

    def setup(self):
        glEnable(GL_DEPTH_TEST)
        self.program = ShaderProgram(Shader(VERTEX_SRC, 'vertex'),
                                     Shader(FRAGMENT_SRC, 'fragment'))
        self.gen_i = 0
        for i, g in enumerate(GENERATORS):
            if g[0] in self.args.flags:
                self.gen_i = i
        self.seed = 20261026
        self.size_i = 1                      # 0/1/2 → 40x28 / 56x40 / 72x52
        self.wire = False
        self.first_person = False
        self.follow = True
        self.walk_t = 0.0
        self.fp_pos = [0.0, 0.0]
        self.fp_yaw = 0.0
        self.cam = OrbitCamera(target=Vec3(0, 0, 0), yaw=38.0, pitch=52.0,
                               dist=52.0, pitch_range=(4.0, 89.0),
                               dist_range=(6.0, 220.0))
        orbit_camera_input(self, self.cam)
        self.model = Mat4()
        # WASD 需要"按住"状态而不是按键事件：pyglet 的 KeyStateHandler 直接
        # 挂在 window 上，读 self.key_state[key.W] 即为当前是否按住。
        self.key_state = key.KeyStateHandler()
        self.window.push_handlers(self.key_state)
        self.rebuild()

    # ------------------------------------------------------------ 重建 --
    def rebuild(self):
        w, h = ((40, 28), (56, 40), (72, 52))[self.size_i]
        name = GENERATORS[self.gen_i][0]
        if name == 'rooms':
            res = generate_dungeon(width=w, height=h, rooms=max(4, w // 7),
                                   seed=self.seed)
            grid = res['grid']
            room_of = {}
            for rid, r in enumerate(res.get('rooms', [])):
                for y in range(r['y'], r['y'] + r['h']):
                    for x in range(r['x'], r['x'] + r['w']):
                        room_of[(x, y)] = rid
            self.n_rooms = len(res.get('rooms', []))
        elif name == 'bsp':
            res = generate_dungeon_bsp(width=w, height=h, seed=self.seed)
            grid = res['grid']
            room_of = {}
            for rid, r in enumerate(res.get('rooms', [])):
                for y in range(r['y'], r['y'] + r['h']):
                    for x in range(r['x'], r['x'] + r['w']):
                        room_of[(x, y)] = rid
            self.n_rooms = len(res.get('rooms', []))
        else:
            res = generate_cave(width=w, height=h, seed=self.seed)
            grid = res['grid']
            room_of = None
            self.n_rooms = 0
        self.grid = grid
        self.regions = res.get('regions')
        self.reach, self.floor_n = flood_reachable(grid)
        self.path = longest_path(grid, self._first_floor(grid))
        pos, nms, cls = build_world_mesh(grid, room_of)
        self.mesh = self.program.vertex_list(
            len(pos) // 3, GL_TRIANGLES,
            position=('f', pos), normals=('f', nms), colors=('f', cls))
        self.n_verts = len(pos) // 3
        self.walk_t = 0.0
        if self.path:
            self._place_fp(self.path[0])
        self.cam.dist = max(28.0, w * 0.95)
        print(f'[dungeonward] gen={name} size={w}x{h} seed={self.seed} '
              f'rooms={self.n_rooms} floor={self.floor_n} '
              f'reach={self.reach} path={len(self.path)} verts={self.n_verts}')

    @staticmethod
    def _first_floor(grid):
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                if grid[y][x] == 1:
                    return (x, y)
        return (0, 0)

    def _place_fp(self, cell):
        h, w = len(self.grid), len(self.grid[0])
        x, y = cell
        self.fp_pos = [(x - w * 0.5) * CELL, (y - h * 0.5) * CELL]

    # ------------------------------------------------------------ 交互 --
    def on_key(self, symbol, modifiers):
        k = key
        if k._1 <= symbol <= k._3 and symbol - k._1 < len(GENERATORS):
            self.gen_i = symbol - k._1
            self.rebuild()
        elif symbol == k.R:
            self.seed = (self.seed * 1103515245 + 12345) & 0x7FFFFFFF
            self.rebuild()
        elif symbol in (k.EQUAL, k.PLUS, k.NUM_ADD):
            self.size_i = min(2, self.size_i + 1)
            self.rebuild()
        elif symbol in (k.MINUS, k.NUM_SUBTRACT):
            self.size_i = max(0, self.size_i - 1)
            self.rebuild()
        elif symbol == k.F:
            self.first_person = not self.first_person
            if self.first_person and self.path:
                self._place_fp(self.path[0])
        elif symbol == k.C:
            self.follow = not self.follow
        elif symbol == k.W and self.first_person is False:
            self.wire = not self.wire

    # ------------------------------------------------------------ 更新 --
    def update(self, dt):
        if self.paused or not self.path:
            return
        h, w = len(self.grid), len(self.grid[0])
        self.walk_t = (self.walk_t + dt * (WALK_SPEED * 0.55)) % max(1, len(self.path) - 1)
        i = int(self.walk_t)
        f = self.walk_t - i
        a = self.path[i]
        b = self.path[min(len(self.path) - 1, i + 1)]
        self.torch = ((a[0] + (b[0] - a[0]) * f - w * 0.5) * CELL, 0.9,
                      (a[1] + (b[1] - a[1]) * f - h * 0.5) * CELL)
        if self.first_person and self.follow:
            self.fp_pos = [self.torch[0], self.torch[2]]
            dx = (b[0] - a[0]) * CELL
            dz = (b[1] - a[1]) * CELL
            if dx or dz:
                self.fp_yaw = math.degrees(math.atan2(dx, dz))
        if self.first_person and not self.follow:
            self._walk(dt, w, h)
        if not self.first_person:
            self.cam.yaw += dt * 0.5

    def _walk(self, dt, w, h):
        """第一人称自由行走：位移前先按轴分别试，撞墙只损失该轴分量（贴墙滑行）。"""
        ks = self.key_state
        yaw = math.radians(self.fp_yaw)
        fx, fz = math.sin(yaw), math.cos(yaw)
        rx, rz = math.cos(yaw), -math.sin(yaw)
        mx = mz = 0.0
        if ks[key.W]:
            mx += fx
            mz += fz
        if ks[key.S]:
            mx -= fx
            mz -= fz
        if ks[key.D]:
            mx += rx
            mz += rz
        if ks[key.A]:
            mx -= rx
            mz -= rz
        L = math.hypot(mx, mz)
        if L < 1e-6:
            return
        step = WALK_SPEED * dt / L
        for dx, dz in ((mx * step, 0.0), (0.0, mz * step)):
            if dx == 0.0 and dz == 0.0:
                continue
            nx, nz = self.fp_pos[0] + dx, self.fp_pos[1] + dz
            gx = int(nx / CELL + w * 0.5)
            gy = int(nz / CELL + h * 0.5)
            if 0 <= gy < h and 0 <= gx < w and self.grid[gy][gx] == 1:
                self.fp_pos[0] = nx
                self.fp_pos[1] = nz

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.first_person:
            self.fp_yaw += dx * 0.25
            return
        return super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)

    # ------------------------------------------------------------ 渲染 --
    def render(self):
        w, h = self.window.width, self.window.height
        aspect = w / max(1, h)
        glClearColor(0.05, 0.05, 0.07, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.window.projection = Mat4.perspective_projection(
            aspect, z_near=0.25, z_far=900.0, fov=self.FOV)

        if self.first_person:
            yaw = math.radians(self.fp_yaw)
            eye = Vec3(self.fp_pos[0], EYE_H, self.fp_pos[1])
            tgt = Vec3(eye.x + math.sin(yaw), EYE_H, eye.z + math.cos(yaw))
            self.window.view = Mat4.look_at(eye, tgt, Vec3(0.0, 1.0, 0.0))
            torch = (eye.x, EYE_H * 0.6, eye.z) if not self.follow else self.torch
        else:
            self.window.view = self.cam.view_matrix()
            torch = self.torch

        prog = self.program
        prog.use()
        prog['u_model'] = self.model
        prog['u_sun'] = (0.40, 0.82, 0.40)
        prog['u_fog'] = (0.05, 0.05, 0.07)
        prog['u_torch'] = torch
        prog['u_torch_on'] = 2.6 if self.first_person else 1.5
        if self.wire:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        self.mesh.draw(GL_TRIANGLES)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.window.projection = self.perspective

    # ------------------------------------------------------------ 状态 --
    def status_text(self):
        reg = f'  regions {self.regions}' if self.regions is not None else ''
        ok = 'OK' if self.reach == self.floor_n else 'SPLIT'
        return (f'{GENERATORS[self.gen_i][0]}  {len(self.grid[0])}x{len(self.grid)}'
                f'  seed {self.seed}  rooms {self.n_rooms}  floor {self.floor_n}'
                f'  reach {self.reach} [{ok}]{reg}  path {len(self.path)}'
                f'  {"FP" if self.first_person else "ORBIT"}'
                f'  {self.fps:5.1f} fps'
                f'{"  [PAUSED]" if self.paused else ""}')


if __name__ == '__main__':
    run(Dungeonward)
