# -*- coding: utf-8 -*-
"""《坍界 / Collapseward》—— WFC 波函数坍缩可视化（logic/wfc）。

**为什么值得单独一个 demo**
----------------------------
``world3d.logic.wfc`` 是瓦片平铺模型的 WFC 求解器：观察（最低熵格坍缩）
→ 传播（AC-3 式收缩）→ 矛盾就整局重启。``solve()`` 只返回终局，
而**这个 demo 的全部看点恰恰是过程**——

* 用模块自己的 ``_observe`` / ``_propagate`` / ``_full_grid`` 逐步驱动求解循环，
  每一步把「波函数快照」压成 3D：未坍缩格是一根柱子，**候选越多越矮越冷**，
  传播收缩一路把柱子"焐高烧红"，最后啪地定格成地形；
* 用 ``check_adjacency`` 在终局做独立正确性校验（模块提供的不变量）。

两个刻意的实现点
----------------
1. **闪光不重建网格**：坍缩 / 传播事件写进顶点属性 —— ``colors.a`` 存事件
   时刻、``uvs.x`` 存幅度（没开混合，alpha 通道是免费的），片元里
   ``exp(-(u_time - a) * k) * amp`` 算出余晖。这样网格只在**波函数真的
   变了**的时候重建，闪光是零成本的 uniform 时间驱动。
2. **步进与帧率解耦**：步速是「格/秒」，update 里用累加器消化 dt 再重建，
   所以 ``--shot`` 的画面与运行帧率无关，可复现。

瓦片集：8 块「道路」瓦片（空地 / 直路×2 / 弯×4 / 十字），边插槽
``'g'/'r'``（草地 / 道路）。权重压低空地，让求解器织出路网而不是一片草原。

用法::

    python examples/collapseward.py                正常运行
    python examples/collapseward.py --smoke 3      冒烟
    python examples/collapseward.py --shot         抓剧照到 docs/collapseward_01/02.png
    python examples/collapseward.py --flag hold    关掉"解完自动换seed"

按键：SPACE 暂停   N 单步   S 步速   1/2/3 网格大小   R 换seed   ESC 退出
"""
import math

from pyglet.gl import (GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
                       GL_FILL, GL_FRONT_AND_BACK, GL_LINE, GL_TRIANGLES,
                       glClear, glClearColor, glDisable, glEnable,
                       glPolygonMode)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key

from world3d.logic.camera import OrbitCamera                       # noqa: E402
from world3d.logic.wfc import Tile, WFC, check_adjacency           # noqa: E402
from world3d.platform.app import (DemoApp, orbit_camera_input,     # noqa: E402
                                  run)

# ------------------------------------------------------------------ 着色器 --
VERTEX_SRC = '''#version 330 core
in vec3 position;
in vec3 normals;
in vec4 colors;
in vec2 uvs;
in float kind;

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;
uniform mat4 u_model;

out vec3 v_col;
out vec3 v_nrm;
out vec2 v_uv;
out float v_kind;
out float v_t;
out vec3 v_world;

void main() {
    vec4 world = u_model * vec4(position, 1.0);
    v_world = world.xyz;
    v_col = colors.rgb;
    v_t = colors.a;        // 事件时刻（-1 = 无事件）
    v_nrm = normals;
    v_uv = uvs;            // uv.x = 闪光幅度
    v_kind = kind;
    gl_Position = window.projection * window.view * world;
}'''

FRAGMENT_SRC = '''#version 330 core
in vec3 v_col;
in vec3 v_nrm;
in vec2 v_uv;
in float v_kind;
in float v_t;
in vec3 v_world;

uniform vec3 u_sun_dir;
uniform vec3 u_sun_col;
uniform vec3 u_amb_sky;
uniform vec3 u_amb_ground;
uniform vec3 u_fog_col;
uniform float u_fog_density;
uniform vec3 u_eye;
uniform float u_time;

out vec4 final_colors;

void main() {
    vec3 n = normalize(v_nrm);
    float ndl = max(dot(n, u_sun_dir), 0.0);
    vec3 hemi = mix(u_amb_ground, u_amb_sky, n.y * 0.5 + 0.5);
    vec3 lit = v_col * (hemi + u_sun_col * ndl);

    // 事件余晖：age<0 说明是"未来的事件"（时钟回绕/提前写入），不发光
    float age = u_time - v_t;
    float flash = (age >= 0.0 && age < 3.0) ? exp(-age * 3.2) * v_uv.x : 0.0;
    lit += vec3(1.0, 0.96, 0.82) * flash;

    float d = distance(v_world, u_eye);
    lit = mix(lit, u_fog_col, 1.0 - exp(-d * u_fog_density));
    final_colors = vec4(lit, 1.0);
}'''

# ------------------------------------------------------------------ 瓦片集 --
#: 边插槽 (N, E, S, W)：'g' 草地 / 'r' 道路。相容 = 字符串相等（模块规则）。
#: 权重压低空地，路网才织得出来 —— 全 'g' 永远是合法解，不压权重就退化成草原。
TILES = (
    Tile('empty', ('g', 'g', 'g', 'g')),
    Tile('ns',    ('r', 'g', 'r', 'g')),
    Tile('ew',    ('g', 'r', 'g', 'r')),
    Tile('ne',    ('r', 'r', 'g', 'g')),
    Tile('es',    ('g', 'r', 'r', 'g')),
    Tile('sw',    ('g', 'g', 'r', 'r')),
    Tile('wn',    ('r', 'g', 'g', 'r')),
    Tile('cross', ('r', 'r', 'r', 'r')),
)
WEIGHTS = {'empty': 1.1, 'ns': 1.8, 'ew': 1.8, 'ne': 1.5, 'es': 1.5,
           'sw': 1.5, 'wn': 1.5, 'cross': 0.8}
#: 名字 → 路面延伸方向集合（N=+z, E=+x, S=-z, W=-x，网格坐标）
ROAD_DIRS = {'empty': (), 'ns': (0, 2), 'ew': (1, 3), 'ne': (0, 1),
             'es': (1, 2), 'sw': (2, 3), 'wn': (3, 0), 'cross': (0, 1, 2, 3)}

GRASS = (0.23, 0.35, 0.17, 1.0)
ROAD = (0.30, 0.30, 0.33, 1.0)
BASE = (0.13, 0.15, 0.19, 1.0)
#: 熵热图：候选多 = 冷的深蓝灰（低柱子），候选少 = 灼热的琥珀（高柱子）
HEAT_COLD = (0.16, 0.22, 0.34)
HEAT_HOT = (1.00, 0.60, 0.16)

SIZES = ((20, 14), (28, 20), (40, 28))
SIZE_NAMES = ('S', 'M', 'L')
SPEEDS = (15.0, 45.0, 110.0, 260.0)
CELL = 2.0
SEED = 20261024


# ------------------------------------------------------------------- 网格 --
class MeshBuilder:
    """扁平顶点累加器（position/normals/uvs/colors/kind）。"""

    def __init__(self):
        self.pos, self.nrm, self.uv, self.col, self.kind = [], [], [], [], []

    def tri(self, a, b, c, n, col, uv, kind):
        for p, t in ((a, uv[0]), (b, uv[1]), (c, uv[2])):
            self.pos += (p[0], p[1], p[2])
            self.nrm += (n[0], n[1], n[2])
            self.uv += (t[0], t[1])
            self.col += (col[0], col[1], col[2], col[3])
            self.kind.append(kind)

    def quad(self, a, b, c, d, n, col, uv, kind):
        self.tri(a, b, c, n, col, (uv[0], uv[1], uv[2]), kind)
        self.tri(a, c, d, n, col, (uv[0], uv[2], uv[3]), kind)

    @property
    def vertex_count(self):
        return len(self.pos) // 3


def _box(mb, x0, z0, x1, z1, y0, y1, col, kind, t=-1.0, amp=0.0):
    """轴对齐盒（不含底面）。``t``/``amp`` 是事件时刻与闪光幅度。"""
    c4 = (col[0], col[1], col[2], t)
    uv = ((amp, 0.0),) * 4
    mb.quad((x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0),
            (0.0, 1.0, 0.0), c4, uv, kind)
    # 四个侧面
    sides = (((x0, z0), (x1, z0)), ((x1, z0), (x1, z1)),
             ((x1, z1), (x0, z1)), ((x0, z1), (x0, z0)))
    for (ax, az), (bx, bz) in sides:
        nx, nz = (az - bz), (bx - ax)
        L = math.hypot(nx, nz) or 1.0
        n = (nx / L, 0.0, nz / L)
        mb.quad((ax, y0, az), (bx, y0, bz), (bx, y1, bz), (ax, y1, az),
                n, c4, uv, kind)


def heat_color(n, total):
    """候选数 → 熵热色。n==total 是全波（最冷），n==1 即将坍缩（最热）。"""
    t = (n - 1) / max(1, total - 1)
    return tuple(HEAT_HOT[i] * (1 - t) + HEAT_COLD[i] * t for i in range(3))


# --------------------------------------------------------------------- App --
class Collapseward(DemoApp):
    TITLE = 'COLLAPSEward - WFC tile collapse (logic/wfc)'
    HELP = ('SPACE pause  N step  S speed  1/2/3 grid  R reseed  ESC')
    DURATION = None
    SHOTS = ((3.2, 'docs/collapseward_01.png'),
             (9.0, 'docs/collapseward_02.png'))
    FOV = 42.0

    # ------------------------------------------------------------- 生命周期 --
    def setup(self):
        glEnable(GL_DEPTH_TEST)
        self.program = ShaderProgram(Shader(VERTEX_SRC, 'vertex'),
                                     Shader(FRAGMENT_SRC, 'fragment'))
        self.size_i = 1
        self.speed_i = 2
        self.seed = SEED
        self.paused = False
        self.hold = 'hold' in self.args.flags
        self.acc = 0.0
        self.done_timer = 0.0
        self.events = {}               # (x, y) -> (时刻, 幅度)
        W, H = SIZES[self.size_i]
        span = max(W, H) * CELL
        self.cam = OrbitCamera(target=Vec3(0.0, 1.0, 0.0), yaw=38.0,
                               pitch=54.0, dist=span * 1.15,
                               pitch_range=(10.0, 88.0),
                               dist_range=(8.0, 260.0))
        orbit_camera_input(self, self.cam)
        self._new_problem()

    def _new_problem(self):
        W, H = SIZES[self.size_i]
        self.wfc = WFC(W, H, TILES, WEIGHTS, seed=self.seed)
        self.grid = self.wfc._full_grid()
        self.total = len(TILES)
        self.steps = 0
        self.restarts = 0
        self.solved = False
        self.events.clear()
        self.adj_ok = None
        self.rebuild()
        # 相机拉到能框住整个网格
        span = max(W, H) * CELL
        self.cam.dist = span * 1.15
        self.cam.target = Vec3(0.0, 1.0, 0.0)

    # ----------------------------------------------------------- 求解推进 --
    def _do_step(self):
        """观察一格 + 传播一轮；矛盾就整局重启（与模块 solve() 同一循环）。"""
        w = self.wfc
        cell = w._observe(self.grid)
        if cell is None:                       # 没有候选 >1 的格子 = 解完
            self._finish()
            return
        before = [[len(s) for s in row] for row in self.grid]
        ok = w._propagate(self.grid, [cell])
        self.steps += 1
        now = self.time
        self.events[cell] = (now, 1.0)         # 坍缩：白闪
        if not ok:                             # 矛盾 → 整局重启（重闪一下）
            self.restarts += 1
            self.events[cell] = (now, 2.6)
            self.grid = w._full_grid()
            return
        for y in range(self.wfc.H):
            row = self.grid[y]
            for x in range(self.wfc.W):
                if len(row[x]) != before[y][x] and (x, y) != cell:
                    self.events[(x, y)] = (now, 0.35)   # 传播波：弱闪

    def _finish(self):
        self.solved = True
        names = [[TILES[next(iter(self.grid[y][x]))].name
                  for x in range(self.wfc.W)] for y in range(self.wfc.H)]
        self.adj_ok = check_adjacency(names, TILES)
        print(f'[collapseward] seed={self.seed} grid={self.wfc.W}x{self.wfc.H} '
              f'steps={self.steps} restarts={self.restarts} '
              f'adjacency_ok={self.adj_ok}')

    # ------------------------------------------------------------- 重建 --
    def rebuild(self):
        """把波函数快照压成 3D：坍缩格 = 地形，未坍缩格 = 熵柱。"""
        mb = MeshBuilder()
        W, H = self.wfc.W, self.wfc.H
        ox, oz = -W * CELL / 2, -H * CELL / 2
        # 底板：比网格大一圈，把"漂浮感"压下去
        _box(mb, ox - 6, oz - 6, ox + W * CELL + 6, oz + H * CELL + 6,
             -0.8, -0.05, BASE, 0.0)
        now = self.time
        for y in range(H):
            for x in range(W):
                wave = self.grid[y][x]
                n = len(wave)
                cx0 = ox + x * CELL
                cz0 = oz + y * CELL
                cx1, cz1 = cx0 + CELL, cz0 + CELL
                ev = self.events.get((x, y))
                t, amp = ev if ev else (-1.0, 0.0)
                if n == 1:
                    name = TILES[next(iter(wave))].name
                    _box(mb, cx0, cz0, cx1, cz1, -0.05, 0.14, GRASS, 1.0,
                         t, amp)
                    # 路面：从格心伸向每条 'r' 边。
                    # 方向约定：0=+z(N) 1=+x(E) 2=-z(S) 3=-x(W)，与 (N,E,S,W) 对齐
                    h = 0.62
                    mx, mz = (cx0 + cx1) / 2, (cz0 + cz1) / 2
                    w0, w1 = cz0 + CELL * 0.28, cz0 + CELL * 0.72
                    for d in ROAD_DIRS[name]:
                        if d == 0:                       # N → +z
                            _box(mb, mx - CELL * 0.22, mz, mx + CELL * 0.22,
                                 cz1, 0.14, h, ROAD, 2.0, t, amp)
                        elif d == 1:                     # E → +x
                            _box(mb, mx, w0, cx1, w1, 0.14, h, ROAD, 2.0,
                                 t, amp)
                        elif d == 2:                     # S → -z
                            _box(mb, mx - CELL * 0.22, cz0, mx + CELL * 0.22,
                                 mz, 0.14, h, ROAD, 2.0, t, amp)
                        else:                            # W → -x
                            _box(mb, cx0, w0, mx, w1, 0.14, h, ROAD, 2.0,
                                 t, amp)
                else:
                    hh = 0.14 + 2.1 * (1.0 - (n - 1) / max(1, self.total - 1))
                    col = heat_color(n, self.total) + (1.0,)
                    _box(mb, cx0 + 0.10, cz0 + 0.10, cx1 - 0.10, cz1 - 0.10,
                         -0.05, hh, col, 3.0, t, amp)
        if hasattr(self, 'mesh') and self.mesh is not None:
            self.mesh.delete()
        self.mesh = self.program.vertex_list(
            mb.vertex_count, GL_TRIANGLES,
            position=('f', mb.pos), normals=('f', mb.nrm), uvs=('f', mb.uv),
            colors=('f', mb.col), kind=('f', mb.kind))
        self._verts = mb.vertex_count

    # ------------------------------------------------------------- 交互 --
    def on_key(self, symbol, modifiers):
        k = key
        if symbol == k.SPACE:
            self.paused = not self.paused
        elif symbol == k.N:
            if not self.solved:
                self._do_step()
                self.rebuild()
        elif symbol == k.S:
            self.speed_i = (self.speed_i + 1) % len(SPEEDS)
        elif symbol in (k._1, k._2, k._3):
            self.size_i = symbol - k._1
            self.seed = SEED
            self._new_problem()
        elif symbol == k.R:
            self.seed = (self.seed * 1103515245 + 12345) & 0x7FFFFFFF
            self._new_problem()

    def update(self, dt):
        if self.paused or self.solved:
            if self.solved and not self.hold:
                # 解完停 5 秒展示，然后自动换 seed 再来一局
                self.done_timer += dt
                if self.done_timer > 5.0:
                    self.done_timer = 0.0
                    self.seed = (self.seed * 1103515245 + 12345) & 0x7FFFFFFF
                    self._new_problem()
            return
        self.acc += dt * SPEEDS[self.speed_i]
        stepped = False
        while self.acc >= 1.0 and not self.solved:
            self.acc -= 1.0
            self._do_step()
            stepped = True
            if self.solved:
                break
        if stepped:
            self.rebuild()
        self.cam.yaw += dt * 0.35

    # ------------------------------------------------------------- 渲染 --
    def render(self):
        w, h = self.window.width, self.window.height
        aspect = w / max(1, h)
        fog = (0.10, 0.12, 0.17)
        glClearColor(fog[0], fog[1], fog[2], 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.window.projection = Mat4.perspective_projection(
            aspect, z_near=0.5, z_far=900.0, fov=self.FOV)
        self.window.view = self.cam.view_matrix()

        prog = self.program
        prog.use()
        sun = Vec3(0.45, 0.78, 0.30).normalize()
        prog['u_model'] = Mat4()
        prog['u_time'] = self.time
        prog['u_sun_dir'] = (sun.x, sun.y, sun.z)
        prog['u_sun_col'] = (1.05, 1.00, 0.90)
        prog['u_amb_sky'] = (0.42, 0.46, 0.56)
        prog['u_amb_ground'] = (0.16, 0.14, 0.12)
        prog['u_fog_col'] = fog
        prog['u_fog_density'] = 0.0038
        eye = self.cam.eye
        prog['u_eye'] = (eye.x, eye.y, eye.z)
        self.mesh.draw(GL_TRIANGLES)
        self.window.projection = self.perspective

    def status_text(self):
        W, H = self.wfc.W, self.wfc.H
        # 熵统计：还在波函数里的格子的候选数分布
        live = [len(self.grid[y][x])
                for y in range(H) for x in range(W) if len(self.grid[y][x]) > 1]
        ent = f'min={min(live) if live else 0} live={len(live)}'
        mark = 'SOLVED' if self.solved else 'running'
        return (f'{SIZE_NAMES[self.size_i]} {W}x{H}  steps {self.steps}  '
                f'restarts {self.restarts}  entropy {ent}  '
                f'x{SPEEDS[self.speed_i]:.0f}/s  {mark}'
                f'  {"[PAUSED]" if self.paused else ""}')


if __name__ == '__main__':
    run(Collapseward)
