# -*- coding: utf-8 -*-
"""《林氏 / Lindenwood》—— L-System 生成 3D 分形植物（风摆 + 四季）。

**为什么值得单独一个 demo**
----------------------------
``world3d.logic.lsystem`` 是一块"算法核"：字符串重写（``expand``，支持
确定性产生式与带权随机产生式）+ 2D 龟图 + **3D 龟图**（``Turtle3D``，
用航向 H / 上向量 U 与偏航俯仰翻滚维护姿态）。``tests/test_lsystem.py``
把重写与龟图都测了，但演示仓里**零个 demo 用过它**。

本 demo 做的是"把线段变成植物"这一段 —— 这一步核里没有，也不该有：

* ``expand`` 给出一串指令，``Turtle3D.run`` 给出一堆**线段**（没有父子关系）；
* 用**端点重合**还原出树：每段的起点等于父段的终点，据此建父子表、算子树叶片数；
* 半径按 **da Vinci 定律** ``r ∝ (n_leaves)^(1/2.4)`` 从叶片数反推 ——
  于是主干粗、末梢细，而不是靠"深度乘一个衰减系数"糊出来；
* 每根**枝链**（连续的独子链）走 ``engine.geometry.tube`` 扫成变径管体。

两个刻意的点
------------
1. **风在顶点着色器里算**：位移 ∝ ``pow(y/H, 1.6)`` —— 根部不动、梢头摆得远，
   这是"树在摇"而不是"整棵树在平移"的关键。
2. **四季只改一个 uniform**：叶色与落叶程度全在片元里由 ``u_leaf`` / ``u_drop``
   决定，换季不重建几何。

用法::

    python examples/lindenwood.py                正常运行
    python examples/lindenwood.py --smoke 3      冒烟
    python examples/lindenwood.py --shot         抓剧照到 docs/lindenwood_01/02.png
    python examples/lindenwood.py --flag autumn  起手就是深秋
    python examples/lindenwood.py --flag still   关掉风（对照用）

按键：1-5 换物种   R 换种子（随机产生式）   +/- 迭代层数   S 四季开关
      W 线框   SPACE 暂停   鼠标拖动/滚轮 看视角   ESC 退出
"""
import math
import random

from pyglet.gl import (GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
                       GL_FILL, GL_FRONT_AND_BACK, GL_LINE, GL_TRIANGLES, glClear,
                       glClearColor, glDisable, glEnable, glPolygonMode)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key

# ⚠️ 不要再加 sys.path 引导：引擎由 `pip install 3dworld` 提供
# （见 requirements.txt 与 docs/依赖与版本.md）。
from world3d.engine.geometry import tube                          # noqa: E402
from world3d.logic.camera import OrbitCamera                      # noqa: E402
from world3d.logic.lsystem import Turtle3D, expand                # noqa: E402
from world3d.platform.app import (DemoApp, orbit_camera_input,   # noqa: E402
                                  run)

VERTEX_SRC = '''#version 330 core
in vec3 position;
in vec3 normals;
in vec4 colors;
in vec2 uvs;
in float kind;      // 0 = 木质, 1 = 叶片
in float wind;      // 0 = 根部, 1 = 梢顶

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;
uniform mat4 u_model;
uniform float u_time;
uniform float u_amp;
uniform vec2 u_gust;

out vec3 v_col;
out float v_rnd;
out vec3 v_nrm;
out vec2 v_uv;
out float v_kind;
out float v_wind;
out float v_viewd;

void main() {
    vec3 p = position;
    // 风：位移按高度的 1.6 次幂放大 —— 根不动、梢头摆幅最大。
    // 两段不同频率的正弦叠出"来回摆 + 高频抖"，单频看起来像机械臂。
    float b = pow(clamp(wind, 0.0, 1.0), 1.6);
    float s1 = sin(u_time * 1.45 + p.y * 0.22 + u_gust.x);
    float s2 = sin(u_time * 3.10 + p.y * 0.55 + u_gust.y);
    p.x += (s1 * 0.75 + s2 * 0.25) * b * u_amp;
    p.z += (cos(u_time * 1.15 + p.y * 0.19 + u_gust.y) * 0.6
          + sin(u_time * 2.7 + p.y * 0.5) * 0.2) * b * u_amp;

    vec4 world = u_model * vec4(p, 1.0);
    vec3 n = normals;
    // 叶片是双面薄片：位移把法线也带走一点，否则摆起来高光不动
    v_col = colors.rgb;
    v_rnd = colors.a;          // 每片叶子一个恒定随机数（存在 color.a 里）
    v_nrm = n;
    v_uv = uvs;
    v_kind = kind;
    v_wind = wind;
    vec4 vp = window.view * world;
    v_viewd = -vp.z;
    gl_Position = window.projection * vp;
}'''

FRAGMENT_SRC = '''#version 330 core
in vec3 v_col;
in float v_rnd;
in vec3 v_nrm;
in vec2 v_uv;
in float v_kind;
in float v_wind;
in float v_viewd;

uniform vec3 u_sun_dir;
uniform vec3 u_sun_col;
uniform vec3 u_amb_sky;
uniform vec3 u_amb_ground;
uniform vec3 u_leaf;       // 当季叶色
uniform float u_drop;      // 落叶程度 0..1
uniform vec3 u_fog_col;
uniform vec2 u_fog_range;  // (near, far)

out vec4 final_colors;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// 线性雾。只混到 0.55，远处留一点形体，别把树彻底吃掉
vec3 apply_fog(vec3 c) {
    float f = clamp((v_viewd - u_fog_range.x)
                    / max(u_fog_range.y - u_fog_range.x, 0.001), 0.0, 1.0);
    return mix(c, u_fog_col, f * 0.55);
}

void main() {
    vec3 n = normalize(v_nrm);
    vec3 base = v_col;

    if (v_kind > 0.5) {
        // --- 叶片：整片按"叶片哈希"决定落没落 ---
        // 注意随机数必须**逐叶片恒定**（走 varying 的 color.a），
        // 不能用插值出来的 uv 去哈希 —— 那样同一片叶子里三个顶点得到
        // 三个不同结果，discard 会把叶片啃成锯齿碎块。
        float r = v_rnd;
        if (r < u_drop * 1.06 - 0.06) {
            discard;
        }
        vec2 q = v_uv - 0.5;
        float rho = clamp(length(q * vec2(1.0, 1.15)) * 2.0, 0.0, 1.0);
        base = u_leaf * (0.78 + 0.44 * r);
        base *= 1.0 - 0.30 * rho * rho;                       // 叶缘压暗
        base *= 1.0 - 0.12 * smoothstep(0.02, 0.16, abs(q.y)); // 主脉留亮线
        // 薄叶透光：背光面加一点随叶色走的透射
        float back = max(0.0, -dot(n, u_sun_dir));
        float ndl = abs(dot(n, u_sun_dir));      // 双面：两面都受光
        vec3 hemi = mix(u_amb_ground, u_amb_sky, 0.5 + 0.5 * abs(n.y));
        vec3 c = base * (hemi + u_sun_col * ndl)
               + base * u_sun_col * back * 0.55;
        final_colors = vec4(apply_fog(c), 1.0);
        return;
    }

    // --- 木质：正常兰伯特 ---
    float ndl = max(dot(n, u_sun_dir), 0.0);
    vec3 hemi = mix(u_amb_ground, u_amb_sky, n.y * 0.5 + 0.5);
    vec3 lit = base * (hemi + u_sun_col * ndl);

    // 梢头稍微提亮一点（树皮在细枝上更浅）
    lit *= 0.86 + 0.30 * v_wind;

    final_colors = vec4(apply_fog(lit), 1.0);
}'''

GROUND_VERT = '''#version 330 core
in vec3 position;
in vec4 colors;
out vec2 v_uv;
out vec3 v_wpos;
uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;
void main() {
    v_uv = position.xz * 0.08;
    v_wpos = position;
    gl_Position = window.projection * window.view * vec4(position, 1.0);
}'''

GROUND_FRAG = '''#version 330 core
in vec2 v_uv;
in vec3 v_wpos;
out vec4 final_colors;
uniform vec3 u_sun_col;
uniform vec3 u_grass;
uniform vec3 u_fog_col;
uniform float u_height;     // 当前树高，用来定接触阴影半径

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1, 0)), f.x),
               mix(hash21(i + vec2(0, 1)), hash21(i + vec2(1, 1)), f.x), f.y);
}

void main() {
    float n = vnoise(v_uv * 1.7) * 0.6 + vnoise(v_uv * 6.0) * 0.4;
    vec3 c = u_grass * (0.72 + 0.55 * n);
    float r = length(v_wpos.xz);
    // 以树为圆心的暗角：把视线收拢到树根
    c *= 1.0 - 0.38 * smoothstep(8.0, 46.0, r);
    // 接触阴影：树根附近压暗，树才像"站在地上"而不是浮着
    float sr = max(0.8, u_height * 0.42);
    c *= 1.0 - 0.62 * exp(-(r * r) / (sr * sr));
    c *= 0.55 + 0.45 * u_sun_col;
    // 远处淡入雾色 —— 不然 1200 见方的地面会在天边切出一条硬边
    float f = smoothstep(320.0, 1250.0, r);
    c = mix(c, u_fog_col, f);
    final_colors = vec4(c, 1.0);
}'''

# ------------------------------------------------------------------- 天空 --
#: 一个铺满屏幕的三角形。片元着色器用相机基向量把屏幕坐标反推成视线方向，
#: 所以天空渐变是**跟随视角**的 —— 抬头天更蓝、低头地平线更亮，
#: 而不是贴在屏幕上的死板横向色带。
SKY_VERT = '''#version 330 core
in vec3 position;
out vec2 v_ndc;
void main() {
    v_ndc = position.xy;
    gl_Position = vec4(position.xy, 1.0, 1.0);
}'''

SKY_FRAG = '''#version 330 core
in vec2 v_ndc;
out vec4 final_colors;
uniform vec3 u_fwd;
uniform vec3 u_right;
uniform vec3 u_up;
uniform vec2 u_tan;        // (tan(fov/2)*aspect, tan(fov/2))
uniform vec3 u_top;
uniform vec3 u_hor;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_col;

void main() {
    vec3 d = normalize(u_fwd + u_right * v_ndc.x * u_tan.x
                              + u_up * v_ndc.y * u_tan.y);
    float t = pow(clamp(d.y, 0.0, 1.0), 0.55);
    vec3 c = mix(u_hor, u_top, t);
    float sd = max(dot(d, u_sun_dir), 0.0);
    c += u_sun_col * (pow(sd, 120.0) * 0.60 + pow(sd, 7.0) * 0.05);
    final_colors = vec4(c, 1.0);
}'''

# ------------------------------------------------------------------ 物种表 --
#: 每个预设：公理 / 产生式 / 迭代 / 转角（度）/ 主干半径。
#: 语法限 Turtle3D 支持的字符：F f + - & ^ \\ / [ ]
PRESETS = (
    dict(name='橡树 oak', axiom='A',
         rules={'A': '[&FA]/////[&FA]///////[&FA]', 'F': 'FF'},
         iters=5, angle=26.0, r0=0.42),
    dict(name='灌木 shrub', axiom='A',
         rules={'A': 'F[+A][-A][&A]', 'F': 'FF'},
         iters=5, angle=33.0, r0=0.30),
    dict(name='垂枝 elm', axiom='FFFA',
         rules={'A': '[&FA]/////[&FA]///////[&FA]'},
         iters=6, angle=24.0, r0=0.30),
    dict(name='幼树 sapling', axiom='X',
         rules={'X': 'F[+X][-X]FX', 'F': 'FF'},
         iters=5, angle=24.0, r0=0.13),
    dict(name='矮树 bush', axiom='A',
         rules={'A': 'F[&A]/[\\A]^A', 'F': 'F'},
         iters=5, angle=19.0, r0=0.28),
)

#: 四季叶色（0 = 初夏浓绿, 1 = 深冬光杆）
SEASON_COLS = (
    (0.34, 0.62, 0.20),      # 春：嫩绿
    (0.22, 0.50, 0.16),      # 夏：浓绿
    (0.72, 0.55, 0.14),      # 秋初：转黄
    (0.78, 0.34, 0.10),      # 深秋：橙红
    (0.34, 0.24, 0.15),      # 冬：枯褐
)
SEASON_LEN = 26.0


# ------------------------------------------------------------------- 树重建 --
def _key(p):
    return (round(p[0], 5), round(p[1], 5), round(p[2], 5))


def build_tree(segments):
    """由线段表还原父子关系。

    ``Turtle3D.run`` 只给线段，不给结构；但龟图是**深度优先**走的，
    所以"某段的起点 == 另一段的终点"且"后者下标更小"就唯一确定了父段。
    返回 ``(parent, children, leaves)``：children 为邻接表，leaves 为子树叶片数。
    """
    n = len(segments)
    ends = {}
    for i, (_, b) in enumerate(segments):
        ends[_key(b)] = i
    parent = [-1] * n
    children = [[] for _ in range(n)]
    for i, (a, _) in enumerate(segments):
        k = _key(a)
        if k in ends:
            j = ends[k]
            if j < i and parent[i] == -1:
                parent[i] = j
                children[j].append(i)
    leaves = [1] * n
    for i in range(n - 1, -1, -1):          # 逆序 = 后序遍历（父一定排在子前）
        tot = 0
        for c in children[i]:
            tot += leaves[c]
        leaves[i] = tot if tot else 1
    return parent, children, leaves


def branch_chains(parent, children):
    """把树切成枝链：起点是根或分叉点，沿着独子一路走到头。"""
    chains = []
    for i in range(len(parent)):
        if parent[i] != -1 and len(children[parent[i]]) == 1:
            continue                        # 不是链首，跳过
        chain = [i]
        cur = i
        while len(children[cur]) == 1:
            cur = children[cur][0]
            chain.append(cur)
        chains.append(chain)
    return chains


class Builder:
    """扁平顶点累加器（position/normals/uvs/colors/kind/wind）。"""

    def __init__(self):
        self.pos, self.nrm, self.uv, self.col = [], [], [], []
        self.kind, self.wind = [], []
        self.ymin, self.ymax = 1e9, -1e9

    def add_mesh(self, md, color, kind, ymin, ymax):
        """把 ``MeshData`` 并入扁平累加器。

        **必须展开 indices**：``geometry.tube`` 返回的是带索引的网格（顶点去重），
        直接拿 ``positions`` 当三角汤喂给 ``vertex_list`` 会按"每 3 个顶点一个
        三角形"重新配对，画出来是一团乱麻 —— 这是索引网格最常见的踩坑点。
        """
        span = max(1e-6, ymax - ymin)
        P = md.positions
        N = md.normals or [0.0, 1.0, 0.0] * (len(P) // 3)
        U = md.uvs or [0.0, 0.0] * (len(P) // 3)
        if md.indices:
            for idx in md.indices:
                self.pos.extend(P[idx * 3:idx * 3 + 3])
                self.nrm.extend(N[idx * 3:idx * 3 + 3])
                self.uv.extend(U[idx * 2:idx * 2 + 2])
                self.col.extend(color)
                self.kind.append(kind)
                self.wind.append(min(1.0, max(0.0,
                                             (P[idx * 3 + 1] - ymin) / span)))
        else:
            nv = len(P) // 3
            self.pos.extend(P)
            self.nrm.extend(N)
            self.uv.extend(U)
            self.col.extend(list(color) * nv)
            self.kind.extend([kind] * nv)
            for i in range(0, len(P), 3):
                self.wind.append(min(1.0, max(0.0, (P[i + 1] - ymin) / span)))

    def _tri(self, a, b, c, uva, uvb, uvc, col, kind, ymin, ymax):
        n = _normal(a, b, c)
        span = max(1e-6, ymax - ymin)
        for p, t in ((a, uva), (b, uvb), (c, uvc)):
            self.pos.extend(p)
            self.nrm.extend(n)
            self.uv.extend(t)
            self.col.extend(col)
            self.kind.append(kind)
            self.wind.append(min(1.0, max(0.0, (p[1] - ymin) / span)))

    def leaf(self, c, t, bn, half_len, half_wid, rnd, ymin, ymax):
        """一片尖头薄叶：沿叶脉折一下的四点菱形。

        折痕是关键 —— 完全平的一块四边形在阳光下就是一整片死色斑，
        中间往背光侧压下去一点，立刻分出受光面/背光面，眼睛才认得出是叶子。
        随机数写进 color.a，交给顶点着色器逐叶传下去做落叶判定。
        """
        cx, cy, cz = c
        tx, ty, tz = t
        bx, by, bz = bn
        n = _normal((cx, cy, cz),
                    (cx + tx, cy + ty, cz + tz),
                    (cx + bx, cy + by, cz + bz))
        fold = half_wid * 0.55
        a = (cx - tx * half_len, cy - ty * half_len, cz - tz * half_len)
        p = (cx + tx * half_len, cy + ty * half_len, cz + tz * half_len)
        l = (cx + bx * half_wid - n[0] * fold,
             cy + by * half_wid - n[1] * fold,
             cz + bz * half_wid - n[2] * fold)
        r = (cx - bx * half_wid - n[0] * fold,
             cy - by * half_wid - n[1] * fold,
             cz - bz * half_wid - n[2] * fold)
        col = (1.0, 1.0, 1.0, rnd)
        self._tri(a, l, p, (0.0, 0.5), (0.5, 0.0), (1.0, 0.5),
                  col, 1.0, ymin, ymax)
        self._tri(a, p, r, (0.0, 0.5), (1.0, 0.5), (0.5, 1.0),
                  col, 1.0, ymin, ymax)

    def quad(self, a, b, c, d, col, kind, ymin, ymax):
        self._tri(a, b, c, (0.0, 0.0), (1.0, 0.0), (1.0, 1.0),
                  col, kind, ymin, ymax)
        self._tri(a, c, d, (0.0, 0.0), (1.0, 1.0), (0.0, 1.0),
                  col, kind, ymin, ymax)

    @property
    def vertex_count(self):
        return len(self.pos) // 3


def _normal(a, b, c):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    L = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return (nx / L, ny / L, nz / L)


def build_plant(preset, seed, extra_iters=0):
    """跑 L-System → 生成 (Builder, 高度, 枝段数, 叶片数)。"""
    rng = random.Random(seed)
    iters = max(1, preset['iters'] + extra_iters)
    cmd = expand(preset['axiom'], preset['rules'], iters)
    turtle = Turtle3D(pos=(0.0, 0.0, 0.0), heading=(0.0, 1.0, 0.0),
                      up=(0.0, 0.0, 1.0), step=1.0,
                      angle=math.radians(preset['angle']))
    segs = turtle.run(cmd)
    if not segs:
        segs = [((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))]
    parent, children, leaves = build_tree(segs)
    chains = branch_chains(parent, children)

    ys = [p[1] for s in segs for p in s]
    ymin, ymax = min(ys), max(ys)
    total_leaves = max(1, leaves[0])
    b = Builder()
    r0 = preset['r0'] * max(1.0, (ymax - ymin) / 9.0) ** 0.45

    # --- 补一段真树干 ---
    # 龟图从原点出发就直接分叉的物种（橡树的 'A' 一上来就三分），
    # 画出来是"三根柱子插在地上"的怪异三脚架。这里在被分叉点下面
    # 接一根锥形主干，把整棵树抬起来，重心和体感立刻对了。
    n_root = sum(1 for a, _ in segs if max(abs(a[0]), abs(a[1]), abs(a[2])) < 1e-6)
    trunk_h = 0.0
    if n_root >= 2:
        trunk_h = (ymax - ymin) * 0.24
        segs = [((a[0], a[1] + trunk_h, a[2]), (b2[0], b2[1] + trunk_h, b2[2]))
                for a, b2 in segs]
        ymin, ymax = 0.0, ymax + trunk_h
        md = tube([(0.0, 0.0, 0.0), (0.0, trunk_h * 0.62, 0.0),
                   (0.0, trunk_h, 0.0)],
                  radius=[r0 * 1.45, r0 * 1.12, r0 * 0.98],
                  radial=8, caps=False)
        b.add_mesh(md, (0.33, 0.24, 0.16, 1.0), 0.0, ymin, ymax)

    # --- 枝干：每条枝链扫成一根变径管 ---
    for chain in chains:
        pts = [segs[chain[0]][0]] + [segs[n][1] for n in chain]
        # da Vinci：半径 ∝ 该段子树叶片数^(1/2.4)。主干粗、末梢细，是算出来的。
        radii = []
        for n in chain:
            frac = leaves[n] / total_leaves
            radii.append(max(0.010, r0 * frac ** 0.416))
        radii.append(radii[-1] * 0.55)
        md = tube(pts, radius=radii, radial=5, caps=False)
        shade = 0.92 + 0.16 * rng.random()
        bark = (0.34 * shade, 0.25 * shade, 0.17 * shade, 1.0)
        b.add_mesh(md, bark, 0.0, ymin, ymax)

    # --- 叶片：沿外侧枝段撒布 ---
    # 挂点不只选末梢：子树叶片占比很小的枝（接近末梢）都算，
    # 而且叶位沿整条线段随机取 —— 否则长枝中段光秃秃，
    # 冠层缩成枝头上的几团"绿爆米花"（垂柳第一批版本就栽在这）。
    leaf_count = 0
    host_frac = 0.030
    hosts = [i for i in range(len(segs))
             if leaves[i] <= max(4.0, total_leaves * host_frac)]
    rng.shuffle(hosts)
    cap = 12000                                 # 顶点预算，超了就不再加叶子
    # 叶片大小跟着树高走：幼树 62 个单位高，用橡树的叶片尺寸就成了绿点
    lscale = min(2.8, max(0.65, (ymax - ymin) / 28.0))
    for i in hosts:
        if b.vertex_count > cap * 6:
            break
        a, bpt = segs[i]
        tt = rng.random() * 0.9 + 0.05          # 沿枝段取叶位（避开节点）
        tip = (a[0] + (bpt[0] - a[0]) * tt,
               a[1] + (bpt[1] - a[1]) * tt,
               a[2] + (bpt[2] - a[2]) * tt)
        for _ in range(12):
            s = lscale * (0.30 + 0.26 * rng.random())
            # 随机朝向的薄片：取一个随机法线，再在其平面内张开
            nx, ny, nz = (rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))
            L = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            nx, ny, nz = nx / L, ny / L, nz / L
            tx, ty, tz = -ny, nx, 0.0
            Lt = math.sqrt(tx * tx + ty * ty + tz * tz) or 1.0
            if Lt < 1e-3:
                tx, ty, tz = 1.0, 0.0, 0.0
                Lt = 1.0
            tx, ty, tz = tx / Lt, ty / Lt, tz / Lt
            bx = ny * tz - nz * ty
            by = nz * tx - nx * tz
            bz = nx * ty - ny * tx
            ox = (rng.random() - 0.5) * 0.55
            oy = (rng.random() - 0.5) * 0.55
            oz = (rng.random() - 0.5) * 0.55
            c = (tip[0] + ox, tip[1] + oy, tip[2] + oz)
            b.leaf(c, (tx, ty, tz), (bx, by, bz),
                   s * 0.52, s * 0.30, rng.random(), ymin, ymax)
            leaf_count += 1

    return b, (ymax - ymin), len(segs), leaf_count


class Lindenwood(DemoApp):
    TITLE = 'LINDENwood - L-System plants (logic/lsystem)'
    HELP = ('1-5 species  R reseed  +/- iterations  S seasons  W wire  '
            'drag/scroll  SPACE pause  ESC')
    DURATION = None
    SHOTS = ((6.0, 'docs/lindenwood_01.png'), (20.0, 'docs/lindenwood_02.png'))
    FOV = 45.0

    def setup(self):
        glEnable(GL_DEPTH_TEST)
        self.program = ShaderProgram(Shader(VERTEX_SRC, 'vertex'),
                                     Shader(FRAGMENT_SRC, 'fragment'))
        self.ground_prog = ShaderProgram(Shader(GROUND_VERT, 'vertex'),
                                         Shader(GROUND_FRAG, 'fragment'))
        self.sky_prog = ShaderProgram(Shader(SKY_VERT, 'vertex'),
                                      Shader(SKY_FRAG, 'fragment'))
        self.sky = self.sky_prog.vertex_list(
            3, GL_TRIANGLES,
            position=('f', (-1.0, -1.0, 0.0, 3.0, -1.0, 0.0, -1.0, 3.0, 0.0)))
        self.preset_i = 0
        for i, pr in enumerate(PRESETS):
            if pr['name'].split()[-1] in self.args.flags:
                self.preset_i = i
        self.seed = 20261024
        self.extra_iters = 0
        self.season_on = True
        self.wire = False
        self.amp = 0.0 if 'still' in self.args.flags else 1.0
        self.season_t = 0.42 if 'autumn' in self.args.flags else 0.0
        self.cam = OrbitCamera(target=Vec3(0, 6, 0), yaw=38.0, pitch=16.0,
                               dist=34.0, pitch_range=(2.0, 88.0),
                               dist_range=(6.0, 220.0))
        orbit_camera_input(self, self.cam)
        self.model = Mat4()
        self._make_ground()
        self.rebuild()

    def _make_ground(self):
        R = 1500.0
        verts = (-R, 0.0, -R, R, 0.0, -R, R, 0.0, R,
                 -R, 0.0, -R, R, 0.0, R, -R, 0.0, R)
        cols = (1.0, 1.0, 1.0, 1.0) * 6
        self.ground = self.ground_prog.vertex_list(
            6, GL_TRIANGLES, position=('f', verts), colors=('f', cols))

    # ------------------------------------------------------------ 重建 --
    def rebuild(self):
        preset = PRESETS[self.preset_i]
        b, height, nseg, nleaf = build_plant(preset, self.seed, self.extra_iters)
        self.tree = self.program.vertex_list(
            b.vertex_count, GL_TRIANGLES,
            position=('f', b.pos), normals=('f', b.nrm), uvs=('f', b.uv),
            colors=('f', b.col), kind=('f', b.kind), wind=('f', b.wind))
        self.height = height
        self.n_segments = nseg
        self.n_leaves = nleaf
        self.n_verts = b.vertex_count
        self.cam.dist = max(12.0, height * 1.55)
        self.cam.target = Vec3(0, height * 0.45, 0)
        print(f'[lindenwood] {preset["name"]!r} seed={self.seed} '
              f'segs={nseg} leaves={nleaf} verts={self.n_verts} '
              f'height={height:.1f}')

    # ------------------------------------------------------------ 交互 --
    def on_key(self, symbol, modifiers):
        k = key
        if symbol in (k._1, k._2, k._3, k._4, k._5):
            self.preset_i = min(len(PRESETS) - 1, symbol - k._1)
            self.extra_iters = 0
            self.seed = 20261024
            self.rebuild()
        elif symbol == k.R:
            self.seed = (self.seed * 1103515245 + 12345) & 0x7FFFFFFF
            self.rebuild()
        elif symbol in (k.EQUAL, k.PLUS, k.NUM_ADD):
            self.extra_iters = min(2, self.extra_iters + 1)
            self.rebuild()
        elif symbol in (k.MINUS, k.NUM_SUBTRACT):
            self.extra_iters = max(-3, self.extra_iters - 1)
            self.rebuild()
        elif symbol == k.S:
            self.season_on = not self.season_on
        elif symbol == k.W:
            self.wire = not self.wire

    def season_colors(self):
        t = self.season_t % 1.0
        f = t * (len(SEASON_COLS) - 1)
        i = min(len(SEASON_COLS) - 2, int(f))
        a = f - i
        c0, c1 = SEASON_COLS[i], SEASON_COLS[i + 1]
        col = tuple(c0[j] * (1 - a) + c1[j] * a for j in range(3))
        drop = max(0.0, (t - 0.80) / 0.20) if t > 0.80 else 0.0
        return col, drop

    def update(self, dt):
        if self.paused:
            return
        if self.season_on:
            self.season_t = (self.season_t + dt / SEASON_LEN) % 1.0
        self.cam.yaw += dt * 1.6

    # ------------------------------------------------------------ 渲染 --
    def render(self):
        w, h = self.window.width, self.window.height
        aspect = w / max(1, h)
        leaf, drop = self.season_colors()
        # 阳光角度随季节压低：冬天太阳更低更冷
        elev = 0.55 - 0.18 * math.sin(self.season_t * math.tau)
        sun = Vec3(0.42, elev, 0.30).normalize()
        warm = 1.0 - 0.25 * math.sin(self.season_t * math.tau)
        sun_col = (1.30 * warm, 1.16 * warm, 0.95 * warm)
        sky = (0.42, 0.52, 0.66)
        grd = (0.20, 0.17, 0.12)
        fog = (0.58, 0.66, 0.76)

        glClearColor(fog[0], fog[1], fog[2], 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.window.projection = Mat4.perspective_projection(
            aspect, z_near=0.5, z_far=4000.0, fov=self.FOV)
        self.window.view = self.cam.view_matrix()

        # --- 天空（先画，不写深度，让地面和树盖在上面） ---
        eye = self.cam.eye
        tgt = self.cam.target
        fx, fy, fz = tgt.x - eye.x, tgt.y - eye.y, tgt.z - eye.z
        fl = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
        fx, fy, fz = fx / fl, fy / fl, fz / fl
        # right = normalize(fwd × 世界上方)；up = right × fwd
        rx, ry, rz = -fz, 0.0, fx
        rl = math.sqrt(rx * rx + rz * rz) or 1.0
        rx, rz = rx / rl, rz / rl
        ux = ry * fz - rz * fy
        uy = rz * fx - rx * fz
        uz = rx * fy - ry * fx
        ty = math.tan(math.radians(self.FOV) * 0.5)
        sp = self.sky_prog
        sp.use()
        sp['u_fwd'] = (fx, fy, fz)
        sp['u_right'] = (rx, ry, rz)
        sp['u_up'] = (ux, uy, uz)
        sp['u_tan'] = (ty * aspect, ty)
        sp['u_top'] = (0.30, 0.45, 0.68)
        sp['u_hor'] = fog
        sp['u_sun_dir'] = (sun.x, sun.y, sun.z)
        sp['u_sun_col'] = sun_col
        glDisable(GL_DEPTH_TEST)
        self.sky.draw(GL_TRIANGLES)
        glEnable(GL_DEPTH_TEST)
        glClear(GL_DEPTH_BUFFER_BIT)

        # --- 地面 ---
        gp = self.ground_prog
        gp.use()
        gp['u_sun_col'] = sun_col
        gp['u_grass'] = (0.29, 0.36, 0.19)
        gp['u_fog_col'] = fog
        gp['u_height'] = self.height
        self.ground.draw(GL_TRIANGLES)

        # --- 树 ---
        if self.wire:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        prog = self.program
        prog.use()
        prog['u_model'] = self.model
        prog['u_time'] = self.time
        prog['u_amp'] = self.amp * (0.55 + 0.25 * math.sin(self.time * 0.21))
        prog['u_gust'] = (math.sin(self.time * 0.13) * 2.0,
                          math.cos(self.time * 0.17) * 2.0)
        prog['u_sun_dir'] = (sun.x, sun.y, sun.z)
        prog['u_sun_col'] = sun_col
        prog['u_amb_sky'] = sky
        prog['u_amb_ground'] = grd
        prog['u_leaf'] = leaf
        prog['u_drop'] = drop
        prog['u_fog_col'] = fog
        # 雾的远近随机距走：橡树高 35、垂枝 elm 高 62，写死一组数值必然顾此失彼
        prog['u_fog_range'] = (self.cam.dist * 0.55, self.cam.dist * 2.4)
        self.tree.draw(GL_TRIANGLES)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.window.projection = self.perspective

    def status_text(self):
        names = ('春', '夏', '秋', '冬')
        t = self.season_t % 1.0
        season = names[min(3, int(t * 4))]
        return (f'{PRESETS[self.preset_i]["name"]}  iters{self.extra_iters:+d}  '
                f'segs {self.n_segments}  leaves {self.n_leaves}  '
                f'H {self.height:4.1f}  {season}  {self.fps:5.1f} fps'
                f'{"  [PAUSED]" if self.paused else ""}')


if __name__ == '__main__':
    run(Lindenwood)
