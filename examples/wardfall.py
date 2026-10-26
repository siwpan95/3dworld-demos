# -*- coding: utf-8 -*-
"""《城邦 / Wardfall》—— 程序化中世纪城市（Voronoi 划区 → 城墙 → 路网 → 建筑）。

**为什么值得单独一个 demo**
----------------------------
``world3d.logic.town`` 是引擎里最完整、却从没在演示仓露过面的一块能力：
移植自 watabou 的 Medieval Fantasy City Generator 管线（Voronoi 划区 →
城墙 → 路网 → ward 分配 → 建筑几何），``generate_town(seed)`` 一次性吐出
bounds / wall / streets / wards / citadel / plaza，确定性可复现（LCG 随机源）。
它有 ``tests/test_town.py``（K0/K1/K2 三级自测），但**零个 demo 消费它**。
本 demo 把它第一次完整渲染出来；几何全部由城镇数据现算，零外部素材。

数据 → 几何的映射
-----------------
* ``wards[].buildings``：每栋一个 Polygon（3~16 顶点）→ **耳切三角化** +
  挤出 + 屋顶（紧凑地块戴尖顶、大地块平顶）；层高按 ward 类型加权。
* ``wall.shape``：它的顶点是**外轮廓**而非环 → 沿相邻顶点对生成一段段厚城墙，
  ``wall.towers`` 的点补圆柱塔楼。
* ``streets``：变长折线 → 按宽度沿走向铺成薄板路面。
* ``wards[].polygon``：地块色板（``ward.color`` 是 0~255 的三元组分区配色）。

两处刻意的取舍
--------------
1. **窗户在片元里现算**：立面 uv 当窗格，夜里只有格子哈希命中的那部分亮灯，
   且抽得随机——这样城才像有人住，不像一排整齐的灯泡阵列。
2. **单趟合并三角汤**：200~800 栋楼一次合进一个 vertex_list，一次 draw call
   画完全城；几何是一次性生成、不做逐帧变换，没必要走 InstancedMesh 那层。

用法::

    python examples/wardfall.py                正常运行
    python examples/wardfall.py --smoke 3      冒烟
    python examples/wardfall.py --shot         抓剧照到 docs/wardfall_01/02.png
    python examples/wardfall.py --flag auto    电影机位自动巡游（录成片用）
    python examples/wardfall.py --flag seed123 开局换种子
    python examples/wardfall.py --flag big     起手 64 wards 的大城

按键：N 换城   D 昼夜   W 线框   1/2/3 城市规模   C 自动机位   SPACE 暂停   ESC 退出
"""
import math
import random

from pyglet.gl import (GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
                       GL_DEPTH_TEST, GL_FILL, GL_FRONT_AND_BACK, GL_LINE,
                       GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_TRIANGLES,
                       glBlendFunc, glClear, glClearColor, glDepthMask,
                       glDisable, glEnable, glPolygonMode)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key

# ⚠️ 不要再加 sys.path 引导：引擎由 `pip install 3dworld` 提供
# （见 requirements.txt 与 docs/依赖与版本.md）。
from world3d.logic.camera import OrbitCamera                        # noqa: E402
from world3d.logic.town import generate_town                        # noqa: E402
from world3d.platform.app import (DemoApp, make_fullscreen_quad,          # noqa: E402
                                  orbit_camera_input, run)

# 本 demo 的诺德组件拆在私有模块里（仿 examples/_harness.py 约定）。
# 加 examples/ 到 sys.path 是为了让 verify_demos / record_all 用
# importlib 加载本文件时也能解析 `_wardfall_nordic`（仅加本目录，不碰引擎路径）。
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
from _wardfall_nordic import (                                       # noqa: E402
    terrain_height, build_terrain, segment_box_terrain,
    build_pine_forest, build_longhouse, build_rune_wall,
    build_citadel, build_plaza, build_gatehouses,
    dragon_body_mesh, dragon_wing_mesh, dragon_eye_mesh,
    dragon_fire_mesh, dragon_city_pose, SnowField, nordic_ost,
    wardfall_ost)

VERTEX_SRC = '''#version 330 core
in vec3 position;
in vec3 normals;
in vec4 colors;
in vec2 uvs;
in float kind;          // 0地块 1路面 2预留 3楼身 4屋顶 5城墙 6塔楼

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;
uniform mat4 u_model;

out vec3 v_col;
out vec3 v_nrm;
out vec2 v_uv;
out float v_kind;
out float v_alpha;
out vec3 v_world;

void main() {
    vec4 world = u_model * vec4(position, 1.0);
    v_world = world.xyz;
    v_col = colors.rgb;
    v_alpha = colors.a;
    v_nrm = normals;
    v_uv = uvs;
    v_kind = kind;
    gl_Position = window.projection * window.view * world;
}'''

FRAGMENT_SRC = '''#version 330 core
in vec3 v_col;
in vec3 v_nrm;
in vec2 v_uv;
in float v_kind;
in float v_alpha;
in vec3 v_world;

uniform vec3 u_sun_dir;      // 指向太阳
uniform vec3 u_sun_col;
uniform vec3 u_moon_dir;     // 指向月亮（太阳落下时升起的对侧）
uniform vec3 u_amb_sky;
uniform vec3 u_amb_ground;
uniform float u_night;       // 0 = 白天, 1 = 深夜
uniform vec3 u_fog_col;
uniform float u_fog_density;
uniform vec3 u_eye;

uniform float u_snow;          // 0 = 盛夏, 1 = 深冬
uniform float u_rune;         // 龙语墙/龙眼 辉光强度
uniform float u_fade;         // 龙/龙焰 透明淡入（城市绘制前复位 1.0）
// 4 盏点光：pyglet 不支持向量数组 uniform（vec3[4]/vec4[4] 都被当成标量数组），
// 故拆成 4 组独立 vec4：xyz 位置 / 颜色 + w 半径（颜色 w 预留）。
uniform vec4 u_pl0;   // 火塘
uniform vec4 u_pl1;   // 龙语墙辉
uniform vec4 u_pl2;   // 月华冷光
uniform vec4 u_pl3;   // 龙焰（跟着龙）
uniform vec4 u_plc0;
uniform vec4 u_plc1;
uniform vec4 u_plc2;
uniform vec4 u_plc3;

out vec4 final_colors;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// 值噪声（地面/田垄的慢变斑块用，只要低频，够用即可）
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i), b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0)), d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// 单盏点光（距离平方衰减，半径越大约束光晕范围）；pyglet 不支持向量数组 uniform，
// 故 4 盏灯走 4 组独立 vec4，这里用函数避免重复四遍。
vec3 point_light(vec4 p, vec4 c, vec3 w) {
    vec3 dd = p.xyz - w;
    float dist2 = dot(dd, dd);
    return c.rgb * (p.w / (1.0 + dist2 * 0.04));
}

void main() {
    vec3 n = normalize(v_nrm);
    if (n.y < -0.5 && v_kind < 6.5) { n.y = -n.y; }

    vec3 base = v_col;
    // --- 地面/地块：低频斑块打散"一整片纯色" ---
    if (v_kind < 1.5) {
        float n1 = vnoise(v_world.xz * 0.012);
        float n2 = vnoise(v_world.xz * 0.055);
        base *= 0.80 + 0.34 * n1 + 0.14 * n2;
        if (v_kind < 0.5) { base *= 0.92 + 0.22 * vnoise(v_world.xz * 0.006); }
    }

    // --- 半兰伯特 + 半球环境光 ---
    float ndl = max(dot(n, u_sun_dir), 0.0);
    float wrap = (ndl + 0.35) / 1.35;
    vec3 hemi = mix(u_amb_ground, u_amb_sky, n.y * 0.5 + 0.5);
    vec3 lit = base * (hemi + u_sun_col * wrap * ndl);

    // --- 月光：太阳落山后由对侧补一盏冷光，否则城里一片纯黑只剩窗 ---
    float mdl = max(dot(n, u_moon_dir), 0.0);
    lit += base * vec3(0.13, 0.17, 0.30) * mdl * u_night * 1.25;

    // --- 窗户：按立面 uv 现算窗格，夜里只有一部分亮灯 ---
    if (v_kind > 2.5 && v_kind < 3.5) {
        vec2 g = floor(v_uv);
        vec2 f = fract(v_uv);
        float frame = smoothstep(0.06, 0.14, f.x) * smoothstep(1.0, 0.86, f.x)
                    * smoothstep(0.06, 0.14, f.y) * smoothstep(1.0, 0.86, f.y);
        lit *= mix(0.58, 1.0, frame);
        float on = step(0.52, hash21(g + v_col.rg * 7.0));
        float lamp = on * frame
                   * smoothstep(0.10, 0.20, v_uv.y)
                   * smoothstep(1.05, 0.55, v_uv.y);
        lit += u_night * lamp * vec3(1.00, 0.71, 0.33) * 1.45;
    }

    // --- 城墙 / 塔楼：石缝 ---
    if (v_kind > 4.5) {
        float row = floor(v_uv.y * 5.0);
        float brick = hash21(vec2(floor(v_uv.x * 3.0) + row * 0.7, row));
        lit *= 0.86 + 0.20 * brick;
    }

    // --- 距离雾：远处城墙融进天色 ---
    float d = length(v_world - u_eye);
    float fog = 1.0 - exp(-pow(d * u_fog_density, 2.0));
    lit = mix(lit, u_fog_col, clamp(fog, 0.0, 0.94));

    // --- 积雪：朝上的表面深冬被雪覆盖（地面/屋顶/墙/木/石/广场）---
    float upf = smoothstep(0.5, 0.92, n.y);
    float snow = u_snow * upf;
    if (v_kind < 0.5 || (v_kind > 3.5 && v_kind < 4.5)
        || (v_kind > 4.5 && v_kind < 10.5)
        || (v_kind > 11.5 && v_kind < 12.5)) {
        vec3 sc = mix(vec3(0.86, 0.89, 0.95), vec3(0.70, 0.78, 0.92),
                      clamp(v_world.y * 0.02, 0.0, 1.0));
        lit = mix(lit, sc, snow * 0.9);
    }

    // --- 龙语墙 / 龙眼：emissive 由 u_rune 驱动 ---
    if (v_kind > 10.5 && v_kind < 11.5) {
        lit += u_rune * vec3(0.95, 1.02, 1.12);
    }

    // --- 火塘：暖光自发 ---
    if (v_kind > 13.5 && v_kind < 14.5) {
        lit += vec3(1.6, 0.7, 0.2) * 1.4;
    }

    // --- 龙家族：自发光 + 透明淡入（u_fade 由调用方在画龙前设置）---
    if (v_kind > 19.5) {
        vec3 emis = vec3(0.0);
        if (v_kind < 20.5)      emis = vec3(0.25, 0.22, 0.20) * 0.6;
        else if (v_kind < 21.5) emis = vec3(0.20, 0.18, 0.16) * 0.5;
        else if (v_kind < 22.5) emis = vec3(2.2, 1.7, 0.4);
        else                    emis = vec3(2.4, 1.1, 0.3);
        lit = lit * 0.4 + emis;
    }

    // --- 4 盏点光（火塘 / 龙语墙辉 / 月华 / 龙焰）---
    lit += point_light(u_pl0, u_plc0, v_world);
    lit += point_light(u_pl1, u_plc1, v_world);
    lit += point_light(u_pl2, u_plc2, v_world);
    lit += point_light(u_pl3, u_plc3, v_world);

    // Reinhard 软肩：正午直射会把屋顶打到 0.7 以上纯白，陡坡全糊成纸片
    lit *= 1.30;
    lit = lit / (lit + vec3(0.95));
    lit = pow(max(lit, 0.0), vec3(1.02));
    float a = v_alpha;
    if (v_kind > 19.5 && v_kind < 23.5) { a = u_fade; }
    final_colors = vec4(lit, a);
}'''

SKY_VERT = '''#version 330 core
in vec3 position;
in vec4 colors;
out vec2 v_uv;
void main() {
    v_uv = position.xy * 0.5 + 0.5;
    gl_Position = vec4(position.xy, 0.999, 1.0);
}'''

SKY_FRAG = '''#version 330 core
in vec2 v_uv;
out vec4 final_colors;

uniform vec3 u_right;
uniform vec3 u_up;
uniform vec3 u_fwd;
uniform vec3 u_sun;
uniform vec3 u_horizon;
uniform vec3 u_zenith;
uniform vec3 u_glow;
uniform float u_ty;       // tan(fovy/2)
uniform float u_aspect;
uniform float u_snow;      // 深冬：天色偏冷、太阳盘减弱

void main() {
    // 由屏幕坐标反推视线方向（相机基向量已在 CPU 侧算好）。
    // 关键：用**视线仰角**而不是屏幕 y 做渐变 —— 地平线（ray.y≈0）因此恒定落在
    // 地平线色上，与地面盘被雾吃成的颜色严丝合缝，接缝不可见。
    vec2 p = v_uv * 2.0 - 1.0;
    vec3 ray = normalize(u_right * (p.x * u_ty * u_aspect)
                       + u_up * (p.y * u_ty) + u_fwd);
    vec3 c = mix(u_horizon, u_zenith, smoothstep(-0.05, 0.62, ray.y));
    // 深冬：天色整体往冷蓝偏
    c = mix(c, mix(c, vec3(0.32, 0.40, 0.58), 0.6), u_snow * 0.6);

    // 太阳：核心 + 两层光晕（角度距离，与相机朝向无关）
    float d = max(dot(ray, u_sun), 0.0);
    c += u_glow * (pow(d, 1200.0) * 8.0 + pow(d, 32.0) * 0.28)
              * (1.0 - 0.5 * u_snow);
    final_colors = vec4(c, 1.0);
}'''


# 飘雪：极简 billboard shader（只吃 WindowBlock 的投影/视图，直写 position 缓冲）。
# u_alpha 由季节驱动：盛夏(snow→0)雪场淡出为不可见。
SNOW_VERT = '''#version 330 core
in vec3 position;
uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;
void main() {
    gl_Position = window.projection * window.view * vec4(position, 1.0);
}'''

SNOW_FRAG = '''#version 330 core
out vec4 final_colors;
uniform float u_alpha;
void main() {
    final_colors = vec4(0.90, 0.94, 1.00, 0.85 * u_alpha);
}'''


def _md_vlist(program, md, kind_val, color=None):
    """把 world3d.engine.geometry 产出的（已非索引）MeshData 立成独立 vertex_list。

    geometry 的 tube/box/sphere/lathe/merge_meshes **不写 kind 字段**，这里补一个
    常数 kind（龙家族用 20~23，让片元着色器走自发光 + u_fade 透明分支）。
    颜色若为空（几何体没带 color）就给一个冷灰默认，避免黑块。
    """
    n = md.vertex_count
    kind_arr = [float(kind_val)] * n
    if md.colors:
        cols = md.colors
    else:
        c = color if color else (0.85, 0.85, 0.90, 1.0)
        cols = list(c) * n
    return program.vertex_list(
        n, GL_TRIANGLES,
        position=('f', md.positions),
        normals=('f', md.normals),
        uvs=('f', md.uvs),
        colors=('f', cols),
        kind=('f', kind_arr))


def _upload_point_lights(prog, pl_pos, pl_col, pl_rad):
    """把 4 盏点光（每盏 xyz + 半径 / rgb）写到 8 个独立 vec4 uniform。

    pyglet 不支持向量数组 uniform（vec3[4]/vec4[4] 都被当成标量数组），
    所以拆成 u_pl{i} = (x,y,z,半径) 与 u_plc{i} = (r,g,b,1)。
    """
    prog['u_pl0'] = (pl_pos[0], pl_pos[1], pl_pos[2], pl_rad[0])
    prog['u_plc0'] = (pl_col[0], pl_col[1], pl_col[2], 1.0)
    prog['u_pl1'] = (pl_pos[3], pl_pos[4], pl_pos[5], pl_rad[1])
    prog['u_plc1'] = (pl_col[3], pl_col[4], pl_col[5], 1.0)
    prog['u_pl2'] = (pl_pos[6], pl_pos[7], pl_pos[8], pl_rad[2])
    prog['u_plc2'] = (pl_col[6], pl_col[7], pl_col[8], 1.0)
    prog['u_pl3'] = (pl_pos[9], pl_pos[10], pl_pos[11], pl_rad[3])
    prog['u_plc3'] = (pl_col[9], pl_col[10], pl_col[11], 1.0)


# 龙在局部坐标系里的尺寸太小（脊线约 8 单位），相对半径 ~70 的城显得像只麻雀。
# 整体放大 5 倍后翼展约 40 单位，绕城飞才够分量。
DRAGON_SCALE = 5.0
# 龙眼/龙焰相对龙躯的局部偏移（脊线 +z 朝前、原点为肩）。
_DRAGON_EYE_L = (-0.18, 0.42, 3.05)
_DRAGON_EYE_R = (0.18, 0.42, 3.05)
_DRAGON_FIRE_T = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0.40, 3.00, 1]


# --------------------------------------------------------------------- 几何 --
class MeshBuilder:
    """扁平列表累加器：position / normals / uvs / colors(vec4) / kind。

    直接压扁平数组一次成型：城镇重建要拼十几万个顶点，省掉一层
    MeshData 对象的构造开销值得。
    """

    def __init__(self):
        self.pos = []
        self.nrm = []
        self.uv = []
        self.col = []
        self.kind = []

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


def clip_halfplane(poly, nx, nz, d):
    """Sutherland–Hodgman：保留 ``n·p <= d`` 的部分（半平面裁剪）。

    用于把街区切成联排小屋。凹多边形被切后可能出多块，这里取第一块即可 ——
    地块基本都是凸的（Voronoi 单元 + bisect 切分），这个误差可以接受。
    """
    out = []
    n = len(poly)
    for i in range(n):
        ax, az = poly[i]
        bx, bz = poly[(i + 1) % n]
        da = ax * nx + az * nz - d
        db = bx * nx + bz * nz - d
        if da <= 0:
            out.append((ax, az))
        if (da < 0 < db) or (db < 0 < da):
            t = da / (da - db)
            out.append((ax + (bx - ax) * t, az + (bz - az) * t))
    return out


def subdivide_lot(poly, max_width, rng, out, depth=0):
    """把过大的街区沿最长轴递归切分，直到宽度 <= ``max_width``。

    为什么需要：``generate_town`` 吐的是**街区（lot）**，不是单栋房子 ——
    实测宽度 p50 ≈ 4、p90 ≈ 26、最大 48 城镇单位，大小差一个数量级。
    直接一栋一栋挤出会得到几个巨型方盒子（第一版就是这样：整屏被 3 个大块占满）。
    切分之后街区变成联排，屋顶线才密得起来、城才像城。
    """
    w = abs(signed_area(poly)) ** 0.5
    if w <= max_width or depth >= 8 or len(poly) < 4:
        out.append(poly)
        return
    n = len(poly)
    ai, bi, best = 0, 1, -1.0
    for i in range(n):
        for j in range(i + 1, n):
            d = (poly[i][0] - poly[j][0]) ** 2 + (poly[i][1] - poly[j][1]) ** 2
            if d > best:
                best, ai, bi = d, i, j
    ax, az = poly[ai]
    bx, bz = poly[bi]
    ex, ez = bx - ax, bz - az
    L = math.hypot(ex, ez)
    if L < 1e-6:
        out.append(poly)
        return
    nx, nz = ex / L, ez / L
    ts = [px * nx + pz * nz for px, pz in poly]
    lo, hi = min(ts), max(ts)
    cut = lo + (hi - lo) * (0.42 + 0.16 * rng.random())
    a = clip_halfplane(poly, nx, nz, cut)
    b = clip_halfplane(poly, -nx, -nz, -cut)
    if len(a) < 3 or len(b) < 3:
        out.append(poly)
        return
    subdivide_lot(a, max_width, rng, out, depth + 1)
    subdivide_lot(b, max_width, rng, out, depth + 1)


def signed_area(poly):
    """``poly`` 为 [(x, z), ...]，返回有符号面积（正 = CCW）。"""
    s = 0.0
    n = len(poly)
    for i in range(n):
        x0, z0 = poly[i]
        x1, z1 = poly[(i + 1) % n]
        s += x0 * z1 - x1 * z0
    return s * 0.5


def ensure_ccw(poly):
    return list(poly) if signed_area(poly) >= 0 else list(reversed(poly))


def is_convex(poly):
    """多边形是否凸（允许极小容差）。"""
    pts = ensure_ccw(poly)
    n = len(pts)
    if n < 4:
        return True
    for i in range(n):
        ax, az = pts[i]
        bx, bz = pts[(i + 1) % n]
        cx, cz = pts[(i + 2) % n]
        if (bx - ax) * (cz - az) - (bz - az) * (cx - ax) < -1e-9:
            return False
    return True


def centroid_z(poly):
    n = len(poly)
    return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)


def triangulate(poly):
    """耳切三角化（支持凹多边形）。

    为什么不用扇形三角化：``generate_town`` 的建筑地块是由 ``bisect`` 反复切分
    出来的地界，偶尔切出**凹多边形**（实测 3~16 顶点都有），扇形化会在凹顶点
    处把三角形翻出墙面之外。
    """
    pts = ensure_ccw(poly)
    idx = list(range(len(pts)))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 100000:
        guard += 1
        clipped = False
        for i in range(len(idx)):
            a = idx[(i - 1) % len(idx)]
            b = idx[i]
            c = idx[(i + 1) % len(idx)]
            ax, az = pts[a]
            bx, bz = pts[b]
            cx, cz = pts[c]
            cross = (bx - ax) * (cz - az) - (bz - az) * (cx - ax)
            if cross <= 1e-9:            # 凹耳或退化
                continue
            inside = False
            for j in idx:
                if j in (a, b, c):
                    continue
                px, pz = pts[j]
                w0 = (bx - ax) * (pz - az) - (bz - az) * (px - ax)
                w1 = (cx - bx) * (pz - bz) - (cz - bz) * (px - bx)
                w2 = (ax - cx) * (pz - cz) - (az - cz) * (px - cx)
                if w0 > 1e-9 and w1 > 1e-9 and w2 > 1e-9:
                    inside = True
                    break
            if not inside:
                tris.append((a, b, c))
                idx.pop(i)
                clipped = True
                break
        if not clipped:                  # 兜底：剩余部分扇形化
            break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    elif len(idx) > 3:
        for i in range(1, len(idx) - 1):
            tris.append((idx[0], idx[i], idx[i + 1]))
    return [(pts[a], pts[b], pts[c]) for a, b, c in tris]


def extrude(mb, poly, y0, y1, col_top, col_side, kind_top, kind_side,
            uv_tile=2.4):
    """把 CCW 的 (x,z) 多边形挤出成「侧墙 + 顶」（省掉看不见的底面）。

    侧面 uv：u 沿走向、v 沿高度都按 ``uv_tile`` 米一格铺 —— 于是窗格在
    不同高度、不同周长的楼上保持一致大小，而不是被周长拉伸。
    """
    pts = ensure_ccw(poly)
    n = len(pts)
    for a, b, c in triangulate(pts):
        mb.tri((a[0], y1, a[1]), (b[0], y1, b[1]), (c[0], y1, c[1]),
               (0.0, 1.0, 0.0), col_top,
               ((a[0] / uv_tile, a[1] / uv_tile),
                (b[0] / uv_tile, b[1] / uv_tile),
                (c[0] / uv_tile, c[1] / uv_tile)), kind_top)
    for i in range(n):
        ax, az = pts[i]
        bx, bz = pts[(i + 1) % n]
        dx, dz = bx - ax, bz - az
        L = math.hypot(dx, dz)
        if L < 1e-6:
            continue
        nx, nz = dz / L, -dx / L          # CCW 多边形的向外法线（右转 90°）
        u1 = L / uv_tile
        v01 = (y1 - y0) / uv_tile
        mb.quad((ax, y0, az), (bx, y0, bz), (bx, y1, bz), (ax, y1, az),
                (nx, 0.0, nz), col_side,
                ((0.0, 0.0), (u1, 0.0), (u1, v01), (0.0, v01)), kind_side)


def pyramid_roof(mb, poly, y_base, y_peak, col, kind):
    """尖顶：从屋檐多边形各边向质心上方收成锥面。"""
    cx, cz = centroid_z(poly)
    pts = ensure_ccw(poly)
    for i in range(len(pts)):
        ax, az = pts[i]
        bx, bz = pts[(i + 1) % len(pts)]
        dx, dz = bx - ax, bz - az
        L = math.hypot(dx, dz) or 1.0
        nx, nz = dz / L, -dx / L
        mb.tri((ax, y_base, az), (bx, y_base, bz), (cx, y_peak, cz),
               (nx, 0.35, nz), col,
               ((0.0, 0.0), (1.6, 0.0), (0.8, 1.6)), kind)


def segment_box(mb, a, b, thickness, y0, y1, col, kind):
    """沿 (a→b) 铺一段厚板（城墙段与路面共用）。"""
    dx, dz = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dz)
    if L < 1e-6:
        return
    nx, nz = -dz / L * thickness * 0.5, dx / L * thickness * 0.5
    poly = [(a[0] + nx, a[1] + nz), (b[0] + nx, b[1] + nz),
            (b[0] - nx, b[1] - nz), (a[0] - nx, a[1] - nz)]
    extrude(mb, poly, y0, y1, col, col, kind, kind, uv_tile=3.0)


def disc(mb, cx, cz, r, y0, y1, col, kind, segs=14):
    """圆柱塔楼（圆环挤出即得，不必再对圆盘内部做三角化）。"""
    ring = [(cx + r * math.cos(i / segs * math.tau),
             cz + r * math.sin(i / segs * math.tau)) for i in range(segs)]
    extrude(mb, ring, y0, y1, col, col, kind, kind, uv_tile=3.0)


# --------------------------------------------------------------------- 城镇 --
WARD_HEIGHT = {
    'Cathedral': 1.55, 'Administration': 1.05, 'Patriciate': 0.88,
    'Merchant': 0.72, 'Market': 0.62, 'Craftsmen': 0.50, 'Slum': 0.34,
    # 补全 town 模块其余 ward 类型（ citadel=Castle / plaza=Market 已由专用件接管，
    # 这里覆盖 Military / Park / Common / Farm / GateWard 等）
    'Military': 0.70, 'Park': 0.30, 'Castle': 1.60, 'Common': 0.50,
    'Farm': 0.35, 'GateWard': 0.55,
}
TARGET_R = 70.0          # 城墙半径映射到的世界半径（决定整城的视觉尺度）
LOT_MAX_WIDTH = 5.5      # 街区切分阈值（城镇单位）：超过就沿最长轴对半切
# 立面与屋顶的实测调色板：中世纪的墙是"土/石灰/砖/木架"四类，不是纯色；
# 屋顶是石板与陶瓦。真实配色比任何单一 ward 色都更像一座城。
PLASTER = ((0.86, 0.82, 0.73), (0.79, 0.73, 0.62), (0.71, 0.66, 0.57),
           (0.63, 0.57, 0.49), (0.74, 0.62, 0.52))
ROOF_TILE = ((0.38, 0.24, 0.20), (0.30, 0.31, 0.36), (0.42, 0.31, 0.24),
             (0.24, 0.27, 0.31), (0.35, 0.28, 0.27))


def build_city_mesh(seed, n_wards):
    """``generate_town`` → (MeshBuilder, town, 建筑数, 塔楼数, info)。

    **缩放基准是城墙半径，不是 ``bounds``**（这里踩过一个坑）：
    ``bounds`` 把城墙外的郊区也框进去了，而郊区能把包围盒撑到城墙半径的 4~8 倍；
    按 bounds 归一化会让全城缩成一小撮、城墙变成中心的一圈白线（第一版就是这样）。
    墙内的 ``radius`` 才是"这座城"的真实尺度，墙外部分当郊区，交给雾去淡出。

    地面/路面/建筑/城墙的 base 全部采样 ``terrain_height`` —— 于是全城依**三级
    台地**而建（白漫城式：内城最高、外墙相隔），不再是一块平盘。诺德组件（松林、
    长屋、龙语墙、城堡、广场、城门楼）也并入同一个 mb，保持全城单 draw call。
    """
    town = generate_town(seed=seed, n_wards=n_wards, walled=True, scale_to=900)
    wl = town['wall']
    R = wl['radius'] if wl else 60.0
    S = TARGET_R / max(R, 1e-6)           # 城镇单位 → 世界单位
    cx = town['center'].x
    cy = town['center'].y
    rng = random.Random(seed * 7919 + n_wards)

    def P(pt):                            # town (x, y) → world (x, z)
        return ((pt.x - cx) * S, (pt.y - cy) * S)

    mb = MeshBuilder()
    mbn = MeshBuilder()      # 诺德专属叠加层（长屋 + 龙语墙），可单键 G 显隐

    # --- 地形：三级台地（原平盘被细分地形网格取代，边缘在雾里消失）---
    build_terrain(mb, 900.0, 12.0)

    # 注：不画 ward 地块色板。原版是 2D 俯视渲染，色板是"分区着色"；搬到 3D 里
    # 它就变成一片片压平的大三角面片，把街道和屋顶全盖住（实测画面像一堆彩色纸片）。
    # 分区信息改由**立面色的轻微色偏**承担（见下面 hue 那一行）。

    # --- 路面：碎石路暖灰；沿折线细分、逐顶点贴地（否则跨台地会穿地/悬空）---
    road_col = (0.44, 0.41, 0.36, 1.0)
    for line in town['streets']:
        for i in range(len(line) - 1):
            segment_box_terrain(mb, P(line[i]), P(line[i + 1]),
                                max(1.6, 3.4 * S), road_col, 1.0)

    # --- 建筑（街区 → 递归切分成联排小屋），base 落在地形上 ---
    buildings = 0
    for ward in town['wards']:
        wr, wg, wb = ward['color']
        hue = (wr / 765.0 - 0.29, wg / 765.0 - 0.29, wb / 765.0 - 0.29)
        for gon in ward['buildings']:
            lot = [(v.x, v.y) for v in gon.verts]
            if len(lot) < 3:
                continue
            parts = []
            subdivide_lot(lot, LOT_MAX_WIDTH, rng, parts)
            for part in parts:
                poly = ensure_ccw([((x - cx) * S, (y - cy) * S) for x, y in part])
                area = abs(signed_area(poly))
                if area < 1e-5:
                    continue
                if area > (7.0 * S) ** 2:
                    continue        # 极端漏网的巨地块：直接丢弃，比糊一块板好
                foot = math.sqrt(area)
                # 层高 = 地块尺度 × ward 类型权重 × 逐栋随机（0.68~1.32）。
                # 用地块尺度而不是固定值：大地块自然出高楼，小巷里的小屋就是矮棚，
                # 于是同一片城区自己长出高低错落的天际线。
                hunits = foot * WARD_HEIGHT.get(ward['type'], 0.55) * 0.9 * \
                    (0.68 + 0.64 * rng.random())
                # 硬上限按地块走：分级切分有个深度上限，个别巨型地块会漏网，
                # 一栋 48 宽 × 26 高的方盒子能顶掉半屏（实测最大三角面 683）。
                h = max(0.7, min(2.2 + foot * 0.55, hunits))

                pl = PLASTER[rng.randrange(len(PLASTER))]
                wall = (min(1.0, pl[0] + hue[0] * 0.9),
                        min(1.0, pl[1] + hue[1] * 0.9),
                        min(1.0, pl[2] + hue[2] * 0.9), 1.0)
                rt = ROOF_TILE[rng.randrange(len(ROOF_TILE))]
                roofc = (rt[0], rt[1], rt[2], 1.0)

                by = terrain_height(*centroid_z(poly)) + 0.16
                extrude(mb, poly, by, by + h, roofc, wall, 4.0, 3.0)
                if foot < 12.0 and is_convex(poly) and rng.random() < 0.72:
                    # 屋顶要**陡**：早先给 foot*0.22 的浅坡，从空中看就是一整块
                    # 被太阳打平的浅色三角面片（实测整城像撒了一地纸片）。
                    # 陡坡（≈0.5 倍进深）才出得了中世纪那种尖屋顶的天际线。
                    pyramid_roof(mb, poly, by + h, by + h + min(foot * 0.34, 2.0),
                                 roofc, 4.0)
                buildings += 1

    # --- 城墙（base 落在地形上）---
    towers = 0
    if wl:
        wall_col = (0.66, 0.64, 0.58, 1.0)
        tower_col = (0.55, 0.53, 0.48, 1.0)
        th = max(0.7, 0.95 * S)
        wall_h = max(2.4, 4.4 * S)
        # 墙脊用 **towers** 连：``build_walls`` 里 towers = shape.verts − gates，
        # 所以它本身就是"沿墙一圈、已去掉城门"的点列。按绕城心的极角重排后顺次
        # 相连即得墙脊，城门处自然是缺口。
        #
        # 为什么不直接用 shape 的顶点表顺序：``smooth_vertex_eq`` 平滑后顶点顺序
        # 被打乱、还留下重复点，顺次相连会拉出几根横穿城内的长弦
        # （第一版画面里那几道白色横梁就是这么来的）。
        uniq = {}
        for v in wl.get('towers', []):
            uniq[(round(v.x, 2), round(v.y, 2))] = v   # 平滑会留下重复点
        tw_sorted = sorted(uniq.values(),
                           key=lambda v: math.atan2(v.y - cy, v.x - cx))
        ring = [P(v) for v in tw_sorted]
        n = len(ring)
        if n >= 3:
            spans = [(math.hypot(ring[(i + 1) % n][0] - ring[i][0],
                                 ring[(i + 1) % n][1] - ring[i][1]), i)
                     for i in range(n)]
            # 最长的那几段跨过城门洞，封上就把门堵死了；其余全部连成墙
            open_gates = {i for _, i in sorted(spans, reverse=True)[:3]}
            for ln, i in spans:
                if i in open_gates or ln <= 1e-6:
                    continue
                p0, p1 = ring[i], ring[(i + 1) % n]
                yb = (terrain_height(p0[0], p0[1]) + terrain_height(p1[0], p1[1])) \
                    * 0.5 - 0.3
                segment_box(mb, p0, p1, th, yb, yb + wall_h, wall_col, 5.0)
        for p in wl.get('towers', []):
            q = P(p)
            tb = terrain_height(q[0], q[1]) - 0.3
            disc(mb, q[0], q[1], max(0.5, 0.85 * S), tb, tb + wall_h * 1.40,
                 tower_col, 6.0)
        towers = len(wl.get('towers', []))

    # --- 诺德组件 ---
    # 松林 / 城堡 / 广场 / 城门楼并入主 mesh（mb，仍单 draw call）；
    # 长屋 + 龙语墙这两个最具诺德辨识度的件单独放进 mbn，可被 G 键整体显隐。
    info = {'forest': 0, 'longhouse': 0, 'rune': 0, 'gate': 0,
            'citadel': False, 'plaza': False,
            'hearth': [], 'rune_pos': (0.0, 0.0, 0.0),
            'longhouse_pos': []}
    # 松林：城墙外环
    info['forest'] = build_pine_forest(mb, seed, inner=TARGET_R + 8.0,
                                       outer=TARGET_R + 150.0, count=260)
    # 长屋（船形屋顶）：一座近城堡/城心，一座在中环
    lh_a = 2.2
    if town['citadel']:
        ccx = sum(P(v)[0] for v in town['citadel'].shape.verts) \
            / len(town['citadel'].shape.verts)
        ccz = sum(P(v)[1] for v in town['citadel'].shape.verts) \
            / len(town['citadel'].shape.verts)
        lh0 = (ccx + 7.0, ccz + 4.0, 0.5)
    else:
        lh0 = (9.0, 11.0, 0.4)
    lh1 = (math.cos(lh_a) * 30.0, math.sin(lh_a) * 30.0, lh_a + 1.2)
    for (x, z, yaw) in (lh0, lh1):
        hp = build_longhouse(mbn, x, z, yaw)
        info['longhouse'] += 1
        info['hearth'].append(hp)
        info['longhouse_pos'].append((x, z))
    # 龙语墙：城外山坡，面朝城心
    ra = lh_a + math.pi
    rrad = TARGET_R + 30.0
    rx, rz = math.cos(ra) * rrad, math.sin(ra) * rrad
    info['rune_pos'] = build_rune_wall(mbn, rx, rz, ra + math.pi / 2.0)
    info['rune'] = 1
    # 城堡（云顶区主堡）
    if town['citadel']:
        info['citadel'] = True
        build_citadel(mb, *_citadel_center(town, P))
    # 广场（中央石板；Market ward 已产出市集建筑，这里补一块铺地）
    if town['plaza']:
        info['plaza'] = True
        pcx = sum(P(v)[0] for v in town['plaza'].shape.verts) \
            / len(town['plaza'].shape.verts)
        pcz = sum(P(v)[1] for v in town['plaza'].shape.verts) \
            / len(town['plaza'].shape.verts)
        build_plaza(mb, pcx, pcz, R=10.0)
    # 城门楼（用 wall['gates']）
    if wl and wl.get('gates'):
        gate_pts = [P(g) for g in wl['gates']]
        info['gate'] = build_gatehouses(mb, gate_pts)

    return mb, mbn, town, buildings, towers, info


def _citadel_center(town, P):
    """城堡 patch 的质心（世界坐标）。"""
    vs = town['citadel'].shape.verts
    cxw = sum(P(v)[0] for v in vs) / len(vs)
    czw = sum(P(v)[1] for v in vs) / len(vs)
    return cxw, czw


class Wardfall(DemoApp):
    TITLE = 'WARDfall - procedural Nordic city-state (logic/town + nordic_saga)'
    HELP = ('N 换城  D 昼夜  W 线框  1/2/3 规模  C 自动机位  S 季节  '
            'F 龙  L 龙语辉光  G 诺德层  M 静音  SPACE 暂停  ESC')
    DURATION = None
    SHOTS = ((3.0, 'docs/wardfall_01.png'),
             (26.0, 'docs/wardfall_02.png'),
             (34.0, 'docs/wardfall_03_winter_dragon.png'),
             (50.0, 'docs/wardfall_04_summer.png'))

    SIZES = (24, 40, 64)
    DAY_LEN = 48.0
    FOV = 42.0

    def setup(self):
        glEnable(GL_DEPTH_TEST)
        self.program = ShaderProgram(Shader(VERTEX_SRC, 'vertex'),
                                     Shader(FRAGMENT_SRC, 'fragment'))
        self.sky_prog = ShaderProgram(Shader(SKY_VERT, 'vertex'),
                                      Shader(SKY_FRAG, 'fragment'))
        self.sky_quad = make_fullscreen_quad(self.sky_prog)
        # 飘雪用独立极简 shader（只吃 WindowBlock，不碰城市 uniform）
        self.snow_prog = ShaderProgram(Shader(SNOW_VERT, 'vertex'),
                                      Shader(SNOW_FRAG, 'fragment'))
        seed = 20261024
        size_i = 1
        for f in self.args.flags:
            if f.startswith('seed'):
                try:
                    seed = int(f[4:])
                except ValueError:
                    pass
            elif f == 'big':
                size_i = 2
            elif f == 'small':
                size_i = 0
        self.seed = seed
        self.size_i = size_i
        self.snow_field = SnowField(self.snow_prog, count=1500,
                                    half=(160.0, 75.0, 160.0), size=0.55,
                                    speed=7.0, seed=seed)
        self.auto_cam = 'auto' in self.args.flags
        self.day_cycle = True
        self.wire = False
        self.day_phase = 0.22             # 起手接近清晨：第一帧就有长影子
        self.cam = OrbitCamera(target=Vec3(0, 4, 0), yaw=35.0, pitch=16.0,
                               dist=196.0, pitch_range=(6.0, 88.0),
                               dist_range=(20.0, 700.0))
        orbit_camera_input(self, self.cam)
        self.model = Mat4()

        # --- 龙：5 个独立 vertex_list（躯体 / 右翼 / 左翼 / 眼 / 焰）---
        self.dragon_body = _md_vlist(self.program, dragon_body_mesh(), 20)
        self.dragon_wing_r = _md_vlist(self.program, dragon_wing_mesh(1.0), 21)
        self.dragon_wing_l = _md_vlist(self.program, dragon_wing_mesh(-1.0), 21)
        self.dragon_eye = _md_vlist(self.program, dragon_eye_mesh(), 22)
        fm = dragon_fire_mesh()
        fm.transform(_DRAGON_FIRE_T)              # 挪到龙嘴前
        self.dragon_fire = _md_vlist(self.program, fm, 23)

        # --- 状态机 ---
        self.snow = 1.0            # 当前积雪强度（季节插值）
        self.snow_target = 1.0     # S 键：1 深冬 / 0 盛夏
        self._auto_season = True   # 录制时自动由深冬过渡到盛夏（手动按 S 即取消）
        self.show_dragon = True    # F 键
        self.show_nordic = True    # G 键：长屋 + 龙语墙层
        self.rune_on = True        # L 键：龙语墙辉光
        self.muted = False         # M 键：静音
        self.audio = wardfall_ost(duration=60.0)   # 多段情绪 OST（回退 nordic_ost）
        if self.audio is not None:
            try:
                self.audio.play()
            except Exception as e:                      # noqa: BLE001
                print(f'[audio] play failed: {e}')

        self.rebuild()

    # ------------------------------------------------------------ 重建 --
    def rebuild(self):
        mb, mbn, town, n_b, n_t, info = build_city_mesh(
            self.seed, self.SIZES[self.size_i])
        self.city = self.program.vertex_list(
            mb.vertex_count, GL_TRIANGLES,
            position=('f', mb.pos), normals=('f', mb.nrm),
            uvs=('f', mb.uv), colors=('f', mb.col), kind=('f', mb.kind))
        self.nordic = self.program.vertex_list(
            mbn.vertex_count, GL_TRIANGLES,
            position=('f', mbn.pos), normals=('f', mbn.nrm),
            uvs=('f', mbn.uv), colors=('f', mbn.col), kind=('f', mbn.kind))
        self.info = info
        self.n_buildings = n_b
        self.n_wards = len(town['wards'])
        self.n_streets = len(town['streets'])
        self.n_towers = n_t
        self.n_tris = (mb.vertex_count + mbn.vertex_count) // 3
        print(f'[wardfall] seed={self.seed} wards={self.SIZES[self.size_i]} '
              f'buildings={n_b} towers={n_t} forest={info["forest"]} '
              f'longhouse={info["longhouse"]} rune={info["rune"]} '
              f'gate={info["gate"]} tris={self.n_tris}')

    # ------------------------------------------------------------ 交互 --
    def on_key(self, symbol, modifiers):
        k = key
        if symbol == k.N:
            self.seed = (self.seed * 1103515245 + 12345) & 0x7FFFFFFF
            self.rebuild()
        elif symbol in (k._1, k._2, k._3):
            self.size_i = symbol - k._1
            self.rebuild()
        elif symbol == k.D:
            self.day_cycle = not self.day_cycle
        elif symbol == k.W:
            self.wire = not self.wire
        elif symbol == k.C:
            self.auto_cam = not self.auto_cam
        elif symbol == k.S:
            self._auto_season = False
            self.snow_target = 0.0 if self.snow_target > 0.5 else 1.0
        elif symbol == k.F:
            self.show_dragon = not self.show_dragon
        elif symbol == k.L:
            self.rune_on = not self.rune_on
        elif symbol == k.G:
            self.show_nordic = not self.show_nordic
        elif symbol == k.M:
            self.muted = not self.muted
            if self.audio is not None:
                try:
                    if self.muted:
                        self.audio.pause()
                    else:
                        self.audio.play()
                except Exception:                      # noqa: BLE001
                    pass

    # ------------------------------------------------------------ 昼夜 --
    def env_colors(self):
        """由太阳高度角推出一套 PBS-ish 的三色环境参数 + 夜权重。"""
        ang = self.day_phase * math.tau
        el = math.sin(ang)
        az = math.cos(ang)
        sun = Vec3(az * 0.75, el, 0.38).normalize()
        night = max(0.0, min(1.0, (0.06 - el) / 0.22))
        horizon = 1.0 - min(1.0, abs(el) * 3.2)
        warm = (1.00, 0.46, 0.20)
        noon = (1.00, 0.97, 0.90)
        k = max(0.0, el) * 1.45
        sun_col = tuple((warm[i] * horizon + noon[i] * (1 - horizon)) * k
                        for i in range(3))
        mix3 = lambda a, b: tuple(a[i] * (1 - night) + b[i] * night            # noqa: E731
                                  for i in range(3))
        sky = mix3((0.20, 0.26, 0.38), (0.045, 0.062, 0.125))
        grd = mix3((0.15, 0.13, 0.10), (0.020, 0.026, 0.048))
        fog = mix3((0.60, 0.68, 0.80), (0.035, 0.045, 0.085))
        return sun, sun_col, sky, grd, fog, night

    def update(self, dt):
        if self.paused:
            return
        if self.day_cycle:
            self.day_phase = (self.day_phase + dt / self.DAY_LEN) % 1.0
        if self.auto_cam:
            self.cam.yaw += dt * 5.0
            self.cam.pitch = 15.0 + 5.0 * math.sin(self.time * 0.11)
            self.cam.dist = 190.0 + 24.0 * math.sin(self.time * 0.07)
        # 季节插值：S 键切换 snow_target，雪量平滑过渡（盛夏↔深冬）
        if self._auto_season and self.time > 38.0:
            self.snow_target = 0.0          # 录制中段自动由深冬转入盛夏
            self._auto_season = False
        self.snow += (self.snow_target - self.snow) * min(1.0, dt * 1.5)
        # 飘雪跟随相机 XZ 更新（CPU 直写顶点缓冲）
        eye = self.cam.eye
        fwd = (self.cam.target - eye).normalize()
        self.snow_field.update(dt, eye, fwd)

    # ------------------------------------------------------------ 渲染 --
    def render(self):
        w, h = self.window.width, self.window.height
        aspect = w / max(1, h)
        sun, sun_col, sky, grd, fog, night = self.env_colors()
        glClearColor(fog[0], fog[1], fog[2], 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 相机基向量：天空着色器靠它把屏幕像素反推成视线方向
        fwd = (self.cam.target - self.cam.eye).normalize()
        right = fwd.cross(Vec3(0, 1, 0)).normalize()
        up = right.cross(fwd)

        # --- 天空：地平线色 == 雾色，于是地面与天的接缝不可见 ---
        glDisable(GL_DEPTH_TEST)
        sp = self.sky_prog
        sp.use()
        sp['u_right'] = (right.x, right.y, right.z)
        sp['u_up'] = (up.x, up.y, up.z)
        sp['u_fwd'] = (fwd.x, fwd.y, fwd.z)
        sp['u_sun'] = (sun.x, sun.y, sun.z)
        sp['u_horizon'] = fog
        sp['u_zenith'] = (sky[0] * 1.55 + 0.02, sky[1] * 1.62 + 0.03,
                          sky[2] * 1.75 + 0.06)
        sp['u_glow'] = sun_col
        sp['u_ty'] = math.tan(math.radians(self.FOV) * 0.5)
        sp['u_aspect'] = aspect
        self.sky_quad.draw(GL_TRIANGLES)
        glEnable(GL_DEPTH_TEST)

        if self.wire:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        self.window.projection = Mat4.perspective_projection(
            aspect, z_near=0.5, z_far=4000.0, fov=self.FOV)
        self.window.view = self.cam.view_matrix()
        prog = self.program
        prog.use()
        prog['u_model'] = self.model
        prog['u_sun_dir'] = (sun.x, sun.y, sun.z)
        prog['u_sun_col'] = sun_col
        # 月亮在太阳的对侧（太阳落下 ⇒ 月亮升起），改由它给夜里补一盏冷光
        prog['u_moon_dir'] = (-sun.x, max(0.12, -sun.y), -sun.z)
        prog['u_amb_sky'] = sky
        prog['u_amb_ground'] = grd
        prog['u_night'] = night
        prog['u_fog_col'] = fog
        prog['u_fog_density'] = 0.0017
        eye = self.cam.eye
        prog['u_eye'] = (eye.x, eye.y, eye.z)
        # --- 季节 / 龙语辉光 / 4 盏点光（pyglet 不支持向量数组 uniform，
        #     故拆成 8 个 vec4：u_pl{i} = (x,y,z,半径) / u_plc{i} = (r,g,b,1)）---
        prog['u_snow'] = self.snow
        prog['u_rune'] = 1.0 if self.rune_on else 0.0
        prog['u_fade'] = 1.0
        pl_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pl_col = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pl_rad = [0.0, 0.0, 0.0, 0.0]
        if self.info['hearth']:
            hx, hy, hz = self.info['hearth'][0]
            pl_pos[0:3] = [hx, hy, hz]
            pl_col[0:3] = [1.6, 0.7, 0.25]
            pl_rad[0] = 26.0
        rx, ry, rz = self.info['rune_pos']
        pl_pos[3:6] = [rx, ry, rz]
        pl_col[3:6] = [0.5, 1.4, 1.8] if self.rune_on else [0, 0, 0]
        pl_rad[1] = 44.0
        pl_pos[6:9] = [0.0, 45.0, 0.0]            # 月华冷光（高空补光）
        pl_col[6:9] = [0.25, 0.32, 0.55]
        pl_rad[2] = 130.0
        _upload_point_lights(prog, pl_pos, pl_col, pl_rad)

        # --- 城市 + 诺德叠加层（同 program，单 draw call 各一）---
        if self.wire:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        self.city.draw(GL_TRIANGLES)
        if self.show_nordic:
            self.nordic.draw(GL_TRIANGLES)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # --- 龙：绕城圆形轨迹，5 段各自带模型矩阵 ---
        if self.show_dragon:
            pos, yaw, wing, alpha = dragon_city_pose(self.time)
            M = (Mat4.from_translation(Vec3(*pos))
                 @ Mat4.from_rotation(yaw, Vec3(0, 1, 0))
                 @ Mat4.from_scale(Vec3(DRAGON_SCALE, DRAGON_SCALE,
                                         DRAGON_SCALE)))
            Rw = Mat4.from_rotation(wing, Vec3(0, 0, 1))
            wing_r_mat = M @ Rw
            wing_l_mat = M @ Mat4.from_rotation(-wing, Vec3(0, 0, 1))
            eye_l_mat = M @ Mat4.from_translation(Vec3(*_DRAGON_EYE_L))
            eye_r_mat = M @ Mat4.from_translation(Vec3(*_DRAGON_EYE_R))
            pl_pos[9:12] = [pos[0], pos[1], pos[2]]   # 龙焰点光跟着龙
            pl_col[9:12] = [2.4, 1.1, 0.3]
            pl_rad[3] = 30.0
            _upload_point_lights(prog, pl_pos, pl_col, pl_rad)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(False)
            prog['u_fade'] = alpha
            prog['u_model'] = M
            self.dragon_body.draw(GL_TRIANGLES)
            prog['u_model'] = eye_l_mat
            self.dragon_eye.draw(GL_TRIANGLES)
            prog['u_model'] = eye_r_mat
            self.dragon_eye.draw(GL_TRIANGLES)
            prog['u_model'] = wing_r_mat
            self.dragon_wing_r.draw(GL_TRIANGLES)
            prog['u_model'] = wing_l_mat
            self.dragon_wing_l.draw(GL_TRIANGLES)
            prog['u_model'] = M
            self.dragon_fire.draw(GL_TRIANGLES)
            glDepthMask(True)
            glDisable(GL_BLEND)
            prog['u_fade'] = 1.0
        else:
            _upload_point_lights(prog, pl_pos, pl_col, pl_rad)

        # --- 飘雪（独立极简 shader，跟相机 XZ，深冬才可见）---
        if self.snow > 0.01:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(False)
            sn = self.snow_prog
            sn.use()
            sn['u_alpha'] = self.snow
            self.snow_field.vlist.draw(GL_TRIANGLES)
            glDepthMask(True)
            glDisable(GL_BLEND)
            prog.use()

        self.window.projection = self.perspective

    def status_text(self):
        hh = (self.day_phase * 24.0) % 24.0
        flags = ''
        flags += '❄' if self.snow_target > 0.5 else '☀'
        flags += 'D' if self.show_dragon else '-'
        flags += 'R' if self.rune_on else '-'
        flags += 'N' if self.show_nordic else '-'
        flags += ('M' if self.muted else '♪')
        return (f'seed {self.seed}  {self.SIZES[self.size_i]} wards  '
                f'{self.n_buildings} buildings  {self.n_streets} streets  '
                f'{self.n_towers} towers  {self.n_tris} tris  {hh:4.1f}h  '
                f'{flags}  {self.fps:5.1f} fps'
                f'{"  [PAUSED]" if self.paused else ""}')

    def close(self):
        # TimelineAudio.close() 会 pause() + delete() 播放器，确保 pyglet.app.run()
        # 在窗口关闭后能正常返回（否则 loop=True 的 Player 会拽住事件循环、进程不退）。
        if self.audio is not None:
            try:
                self.audio.close()
            except Exception:                      # noqa: BLE001
                pass
        self.audio = None
        # ⚠️ 别漏了这行（2026-10-26 实测）：基类 DemoApp.close() 才是
        # ``self.window.close()``。只关音频不关窗口 ⇒ pyglet.app.run() 永不返回
        # ⇒ --smoke 挂到超时被杀（批量验收 rc=-9），而交互式运行因为点了
        # 窗口的 ✕（pyglet 自己关窗口）看不出问题。
        # 对照 nordic_saga.py 的同款 close：它就带了 super().close()。
        super().close()


if __name__ == '__main__':
    run(Wardfall)
