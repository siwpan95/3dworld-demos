# -*- coding: utf-8 -*-
"""《远景 / CDLOD Terrain》—— 大世界地形的两条 LOD 路线（terrain_lod / terrain_clipmap）。

**为什么值得单独一个 demo**
----------------------------
``engine/terrain_lod``（CDLOD）与 ``engine/terrain_clipmap``（几何 clipmap）是引擎里
单测最密的两块：前者 18 项（选层无缝无重叠 / 相邻层差 ≤1 / morph 消缝 / 三角数亚线性 /
模板网格），后者 15 项（嵌套 ring / 刚性吸附 / 顶点位移 / annulus 拓扑）。
它们解决的是同一个问题 —— **`geometry.heightfield()` 是 1024² 就 100 万顶点的
静态全分辨率网格，视距一拉远面数线性爆炸** —— 但手段相反：

* **CDLOD**（Strugar 2010）：四叉树选层，**所有节点共用一张 ``grid×grid`` 模板**，
  按"节点包围盒到相机的距离"决定细分；相邻层交界处把细网格顶点**插值吸附到粗网格
  格点**（morph），因此不需要 skirts 也不需要缝合格。
* **几何 clipmap**：反过来实现"以相机为中心的一叠嵌套 ring"，每层格距翻倍、外层挖
  掉被内层盖住的洞；关键不变式是**所有 ring 的中心都吸附到最细格距**，于是相机在
  一个最细格距内移动时，网格是**刚性**的 —— 顶点位置不变，只有高度数据滚动。

两块都有单测，**演示仓此前零个 demo 用过它们**。本 demo 把它们做成可切换、可对照，
并把单测里断言的那些不变式搬到 HUD 上，让"无缝无重叠""刚性吸附"变成看得见的数。

怎么看
------
* ``L`` 键 —— **LOD 层染色**：近处暖、远处冷，四叉树的选层一眼可见。
* ``M`` 键 —— **关掉 morph**：相邻 LOD 层的交界会立刻裂开缝（按下就知道这个
  0.25 的常数在干什么）。
* ``C`` 键 —— 在 CDLOD 与 clipmap 两条路线之间切换，HUD 指标也跟着换。
* HUD 常驻：
  - CDLOD 的 ``cov min/max`` 来自 ``coverage_report()``，正常时应为 **1/1**
    （无缝、无重叠）—— 这是单测里的硬断言，这里让它实时可见；
  - clipmap 的 ``rebuild/reuse`` 直接说明"刚性吸附"省了多少次重建：相机在
    最细格距内移动时不重建，跨过格点才重建一次。

实现要点
--------
* 高度只在**一张 R32F 纹理**里：顶点着色器采样它做位移与有限差分法线，
  网格本身是常驻的模板（每节点只改 4 个 uniform）。这是 CDLOD 该有的用法
  （引擎模块注释明写：真渲染用 ``MORPH_GLSL``，**不要每帧重建网格**）。
* clipmap 侧反过来：ring 是 CPU 端 ``build_ring_mesh`` 生成的，只有 snap 后的
  中心变化才重建 —— 缓存命中时整帧零重建。
* 相机默认**自动巡航**（``F`` 开关），因为 LOD 只有动起来才看得出变化；
  巡航基于累积时间，所以 ``--shot`` 的取景时刻是可复现的。

用法::

    python examples/cdlod_terrain.py                正常运行
    python examples/cdlod_terrain.py --smoke 3      冒烟
    python examples/cdlod_terrain.py --shot         抓剧照到 docs/cdlod_terrain_01/02.png
    python examples/cdlod_terrain.py --flag clip    起手就是 clipmap 路线

按键：C 换路线  L 层染色  M morph 开关  W 线框  R 换地形种子
      +/- 层数   F 自动巡航  WASD 飞  QE 升降  拖拽转头  SPACE 暂停  ESC 退出
"""
import ctypes
import math

import numpy as np
from pyglet.gl import (GL_CLAMP_TO_EDGE, GL_COLOR_BUFFER_BIT,
                       GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_FILL,
                       GL_FRONT_AND_BACK, GL_LINE, GL_LINEAR, GL_R32F,
                       GL_RED, GL_TEXTURE0, GL_TEXTURE_2D,
                       GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER,
                       GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TRIANGLES,
                       GL_FLOAT, glActiveTexture, glBindTexture, glClear,
                       glClearColor, glDeleteTextures, glEnable,
                       glGenTextures, glPolygonMode, glTexImage2D,
                       glTexParameteri)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key

# ⚠️ 不要再加 sys.path 引导：引擎由 `pip install 3dworld` 提供
# （见 requirements.txt 与 docs/依赖与版本.md）。
from world3d.engine.terrain_clipmap import (ClipmapConfig, build_clipmap,    # noqa: E402
                                            build_ring_mesh, clipmap_stats)
from world3d.engine.terrain_lod import (LodConfig, TerrainLod)               # noqa: E402
from world3d.logic.noise_vec import fbm                                      # noqa: E402
from world3d.platform.app import DemoApp, run                                # noqa: E402

HM = 512                       # 高度图分辨率（每边）
WORLD = 2048.0                 # 地形世界尺寸（米）
HEIGHT_SCALE = 300.0
# 一个高度纹素覆盖多少世界米 —— 顶点着色器用它当差分步长，保证法线尺度与视觉一致
TEXEL_WORLD = WORLD / HM

# 每层一个色：LOD 染色时"近暖远冷"
LEVEL_COLS = ((1.00, 0.52, 0.34), (1.00, 0.72, 0.38), (0.86, 0.86, 0.52),
              (0.56, 0.84, 0.70), (0.48, 0.70, 0.94), (0.60, 0.62, 0.98),
              (0.72, 0.64, 0.96))


# ------------------------------------------------------------- 高度场 --
def make_heightmap(seed, octaves=5):
    """fBm 高度场，值域归一化到 [0,1]（向量化噪声，512² 也是毫秒级）。"""
    axis = np.linspace(-4.0, 4.0, HM, dtype=np.float64)
    xs, ys = np.meshgrid(axis, axis)
    h = fbm(xs, ys, octaves=octaves, lacunarity=2.03, gain=0.5, seed=seed)
    h = np.asarray(h, dtype=np.float64)
    # 岛屿化：中心高、边缘沉到水下，这样有海岸线可看
    r = np.hypot(xs, ys) / 4.0
    h = h * 0.72 + (1.0 - np.clip(r, 0.0, 1.0)) * 0.42
    lo, hi = float(h.min()), float(h.max())
    return (h - lo) / max(1e-9, hi - lo)


def make_height_texture(hm):
    """R32F 高度纹理（LINEAR + CLAMP_TO_EDGE）。

    用 R32F 而不是 R8：8-bit 只有 256 级，300 米高差下会带来 ~1.2 米的台阶，
    在近处地形上肉眼可见（cellward 那条"8-bit 通道要过 LUT"是同一个坑的另一种表现）。
    """
    data = np.ascontiguousarray(hm, dtype=np.float32)
    tid = ctypes.c_uint()
    glGenTextures(1, ctypes.byref(tid))
    glBindTexture(GL_TEXTURE_2D, tid)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, HM, HM, 0, GL_RED, GL_FLOAT,
                 data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    return tid


# --------------------------------------------------------------- 着色 --
# 顶点位移 + morph 吸附：逻辑与引擎 terrain_lod.MORPH_GLSL 完全一致
# （snap_even 把顶点吸到"父层格点"，k=1 时细网格完全落在粗网格上 ⇒ 无缝）。
CDLOD_VERT = '''#version 330 core
in vec3 position;      // 模板顶点，xz ∈ [0,1]
in vec2 uvs;

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;
uniform vec2 u_node_origin;
uniform float u_node_size;
uniform float u_morph_k;
uniform float u_cell;
uniform float u_level;
uniform float u_world_size;
uniform float u_height_scale;
uniform float u_texel;
uniform sampler2D u_height;

out vec3 v_world;
out vec3 v_nrm;
out vec2 v_uv;
out float v_level;

float snap_even(float v, float spacing) {
    float s = spacing * 2.0;
    return floor(v / s + 0.5) * s;
}

void main() {
    vec2 local = position.xz * u_node_size;
    vec2 snapped = vec2(snap_even(local.x, u_cell), snap_even(local.y, u_cell));
    vec2 morphed = mix(local, snapped, u_morph_k);
    vec2 wpos = u_node_origin + morphed;
    vec2 uvw = (wpos + 0.5 * u_world_size) / u_world_size;
    float h = texture(u_height, uvw).r * u_height_scale;
    v_world = vec3(wpos.x, h, wpos.y);
    // 法线用高度图的邻域差分：步长取一个纹素对应的世界距离，
    // 步长写死会导致近处过平滑、远处过锐利。
    float e = u_texel / u_world_size;
    float hx1 = texture(u_height, uvw + vec2(e, 0.0)).r * u_height_scale;
    float hx0 = texture(u_height, uvw - vec2(e, 0.0)).r * u_height_scale;
    float hz1 = texture(u_height, uvw + vec2(0.0, e)).r * u_height_scale;
    float hz0 = texture(u_height, uvw - vec2(0.0, e)).r * u_height_scale;
    v_nrm = normalize(vec3(hx0 - hx1, 2.0 * u_texel, hz0 - hz1));
    v_uv = uvw;
    v_level = u_level;
    gl_Position = window.projection * window.view * vec4(v_world, 1.0);
}'''

# clipmap 的 ring 顶点已是世界坐标，着色器只管上色
RING_VERT = '''#version 330 core
in vec3 position;
in vec3 normals;
in vec2 uvs;

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;

out vec3 v_world;
out vec3 v_nrm;
out vec2 v_uv;
out float v_level;

void main() {
    v_world = position;
    v_nrm = normals;
    v_uv = uvs;
    v_level = 0.0;
    gl_Position = window.projection * window.view * vec4(position, 1.0);
}'''

FRAG = '''#version 330 core
in vec3 v_world;
in vec3 v_nrm;
in vec2 v_uv;
in float v_level;
out vec4 final_colors;

uniform vec3 u_sun;
uniform vec3 u_fog;
uniform float u_levels;     // 总层数（LOD 染色归一化用）
uniform float u_tint;       // 1 = LOD 层染色，0 = 按高度上色
uniform float u_ring_level; // clipmap 时逐 ring 传层数

vec3 level_color(float lv) {
    float t = clamp(lv / max(1.0, u_levels - 1.0), 0.0, 1.0);
    // 近暖远冷：t=0 用暖橙，t=1 用冷紫，中间插值
    vec3 near = vec3(1.00, 0.56, 0.36);
    vec3 far = vec3(0.58, 0.62, 0.98);
    return mix(near, far, t);
}

void main() {
    float h = v_world.y;
    vec3 n = normalize(v_nrm);
    // 按高度分层：水下 → 沙滩 → 草 → 岩 → 雪
    vec3 rock = vec3(0.36, 0.34, 0.32);
    vec3 col = mix(vec3(0.10, 0.20, 0.32), vec3(0.52, 0.48, 0.34),
                   smoothstep(-6.0, 12.0, h));
    col = mix(col, vec3(0.26, 0.40, 0.22), smoothstep(8.0, 42.0, h));
    col = mix(col, rock, smoothstep(70.0, 150.0, h));
    col = mix(col, vec3(0.92, 0.94, 0.98), smoothstep(210.0, 280.0, h));
    // 陡坡露岩：坡度大就不长草
    float slope = 1.0 - clamp(n.y, 0.0, 1.0);
    col = mix(col, rock, smoothstep(0.35, 0.75, slope));

    // CDLOD 走 v_level（逐节点），clipmap 走 u_ring_level（逐 ring），另一路恒为 0
    float lv = v_level + u_ring_level;
    vec3 base = mix(col, level_color(lv), u_tint);
    float lam = max(dot(n, u_sun), 0.0);
    vec3 lit = base * (0.28 + 0.86 * lam);
    lit = mix(lit, u_fog, clamp(length(v_world) * 0.00042, 0.0, 0.75));
    final_colors = vec4(lit, 1.0);
}'''


# ------------------------------------------------------------------- app --
class CdlodTerrain(DemoApp):
    TITLE = 'CDLOD terrain - quadtree LOD vs geometry clipmap'
    HELP = ('C route  L level colors  M morph off  W wire  R new terrain  '
            'F cruise  WASD fly  QE up/down  SPACE pause  ESC')
    DURATION = None
    SHOTS = ((6.0, 'docs/cdlod_terrain_01.png'),
             (18.0, 'docs/cdlod_terrain_02.png'))
    FOV = 60.0

    def setup(self):
        glEnable(GL_DEPTH_TEST)
        self.cdlod_prog = ShaderProgram(Shader(CDLOD_VERT, 'vertex'),
                                        Shader(FRAG, 'fragment'))
        self.ring_prog = ShaderProgram(Shader(RING_VERT, 'vertex'),
                                       Shader(FRAG, 'fragment'))
        self.use_clipmap = 'clip' in self.args.flags
        self.lod_colors = True
        self.morph_on = True
        self.wire = False
        self.cruise = True
        self.seed = 20261026
        self.level_n = 6                     # CDLOD 层数（clipmap 时另算）
        self.rebuilds = 0
        self.reuses = 0
        self.ring_meshes = []
        self.ring_snap = None
        self.height_tex = None
        self.pos = [0.0, 260.0, 0.0]
        self.yaw = 0.0
        self.pitch = -0.35
        self.t = 0.0
        self.frame = 0
        # WASD 需要"按住"状态：pyglet 的 KeyStateHandler 直接挂在 window 上
        self.key_state = key.KeyStateHandler()
        self.window.push_handlers(self.key_state)
        self._build_terrain()

    # ------------------------------------------------------------ 地形 --
    def _build_terrain(self):
        if self.height_tex is not None:
            glDeleteTextures(1, ctypes.byref(ctypes.c_uint(self.height_tex)))
        self.hm = make_heightmap(self.seed)
        self.height_tex = make_height_texture(self.hm)
        self.terrain = TerrainLod(
            self.hm, size=WORLD, height_scale=HEIGHT_SCALE,
            config=LodConfig(grid=16, levels=self.level_n, leaf_size=32.0,
                             leaf_range=52.0))
        md = self.terrain.build_template()
        self.tpl = self.cdlod_prog.vertex_list_indexed(
            len(md.positions) // 3, GL_TRIANGLES,
            [int(i) for i in md.indices],
            position=('f', md.positions), uvs=('f', md.uvs))
        self.ring_snap = None                # 换地形则强制重建 clipmap
        print(f'[cdlod] seed={self.seed} levels={self.level_n} '
              f'grid=16 leaf=32 world={WORLD:.0f} hscale={HEIGHT_SCALE:.0f}')

    # ------------------------------------------------------------ 交互 --
    def on_key(self, symbol, modifiers):
        k = key
        if symbol == k.C:
            self.use_clipmap = not self.use_clipmap
            self.ring_snap = None
        elif symbol == k.L:
            self.lod_colors = not self.lod_colors
        elif symbol == k.M:
            self.morph_on = not self.morph_on
        elif symbol == k.W and not self.cruise:
            self.wire = not self.wire
        elif symbol == k.R:
            self.seed = (self.seed * 1103515245 + 12345) & 0x7FFFFFFF
            self._build_terrain()
        elif symbol in (k.EQUAL, k.PLUS, k.NUM_ADD):
            self.level_n = min(8, self.level_n + 1)
            self._build_terrain()
        elif symbol in (k.MINUS, k.NUM_SUBTRACT):
            self.level_n = max(3, self.level_n - 1)
            self._build_terrain()
        elif symbol == k.F:
            self.cruise = not self.cruise

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.yaw -= dx * 0.22
        self.pitch = max(-1.4, min(1.2, self.pitch + dy * 0.006))

    # ------------------------------------------------------------ 更新 --
    def update(self, dt):
        if self.paused:
            return
        self.t += dt
        if self.cruise:
            # 自动巡航：绕着地形中心转，LOD 只有动起来才看得出变化
            a = self.t * 0.06
            r = 420.0 + 260.0 * math.sin(self.t * 0.031)
            self.pos[0] = math.cos(a) * r
            self.pos[2] = math.sin(a) * r
            self.yaw = math.degrees(-a) - 90.0
            self.pitch = -0.32
            self.pos[1] = self.terrain.height_at(self.pos[0], self.pos[2]) + 95.0
        else:
            ks = getattr(self, 'key_state', None)
            if ks is not None:
                sp = 160.0 * dt
                fx = math.sin(math.radians(self.yaw))
                fz = math.cos(math.radians(self.yaw))
                if ks[key.W]:
                    self.pos[0] += fx * sp
                    self.pos[2] += fz * sp
                if ks[key.S]:
                    self.pos[0] -= fx * sp
                    self.pos[2] -= fz * sp
                if ks[key.A]:
                    self.pos[0] -= fz * sp
                    self.pos[2] += fx * sp
                if ks[key.D]:
                    self.pos[0] += fz * sp
                    self.pos[2] -= fx * sp
                if ks[key.Q]:
                    self.pos[1] -= sp
                if ks[key.E]:
                    self.pos[1] += sp
            # 别让相机钻进地里
            ground = self.terrain.height_at(self.pos[0], self.pos[2]) + 18.0
            self.pos[1] = max(self.pos[1], ground)

    # ------------------------------------------------------------ 渲染 --
    def render(self):
        w, h = self.window.width, self.window.height
        glClearColor(0.52, 0.60, 0.70, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.window.projection = Mat4.perspective_projection(
            w / max(1, h), z_near=1.0, z_far=6000.0, fov=self.FOV)
        yaw = math.radians(self.yaw)
        pitch = self.pitch
        eye = Vec3(self.pos[0], self.pos[1], self.pos[2])
        fwd = Vec3(math.sin(yaw) * math.cos(pitch), math.sin(pitch),
                   math.cos(yaw) * math.cos(pitch))
        self.window.view = Mat4.look_at(eye, eye + fwd, Vec3(0.0, 1.0, 0.0))
        if self.wire:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        if self.use_clipmap:
            self._render_clipmap()
        else:
            self._render_cdlod()

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.window.projection = self.perspective

    def _render_cdlod(self):
        nodes = self.terrain.select(camera=(self.pos[0], self.pos[2]),
                                    view_range=1400.0)
        prog = self.cdlod_prog
        prog.use()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.height_tex)
        prog['u_height'] = 0
        prog['u_world_size'] = WORLD
        prog['u_height_scale'] = HEIGHT_SCALE
        prog['u_texel'] = TEXEL_WORLD
        prog['u_sun'] = (0.42, 0.80, 0.42)
        prog['u_fog'] = (0.52, 0.60, 0.70)
        prog['u_levels'] = float(self.level_n)
        prog['u_tint'] = 1.0 if self.lod_colors else 0.0
        prog['u_ring_level'] = 0.0
        grid = float(self.terrain.config.grid)
        for n in nodes:
            prog['u_node_origin'] = (n.x0, n.z0)
            prog['u_node_size'] = n.size
            prog['u_morph_k'] = n.morph if self.morph_on else 0.0
            prog['u_cell'] = n.size / grid
            prog['u_level'] = float(n.level)
            self.tpl.draw(GL_TRIANGLES)
        self.last_nodes = nodes
        self.last_stats = self.terrain.stats(nodes)
        # coverage_report 要栅格化 64×64 再逐节点累加，每帧算纯属浪费：
        # 它验证的是"选层结果的拓扑不变式"，30 帧刷一次足够盯住。
        self.frame += 1
        if self.frame % 30 == 0 or not hasattr(self, 'last_cov'):
            self.last_cov = self.terrain.coverage_report(nodes,
                                                         view_range=1400.0)

    def _render_clipmap(self):
        cfg = ClipmapConfig(levels=min(6, self.level_n), grid=32,
                            base_cell=2.0)
        rings = build_clipmap((self.pos[0], self.pos[2]), cfg)
        snap = (cfg.snap(self.pos[0]), cfg.snap(self.pos[2]))
        # 刚性吸附的不变式：中心 snap 不变 ⇒ 顶点位置一格没动，只有高度数据在滚。
        # 所以命中时整帧零重建 —— HUD 的 rebuild/reuse 就是在报这件事。
        if self.ring_snap != snap or len(self.ring_meshes) != len(rings):
            self._rebuild_rings(rings)
            self.rebuilds += 1
            self.ring_snap = snap
        else:
            self.reuses += 1
        prog = self.ring_prog
        prog.use()
        prog['u_sun'] = (0.42, 0.80, 0.42)
        prog['u_fog'] = (0.52, 0.60, 0.70)
        prog['u_levels'] = float(len(rings))
        prog['u_tint'] = 1.0 if self.lod_colors else 0.0
        for lv, vl in zip(rings, self.ring_meshes):
            prog['u_ring_level'] = float(lv.level)
            vl.draw(GL_TRIANGLES)
        self.last_nodes = rings
        self.last_stats = clipmap_stats(rings)
        self.last_cov = {'min': 1, 'max': 1, 'holes': 0}

    def _rebuild_rings(self, rings):
        for vl in self.ring_meshes:
            try:
                vl.delete()
            except Exception:
                pass
        self.ring_meshes = []
        for ring in rings:
            md = build_ring_mesh(ring, self.terrain.height_at,
                                 world_size=WORLD)
            vl = self.ring_prog.vertex_list_indexed(
                len(md.positions) // 3, GL_TRIANGLES,
                [int(i) for i in md.indices],
                position=('f', md.positions), normals=('f', md.normals),
                uvs=('f', md.uvs))
            self.ring_meshes.append(vl)

    # ------------------------------------------------------------ 状态 --
    def status_text(self):
        st = self.last_stats
        cov = self.last_cov
        route = 'clipmap' if self.use_clipmap else 'cdlod'
        if self.use_clipmap:
            extra = (f'  rings {st.get("rings", len(self.last_nodes))}'
                     f'  rebuild {self.rebuilds}/reuse {self.reuses}')
        else:
            extra = (f'  nodes {st.get("nodes", 0)}'
                     f'  tris {st.get("triangles", 0)}'
                     f'  L{st.get("max_level", 0)}'
                     f'  cov {cov.get("min", 0)}/{cov.get("max", 0)}'
                     f'  holes {cov.get("holes", 0)}'
                     f'  morph {"on" if self.morph_on else "OFF"}')
        return (f'{route}{extra}  levels {self.level_n}'
                f'  {self.fps:5.1f} fps'
                f'{"  [PAUSED]" if self.paused else ""}')


if __name__ == '__main__':
    run(CdlodTerrain)
