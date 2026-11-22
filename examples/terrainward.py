# -*- coding: utf-8 -*-
"""《地形志 / Terrainward》—— 程序化地形高度场 + 顶点烘焙着色（logic/terrain 首秀）。

**为什么值得单独一个 demo**
----------------------------
``world3d.logic.terrain`` 是「程序化地形」能力的引擎侧落点（从已删除的
``terrain_flyover.py`` 上提，被 ``tests/test_terrain_row_emitter`` 逐位钉死契约）。
它把"高度场生成 + 分层着色 + 网格发射"拆成一组**纯函数**：

    height_at(wx, wz, seed)        世界坐标 -> 高程（三层 fBm：大起伏 / 山脊 / 谷地）
    terrain_color(h, slope, lam, j) (高度, 坡度, 面向光, 抖动) -> 烘焙好的 RGB
    emit_row(pos, col, hs, cols, n, step, ox, oz, j)
                                    把第 j 行四边形带写成三角形汤（顶点色平滑着色）

本 demo 把这条"逻辑流水线"完整跑一遍并看得见：**规则网格采样高度场 -> 逐点算
坡度/面向光 -> terrain_color 烘焙顶点色 -> emit_row 逐行拼出三角汤 -> 轨道相机
俯视**。着色全在 CPU 顶点色里完成（片元着色器只做按视距混雾），这正是
``logic/terrain`` 的设计意图：顶点着色器直接复用模块自带的 ``VERTEX_SRC``
（WindowBlock 名为 ``window``、``colors`` 为 ``vec4``、无需 normals），片元着色器
沿用同一套属性/uniform 约定、仅把雾系数调小以便俯视看得清——本 demo 不重写引擎
契约，只是"翻译"成更合适的雾强度。

与 cdlod_terrain.py 的关系（互补，不重复）
------------------------------------------
``cdlod_terrain.py`` 演示的是 ``engine.terrain_lod`` / ``clipmap`` 的 **LOD 渲染**
（四叉树细分、视相关精度、Clipmap 环形缓冲）；它关心"怎么高效把一大片地形画出来"。
本 demo 关心的是**更底层**：高度场是怎么由纯函数算出来的、顶点色是怎么按高度/坡度/
光照烘焙的、网格是怎么由 ``emit_row`` 一行行发射的。**一个是渲染层，一个是逻辑/着色
层** —— 看 cdlod 学"渲染金字塔"，看本 demo 学"地形数据从哪来"。

用到的引擎 API
--------------
* ``world3d.logic.terrain``：``height_at`` / ``terrain_color`` / ``emit_row``，
  以及自带的 GLSL 顶点着色器 ``VERTEX_SRC``、天空色 ``SKY``、主光方向
  ``LIGHT_DIR``（片元着色器沿用同约定、仅调小雾系数，见模块内 ``FRAG_SRC``）。
* ``world3d.logic.camera.OrbitCamera``：俯视轨道相机（``view_matrix()`` /
  ``eye()``）；``orbit_camera_input`` 接鼠标拖拽/滚轮。
* ``world3d.platform.app``：``DemoApp`` 子类范式（``setup/update/render/
  status_text/on_key``），``run`` 入口。

HUD（status_text）读取并显示
-----------------------------
种子 seed、网格边长 ``grid N x N``、地形 min/max 高程、当前上色模式（SHADE=完整
坡面光照 / FLAT=纯高度分层）、实时帧率 fps。

控制键
------
R 重新播种（确定性 LCG 推进 seed，非随机）   W 线框
SPACE 暂停（同时停掉相机自动旋转）          L 切换 SHADE / FLAT 上色
拖拽/滚轮 转视角/缩放                        ESC 退出

用法::

    python examples/terrainward.py                正常运行
    python examples/terrainward.py --smoke 3      冒烟
    python examples/terrainward.py --shot         抓剧照到 docs/terrainward_01/02.png
"""
import math

from pyglet.gl import (GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
                       GL_FILL, GL_FRONT_AND_BACK, GL_LINE, GL_TRIANGLES, glClear,
                       glClearColor, glEnable, glPolygonMode)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key

# ⚠️ 不要再加 sys.path 引导：引擎由 `pip install 3dworld` 提供
# （见 requirements.txt 与 docs/依赖与版本.md）。禁止 noise_vec（不在 wheel）。
from world3d.logic.camera import OrbitCamera                            # noqa: E402
from world3d.logic.terrain import (LIGHT_DIR, SKY, VERTEX_SRC, emit_row, # noqa: E402
                                   height_at, terrain_color)
from world3d.platform.app import DemoApp, orbit_camera_input, run       # noqa: E402

# 片元着色器：与 logic/terrain.FRAGMENT_SRC 同一套属性/uniform 约定（330 +
# WindowBlock `window`，colors 为 vec4），仅把雾系数调小让俯视地形更清晰。
FRAG_SRC = '''#version 330 core
in vec4 v_col;
in vec3 v_world;
out vec4 final_colors;
uniform vec3 u_eye;
uniform vec3 u_fog;       // 必须等于清屏色，否则地平线一条硬边
void main() {
    float d = distance(v_world, u_eye);
    float fog = 1.0 - exp(-d * 0.0030);
    final_colors = vec4(mix(v_col.rgb, u_fog, clamp(fog, 0.0, 1.0)), 1.0);
}'''

GRID = 128                      # 每边采样点数（网格边长）
STEP = 2.0                      # 采样点间距（米）
SEED0 = 20261026                # 固定初始种子，保证 smoke / shot 可复现
ROT_SPEED = 0.25                # 相机自动旋转（弧度/秒）

# 把主光方向归一化一次（terrain_color 的 lambert 用未归一方向也行，这里归一更稳）。


def _norm(v):
    L = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return (v[0] / L, v[1] / L, v[2] / L)


class Terrainward(DemoApp):
    TITLE = 'TERRAINward - procedural height field + baked shading (logic/terrain)'
    HELP = ('R reseed  W wire  SPACE pause  L shade/flat  '
            'drag/scroll  ESC')
    DURATION = None
    SHOTS = ((6.0, 'docs/terrainward_01.png'),
             (16.0, 'docs/terrainward_02.png'))
    FOV = 55.0
    Z_NEAR, Z_FAR = 0.5, 2000.0

    def setup(self):
        glEnable(GL_DEPTH_TEST)
        # 顶点着色器直接复用 logic/terrain 自带的 VERTEX_SRC（WindowBlock 名为
        # window，colors 为 vec4，无需 normals 属性）。片元着色器沿用同一套
        # 属性/uniform 约定，仅把雾系数从模块默认的 0.0105 调小，让俯视视角下
        # 的地形更清晰（远景仍保留一点大气透视，雾色恒等于清屏色 SKY）。
        self.prog = ShaderProgram(Shader(VERTEX_SRC, 'vertex'),
                                  Shader(FRAG_SRC, 'fragment'))
        self.n = GRID
        self.step = STEP
        self.seed = SEED0
        self.flat = False                 # False=完整坡面光照，True=纯高度分层
        self.wire = 'wire' in self.args.flags      # --flag wire：线框对照（也用于验证 GL_LINE 路径）
        self.L = _norm(LIGHT_DIR)
        self.cam = OrbitCamera(target=Vec3(0, 0, 0), yaw=35.0, pitch=56.0,
                               dist=220.0, pitch_range=(8.0, 85.0),
                               dist_range=(40.0, 700.0))
        orbit_camera_input(self, self.cam)
        self.mesh = None
        self._sample()

    # ----------------------------------------------------- 高度场 + 着色 --
    def _sample(self):
        """采样高度场 -> 逐点算坡度/面向光 -> terrain_color 烘焙顶点色。

        先一次性填好扁平数组 ``hs``（高程）与 ``cols_full`` / ``cols_flat``
        （4 元组，因为着色器 colors 是 vec4），再由 ``emit_row`` 逐行拼几何。
        """
        N, step = self.n, self.step
        ox = -step * (N - 1) * 0.5
        oz = -step * (N - 1) * 0.5
        L = self.L
        hs = [0.0] * (N * N)
        cols_full = [None] * (N * N)
        cols_flat = [None] * (N * N)

        # 1) 高程场
        for j in range(N):
            wz = oz + j * step
            base = j * N
            for i in range(N):
                hs[base + i] = height_at(ox + i * step, wz, self.seed)

        # 2) 逐点：有限差分求法线 -> 坡度(0平1陡) + 面向光亮度 -> 烘焙色
        for j in range(N):
            base = j * N
            jd = max(0, j - 1)
            ju = min(N - 1, j + 1)
            for i in range(N):
                idx = base + i
                h = hs[idx]
                il = max(0, i - 1)
                ir = min(N - 1, i + 1)
                dhdx = (hs[base + ir] - hs[base + il]) / ((ir - il) * step)
                dhdz = (hs[ju * N + i] - hs[jd * N + i]) / ((ju - jd) * step)
                # 法线 = (-dh/dx, 1, -dh/dz) 归一
                nx, ny, nz = -dhdx, 1.0, -dhdz
                inv = 1.0 / (math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0)
                nx, ny, nz = nx * inv, ny * inv, nz * inv
                lam = max(nx * L[0] + ny * L[1] + nz * L[2], 0.0)
                slope = 1.0 - ny                       # ny=1 平 -> 0；越陡越近 1
                jit = ((i * 928371 + j * 123457) & 1023) / 1023.0   # 确定性抖动
                cf = terrain_color(h, slope, lam, jit)
                cols_full[idx] = (cf[0], cf[1], cf[2], 1.0)
                cflat = terrain_color(h, 0.0, 1.0, jit)            # 关掉坡/光
                cols_flat[idx] = (cflat[0], cflat[1], cflat[2], 1.0)

        self.hs = hs
        self.cols_full = cols_full
        self.cols_flat = cols_flat
        self.min_h = min(hs)
        self.max_h = max(hs)
        self._build_mesh()

    def _build_mesh(self):
        """用 emit_row 把整张地形拼成一次 draw 的三角汤。

        emit_row 追加写入 pos / col（颜色已是 4 元组，对应 vec4 属性）。
        """
        N, step = self.n, self.step
        ox = -step * (N - 1) * 0.5
        oz = -step * (N - 1) * 0.5
        cols = self.cols_flat if self.flat else self.cols_full
        pos, col = [], []
        for j in range(N - 1):                  # 逐行发射（与整块拼接逐位相同）
            emit_row(pos, col, self.hs, cols, N, step, ox, oz, j)
        if self.mesh is not None:
            self.mesh.delete()
        self.mesh = self.prog.vertex_list(
            len(pos) // 3, GL_TRIANGLES,
            position=('f', pos), colors=('f', col))
        self.n_verts = len(pos) // 3

    # ------------------------------------------------------------ 交互 --
    def on_key(self, symbol, modifiers):
        k = key
        if symbol == k.R:
            # 确定性换种子（LCG），保证 R 之后仍可复现。
            self.seed = (self.seed * 1103515245 + 12345) & 0x7FFFFFFF
            self._sample()
            print(f'[terrainward] reseed -> seed={self.seed}  '
                  f'min={self.min_h:.1f} max={self.max_h:.1f} '
                  f'verts={self.n_verts}')
        elif symbol == k.W:
            self.wire = not self.wire
        elif symbol == k.L:
            self.flat = not self.flat
            self._build_mesh()

    # ------------------------------------------------------------ 更新 --
    def update(self, dt):
        if self.paused:                 # SPACE 暂停同时停掉自动旋转
            return
        self.cam.yaw += dt * ROT_SPEED

    # ------------------------------------------------------------ 渲染 --
    def render(self):
        w, h = self.window.width, self.window.height
        glClearColor(*SKY, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.window.projection = Mat4.perspective_projection(
            w / max(1, h), z_near=self.Z_NEAR, z_far=self.Z_FAR, fov=self.FOV)
        self.window.view = self.cam.view_matrix()

        prog = self.prog
        prog.use()
        eye = self.cam.eye                        # 世界坐标 eye（@property，Vec3）
        prog['u_eye'] = (eye.x, eye.y, eye.z)
        prog['u_fog'] = SKY                         # 必须等于清屏色，免地平线硬边
        if self.wire:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        self.mesh.draw(GL_TRIANGLES)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.window.projection = self.perspective

    # ------------------------------------------------------------ 状态 --
    def status_text(self):
        return (f'seed {self.seed}  grid {self.n}x{self.n}  '
                f'h {self.min_h:6.1f}..{self.max_h:6.1f}  '
                f'{"FLAT" if self.flat else "SHADE"}  '
                f'{self.fps:5.1f} fps'
                f'{"  [PAUSED]" if self.paused else ""}')


if __name__ == '__main__':
    run(Terrainward)
