# -*- coding: utf-8 -*-
"""《织造 / Texward》—— 贴图全链路：文件解码 / sRGB 分色空间 / ORM 打包 / AO 分级。

`proctex` 解决的是"**生成**像素"（程序化图案）。本 demo 演示另一半：
像素如何**正确落地**到 GPU 并被 PBR 消费——正是《游戏贴图全链路实操手册》
里最容易静默出错的那几段：

1. **文件通道**（左球）：proctex 产物先落盘 PNG，再用 ``texture.from_image_file``
   走真实解码器回来，按 ``role='base'`` 以 **sRGB 内部格式**上传（GL_SRGB8_ALPHA8，
   采样时硬件解码）。误标成线性会把颜色整体洗亮——"看起来还行"的经典错误。
2. **内存通道**（中球）：``texture.from_pixels`` 直接上传 proctex 字节，
   法线按 ``role='normal'`` 以线性格式上传，``normalScale`` 烘进像素。
3. **ORM 单图三通道**（右球）：把 AO（R）/ 粗糙（G）/ 金属（B）打包成**一张**
   ORM 贴图，同时绑到 ``mr_tex`` 与 ``occlusion_tex``——这正是 UE/glTF 的省槽
   做法；通道位置由 ``texture.unpack_packed`` 逐像素核对（HUD 实时显示）。

另外演示两条手册口径的修正：
- **AO 只衰减间接光**，且镜面 IBL 用 ``mix(1, ao, 0.4)`` 弱化（全量压镜面会在
  缝隙里出现死黑）——A 键在 ``occlusion_strength=1.0 / 0.35`` 间切换可对照；
- **法线 mip 重归一化**：`from_pixels(role='normal')` 上传前做一次 XY 缩放 +
  归一化（规范顺序：先缩放后归一化）。

用法::

    python examples/texward.py
    python examples/texward.py --smoke 3
    python examples/texward.py --shot

按键：1/2/3 选材质（HUD 显示其账务）  N 法线开关  A AO 强度  D 调试通道
      SPACE 暂停  拖动旋转  滚轮缩放  ESC 退出

开销账务（HUD 实时）：纹素密度 px/m、RGBA8 与 BC7 的含 mip 显存、padding 推荐值。
"""
from __future__ import annotations

import math
import os
import tempfile

import pyglet
from pyglet.math import Vec3

from world3d.engine.app import EngineApp, run
from world3d.engine.camera import OrbitCamera
from world3d.engine.env import LightEnvironment
from world3d.engine.geometry import plane, sphere
from world3d.engine.material import Material
from world3d.engine.mesh import Mesh
from world3d.engine import proctex
from world3d.engine import texture as T
from world3d.engine.renderer import Renderer
from world3d.engine.scene import Scene
from world3d.engine.shaders import get

TITLE = 'texward · 织造（贴图全链路：文件 / sRGB / ORM 打包 / AO 分级）'
HELP = ('1/2/3 material  N normal  A ao-strength  D debug  SPACE pause  '
        'drag orbit  wheel zoom  ESC quit')
SHOTS = ((4.0, 'docs/texward_01.png'),
         (10.0, 'docs/texward_02.png'),
         (16.0, 'docs/texward_03.png'))

#: (proctex 图案, 标签, 落地方式)
SPECS = (
    ('marble', '大理石 · 文件通道(sRGB)', 'file'),
    ('brushed', '拉丝金属 · 内存通道(线性)', 'mem'),
    ('wood_planks', '木板 · ORM 单图三通道', 'orm'),
)

TEX_SIZE = 256
SPHERE_R = 0.9


class TexwardApp(EngineApp):
    TITLE = TITLE
    HELP = HELP
    SHOTS = SHOTS

    # ------------------------------------------------------------ 构建 --
    def setup(self):
        w, h = self.window.width, self.window.height
        self.env = LightEnvironment(
            sun_dir=(0.35, 0.85, 0.30), sun_color=(1.0, 0.94, 0.84),
            sun_intensity=3.2, ibl_intensity=0.95,
            sky_zenith=(0.10, 0.14, 0.26), sky_horizon=(0.34, 0.37, 0.44),
            sky_ground=(0.10, 0.10, 0.10)).bake()
        self.renderer = Renderer(w, h, environment=self.env)
        self.scene = Scene(name='texward', environment=self.env)
        pbr = get('pbr')

        ground = Mesh(plane(12.0, 6.0, 1, y=0.0), pbr,
                      Material('ground', (0.34, 0.33, 0.31), 0.0, 0.55))
        self.scene.add_mesh(ground, name='ground')

        # 三只球：三条贴图落地路径
        self.mats = []
        self.info = []
        for i, (pattern, label, mode) in enumerate(SPECS):
            mat, meta = self._build_material(pattern, mode, seed=i + 1)
            self.mats.append(mat)
            self.info.append((label, mode, meta))
            m = Mesh(sphere(SPHERE_R, segments=40, rings=26), pbr, mat)
            self.scene.add_mesh(m, name=f'sphere_{pattern}',
                                translation=(-2.4 + i * 2.4, SPHERE_R, 0.0))

        # 一盏暖点光做侧面补光（IBL 之外给贴图一点可读的高光）
        self.renderer.point_lights = [((3.5, 4.0, 4.5), (1.3, 1.05, 0.8))]

        self.camera = OrbitCamera(target=(0.0, 0.9, 0.0), yaw=28.0, pitch=12.0,
                                  dist=8.6, dist_range=(4.0, 16.0), fov=45.0)
        self.normal_on = True
        self._saved_normals = [m.normal_tex for m in self.mats]
        self.ao_strength = 1.0
        self.debug = 0
        self.selected = 0
        self.scene.update()             # 静态变换也要刷一次世界矩阵（否则全叠在原点）

    def _build_material(self, pattern, mode, seed):
        """proctex 图案 → Material（贴图按 role 上传；ORM 走单图三通道）。"""
        pat = proctex.PATTERNS[pattern](size=TEX_SIZE, seed=seed)
        size = pat['size']
        if mode == 'file':
            # 落盘 → 真实解码器 → sRGB 上传（文件通道）
            path = os.path.join(tempfile.gettempdir(), f'texward_{pattern}_base.png')
            pyglet.image.ImageData(size, size, 'RGBA', pat['base']).save(path)
            base = T.from_image_file(path, role='base')
            mr = T.from_pixels(size, size, pat['mr'], role='mr')
            occ = self._ao_texture(pat)
            meta = f'file:{os.path.basename(path)}'
        elif mode == 'mem':
            base = T.from_pixels(size, size, pat['base'], role='base')
            mr = T.from_pixels(size, size, pat['mr'], role='mr')
            occ = self._ao_texture(pat)
            meta = 'from_pixels(sRGB)'
        else:
            # ORM：AO(R) 从 height 派生 / 粗糙(G) 与金属(B) 取自 glTF MR 图
            orm = self._pack_orm(pat)
            base = T.from_pixels(size, size, pat['base'], role='base')
            mr = T.from_pixels(size, size, orm, role='orm')
            occ = mr                       # 同一张图同时作 MR 与 Occlusion
            meta = 'one ORM map → mr+occlusion'
        normal = T.from_pixels(size, size, pat['normal'], role='normal')
        m = Material(name=pattern, base_color=(1.0, 1.0, 1.0),
                     metallic=1.0, roughness=1.0,
                     base_tex=base, mr_tex=mr, normal_tex=normal,
                     occlusion_tex=occ)
        return m, meta

    @staticmethod
    def _ao_texture(pat):
        """由高度场派生 AO（凹处更暗）：255 - 高度的高频部分。"""
        size = pat['size']
        hmap = pat['height']
        mx = max(hmap) or 1
        ao = bytearray(size * size * 4)
        for i in range(size * size):
            v = 255 - int(255 * (hmap[i] / mx) ** 0.6 * 0.55)
            ao[i * 4:i * 4 + 4] = bytes((v, v, v, 255))
        return T.from_pixels(size, size, bytes(ao), role='occlusion')

    @staticmethod
    def _pack_orm(pat):
        """AO(R)+粗糙(G)+金属(B) 单图三通道（UE ORM 口径）。"""
        size = pat['size']
        mr, hmap = pat['mr'], pat['height']
        mx = max(hmap) or 1
        out = bytearray(size * size * 4)
        for i in range(size * size):
            ao = 255 - int(255 * (hmap[i] / mx) ** 0.6 * 0.55)
            out[i * 4 + 0] = ao
            out[i * 4 + 1] = mr[i * 4 + 1]      # G = roughness（glTF MR 同位置）
            out[i * 4 + 2] = mr[i * 4 + 2]      # B = metallic
            out[i * 4 + 3] = 255
        return bytes(out)

    # ------------------------------------------------------------ 每帧 --
    def update(self, dt):
        if self.paused:
            return
        self.camera.advance(dt * 0.25)          # 慢速环绕
        self.scene.update()

    def render(self):
        aspect = self.window.width / max(1, self.window.height)
        # uDebugMode 是 program 级 uniform：必须在 renderer 取用之前设好
        self.renderer.program('pbr')['uDebugMode'] = self.debug
        self.renderer.render(self.camera, self.scene, aspect=aspect,
                             window=self.window)

    # ------------------------------------------------------------ 交互 --
    def on_key(self, symbol, modifiers):
        k = pyglet.window.key
        if symbol in (k._1, k._2, k._3):
            self.selected = symbol - k._1
        elif symbol == k.N:
            self.normal_on = not self.normal_on
            if self.normal_on:
                for m, saved in zip(self.mats, self._saved_normals):
                    m.normal_tex = saved
            else:
                # Material 有 __slots__，不能在实例上挂临时属性——存回 App
                self._saved_normals = [m.normal_tex for m in self.mats]
                for m in self.mats:
                    m.normal_tex = None
        elif symbol == k.A:
            self.ao_strength = 0.35 if self.ao_strength > 0.9 else 1.0
            for m in self.mats:
                m.occlusion_strength = self.ao_strength
        elif symbol == k.D:
            self.debug = (self.debug + 1) % 6
        else:
            return pyglet.event.EVENT_UNHANDLED
        return None

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.camera.orbit(dx, dy)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.camera.zoom(scroll_y)

    # ------------------------------------------------------------ HUD --
    def status_text(self):
        label, mode, meta = self.info[self.selected]
        pat = proctex.PATTERNS[SPECS[self.selected][0]](size=32, seed=self.selected + 1)
        packed = T.unpack_packed(self._pack_orm(pat) if mode == 'orm'
                                 else pat['mr'], 'orm')
        bc7 = T.estimate_bytes(TEX_SIZE, TEX_SIZE, 'bc7') / 1e6
        dens = T.texel_density(TEX_SIZE, 2 * SPHERE_R)      # 球直径 1.8 m 的代尺
        dbg = ('final', 'albedo', 'normal', 'ibl-spec', 'direct', 'ibl-amb')[self.debug]
        # HUD 标题在左、状态在右，过长会压上去——只留最关键的四个数
        return (f'{SPECS[self.selected][2]} · {dens:.0f}px/m · BC7 {bc7:.2f}MB · '
                f'AO {packed["ao"]:.2f} R{packed["roughness"]:.2f} '
                f'M{packed["metallic"]:.2f} · ×{self.ao_strength:.2f} · '
                f'{"N" if self.normal_on else "-"}{dbg[0]} · {self.fps:.0f}fps')


if __name__ == '__main__':
    run(TexwardApp)
