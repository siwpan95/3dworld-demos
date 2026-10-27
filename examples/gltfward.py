# -*- coding: utf-8 -*-
"""《匣境 / Gltfward》—— AI 资产管线 demo：外部 GLB 资产 → 3dworld 引擎渲染。

消费引擎的 glTF 2.0 加载栈（``world3d.engine.gltf.GLTF``），把**外部 GLB**
（含 AI 生成资产，如 TRELLIS.2 输出）加载进引擎：骨骼资产走蒙皮管线并驱动
**真实 glTF 动画**（非手写正弦），静态资产走普通 PBR 管线——
任何合法 GLB 都应能灌进来渲染。

默认资产：Khronos glTF-Sample-Assets 的 `Fox`（骨骼动画，3 个 clip）::

    python examples/gltfward.py
    python examples/gltfward.py --smoke 3
    python examples/gltfward.py --shot
    python examples/gltfward.py --glb PATH/to/model.glb   # 灌任意 GLB

资产许可（Fox，https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Fox）::

    模型      (c) 2014 PixelMannen          CC0 v1.0 Universal
    绑定+动画 (c) 2014 tomkranis            CC-BY 4.0 International
    glTF 转换 (c) 2017 @AsoboStudio & @scurest  CC-BY 4.0 International

按键：1/2/3=切换动画 clip  SPACE=暂停  R=重置  拖动=旋转视角  滚轮=缩放  ESC=退出
"""
from __future__ import annotations

import os
import sys

import pyglet                                                   # noqa: E402

from world3d.engine.app import EngineApp, run                   # noqa: E402
from world3d.platform.app import PROJECT_ROOT                   # noqa: E402
from world3d.engine.camera import OrbitCamera                   # noqa: E402
from world3d.engine.env import LightEnvironment                 # noqa: E402
from world3d.engine.gltf import GLTF                            # noqa: E402
from world3d.engine.material import PRESETS                     # noqa: E402
from world3d.engine.mesh import Mesh, SkinnedMesh               # noqa: E402
from world3d.engine.renderer import Renderer                    # noqa: E402
from world3d.engine.shaders import get                          # noqa: E402
from world3d.engine.skin import make_skin_from_gltf             # noqa: E402

GLB_PATH = os.path.join(PROJECT_ROOT, 'assets', 'gltfward_fox.glb')
# --glb PATH 覆盖默认资产（main() 里解析后经环境变量传入，见文件尾 main()）

TITLE = 'gltfward · 匣境（AI 资产管线 · 外部 GLB → 3dworld）'
HELP = ('1/2/3 clip  SPACE pause  R reset  drag orbit  wheel zoom  ESC quit')
SHOTS = ((0.8, 'docs/gltfward_01.png'),
         (1.6, 'docs/gltfward_02.png'),
         (2.4, 'docs/gltfward_03.png'),
         (3.2, 'docs/gltfward_04.png'))


class GltfwardApp(EngineApp):
    TITLE = TITLE
    HELP = HELP
    SHOTS = SHOTS

    # ------------------------------------------------------------ 构建 --
    def setup(self):
        w, h = self.window.width, self.window.height

        self.env = LightEnvironment(
            sun_dir=(0.42, 0.68, 0.38), sun_color=(1.0, 0.95, 0.86),
            sun_intensity=1.9, ibl_intensity=0.9,
            sky_zenith=(0.15, 0.33, 0.70), sky_horizon=(0.45, 0.61, 0.82),
            sky_ground=(0.10, 0.095, 0.09)).bake()
        self.renderer = Renderer(w, h, environment=self.env)

        # ---- 载入外部 GLB（默认 Fox，--glb 可换任意资产）----
        glb_path = os.environ.get('GLBWARD_GLB', GLB_PATH)
        self.glb_name = os.path.basename(glb_path)
        self.gltf = GLTF.load(glb_path)
        scene, pending, nmap = self.gltf.build_scene_ex()
        self.scene = scene
        self.scene.environment = self.env
        self.node_map = nmap
        # 只对 root 调一次：update_world 会递归（逐节点调会丢父链）
        scene.root.update_world(force=True)

        # 各节点 bind TRS（clip 切换 / R 重置时恢复；未覆盖通道保持 bind）
        self.bind_trs = {i: (n.translation, n.rotation, n.scale)
                         for i, n in nmap.items()}

        # ---- 蒙皮（静态资产无 skins[] 时走普通 Mesh 回退）----
        skins = self.gltf.skins()
        self.skin = make_skin_from_gltf(self.gltf, nmap, 0) if skins else None
        self.joints = self.skin.joints if self.skin else []

        # ---- 网格（每个 primitive 一个 Mesh/SkinnedMesh，挂到对应节点）----
        self._mats = {}
        prog_pbr = get('pbr')
        prog_skin = get('pbr_skin') if self.skin else None
        self.meshes = []
        for node, md, mat_idx in pending:
            mat = self._material(mat_idx)
            if self.skin is not None and md.skinned:
                m = SkinnedMesh(md, prog_skin, self.skin, mat)
            else:                                   # 静态 GLB 回退（AI 资产常态）
                m = Mesh(md, prog_pbr, mat)
            node.drawable = m
            self.meshes.append(m)
        self.tri_count = sum(m.data.triangle_count for m in self.meshes)
        # ⚠️ Scene.drawables() 有缓存——setup 里新挂的 drawable 必须 invalidate
        self.scene.invalidate()

        # ---- 相机自动取景：pending 几何按节点世界矩阵算 AABB ----
        center, size = self._frame_aabb(pending)
        self.Z_NEAR = max(0.05, size * 0.005)
        self.Z_FAR = max(500.0, size * 12.0)
        self.camera = OrbitCamera(target=center, yaw=36.0, pitch=14.0,
                                  dist=size * 1.8,
                                  dist_range=(size * 0.15, size * 6.0),
                                  fov=self.FOV, far=self.Z_FAR)

        # ---- 动画 clip（真实 glTF 动画驱动）----
        self.clips = self.gltf.animations()
        self.clip_i = 0
        self.t = 0.0
        self.paused = False
        self._sanity_print()
        self.scene.update()

    # ------------------------------------------------------------ 材质 --
    def _material(self, mat_idx):
        """gltf.material() 的 PBR 参数叠到预设克隆上（引擎不加载贴图槽）。

        资产没有贴图时 metallic=1 会渲染成白镜面，故钳到低金属度、中粗糙度，
        保证"无贴图也好看"。
        """
        if mat_idx in self._mats:
            return self._mats[mat_idx]
        m = PRESETS['emerald_light'].clone(name=f'gltfward_m{mat_idx}')
        m.emissive = (0.0, 0.0, 0.0)
        src = self.gltf.material(mat_idx)
        m.base_color = src.base_color
        m.metallic = min(src.metallic, 0.2)
        m.roughness = max(src.roughness, 0.45)
        m.alpha = src.alpha
        m.alpha_mode = src.alpha_mode
        m.alpha_cutoff = src.alpha_cutoff
        m.double_sided = src.double_sided
        self._mats[mat_idx] = m
        return m

    # -------------------------------------------------------- 取景 --
    def _frame_aabb(self, pending):
        """遍历 pending 几何（世界空间）→ (AABB 中心, 最长边)。"""
        INF = float('inf')
        mn = [INF, INF, INF]
        mx = [-INF, -INF, -INF]
        for node, md, _ in pending:
            wm = tuple(node.world_matrix)       # 列主序，与 Mesh.world_sphere 同式
            pos = md.positions
            for i in range(0, len(pos) - 2, 3):
                x, y, z = pos[i], pos[i + 1], pos[i + 2]
                wx = wm[0] * x + wm[4] * y + wm[8] * z + wm[12]
                wy = wm[1] * x + wm[5] * y + wm[9] * z + wm[13]
                wz = wm[2] * x + wm[6] * y + wm[10] * z + wm[14]
                for k, v in enumerate((wx, wy, wz)):
                    mn[k] = min(mn[k], v)
                    mx[k] = max(mx[k], v)
        if mn[0] is INF:
            return (0.0, 0.0, 0.0), 1.0
        size = max(mx[k] - mn[k] for k in range(3)) or 1.0
        return tuple((mn[k] + mx[k]) * 0.5 for k in range(3)), size

    # ------------------------------------------------------------ 动画 --
    def _sanity_print(self):
        """四元数序 sanity check（一次性，ASCII 输出）。

        glTF 存 (x,y,z,w)，引擎 Node 用 (w,x,y,z)。gltf.py 的 _as_value 在
        加载时已完成换序——这里对照 bind pose 打印留档，错序会在姿态上炸开。
        """
        if not self.clips:
            print('[gltfward] no animation clips (static GLB)')
            return
        for ni, path, sp in self.clips[0]['channels']:
            if path != 'rotation':
                continue
            q = sp.sample(0.0)
            node = self.node_map.get(ni)
            if node is None:
                break
            b = tuple(node.rotation)
            print('[gltfward] sanity rot node=%s bind=(%.4f, %.4f, %.4f, %.4f) '
                  't0=(%.4f, %.4f, %.4f, %.4f)  [wxyz]'
                  % (node.name, b[0], b[1], b[2], b[3],
                     q.w, q.x, q.y, q.z))
            break

    def _apply_clip(self, clip, t):
        """采样活动 clip 的全部通道并写入节点 TRS。

        未被通道覆盖的节点保持 bind TRS（规范行为）。rotation 通道的
        Sampler 输出已是引擎序 (w,x,y,z)（加载时换序，见 _sanity_print）。
        """
        for ni, path, sp in clip['channels']:
            node = self.node_map.get(ni)
            if node is None:
                continue
            v = sp.sample(t)
            if path == 'rotation':
                node.set_rotation((v.w, v.x, v.y, v.z))
            elif path == 'translation':
                node.set_translation((v.x, v.y, v.z))
            elif path == 'scale':
                node.set_scale((v.x, v.y, v.z))

    def _restore_bind(self):
        for i, (t0, r0, s0) in self.bind_trs.items():
            n = self.node_map[i]
            n.translation = t0
            n.rotation = r0
            n.scale = s0
            n.mark_dirty()

    def _switch_clip(self, idx):
        if not (0 <= idx < len(self.clips)):
            return
        self.clip_i = idx
        self.t = 0.0
        self._restore_bind()
        self._apply_clip(self.clips[idx], 0.0)
        self.scene.root.update_world()

    # ------------------------------------------------------------ 每帧 --
    def update(self, dt):
        if self.paused:
            return
        if self.clips:
            clip = self.clips[self.clip_i]
            if clip['duration'] > 0:
                self.t = (self.t + dt) % clip['duration']
            self._apply_clip(clip, self.t)
        # 关节世界矩阵与蒙皮矩阵每帧刷新（SkinnedMesh.draw 里读）
        self.scene.root.update_world()

    def render(self):
        aspect = self.window.width / max(1, self.window.height)
        self.renderer.render(self.camera, self.scene, aspect=aspect,
                             window=self.window)

    # ------------------------------------------------------------ 交互 --
    def on_key(self, symbol, modifiers):
        key = pyglet.window.key
        if symbol == key.R:
            self.t = 0.0
            self._restore_bind()
            if self.clips:
                self._apply_clip(self.clips[self.clip_i], 0.0)
            self.scene.root.update_world()
        elif key._1 <= symbol <= key._9:
            self._switch_clip(symbol - key._1)
        else:
            return pyglet.event.EVENT_UNHANDLED
        return None

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.camera.orbit(dx, dy)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.camera.zoom(scroll_y)

    def status_text(self):
        clips = ' | '.join(
            ('*' if i == self.clip_i else '') + (c['name'] or f'clip{i}')
            for i, c in enumerate(self.clips)) or '(static)'
        state = '暂停' if self.paused else '播放'
        return (f'{self.glb_name} · 三角 {self.tri_count} · '
                f'关节 {len(self.joints)} · clip [{clips}] · '
                f'{state} t={self.t:.2f}s · {self.fps:.0f}fps')


def main(argv=None):
    """--glb PATH 在这里预解析（engine.app.parse_args 不认识它），
    剩余参数原样交给 run()。资产路径经环境变量传入 App。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--glb' in argv:
        i = argv.index('--glb')
        if i + 1 >= len(argv):
            raise SystemExit('gltfward: --glb needs a PATH argument')
        os.environ['GLBWARD_GLB'] = os.path.abspath(argv[i + 1])
        del argv[i:i + 2]
    return run(GltfwardApp, argv)


if __name__ == '__main__':
    sys.exit(main())
