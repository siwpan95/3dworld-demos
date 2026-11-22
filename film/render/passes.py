# -*- coding: utf-8 -*-
"""《造界 / Worldwright》渲染趟 —— 一帧 = 场景（声明式）+ 粒子（程序化）+ 后处理。

分工的理由
----------
``scenes/*.scene.json`` 只表达**静态世界**：节点树、材质、环境参数。它能被
``scene_build`` 校验、能 diff、能被 `scene_viewer` 热重载。但有两类东西它表达不了：

* **GPU 粒子**：不是 ``Mesh``，没有材质，走自己的 program + instancing；
* **"哪一趟在哪个顺序"**：CSM / 体积雾 / DoF / Bloom 的先后是工程决策，
  不是场景数据。

所以本模块是"渲染清单"的执行者：它读 scene/shot，按 `draw()` 的固定顺序把活儿派下去。
顺序一旦错了，症状会很隐蔽（Bloom 照不到粒子、雾把 DoF 深度冲掉），
故这里的注释只写"为什么是这个顺序"。
"""
import math
import os

from world3d.engine.camera import FlyCamera
from world3d.engine.env import LightEnvironment
from world3d.engine.particles import ParticleSystem
from world3d.engine.renderer import Renderer
from world3d.engine.scene_build import build_scene, load_json
from world3d.engine.shaders import get
from world3d.engine.shot import Shot, ShotPlayer


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def void_energy(t, dur=24.0):
    """第 I 幕的"起搏"曲线：0.10 → 1.00，后段加速。

    **音画同源**：`film/audio/score.py` 读同一条曲线决定配乐密度，
    所以这里只留纯函数、不带状态 —— 换幕时不需要同步任何计数器。
    """
    u = clamp01(t / max(1e-6, dur))
    return 0.10 + 0.90 * (u ** 2.2)


#: 幕 id → 能量曲线。没登记的幕一律按"恒定 1.0"处理（不猜艺术意图）。
ENERGY = {
    'void': void_energy,
}


def load_act(project_dir, act):
    """读一幕的 scene/shot，返回 (scene_data, shot)。供渲染器与校验脚本共用。

    ``project.json`` 里 12 幕**先全部登记**（时长要能算总片长），未建模的幕
    ``scene``/``shot`` 为 ``null``。这里给出可读的失败信息 —— 否则
    ``os.path.join(dir, None)`` 会抛一个跟"还没建"毫无关系的 TypeError。
    """
    if not act.get('scene') or not act.get('shot'):
        built = '（已建好的幕：status == "built"）'
        raise SystemExit(f'幕 {act["id"]}「{act.get("title", "")}」尚未建模：'
                         f'project.json 里 scene/shot 为 null。{built}')
    scene_data = load_json(os.path.join(project_dir, act['scene']))
    shot = Shot.load(os.path.join(project_dir, act['shot']))
    return scene_data, shot


class ActRenderer:
    """一幕的渲染器：装环境 / 场景 / 相机 / 粒子，按 ``t`` 画一帧。

    ``t`` 是**幕内时间**（秒），不是全局时间 —— 幕间 xfade 由 `film.html`
    与 ffmpeg 负责，渲染层只管单幕的确定性推进。
    """

    def __init__(self, window, project_dir, act, fps=30, particle_count=None):
        self.window = window
        self.act = act
        self.fps = float(fps)
        self.duration = float(act['duration'])
        w, h = window.width, window.height

        scene_data, self.shot = load_act(project_dir, act)

        # --- 水面（Gerstner，引擎 0.8.0）---
        # 幕里写 "water": {"preset": 0, "size": 26, "segs": 32, "y": 2.15}
        wcfg = act.get('water') or {}
        self.water = None
        if wcfg:
            from world3d.engine.water import WaterPlane
            from world3d.logic.waves import scaled_preset
            waves = scaled_preset(int(wcfg.get('preset', 0)))[1]
            self.water = WaterPlane(waves,
                                    size=float(wcfg.get('size', 160.0)),
                                    segs=int(wcfg.get('segs', 32)),
                                    y=float(wcfg.get('y', 0.0)))

        # --- 粒子：**按幕配置**，不是全局常量 ---
        # 第 I 幕是"星尘"，第 II 幕是"地形"，硬画粒子会把上一幕的视觉语言带进来
        # （实测：第 II 幕探针里星盘浮在地形上，看着像两幕叠在一起）。
        # project.json 的幕里写 "particles": {"count": N, "preset": P}；缺省 = 无粒子。
        pcfg = act.get('particles') or {}
        n = pcfg.get('count', 0) if particle_count is None else particle_count
        self.particle_preset = int(pcfg.get('preset', 0))
        self.size_scale = 3.2 * (540.0 / max(1, window.height)) ** 0.55
        self.particles = ParticleSystem(get('particles'), count=n) if n > 0 else None

        # --- 环境：scene JSON 的 environment 段 ↔ LightEnvironment 字段同名 ---
        self.env = LightEnvironment(**(scene_data.get('environment') or {})).bake()

        # --- 场景：空 root 也合法（第 I 幕"空无"就没有几何）---
        self.scene = build_scene(scene_data, get('pbr'))

        # --- 相机：**FlyCamera**，不是 OrbitCamera ---
        # `ShotPlayer._apply_camera` 只写回 position / target / fov（yaw/pitch/dist
        # 会被它直接 return False）；而 OrbitCamera 从 0.7.4 起对 position 赋值无效
        # （眼位只由 yaw/pitch/dist 决定）。两者相乘 = 用 OrbitCamera 接通道会静默不动。
        self.camera = FlyCamera(position=(0.0, 0.55, 26.0), fov=52.0, far=400.0)

        # --- 渲染器 + 后处理 ---
        self.renderer = Renderer(w, h, environment=self.env)
        post = self.renderer.post.settings
        post.bloom_enabled = True
        # 空无幕没有几何：SSAO 只会白耗一遍全屏 pass，关掉（有几何的幕再按需开）
        post.ssao_enabled = bool(scene_data.get('root', {}).get('children'))

        # --- 演出 ---
        self.player = ShotPlayer(self.shot, scene=self.scene,
                                 camera=self.camera, env=self.env)
        self.energy_fn = ENERGY.get(act.get('energy', ''), lambda t, d: 1.0)
        self.last_applied = 0

    # ------------------------------------------------------------------ 画
    def draw(self, t):
        """按固定顺序画一帧。返回 (energy, applied_channels) 便于渲染器打日志。"""
        energy = self.energy_fn(t, self.duration)
        self.last_applied = self.player.apply(t)

        # 相机朝向由引擎 0.7.8 的 FlyCamera.look_at 处理（ShotPlayer 的 target
        # 通道现在会反解 yaw/pitch）。此前这里有一层项目内 _aim() 兜底 ——
        # 反哺落地后必须删掉，否则会变成"第二套实现"（同 shot_path 那次的教训）。

        # 粒子相位按**固定 dt** 推进（不是墙钟），这样同 t 必然同相位 ——
        # 确定性成片的地基。用墙钟会让"同一秒"在慢机器上少走几步。
        pre_post = None
        if self.particles is not None:
            self._phase = self.particles.advance(1.0 / self.fps, energy)

            def _draw_particles():
                """在场景之后、Bloom 之前画粒子。

                顺序理由：Bloom 是"对 scene_target 做阈值+模糊"，粒子必须已经写进
                target 才会发光；若画在 composite 之后，光点就变成贴上去的死点，
                起搏感全无。

                ⚠️ `prog.use()` 不能省：`ParticleSystem.set_uniforms()` 只写 uniform、
                **不绑定 program**（见其 docstring 的用法示例——示例里也没写 use，
                照抄就会得到"满屏没有粒子"）。机制链路（2026-11-22 逐层查证）：

                `program['u_x'] = v` → `Uniform.set()` → `glUniform*`，作用于**当前绑定的
                program**。而平台层为了性能关掉了 pyglet 的逐调用 `glGetError` 巡检
                （见 `platform/app.py` 的注释：它占每帧 Python 耗时 ~44%），
                于是"program 未绑定 → uniform 写到 0 号 → 驱动忽略"这一路
                **既不抛异常、也不 warn**。后果不是"画得淡"，而是 `u_size_scale`
                停在默认 0 ⇒ **整批粒子尺寸为 0 ⇒ 一颗都看不见**。
                """
                from pyglet.gl import (GL_BLEND, GL_ONE, GL_SRC_ALPHA, GL_TRUE,
                                       glBlendFunc, glDepthMask, glDisable, glEnable)
                prog = self.particles.program
                prog.use()
                self.particles.set_uniforms(time=self._phase, energy=energy,
                                            preset=self.particle_preset,
                                            size_scale=self.size_scale)
                glDepthMask(False)          # 粒子自叠自，不写深度
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE)     # 加性
                self.particles.draw()
                glDisable(GL_BLEND)
                glDepthMask(GL_TRUE)        # **必须还原**：否则下一帧深度清不掉

            pre_post = _draw_particles

        if self.water is not None:
            _prev_pp = pre_post

            def _pre_post_all():
                # 水是**不透明**的（自带色调映射），必须先于加性粒子画：
                # 反过来的话粒子会被水的深度写入挡掉。
                self.water.draw(t, self.camera.position, self.env)
                if _prev_pp is not None:
                    _prev_pp()

            pre_post = _pre_post_all

        aspect = self.window.width / max(1.0, float(self.window.height))
        self.renderer.render(self.camera, self.scene, aspect, self.window,
                             pre_post=pre_post)
        return energy, self.last_applied
