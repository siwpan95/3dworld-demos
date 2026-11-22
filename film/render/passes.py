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
    """读一幕的 scene/shot，返回 (scene_data, shot)。供渲染器与校验脚本共用。"""
    scene_data = load_json(os.path.join(project_dir, act['scene']))
    shot = Shot.load(os.path.join(project_dir, act['shot']))
    return scene_data, shot


class ActRenderer:
    """一幕的渲染器：装环境 / 场景 / 相机 / 粒子，按 ``t`` 画一帧。

    ``t`` 是**幕内时间**（秒），不是全局时间 —— 幕间 xfade 由 `film.html`
    与 ffmpeg 负责，渲染层只管单幕的确定性推进。
    """

    def __init__(self, window, project_dir, act, fps=30, particle_count=5000):
        self.window = window
        self.act = act
        self.fps = float(fps)
        self.duration = float(act['duration'])
        w, h = window.width, window.height

        scene_data, self.shot = load_act(project_dir, act)

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

        # --- 粒子 ---
        self.particles = ParticleSystem(get('particles'), count=particle_count)

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

        # 粒子相位按**固定 dt** 推进（不是墙钟），这样同 t 必然同相位 ——
        # 确定性成片的地基。用墙钟会让"同一秒"在慢机器上少走几步。
        phase = self.particles.advance(1.0 / self.fps, energy)
        self._phase = phase

        def _draw_particles():
            """在场景之后、Bloom 之前画粒子。

            顺序理由：Bloom 是"对 scene_target 做阈值+模糊"，粒子必须已经写进
            target 才会发光；若画在 composite 之后，光点就变成贴上去的死点，
            起搏感全无。

            ⚠️ `prog.use()` 不能省：`ParticleSystem.set_uniforms()` 只写 uniform、
            **不绑定 program**（见其 docstring 的用法示例——示例里也没写 use，
            照抄就会得到"满屏没有粒子"）。而 `program['u_x'] = v` 写到未绑定的
            program 上不会响亮报错，只会静默丢值。
            """
            from pyglet.gl import (GL_BLEND, GL_ONE, GL_SRC_ALPHA, GL_TRUE,
                                   glBlendFunc, glDepthMask, glDisable, glEnable)
            prog = self.particles.program
            prog.use()
            self.particles.set_uniforms(time=self._phase, energy=energy,
                                        preset=0, size_scale=3.2)
            glDepthMask(False)          # 粒子自叠自，不写深度
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)     # 加性
            self.particles.draw()
            glDisable(GL_BLEND)
            glDepthMask(GL_TRUE)        # **必须还原**：否则下一帧深度清不掉

        aspect = self.window.width / max(1.0, float(self.window.height))
        self.renderer.render(self.camera, self.scene, aspect, self.window,
                             pre_post=_draw_particles)
        return energy, self.last_applied
