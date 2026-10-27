# -*- coding: utf-8 -*-
"""《粒子境 / Particleward》—— 引擎模块 engine.particles 的 GPU 粒子渲染首秀。

**为什么值得单独一个 demo**
----------------------------
``world3d.engine.particles`` 把「GPU 粒子」这项能力沉到了引擎侧，但演示仓里
还没有任何一个 demo 消费过它（旧 ``examples/particles_gpu.py`` 是手搓
``vertex_list_instanced``，现已下线）。本 demo 通过**模块公共 API** 驱动整场：

    PRESETS                           # 预设表：(名字, 相机距离, 注视点高度)
    base_quad()                       # 基础公告板四边形几何（位置/颜色）
    make_instance_data(count, seed)   # 每实例种子与基色（固定种子 ⇒ 可复现）
    PARTICLES_VERT / PARTICLES_FRAG   # 引擎自带 GLSL 330 着色器
    ParticleSystem                    # 引擎封装的 pyglet 原生实例化（本机不可用，见下）

**本机实测记录（为什么绘制走裸 GL，而不是引擎的 ParticleSystem）**
--------------------------------------------------------------------
本机（RTX 3060 / OpenGL 4.6 / pyglet 2.1.16）上，引擎 ``ParticleSystem`` 的
pyglet 原生 ``vertex_list_instanced`` + 逐个 ``add_instance`` 路径不可用：
12000 实例时整屏被加性混合刷成白色、~7 fps；把同一套数据/着色器换到
pyglet 普通 ``vertex_list``（每实例 6 顶点手动展开、逐 attribute 校验过
GPU buffer 内容与 CPU 完全一致）依然满屏巨大三角形。用裸 GL 的
transform feedback 捕获同一着色器/数据/矩阵的 ``gl_Position``，18 个顶点
全部有限且在视锥内——顶点阶段没问题，问题出在 pyglet 的 domain/VAO
绘制路径本身。同机其余 47 个 demo 的普通绘制全部正常，唯独这条
「4 attribute + 大顶点量」的路径异常，故判定为该组合下的驱动/封装缺陷。

规避方式：保留引擎**着色器源码与全部公共数据 API**，绘制层用裸 GL
（自建 VAO/VBO/program/UBO）。粒子的位置/大小/颜色仍然 100% 在顶点
着色器里由 ``i_seed`` + 累加相位 ``u_time`` 现推——CPU 每帧仍只写
几个标量，粒子数翻倍几乎不增加 CPU 占用，与引擎设计的意图一致。

**用到的引擎 API**
------------------
* ``PARTICLES_VERT / PARTICLES_FRAG``：着色器逐字采用（仅把 ``WindowBlock``
  UBO 适配成两个普通 mat4 uniform，动画/分布/着色公式一行未动）。
* ``base_quad()``：公告板基础几何（6 顶点），逐粒子平铺成顶点缓冲。
* ``make_instance_data(count, seed)``：每实例 4 个 [0,1) 种子 + 基色，
  平铺到 6 顶点（每顶点复制一份——「手动展开实例化」）。
* ``PRESETS``：三种效果与配套取景。

**三种效果（对应着色器 u_preset 分支，按 ``P`` 切换）**
-----------------------------------------------------
* ``0 星系漩涡`` —— 盘状分布 + 差速旋转 + 固定世界坐标的旋臂。
* ``1 喷泉烟花`` —— 抛物线弹道 + 重力 + 寿命循环（分 3 组错峰发射）。
* ``2 湍流云``   —— 多频正弦伪噪声位移 + 极慢自转。

**关键渲染约定（坑位，已在模块 docstring 钉死）**
-------------------------------------------------
* 加性混合自己开：``glEnable(GL_BLEND)`` + ``glBlendFunc(GL_SRC_ALPHA, GL_ONE)``；
  画粒子时 ``glDepthMask(GL_FALSE)``，画完**必须**还原成 ``GL_TRUE``（否则基类
  ``window.clear()`` 连深度都清不掉）。
* 用累加相位而非 ``u_time * energy``，按键瞬间画面不闪跳（只改推进速率）。
* 能量守恒式 alpha 补偿：``u_alpha_scale = 1/size_scale²``，放大后单颗变淡，
  否则加性混合直接饱和成实心色团。

**HUD（status_text）读的内容**
------------------------------
粒子总数（``ps.count``）· 当前预设名 · 活跃粒子数（按着色器 life/alpha
公式在 CPU 估算的可见实例数）· 帧率。另附能量与尺寸倍率。

**控制键**
----------
``R`` 重置/重新发射（相位归零并恢复默认取景）   ``SPACE`` 暂停/继续
``P`` 切换预设（0/1/2 三态循环）                ``W`` 线框（公告板四边形轮廓）
``+``/``-`` 能量   ``[``/``]`` 尺寸倍率   拖拽/滚轮 轨道相机   ``ESC`` 退出

用法::

    python examples/particleward.py                 正常运行
    python examples/particleward.py --smoke 2       冒烟
    python examples/particleward.py --shot          抓剧照到 docs/particleward_01/02.png
"""
import ctypes
import math

import numpy as np
import pyglet
from pyglet.gl import (GL_ARRAY_BUFFER, GL_BLEND, GL_COLOR_BUFFER_BIT,
                       GL_COMPILE_STATUS, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
                       GL_FALSE, GL_FILL, GL_FLOAT, GL_FRAGMENT_SHADER,
                       GL_FRONT_AND_BACK, GL_LINE, GL_LINK_STATUS, GL_ONE,
                       GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_STATIC_DRAW,
                       GL_TRIANGLES, GL_TRUE, GL_VERTEX_SHADER,
                       glAttachShader, glBufferData, glCreateProgram,
                       glCreateShader, glCompileShader, glDisableVertexAttribArray,
                       glDrawArrays, glEnableVertexAttribArray,
                       glGenBuffers, glGenVertexArrays,
                       glGetAttribLocation, glGetBufferSubData,
                       glGetProgramInfoLog, glGetProgramiv,
                       glGetShaderInfoLog, glGetShaderiv,
                       glGetUniformLocation, glUniform1f, glUniform1i,
                       glUniformMatrix4fv, glLinkProgram, glShaderSource,
                       glUseProgram, glVertexAttribPointer,
                       glBlendFunc, glClear, glClearColor, glDepthMask,
                       glEnable, glPolygonMode, glBindVertexArray,
                       glBindBuffer)
from pyglet.math import Mat4, Vec3
from pyglet.window import key

# ⚠️ 不要再加 sys.path 引导：引擎由 `pip install 3dworld` 提供（见 requirements.txt）。
# ⚠️ 禁止 import world3d.logic.noise_vec（不在 wheel 里）。
from world3d.engine.particles import (BASE_VERTS, PARTICLES_FRAG,             # noqa: E402
                                      PARTICLES_VERT, PRESETS, base_quad,
                                      make_instance_data)
from world3d.logic.camera import OrbitCamera                         # noqa: E402
from world3d.platform.app import DemoApp, orbit_camera_input, run     # noqa: E402

# 默认粒子数：4.5 万颗在加性混合下必然糊成一片、单颗软边无从谈起；
# 降到 ~1.2 万 + 放大尺寸（size_scale），才看得到「圆点 + 高斯软边」。
COUNT = 12000
SEED = 20240607                  # 固定种子 ⇒ 截图可复现

# 引擎着色器用 pyglet 的 WindowBlock UBO；裸 GL 路径下适配成两个普通
# mat4 uniform——动画/分布/着色公式一字未动。
_ADAPTED_VERT = (PARTICLES_VERT
                 .replace('uniform WindowBlock { mat4 projection; mat4 view; } window;',
                          'uniform mat4 projection; uniform mat4 view;')
                 .replace('window.projection', 'projection')
                 .replace('window.view', 'view'))


def _compile(src, typ):
    sh = glCreateShader(typ)
    buf = ctypes.c_char_p(src.encode())
    ptr = (ctypes.POINTER(ctypes.c_char) * 1)(
        ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    glShaderSource(sh, 1, ptr, None)
    glCompileShader(sh)
    ok = ctypes.c_int(0)
    glGetShaderiv(sh, GL_COMPILE_STATUS, ctypes.byref(ok))
    if not ok.value:
        log = ctypes.create_string_buffer(4096)
        glGetShaderInfoLog(sh, 4096, None, log)
        raise RuntimeError(f'shader compile failed: {log.value!r}')
    return sh


# ------------------------------------------------------------------- app --
class FlatParticles:
    """引擎粒子系统的裸 GL 绘制层（原因见模块 docstring 的实测记录）。

    数据层完全来自引擎公共 API：``base_quad`` 给公告板几何，
    ``make_instance_data`` 给每实例种子/基色，逐粒子把 4 维种子复制到
    6 个基础顶点（「手动展开实例化」），一次 ``glDrawArrays`` 画完全部。
    CPU 逐帧只写 5 个 uniform，与引擎 ``ParticleSystem`` 的设计意图一致。
    """

    def __init__(self, count, seed=20240607):
        # ---- program：引擎着色器（仅 WindowBlock→两个 mat4 的适配） ----
        self.prog = glCreateProgram()
        glAttachShader(self.prog, _compile(_ADAPTED_VERT, GL_VERTEX_SHADER))
        glAttachShader(self.prog, _compile(PARTICLES_FRAG, GL_FRAGMENT_SHADER))
        glLinkProgram(self.prog)
        ok = ctypes.c_int(0)
        glGetProgramiv(self.prog, GL_LINK_STATUS, ctypes.byref(ok))
        if not ok.value:
            log = ctypes.create_string_buffer(4096)
            glGetProgramInfoLog(self.prog, 4096, None, log)
            raise RuntimeError(f'link failed: {log.value!r}')

        # ---- 数据：引擎 API 给基础几何与每实例种子，逐粒子平铺 ----
        pos, col = base_quad()                       # 6 顶点角点 [-1,1]^2
        seeds, cols = make_instance_data(count, seed=seed)
        n = int(count)
        self.count = n
        v = BASE_VERTS
        quad_pos = np.asarray(pos, dtype=np.float32)
        quad_col = np.asarray(col, dtype=np.float32)
        seeds_a = np.asarray(seeds, dtype=np.float32).reshape(n, 4)
        cols_a = np.asarray(cols, dtype=np.float32).reshape(n, 4)
        flat_pos = np.tile(quad_pos, n)                        # n×18 floats
        flat_col = np.tile(quad_col, n)                        # n×24 floats
        flat_seed = np.repeat(seeds_a, v, axis=0).reshape(-1)  # n*6×4
        flat_iclr = np.repeat(cols_a, v, axis=0).reshape(-1)   # n*6×4

        # ---- GL 资源：VAO + 4 VBO ----
        self.vao = (ctypes.c_uint * 1)()
        glGenVertexArrays(1, self.vao)
        self.vbos = (ctypes.c_uint * 4)()
        glGenBuffers(4, self.vbos)
        loc = {name: glGetAttribLocation(self.prog, name.encode())
               for name in ('position', 'colors', 'i_seed', 'i_color')}
        glBindVertexArray(self.vao[0])
        for i, (arr, name, size) in enumerate((
                (flat_pos, 'position', 3), (flat_col, 'colors', 4),
                (flat_seed, 'i_seed', 4), (flat_iclr, 'i_color', 4))):
            glBindBuffer(GL_ARRAY_BUFFER, self.vbos[i])
            glBufferData(GL_ARRAY_BUFFER, arr.nbytes,
                         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         GL_STATIC_DRAW)
            glEnableVertexAttribArray(loc[name])
            glVertexAttribPointer(loc[name], size, GL_FLOAT,
                                  GL_FALSE, 0, None)
        glBindVertexArray(0)
        self.loc = {k: glGetUniformLocation(self.prog, k.encode())
                    for k in ('u_time', 'u_energy', 'u_preset',
                              'u_size_scale', 'u_alpha_scale')}

    def set_uniforms(self, time, energy=1.0, preset=0, size_scale=1.0):
        """CPU 侧全部工作：把每帧标量写进着色器。"""
        glUseProgram(self.prog)
        glUniform1f(self.loc['u_time'], float(time))
        glUniform1f(self.loc['u_energy'], float(energy))
        glUniform1i(self.loc['u_preset'], int(preset) % len(PRESETS))
        glUniform1f(self.loc['u_size_scale'], float(size_scale))
        # 能量守恒式缩放：面积涨 k² 倍，单颗亮度就除以 k²
        glUniform1f(self.loc['u_alpha_scale'], 1.0 / (size_scale * size_scale))

    def draw(self, projection, view):
        """projection/view 传 pyglet Mat4（demo 相机每帧可动）。"""
        f32 = np.asarray(list(projection) + list(view),
                         dtype=np.float32).ctypes.data_as(
                             ctypes.POINTER(ctypes.c_float))
        glUseProgram(self.prog)
        glUniformMatrix4fv(glGetUniformLocation(self.prog, b'projection'),
                           1, GL_FALSE, f32)
        v16 = np.ascontiguousarray(np.asarray(list(view), dtype=np.float32))
        glUniformMatrix4fv(glGetUniformLocation(self.prog, b'view'),
                           1, GL_FALSE, v16.ctypes.data_as(
                               ctypes.POINTER(ctypes.c_float)))
        glBindVertexArray(self.vao[0])
        glDrawArrays(GL_TRIANGLES, 0, self.count * BASE_VERTS)
        glBindVertexArray(0)


class Particleward(DemoApp):
    TITLE = 'PARTICLEward - GPU particles (engine.particles)'
    HELP = ('R reset  SPACE pause  P preset  W wire  '
            '+/- energy  [ ] size  drag/scroll  ESC')
    DURATION = None              # 循环演示（smoke/shot 按各自计时器收尾）
    SHOTS = ((6.0, 'docs/particleward_01.png'),
             (16.0, 'docs/particleward_02.png'))
    FOV = 60.0

    def setup(self):
        glEnable(GL_DEPTH_TEST)
        # 加性混合：粒子只叠加亮度、不互相遮挡（坑位：画完要还原 depth mask）。
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        # 直接调用模块公共 API 的实例数据（HUD 估算活跃数用）。
        self.seeds, _ = make_instance_data(COUNT, seed=SEED)

        # 「手动展开实例化」版粒子系统（原因见模块 docstring 的实测记录）。
        self.ps = FlatParticles(COUNT, seed=SEED)
        self.count = self.ps.count

        self.preset = 0
        self.energy = 1.0
        self.size_scale = 1.4         # 放大到 1.4 倍，软边才看得清
        self.phase = 0.0
        self.active = self.count

        self.cam = OrbitCamera(target=Vec3(0.0, 1.6, 0.0),
                               yaw=35.0, pitch=22.0, dist=21.0)
        self.on_mouse_drag, self.on_mouse_scroll = orbit_camera_input(self, self.cam)
        self._apply_preset(0)

    # ------------------------------------------------------------ 预设 --
    def _apply_preset(self, i):
        self.preset = i % len(PRESETS)
        _, dist, ty = PRESETS[self.preset]
        self.cam.set_target((0.0, ty, 0.0))
        self.cam.dist = dist

    # ------------------------------------------------------------ 交互 --
    def on_key(self, symbol, modifiers):
        k = key
        if symbol == k.R:                       # 重置/重新发射：相位归零 + 恢复取景
            self.phase = 0.0
            self._apply_preset(self.preset)
        elif symbol == k.P:                     # 切换预设（0/1/2 循环）
            self._apply_preset(self.preset + 1)
            print(f'[particleward] preset={self.preset} '
                  f'name={PRESETS[self.preset][0]}')
        elif symbol == k.W:                     # 线框（公告板四边形轮廓）
            self.wire = not getattr(self, 'wire', False)
        elif symbol in (k.PLUS, k.EQUAL, k.NUM_ADD):
            self.energy = min(3.0, self.energy + 0.2)
        elif symbol in (k.MINUS, k.NUM_SUBTRACT):
            self.energy = max(0.2, self.energy - 0.2)
        elif symbol == k.BRACKETRIGHT:
            self.size_scale = min(5.0, self.size_scale + 0.5)
        elif symbol == k.BRACKETLEFT:
            self.size_scale = max(0.5, self.size_scale - 0.5)

    # ------------------------------------------------------------ 更新 --
    def update(self, dt):
        if self.paused:
            return
        # 能量只改推进速率（累加相位），不闪跳（与引擎 advance 同一公式）。
        self.phase += dt * (0.5 + 0.5 * self.energy)
        self.active = self._estimate_active(self.phase)

    def _estimate_active(self, phase):
        """按着色器 life/alpha 公式在 CPU 估算「本帧可见」的粒子数。

        预设 0（星系）与 2（湍流云）的 alpha 恒 > 0 ⇒ 全部活跃；
        预设 1（喷泉）的 alpha 随寿命衰减到 0（落回地面那刻），故需逐实例算。
        1.2 万次纯 Python 算术 / 帧，远低于一帧预算，只为给 HUD 一个真数字。
        """
        if self.preset != 1:
            return self.count
        p = phase * 0.45                  # 与着色器 rate=0.45 一致
        n = 0
        for s in self.seeds:
            grp = math.floor(s[2] * 3.0)
            life = (p + grp * 0.3333 + s[2] * 0.05) % 1.0
            alpha = 0.45 * (1.0 - life) * (1.0 - life)
            if alpha > 0.02:
                n += 1
        return n

    # ------------------------------------------------------------ 渲染 --
    def render(self):
        w, h = self.window.width, self.window.height
        glClearColor(0.02, 0.02, 0.05, 1.0)   # 深空底色，衬托加性混合的粒子
        # 直接清颜色+深度（基类已清过一次，这里保证暗背景）。
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        projection = Mat4.perspective_projection(
            w / max(1, h), z_near=self.Z_NEAR, z_far=self.Z_FAR, fov=self.FOV)
        view = self.cam.view_matrix()
        self.window.projection = projection
        self.window.view = view

        # 加性混合 + 不写深度（画完还原，否则下一帧深度清不掉）。
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glDepthMask(GL_FALSE)
        if getattr(self, 'wire', False):
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        self.ps.set_uniforms(time=self.phase, energy=self.energy,
                             preset=self.preset, size_scale=self.size_scale)
        self.ps.draw(projection, view)

        glDepthMask(GL_TRUE)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        # 还原成普通 alpha 混合，避免干扰基类 HUD 绘制。
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.window.projection = self.perspective

    # ------------------------------------------------------------ 状态 --
    def status_text(self):
        # HUD 用英文：基类 HUD 对中文逐帧 layout 开销大（实测掉到个位数 fps）
        # 且字形会叠影，全仓其余 demo 的 HUD 也都是英文。
        name_en = ('galaxy vortex', 'fountain fireworks', 'turbulence cloud')[
            min(self.preset, 2)]
        return (f'particles {self.count}  preset {self.preset + 1}/{len(PRESETS)}'
                f' {name_en}  active {self.active}'
                f'  energy {self.energy:.1f}  size x{self.size_scale:.1f}'
                f'  {self.fps:5.1f} fps'
                f'{"  [PAUSED]" if self.paused else ""}')


if __name__ == '__main__':
    run(Particleward)
