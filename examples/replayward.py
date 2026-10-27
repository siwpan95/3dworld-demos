# -*- coding: utf-8 -*-
"""《回响 / Replayward》—— 确定性仿真的录制 / 回放 / 确定性自检（logic/replay 首秀）。

**为什么值得单独一个 demo**
----------------------------
``world3d.logic.replay`` 提供三件套：

* ``Replay`` —— 一份档案的可重演视图：事件流（``events``，记"世界变成了什么样"）
  与动作序列（``actions``，记"当时控制器提议了什么"）**双轨**记录 + 元数据
  （``meta``），并自带 ``timeline / narrative / filter / summary / actions_of``
  等浏览 API，以及 ``to_dict / from_dict / save_json / load_json`` 的 JSON 往返。
* ``save_replay(replay, path)`` —— 把档案的**战报**导出成纯文本（``timeline()``
  逐行），便于 diff / 归档 / 分享。
* ``verify_determinism(run_fn, count=2)`` —— 工厂无参可调用 ``run_fn`` 跑
  ``count`` 遍，逐项深度比较，返回 ``{"identical", "runs", "first_diff"}``，
  回答"这局能不能精确复现"。

本 demo 把"录制 → 回放 → 证明它精确复现"做成**看得见的**：

1. **录制**：一个**确定性**的小仿真（N 个粒子在 seeded RNG + 固定步长下，受一个
   由时间 + 种子相位驱动的'吸引子目标'（这就是每帧的**输入**）牵引并带阻尼）。
   跑满 ``FRAMES`` 帧，把每帧的**状态**（粒子坐标）写进 ``replay.events``、
   把每帧的**输入**（吸引子目标）写进 ``replay.actions``、把运行参数写进
   ``replay.meta``。
2. **回放**：直接遍历 ``replay.events``，把录下的坐标重新画出来——画面可以切
   "实时重算 / 回放 / 对照（左实时、右回放）"三视图（``T`` 键切换）。
3. **证明**：用 ``verify_determinism`` 跑两遍同种子的整条轨迹，证明仿真本身
   确定性；再用 ``Replay.from_dict(replay.to_dict())`` 走一遍 JSON 往返，把
   回放坐标与原始轨迹逐帧比对（``roundtrip_err``）；运行时还把"实时重算"与
   "回放"在当前帧逐粒子比对（``live-vs-replay``）。三者都应 = 0。

**本 demo 严格确定性**：固定种子 + 固定步长 + 不用墙钟，所以 verify_determinism
与"实时==回放"才成立。

用法::
    python examples/replayward.py                 正常运行
    python examples/replayward.py --smoke 3       冒烟
    python examples/replayward.py --shot          抓剧照到 docs/replayward_01/02.png

按键：R 重新录制（换种子重跑）   SPACE 暂停   T 切 实时/回放/对照视图
      drag/scroll 转视角/缩放   ESC 退出

HUD（status_text）读了什么：当前视图模式、帧号、录制种子、verify_determinism 的
PASS/FAIL、JSON 往返误差 roundtrip_err、运行时"实时 vs 回放"逐帧最大误差。
"""
import colorsys
import os
import tempfile

import numpy as np
from pyglet.gl import (GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
                       GL_LINES, GL_POINTS, GL_PROGRAM_POINT_SIZE,
                       GL_SCISSOR_TEST, glClear, glClearColor, glDisable,
                       glEnable, glScissor, glViewport)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key

# ⚠️ 不要再加 sys.path 引导：引擎由 `pip install 3dworld` 提供。
from world3d.logic.camera import OrbitCamera                            # noqa: E402
from world3d.logic.replay import (Replay, save_replay,                  # noqa: E402
                                  verify_determinism)
from world3d.platform.app import DemoApp, orbit_camera_input, run       # noqa: E402

N_PARTICLES = 280
N_FRAMES = 480                       # 8 秒 @ 60Hz
DT = 1.0 / 60.0                      # 固定步长（确定性立身之本）
SPRING = 1.5                         # 牵引刚度（弱 → 粒子成云而非塌成一点）
DAMP = 0.4                           # 速度阻尼
ATTR_R = 4.0                         # 吸引子水平幅度
ATTR_Y = 2.0                         # 吸引子竖直幅度
CLOUD_R = 3.0                         # 每颗粒子相对吸引子的平衡偏移半径

MODES = ('实时', '回放', '对照')      # T 键循环


# ------------------------------------------------------- 确定性仿真内核 --
def _init_state(seed, n):
    """同种子 → 同初始坐标/速度/相位/偏移（确定性入口，run_sim 与 LiveSim 共用）。

    ``off`` 是每颗粒子相对吸引子的平衡偏移：让整团粒子在吸引子周围铺成
    一团会整体漂移的"云"，而不是全部塌到同一个点，画面才看得清。
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(n, 3)) * 3.0
    vel = rng.uniform(-1.0, 1.0, size=(n, 3)) * 0.5
    phase = float(rng.random() * 6.283185307179586)
    off = rng.uniform(-1.0, 1.0, size=(n, 3)) * CLOUD_R
    return pos, vel, phase, off


def _target(t, phase):
    """每帧的'输入'：由时间 + 种子相位驱动的三维吸引子目标。"""
    return np.array([
        ATTR_R * np.sin(0.7 * t + phase),
        ATTR_Y * np.sin(1.3 * t + phase * 1.7),
        ATTR_R * np.cos(0.5 * t + phase),
    ], dtype=float)


def _step(pos, vel, t, phase, off, dt):
    """单步半隐式欧拉：状态完全由 (pos, vel, t, phase, off, dt) 决定。"""
    tgt = _target(t, phase)
    eq = tgt + off                        # 该粒子的平衡位置 = 吸引子 + 自身偏移
    acc = SPRING * (eq - pos) - DAMP * vel
    vel = vel + acc * dt
    pos = pos + vel * dt
    return pos, vel, tgt


def run_sim(seed, n, frames, dt):
    """一次性跑完整条轨迹，返回 ``[(pos_Nx3, tgt_3), ...]``（长度 = frames）。

    这是"原始运行"，也是 verify_determinism 与录制所引用的唯一真相源。
    """
    pos, vel, phase, off = _init_state(seed, n)
    t = 0.0
    traj = []
    for _ in range(frames):
        pos, vel, tgt = _step(pos, vel, t, phase, off, dt)
        t += dt
        traj.append((pos.copy(), tgt.copy()))
    return traj


def run_sim_flat(seed, n, frames, dt):
    """把整条轨迹拍平成单个 Python 列表——供 verify_determinism 做可深度比较的

    结果（``verify_determinism`` 内部用 ``r == first`` 判等，numpy 数组会触发
    "truth value ambiguous"，故这里显式转成 list，比较才是逐元素标量的 ==）。
    """
    traj = run_sim(seed, n, frames, dt)
    flat = []
    for p, t in traj:
        flat.extend(p.ravel().tolist())
        flat.extend(t.ravel().tolist())
    return flat


class LiveSim:
    """有状态的实时重算器：从同种子起跑，每一步与 run_sim 第 k 帧逐位一致。"""

    def __init__(self, seed, n, dt):
        self.pos, self.vel, self.phase, self.off = _init_state(seed, n)
        self.t = 0.0
        self.dt = dt

    def step(self):
        self.pos, self.vel, tgt = _step(self.pos, self.vel, self.t,
                                        self.phase, self.off, self.dt)
        self.t += self.dt
        return self.pos.copy(), tgt.copy()


# ----------------------------------------------------------------- 着色 --
VERT = '''#version 330 core
in vec3 position;
in vec4 colors;                       // 顶点色：vec4（含 alpha）

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;
uniform mat4 u_model;
uniform float u_size;                 // 点大小（粒子 / 吸引子不同）

out vec4 v_col;

void main() {
    vec4 world = u_model * vec4(position, 1.0);
    v_col = colors;
    gl_Position = window.projection * window.view * world;
    gl_PointSize = u_size;
}'''

FRAG = '''#version 330 core
in vec4 v_col;
out vec4 final_colors;

void main() {
    // 把方点裁成圆点，观感更干净
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    final_colors = v_col;
}'''

# 网格用单独的直线着色器：直线没有 gl_PointCoord，不能再做圆点裁剪
# （否则所有线片段都会被 discard 掉，网格就看不见了）。
LINE_FRAG = '''#version 330 core
in vec4 v_col;
out vec4 final_colors;

void main() {
    final_colors = v_col;
}'''


# ------------------------------------------------------------------- app --
class Replayward(DemoApp):
    TITLE = 'REPLAYward - deterministic record / replay / verify (logic/replay)'
    HELP = ('R re-record (new seed)  T live/replay/compare  SPACE pause  '
            'drag/scroll  ESC')
    DURATION = None
    SHOTS = ((6.0, 'docs/replayward_01.png'), (18.0, 'docs/replayward_02.png'))
    FOV = 55.0

    def setup(self):
        glEnable(GL_DEPTH_TEST)
        # 兼容 Profile 下需显式开启，否则 gl_PointSize 被忽略（点会变成 1px 的暗点）
        glEnable(GL_PROGRAM_POINT_SIZE)
        self.prog = ShaderProgram(Shader(VERT, 'vertex'), Shader(FRAG, 'fragment'))
        self.line_prog = ShaderProgram(Shader(VERT, 'vertex'),
                                      Shader(LINE_FRAG, 'fragment'))
        self.mode = 0                       # 0 实时 / 1 回放 / 2 对照
        # 起手视图可被 --flag 指定：replay / compare(split)
        if 'replay' in self.args.flags:
            self.mode = 1
        elif {'compare', 'split'} & self.args.flags:
            self.mode = 2
        self.seed = 20261026
        self.cam = OrbitCamera(target=Vec3(0, 0, 0), yaw=35.0, pitch=22.0,
                               dist=16.0, pitch_range=(2.0, 89.0),
                               dist_range=(6.0, 120.0))
        orbit_camera_input(self, self.cam)
        self._init_buffers()
        self._record()                      # 起手先录制一遍

    # --------------------------------------------------------- 缓冲 --
    def _init_buffers(self):
        n = N_PARTICLES
        # 粒子：每个一颗点，颜色按索引取黄金分割色相（4 元组 vec4）
        base = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            h = (i * 0.618033988749895) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
            base[i] = (r, g, b, 1.0)
        self.base_colors = base
        self.vl = self.prog.vertex_list(
            n, GL_POINTS,
            position=('f', np.zeros(3 * n, dtype=np.float32)),
            colors=('f', base.ravel()))
        self.tvl = self.prog.vertex_list(
            1, GL_POINTS,
            position=('f', np.zeros(3, dtype=np.float32)),
            colors=('f', (1.0, 0.9, 0.2, 1.0)))   # 吸引子目标标记

        # 地面参考网格（GL_LINES，颜色同样走 vec4 顶点色属性，调亮以便看清）
        G, step = 10.0, 2.0
        gpos, gcls = [], []
        gc = (0.45, 0.50, 0.60, 1.0)
        xs = list(np.arange(-G, G + 1e-6, step))
        for x in xs:
            gpos.extend([x, 0.0, -G, x, 0.0, G])
            gcls.extend(gc + gc)
        for z in xs:
            gpos.extend([-G, 0.0, z, G, 0.0, z])
            gcls.extend(gc + gc)
        self.grid_vl = self.line_prog.vertex_list(
            len(gpos) // 3, GL_LINES,
            position=('f', gpos), colors=('f', gcls))

    # --------------------------------------------------------- 录制 --
    def _record(self):
        """跑一遍确定性仿真，把状态/输入双轨写进 Replay，并立即做两项校验。"""
        traj = run_sim(self.seed, N_PARTICLES, N_FRAMES, DT)
        events = [{'seq': i, 'ts': i * DT, 'type': 'frame',
                   'text': f'frame {i}: {N_PARTICLES} particles',
                   'state': p.tolist()}            # 事件流 = 世界状态
                  for i, (p, _t) in enumerate(traj)]
        actions = [{'seq': i, 'actor': 'controller',
                    'input': t.tolist()}           # 动作序列 = 每帧输入
                   for i, (_p, t) in enumerate(traj)]
        self.replay = Replay(
            events=events, actions=actions,
            meta={'seed': self.seed, 'n': N_PARTICLES,
                  'frames': N_FRAMES, 'dt': DT})

        # ① verify_determinism：同种子跑两遍，整条轨迹应逐项一致
        self.det = verify_determinism(
            lambda: run_sim_flat(self.seed, N_PARTICLES, N_FRAMES, DT), 2)

        # ② JSON 往返：from_dict(to_dict()) 后，回放坐标与原始轨迹的误差
        r2 = Replay.from_dict(self.replay.to_dict())
        self.replay_err = max(
            float(np.max(np.abs(np.asarray(r2.events[i]['state']) - traj[i][0])))
            for i in range(len(traj)))

        # 顺便演示 save_replay（战报导出纯文本）——只落临时文件，不污染仓库
        fd, tmp = tempfile.mkstemp(suffix='.txt', prefix='replayward_')
        os.close(fd)
        save_replay(self.replay, tmp)
        with open(tmp, encoding='utf-8') as fh:
            n_lines = sum(1 for _ in fh)
        os.remove(tmp)

        # 实时重算器从同种子起跑
        self.live = LiveSim(self.seed, N_PARTICLES, DT)
        self.frame = 0
        self.live_pos, self.live_tgt = self.live.step()
        self.replay_pos = np.asarray(traj[0][0], dtype=float)
        self.replay_tgt = np.asarray(traj[0][1], dtype=float)
        self.frame_err = 0.0
        print(f'[replayward] seed={self.seed} frames={N_FRAMES} '
              f'det={self.det["identical"]} roundtrip_err={self.replay_err:.2e} '
              f'save_replay_lines={n_lines}')

    # --------------------------------------------------------- 交互 --
    def on_key(self, symbol, modifiers):
        k = key
        if symbol == k.R:
            # 重新录制：换种子重跑一遍，刷新 Replay 与两项校验
            self.seed = (self.seed * 1103515245 + 12345) & 0x7FFFFFFF
            self._record()
        elif symbol == k.T:
            self.mode = (self.mode + 1) % len(MODES)

    # --------------------------------------------------------- 更新 --
    def update(self, dt):
        if self.paused:
            return
        if self.frame >= N_FRAMES - 1:           # 循环：重置实时重算器
            self.live = LiveSim(self.seed, N_PARTICLES, DT)
            self.frame = 0
            self.live_pos, self.live_tgt = self.live.step()
        else:
            self.frame += 1
            self.live_pos, self.live_tgt = self.live.step()
        ev = self.replay.events[self.frame]
        self.replay_pos = np.asarray(ev['state'], dtype=float)
        self.replay_tgt = np.asarray(
            self.replay.actions[self.frame]['input'], dtype=float)
        # 运行时逐帧比对：实时重算 vs 回放
        self.frame_err = float(np.max(np.abs(self.live_pos - self.replay_pos)))

    # --------------------------------------------------------- 渲染 --
    def _draw_scene(self, positions, target):
        # 相机跟着吸引子走，整团粒子（围绕吸引子铺开）才始终居中、看得清
        self.cam.target = Vec3(float(target[0]), float(target[1]), float(target[2]))
        self.window.view = self.cam.view_matrix()
        self.line_prog.use()
        self.line_prog['u_model'] = Mat4()
        self.grid_vl.draw(GL_LINES)
        self.prog.use()
        self.prog['u_model'] = Mat4()
        self.prog['u_size'] = 16.0
        self.vl.position[:] = positions.astype(np.float32).ravel()
        self.vl.draw(GL_POINTS)
        self.prog['u_size'] = 26.0
        self.tvl.position[:] = target.astype(np.float32).ravel()
        self.tvl.draw(GL_POINTS)

    def render(self):
        w, h = self.window.width, self.window.height
        glDisable(GL_SCISSOR_TEST)
        glClearColor(0.04, 0.05, 0.07, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.window.projection = Mat4.perspective_projection(
            w / max(1, h), z_near=0.1, z_far=400.0, fov=self.FOV)
        self.prog.use()
        self.prog['u_model'] = Mat4()

        if self.mode == 0:                       # 实时重算
            glViewport(0, 0, w, h)
            self._draw_scene(self.live_pos, self.live_tgt)
        elif self.mode == 1:                     # 回放
            glViewport(0, 0, w, h)
            self._draw_scene(self.replay_pos, self.replay_tgt)
        else:                                    # 对照：左实时 / 右回放
            hw = w // 2
            glEnable(GL_SCISSOR_TEST)
            glViewport(0, 0, hw, h)
            glScissor(0, 0, hw, h)
            self._draw_scene(self.live_pos, self.live_tgt)
            glViewport(hw, 0, w - hw, h)
            glScissor(hw, 0, w - hw, h)
            self._draw_scene(self.replay_pos, self.replay_tgt)
            glDisable(GL_SCISSOR_TEST)

        # HUD 用全屏视口，务必复位
        self.window.viewport = (0, 0, w, h)

    # --------------------------------------------------------- 状态 --
    def status_text(self):
        det = ('PASS' if self.det['identical']
               else f'FAIL@{self.det["first_diff"]}')
        return (f'MODE={MODES[self.mode]}  frame {self.frame}/{N_FRAMES - 1}'
                f'  seed {self.seed}  DET={det}'
                f'  roundtrip_err={self.replay_err:.2e}'
                f'  live-vs-replay={self.frame_err:.2e}'
                f'  {self.fps:5.1f}fps'
                + ('  [PAUSED]' if self.paused else ''))


if __name__ == '__main__':
    run(Replayward)
