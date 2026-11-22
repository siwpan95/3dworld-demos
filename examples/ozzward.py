# -*- coding: utf-8 -*-
"""《骨鸣 / Ozzward》—— ozz 风格骨架动画最小演示（anim.ozz_lite 首秀）。

**为什么值得单独一个 demo**
----------------------------
``world3d.anim.ozz_lite`` 是引擎的轻量骨架动画层（纯 numpy / 零 GL / headless
可断言），此前演示仓里**零个 demo 消费过**。本 demo 把它的核心四件套
``Skeleton`` / ``Clip`` / ``sample_clip`` / ``compute_skin_matrices`` /
``skin_vertices`` 串成一条肉眼可见的链路：

    手写骨架  →  手写 clip  →  sample_clip 采样局部 pose
             →  compute_skin_matrices 求蒙皮矩阵
             →  skin_vertices 把矩阵作用到蒙皮顶点
             →  逐帧把变形后的网格交给轨道相机渲染

资源策略（重要）
---------------
仓库 ``assets/skinned_snake.gltf`` 只带 skin、**没有嵌入 animation**（验证过：
glTF 顶层没有 ``animations`` 数组），所以本 demo 走「**代码合成装备**」路线：
在 Python 里手动构造一套最小 ozz 装备——一条 7 关节的「摆动触手/尾巴」
（每关节一个父索引 + 各关节绑定变换），以及一个让关节绕 Z/X 轴做行波摆动的
``Clip``。然后完全用 ``ozz_lite`` 的数学 API 做前向运动学 + 蒙皮，不依赖任何
glTF 加载。演示因此自包含、零外部素材、固定种子可复现。

（若日后有「同时带 skin 与 animation」的 glTF，可改用
``skeleton_from_gltf`` + ``clip_from_gltf_animation`` + ``skinned_mesh_playback``
走 glTF 加载路径——本 demo 顶部注释也给出了对应 API 清单，但本仓暂无此资源。）

用到的引擎 API（world3d.anim.ozz_lite）
--------------------------------------
* ``Skeleton(joints)``：``joints[i] = (name, parent_index, (t, r, s))``，
  父索引 -1 表示根；本 demo 用「根 + 6 段」，每段绑定局部平移 (0, SEG, 0)。
* ``Clip(name, duration, tracks)``：``tracks[joint_index] = {'rotation': (times, quats)}``，
  quaternion 序为 ozz 的 (w,x,y,z)；每个关节一条绕 Z 与绕 X 的行波旋转轨。
* ``sample_clip(clip, t, loop=True, bind=skeleton.bind)``：返回 list[(t,r,s)]
  局部 pose。⚠️ **坑**：关节若只有 rotation 轨、没有 translation 轨，不传
  ``bind`` 时平移会被默认成 0（整条尾巴塌到原点）；必须传 ``bind=skeleton.bind``
  才能保留绑定姿态的平移/旋转。本 demo 严格传 bind。
* ``compute_skin_matrices(skeleton, local_pose, inverse_bind)``：
  每个关节 ``jointWorld @ inverseBind``，inverseBind 由绑定姿态世界矩阵求逆得到。
* ``skin_vertices(positions, joints, weights, skin_matrices)``：线性混合蒙皮，
  positions 是 bind-space 顶点，joints/weights 每顶点最多 4 个影响关节。

HUD（status_text）读什么
-------------------------
clip 名、当前时间 / 时长（loop 取模）、关节数、帧率；叠加 [PAUSED] / [BONES] 标记。

控制键
------
拖动=旋转视角  滚轮=缩放  SPACE=暂停  R=重置时间  W=线框  B=显示骨骼（关节球+骨连线）  ESC=退出

用法::

    python examples/ozzward.py                正常运行
    python examples/ozzward.py --smoke 2      冒烟（2s 自动关）
    python examples/ozzward.py --shot         抓剧照到 docs/ozzward_01/02.png
"""
import math
import random

import numpy as np
from pyglet.gl import (GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
                       GL_DEPTH_TEST, GL_FILL, GL_FRONT_AND_BACK, GL_LINE,
                       GL_LINES, GL_ONE_MINUS_SRC_ALPHA, GL_POINTS, GL_SRC_ALPHA,
                       GL_TRIANGLES, glBlendFunc, glClear, glClearColor,
                       glEnable, glPolygonMode)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key

# ⚠️ 不要再加 sys.path 引导：引擎由 `pip install 3dworld` 提供
# （见 requirements.txt 与 docs/依赖与版本.md）。
from world3d.anim.ozz_lite import (Clip, Skeleton, compute_skin_matrices,  # noqa: E402
                                   q_mul, q_normalize, sample_clip,
                                   skin_vertices)
from world3d.logic.camera import OrbitCamera                            # noqa: E402
from world3d.platform.app import DemoApp, orbit_camera_input, run        # noqa: E402

SEED = 20261027
N_JOINTS = 7                      # 根 + 6 段
SEG = 0.9                         # 每段长度（米）
R0 = 0.42                         # 触手根部半径
CLIP_T = 2.5                      # clip 时长（秒）

# ------------------------------------------------------------ 着色器 --
VERT = '''#version 330 core
in vec3 position;
in vec3 normals;
in vec4 colors;

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;
uniform mat4 u_model;

out vec3 v_nrm;
out vec4 v_col;
out vec3 v_world;

void main() {
    vec4 world = u_model * vec4(position, 1.0);
    v_world = world.xyz;
    v_nrm = normalize(mat3(u_model) * normals);
    v_col = colors;
    gl_Position = window.projection * window.view * world;
}'''

FRAG = '''#version 330 core
in vec3 v_nrm;
in vec4 v_col;
in vec3 v_world;
out vec4 final_colors;

uniform vec3 u_sun;
uniform vec3 u_fog;

void main() {
    vec3 n = normalize(v_nrm);
    float lam = max(dot(n, u_sun), 0.0);
    float hemi = 0.5 + 0.5 * n.y;
    vec3 amb = mix(vec3(0.13, 0.13, 0.15), vec3(0.32, 0.35, 0.43), hemi);
    vec3 col = v_col.rgb * (amb + lam * vec3(0.98, 0.94, 0.86));
    float fog = clamp(length(v_world.xz) * 0.012, 0.0, 0.5);
    col = mix(col, u_fog, fog);
    final_colors = vec4(col, 1.0);
}'''

# 骨骼显示用：只吃窗口矩阵（关节世界坐标直接喂入），线段 + 点共用一套
BONE_VERT = '''#version 330 core
in vec3 position;
uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;
void main() {
    gl_Position = window.projection * window.view * vec4(position, 1.0);
    gl_PointSize = 11.0;
}'''

BONE_FRAG = '''#version 330 core
out vec4 final_colors;
uniform vec3 u_col;
void main() { final_colors = vec4(u_col, 1.0); }'''


# -------------------------------------------------- 蒙皮法线（CPU 侧） --
def skin_normals(normals, joints, weights, skin_mats):
    """与 skin_vertices 同构，但只对 3x3 旋转部分做线性混合（无平移）。"""
    normals = np.asarray(normals, dtype=float)
    joints = np.asarray(joints, dtype=int)
    weights = np.asarray(weights, dtype=float)
    V = normals.shape[0]
    out = np.zeros((V, 3), dtype=float)
    for k in range(4):
        ji = joints[:, k]
        w = weights[:, k]
        M3 = np.stack([skin_mats[j][:3, :3] for j in ji])     # (V,3,3)
        tp = (M3 @ normals[:, :, None])[:, :, 0]
        out += w[:, None] * tp
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-6)


def _gradient(t):
    """t in [0,1] → 青绿→品红的 4 元组顶点色。"""
    c0 = (0.25, 0.82, 0.72)
    c1 = (0.93, 0.36, 0.62)
    r = c0[0] + (c1[0] - c0[0]) * t
    g = c0[1] + (c1[1] - c0[1]) * t
    b = c0[2] + (c1[2] - c0[2]) * t
    return (r, g, b, 1.0)


# ------------------------------------------------------------------- app --
class Ozzward(DemoApp):
    TITLE = 'OZZward - ozz-lite skeletal animation (anim.ozz_lite)'
    HELP = ('drag orbit  wheel zoom  SPACE pause  R reset  '
            'W wire  B bones  ESC quit')
    SHOTS = ((6.0, 'docs/ozzward_01.png'), (17.0, 'docs/ozzward_02.png'))
    DURATION = None
    FOV = 55.0

    def setup(self):
        glEnable(GL_DEPTH_TEST)
        self.prog = ShaderProgram(Shader(VERT, 'vertex'), Shader(FRAG, 'fragment'))
        self.bone_prog = ShaderProgram(Shader(BONE_VERT, 'vertex'),
                                       Shader(BONE_FRAG, 'fragment'))
        self.wire = 'wire' in self.args.flags      # --flag wire：线框对照（也用于验证 GL_LINE 路径）
        self.show_bones = False
        self.rng = random.Random(SEED)
        self.n_joints = N_JOINTS
        self.seg = SEG

        # 轨道相机：目标抬到触手中段
        self.cam = OrbitCamera(target=Vec3(0, (N_JOINTS - 1) * SEG * 0.5, 0),
                               yaw=34.0, pitch=16.0, dist=15.0,
                               pitch_range=(4.0, 89.0), dist_range=(4.0, 80.0))
        orbit_camera_input(self, self.cam)
        self.model = Mat4()

        self._build_rig()       # Skeleton + Clip + inverse_bind
        self._build_mesh()      # bind-space 蒙皮网格（逐帧变形）
        self._build_ground()    # 参考地面
        self._build_head()      # 静态根球（关节 0 = 根，恒等变换）
        self._build_bones()     # 骨骼线段 / 关节点（逐帧更新）
        self.pose = None
        self.joint_pos = None
        print(f'[ozzward] rig joints={self.n_joints} clip="{self.clip.name}" '
              f'duration={self.clip.duration:.2f}s  mesh_verts={len(self.rest_pos)}')

    # ----------------------------------------------------- 骨架与 clip --
    def _build_rig(self):
        """手写最小 ozz 装备：根 + 6 段，每段绑定局部平移 (0, SEG, 0)。"""
        rng = self.rng
        joints = [("root", -1, (np.zeros(3), q_normalize([1, 0, 0, 0]), np.ones(3)))]
        for i in range(1, self.n_joints):
            joints.append((f"j{i}", i - 1,
                           (np.array([0.0, self.seg, 0.0]),
                            q_normalize([1, 0, 0, 0]), np.ones(3))))
        sk = Skeleton(joints)

        # clip：每个关节绕 Z 与绕 X 做行波摆动（相位随关节号递增）
        times = np.linspace(0.0, CLIP_T, 9)          # 9 个关键帧 → 平滑插值
        tracks = {}
        for j in range(1, self.n_joints):
            az = 0.55 * np.sin(2 * np.pi * (times / CLIP_T) * 1.5 - j * 0.6)
            ax = 0.30 * np.sin(2 * np.pi * (times / CLIP_T) * 1.5 - j * 0.6 + 1.1)
            quats = np.empty((len(times), 4), dtype=float)
            for k in range(len(times)):
                # 组合：先绕 X 再绕 Z（局部空间）
                qx = np.array([math.cos(ax[k] / 2), math.sin(ax[k] / 2), 0.0, 0.0])
                qz = np.array([math.cos(az[k] / 2), 0.0, 0.0, math.sin(az[k] / 2)])
                quats[k] = q_normalize(q_mul(qz, qx))   # ozz 序 (w,x,y,z)
            tracks[j] = {'rotation': (times, quats)}
        clip = Clip("tail-swing", CLIP_T, tracks)

        # inverseBind：绑定姿态世界矩阵求逆（蒙皮矩阵 = jointWorld @ inverseBind）
        bind_world = sk.world_matrices()
        inverse_bind = [np.linalg.inv(m) for m in bind_world]

        self.skeleton = sk
        self.clip = clip
        self.inverse_bind = inverse_bind

    # ----------------------------------------------------- 蒙皮网格 --
    def _build_mesh(self):
        # 展开成三角汤（不依赖 indices kwarg）：每环 K 段 × NR 段 × 2 三角。
        H = (self.n_joints - 1) * self.seg
        K = 14                       # 每环分段
        NR = 40                      # 环数
        pos, nrm, jt, wt, col = [], [], [], [], []

        def ring_verts(r):
            """第 r 环（0..NR）的顶点列表：每项 (position, normal, joint, weight)。"""
            y = H * r / NR
            frac = y / self.seg
            jl = min(int(math.floor(frac)), self.n_joints - 2)
            ju = jl + 1
            f = frac - jl
            rad = R0 * (1.0 - 0.82 * (y / H)) + 0.04
            out = []
            for kseg in range(K):
                th = 2 * math.pi * kseg / K
                cx, cz = math.cos(th), math.sin(th)
                out.append(((rad * cx, y, rad * cz),
                            (cx, 0.0, cz),
                            (jl, ju, 0, 0),
                            (1.0 - f, f, 0.0, 0.0)))
            return out

        rings = [ring_verts(r) for r in range(NR + 1)]
        for r in range(NR):
            a = rings[r]
            b = rings[r + 1]
            for kseg in range(K):
                k2 = (kseg + 1) % K
                # 两个三角形：a[kseg], a[k2], b[k2] / a[kseg], b[k2], b[kseg]
                tri = (a[kseg], a[k2], b[k2], a[kseg], b[k2], b[kseg])
                for (p, n, j, wv) in tri:
                    pos.extend(p)
                    nrm.extend(n)
                    jt.extend(j)
                    wt.extend(wv)
                    col.extend(_gradient(j[0] / max(1, self.n_joints - 1)))
        self.rest_pos = np.array(pos, dtype=float).reshape(-1, 3)
        self.rest_nrm = np.array(nrm, dtype=float).reshape(-1, 3)
        self.vjoints = np.array(jt, dtype=int).reshape(-1, 4)
        self.vweights = np.array(wt, dtype=float).reshape(-1, 4)
        self.vcolors = np.array(col, dtype=float).reshape(-1, 4)
        self.mesh = self.prog.vertex_list(
            len(pos) // 3, GL_TRIANGLES,
            position=('f', pos), normals=('f', nrm), colors=('f', col))

    def _build_ground(self):
        s = 9.0
        pos = [-s, 0, -s, s, 0, -s, s, 0, s, -s, 0, -s, s, 0, s, -s, 0, s]
        nrm = [0, 1, 0] * 6
        col = (0.16, 0.18, 0.22, 1.0) * 6
        self.ground = self.prog.vertex_list(
            6, GL_TRIANGLES, position=('f', pos), normals=('f', nrm),
            colors=('f', col))

    def _build_head(self):
        """静态根球（关节 0 = 根，恒等 → 不动）；纯装饰，单独 draw。"""
        pos, nrm, col = [], [], []
        seg, ring, r = 14, 10, 0.62
        for i in range(ring):
            t0, t1 = math.pi * i / ring, math.pi * (i + 1) / ring
            y0, y1 = math.cos(t0), math.cos(t1)
            r0, r1 = math.sin(t0), math.sin(t1)
            for kseg in range(seg):
                p0, p1 = 2 * math.pi * kseg / seg, 2 * math.pi * (kseg + 1) / seg
                a = (r0 * math.cos(p0), y0, r0 * math.sin(p0))
                b = (r0 * math.cos(p1), y0, r0 * math.sin(p1))
                c = (r1 * math.cos(p1), y1, r1 * math.sin(p1))
                d = (r1 * math.cos(p0), y1, r1 * math.sin(p0))
                for v in (a, b, c, a, c, d):
                    pos.extend((v[0] * r, v[1] * r, v[2] * r))
                    nrm.extend(v)
                    col.extend((0.95, 0.48, 0.42, 1.0))
        self.head = self.prog.vertex_list(
            len(pos) // 3, GL_TRIANGLES, position=('f', pos),
            normals=('f', nrm), colors=('f', col))

    # ----------------------------------------------------- 骨骼缓冲 --
    def _build_bones(self):
        n_bone_verts = 2 * max(0, self.n_joints - 1)     # 每根骨一条线段
        self.bone_vl = self.bone_prog.vertex_list(
            n_bone_verts, GL_LINES,
            position=('f', [0.0] * (3 * max(1, n_bone_verts))))
        self.joint_vl = self.bone_prog.vertex_list(
            self.n_joints, GL_POINTS,
            position=('f', [0.0] * (3 * self.n_joints)))

    def _update_bones(self, jp):
        bp = []
        for i in range(self.n_joints):
            p = self.skeleton.parents[i]
            if p >= 0:
                bp.extend(jp[p])
                bp.extend(jp[i])
        if bp:
            self.bone_vl.position[:] = bp
        jpflat = jp.reshape(-1).tolist()
        self.joint_vl.position[:] = jpflat

    # ------------------------------------------------------------ 交互 --
    def on_key(self, symbol, modifiers):
        if symbol == key.W:
            self.wire = not self.wire
        elif symbol == key.R:
            self.time = 0.0                   # 重置动画时间
        elif symbol == key.B:
            self.show_bones = not self.show_bones

    # ------------------------------------------------------------ 更新 --
    def update(self, dt):
        if self.paused:
            return
        # ⚠️ 必须传 bind=skeleton.bind，否则只有 rotation 轨的关节平移塌成 0
        self.pose = sample_clip(self.clip, self.time, loop=True,
                                bind=self.skeleton.bind)
        skin_mats = compute_skin_matrices(self.skeleton, self.pose,
                                          self.inverse_bind)
        def_pos = skin_vertices(self.rest_pos, self.vjoints, self.vweights,
                                skin_mats)
        def_nrm = skin_normals(self.rest_nrm, self.vjoints, self.vweights,
                               skin_mats)
        self.mesh.position[:] = def_pos.reshape(-1)
        self.mesh.normals[:] = def_nrm.reshape(-1)
        wm = self.skeleton.world_matrices(self.pose)
        jp = np.array([m[:3, 3] for m in wm], dtype=float)
        self.joint_pos = jp
        self._update_bones(jp)

    # ------------------------------------------------------------ 渲染 --
    def render(self):
        w, h = self.window.width, self.window.height
        glClearColor(0.06, 0.07, 0.10, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.window.projection = Mat4.perspective_projection(
            w / max(1, h), z_near=0.1, z_far=400.0, fov=self.FOV)
        self.window.view = self.cam.view_matrix()

        prog = self.prog
        prog.use()
        prog['u_model'] = self.model
        prog['u_sun'] = (0.40, 0.78, 0.42)
        prog['u_fog'] = (0.06, 0.07, 0.10)

        if self.wire:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        self.ground.draw(GL_TRIANGLES)
        self.head.draw(GL_TRIANGLES)
        self.mesh.draw(GL_TRIANGLES)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        if self.show_bones and self.joint_pos is not None:
            self.bone_prog.use()
            self.bone_prog['u_col'] = (1.0, 0.84, 0.32)
            self.bone_vl.draw(GL_LINES)
            self.bone_prog['u_col'] = (1.0, 0.34, 0.34)
            self.joint_vl.draw(GL_POINTS)

        self.window.projection = self.perspective

    # ------------------------------------------------------------ 状态 --
    def status_text(self):
        tt = self.time % self.clip.duration if self.clip.duration > 0 else 0.0
        flag = '  [PAUSED]' if self.paused else ''
        if self.show_bones:
            flag += '  [BONES]'
        return (f'clip {self.clip.name}  t={tt:5.2f}/{self.clip.duration:.2f}s'
                f'  joints {self.n_joints}  {self.fps:5.1f} fps{flag}')


if __name__ == '__main__':
    run(Ozzward)
