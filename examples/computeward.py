# -*- coding: utf-8 -*-
"""《洪流 / Computeward》—— 消费引擎新模块 ``engine.compute`` 的 GPU 百万粒子模拟。

**本 demo 消费的是什么**
------------------------
``world3d.engine.compute``（0.4.1 wheel 没有、需 editable 安装引擎源码树）是
compute shader / SSBO / memory barrier 的**通用调度层**——本 demo 用它驱动
整场模拟，但粒子系统本身（着色器、预设语义、绘制）全部是本 demo 自己的：

    detect()          能力探测（真实窗口建好之后调用）
    ComputeBuffer     SSBO（带 numpy 初值创建 / bind / upload / release）
    ComputeShader     compute 程序（use / set / dispatch / groups_for_n）
    barrier()         compute 写 -> 顶点属性读 的显存写序

**GPU 路径（主路径）——零回读经典三步**
----------------------------------------
每帧 CPU 只做一件事::

    prog.use(); pos_buf.bind(0); vel_buf.bind(1)
    prog.set(几个标量 uniform)
    prog.dispatch(*prog.groups_for_n(N))     # N = 1_048_576 -> 16384 组 x 64 线程
    barrier()                                # ② compute 写序
    ...同一块 GL buffer 作顶点属性源绘制...   # ③ 零回读

- SSBO x2：``Pos { vec4 p[]; }``（xyz + life）、``Vel { vec4 v[]; }``（xyz + seed）。
- 力场三预设（u_mode 分支，全在 compute shader 里）：
  0 漩涡星系（中心引力 + 切向巡舵 + 竖直振荡）、
  1 湍流云（标量位场取旋度的 **无散度 curl-noise**，GLSL 3D value noise + 有限差分）、
  2 喷泉（上抛初速 + 重力 + 地面反弹）。
- 重生（life<=0 或出界）按 ``hash(index, u_time)`` 在发射域内重置——shader 内完成，
  CPU 不参与。预设切换的重置也走 GPU（u_reset 分支重播种），无 32MB 上传。
- 绘制：裸 GL VAO 把两个 SSBO 直接绑成 position(vec4)/velocity(vec4) 顶点属性
  （为什么裸 GL：本机 pyglet domain/VAO 的 4-attribute 路径有缺陷，实测画满屏
  巨大三角形，particleward 已有完整实测记录与规避先例——本 demo 直接沿用其形态）。
  顶点着色器按速度模长染色（快暖慢冷）、gl_PointSize 按深度衰减；加色混合。

**CPU 回退路径**
----------------
``detect()['available']`` 为 False（或启动参数 ``--flag cpu`` 强制）时自动降级：
numpy 向量化积分 30_000 粒子（同一套预设语义），每帧 ``glBufferSubData`` 写两个
VBO，绘制走同一个裸 GL 程序。status_text 显示「CPU fallback」。

**控制键**
----------
``1/2/3`` 立即切预设并重置该预设   ``R`` 重置当前预设粒子场
``SPACE`` 暂停/继续                拖拽/滚轮 轨道相机      ``ESC`` 退出
无操作时每 4.0s 自动轮播下一个预设（DURATION=12s，保证抓帧落点覆盖三预设）。

用法::

    python examples/computeward.py                     正常运行（GPU 百万粒子）
    python examples/computeward.py --smoke 3           冒烟
    python examples/computeward.py --count 262144      覆盖粒子数
    python examples/computeward.py --shot              抓剧照到 docs/computeward_*.png
    python examples/computeward.py --bench 5           基准（vsync off）
    python examples/computeward.py --flag cpu          强制 CPU 回退路径
"""
from __future__ import annotations

import ctypes
import math
import sys

import numpy as np
import pyglet
from pyglet.gl import (GL_ARRAY_BUFFER, GL_BLEND, GL_COLOR_BUFFER_BIT,
                       GL_COMPILE_STATUS, GL_DEPTH_BUFFER_BIT,
                       GL_DYNAMIC_DRAW, GL_FALSE,
                       GL_FLOAT, GL_FRAGMENT_SHADER, GL_LINK_STATUS,
                       GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                       GL_POINTS, GL_SRC_ALPHA, GL_TRUE, GL_VERTEX_SHADER,
                       glAttachShader, glBlendFunc, glBufferData,
                       glBufferSubData, glClear, glClearColor, glCompileShader,
                       glCreateProgram, glCreateShader, glDeleteBuffers,
                       glDeleteProgram, glDeleteVertexArrays, glDepthMask,
                       glDrawArrays, glEnable, glEnableVertexAttribArray,
                       glGenBuffers, glGenVertexArrays, glGetAttribLocation,
                       glGetProgramInfoLog, glGetProgramiv, glGetShaderInfoLog,
                       glGetShaderiv, glGetUniformLocation, glUniform1f,
                       glUniform1i, glUniformMatrix4fv, glLinkProgram,
                       glShaderSource, glBindVertexArray, glBindBuffer,
                       glUseProgram, glVertexAttribPointer)
from pyglet.math import Mat4, Vec3
from pyglet.window import key

try:    # 0.4.1 wheel 没有 engine.compute（需 editable 安装引擎源码树）；缺失时降级 CPU
    from world3d.engine.compute import (ComputeBuffer, ComputeError,  # noqa: E402
                                        ComputeShader, barrier, detect)
    _HAS_COMPUTE = True
except ImportError:
    ComputeError = Exception           # 占位：setup 里 except ComputeError 不会触发
    ComputeBuffer = ComputeShader = barrier = detect = None
    _HAS_COMPUTE = False
from world3d.logic.camera import OrbitCamera                         # noqa: E402
from world3d.platform.app import DemoApp, orbit_camera_input, run    # noqa: E402

# ------------------------------------------------------------------ 常量 --
COUNT_DEFAULT = 1_048_576          # 2^20，GPU 主路径默认粒子数
COUNT_CPU = 30_000                 # CPU 回退路径粒子数
PRESET_SEC = 4.0                   # 每预设停留时长
MODES_EN = ('galaxy vortex', 'turbulence cloud', 'fountain')
MODE_DAMP = (0.02, 1.0, 0.0)       # 每预设速度阻尼（/s）
# 每预设取景：(相机距离, 注视点高度, 俯仰角)
MODE_CAM = ((36.0, 0.0, 26.0), (15.0, 0.0, 18.0), (22.0, 3.5, 10.0))
# 每预设点大小 / 单颗 alpha / 速度染色尺度（百万粒子加色混合，密度差异大）
MODE_POINT = (1.2, 2.0, 1.1)
MODE_ALPHA = (0.10, 0.35, 0.15)
MODE_CSCALE = (0.14, 0.12, 0.14)
# CPU 回退（3 万粒子）专用：密度低 ~33 倍，点大小/alpha 相应放大；
# 湍流阻尼加强（阻尼主导的慢流态），低帧率下粒子才留在盒域内成缕
CPU_MODE_POINT = (4.5, 10.0, 4.0)
CPU_MODE_ALPHA = (0.70, 0.95, 0.60)
CPU_MODE_DAMP = (0.02, 2.5, 0.0)
CPU_MODE_CAM = ((30.0, 0.0, 26.0), (13.0, 0.0, 18.0), (12.0, 3.5, 12.0))
CLI_COUNT = None                   # --count N（main() 里回填，run() 前生效）

# ---------------------------------------------------- compute 着色器（430）--
COMPUTE_SRC = '''#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 0) buffer PosBuf { vec4 P[]; };  // xyz + life
layout(std430, binding = 1) buffer VelBuf { vec4 V[]; };  // xyz + seed

uniform float u_dt;
uniform float u_time;
uniform float u_damp;    // 速度阻尼（/s，按预设由 CPU 传入）
uniform int   u_mode;
uniform int   u_count;
uniform int   u_reset;   // 1 = 只做（重）播种，不积分

const float PI = 3.14159265;
const float TAU = 6.2831853;
const float MU = 60.0;   // 星系中心引力参数

uint uhash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16; return x;
}
uint g_seed;
float rnd() {
    g_seed = uhash(g_seed);
    return float(g_seed) * (1.0 / 4294967296.0);
}

// ---- 3D value noise（整型 hash，比 sin 版快） + 有限差分 curl ----
float chash(vec3 c) {
    uint h = uint(c.x) * 73856093u ^ uint(c.y) * 19349663u
           ^ uint(c.z) * 83492791u;
    return float(uhash(h)) * (1.0 / 4294967296.0);
}
float vnoise(vec3 q) {
    q += 512.0;                          // 移正域，避免 float->uint 未定义
    vec3 c = floor(q);
    vec3 f = fract(q);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(chash(c), chash(c + vec3(1, 0, 0)), f.x),
            mix(chash(c + vec3(0, 1, 0)), chash(c + vec3(1, 1, 0)), f.x), f.y),
        mix(mix(chash(c + vec3(0, 0, 1)), chash(c + vec3(1, 0, 1)), f.x),
            mix(chash(c + vec3(0, 1, 1)), chash(c + vec3(1, 1, 1)), f.x), f.y),
        f.z) * 2.0 - 1.0;
}
vec3 psi(vec3 q) {
    return vec3(vnoise(q),
                vnoise(q + vec3(31.416, -47.853, 12.793)),
                vnoise(q + vec3(-233.145, 88.23, -54.31)));
}
vec3 curl(vec3 q, float e) {
    vec3 ex = vec3(e, 0, 0), ey = vec3(0, e, 0), ez = vec3(0, 0, e);
    float cx = (psi(q + ey).z - psi(q - ey).z) - (psi(q + ez).y - psi(q - ez).y);
    float cy = (psi(q + ez).x - psi(q - ez).x) - (psi(q + ex).z - psi(q - ex).z);
    float cz = (psi(q + ex).y - psi(q - ex).y) - (psi(q + ey).x - psi(q - ey).x);
    return vec3(cx, cy, cz) / (2.0 * e);
}

// ---- 发射域（按 u_mode），重生与重置共用 ----
void spawn(out vec3 pos, out vec3 vel, out float life) {
    if (u_mode == 0) {                   // 漩涡星系：核球 + 盘 + 双 log-旋臂
        float rad;
        float ang;
        if (rnd() < 0.12) {              // 中心核球
            rad = 1.0 + 3.5 * rnd() * rnd();
            ang = rnd() * TAU;
        } else {                         // 盘面，55% 粒子落在旋臂上
            rad = 3.0 + 16.0 * pow(rnd(), 0.8);
            float a_disc = rnd() * TAU;
            float a_arm = floor(rnd() * 2.0) * PI + log(rad) * 1.7
                        + (rnd() - 0.5) * 0.35;
            ang = (rnd() < 0.55) ? a_arm : a_disc;
        }
        pos = vec3(cos(ang) * rad, (rnd() - 0.5) * 0.9 * exp(-rad * 0.10),
                   sin(ang) * rad);
        float vo = sqrt(MU / max(rad, 0.5));
        vel = vec3(-sin(ang), 0.0, cos(ang)) * vo
            + vec3(0.0, (rnd() - 0.5) * 0.3, 0.0);
        life = 30.0 + rnd() * 40.0;
    } else if (u_mode == 1) {            // 湍流云：盒域慢速
        pos = (vec3(rnd(), rnd(), rnd()) - 0.5) * vec3(12.0, 7.0, 12.0);
        vel = (vec3(rnd(), rnd(), rnd()) - 0.5) * 0.75;
        life = 2.5 + rnd() * 2.5;        // 短寿命：分布留在盒域，构图居中
    } else {                             // 喷泉：6 束离散喷流，抛物线亮弧
        float jet = floor(rnd() * 6.0);
        float ang = (jet / 6.0) * TAU + (rnd() - 0.5) * 0.06;
        vec2 lat = vec2(cos(ang), sin(ang)) * (2.2 + rnd() * 0.7);
        pos = vec3(lat.x * 0.1, 0.05, lat.y * 0.1);
        vel = vec3(lat.x, 8.0 + rnd() * 3.0, lat.y);
        life = 2.8 + rnd() * 1.2;        // 盖过飞行时间，弧线完整可见
    }
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_count)) return;
    g_seed = uhash(i * 747796405u + 2891336453u * (1u + uint(u_time * 173.0)));

    vec4 p4 = P[i];
    vec4 v4 = V[i];
    vec3 pos = p4.xyz;
    vec3 vel = v4.xyz;
    float life = p4.w;

    if (u_reset == 1) {                  // （重）播种：GPU 侧完成，无 CPU 上传
        spawn(pos, vel, life);
        P[i] = vec4(pos, life);
        V[i] = vec4(vel, v4.w);
        return;
    }

    vec3 force = vec3(0.0);
    if (u_mode == 0) {                   // ① 星系：中心引力 + 切向巡舵 + 竖直振荡
        float r = max(length(pos.xz), 0.4);
        vec3 radial = vec3(pos.x, 0.0, pos.z) / r;
        vec3 tangent = vec3(-radial.z, 0.0, radial.x);
        float vo = sqrt(MU / r);
        force += -radial * (MU / (r * r));
        force += tangent * (vo - dot(vel, tangent)) * 0.8;
        force += vec3(0.0, -pos.y * 2.2 - vel.y * 0.5, 0.0);
    } else if (u_mode == 1) {            // ① 湍流：无散度 curl-noise
        vec3 q = pos * 0.10 + vec3(0.0, u_time * 0.15, 0.0);
        force += curl(q, 0.55) * 55.0;   // 快流 + 时变场：旋转流场摊出纤维片
        force += -pos * 0.15;            // 向心弹簧：居中构图 + 防飘散
    } else {                             // ① 喷泉：重力
        force += vec3(0.0, -9.8, 0.0);
    }

    // ② 积分
    vel += force * u_dt;
    vel *= max(0.0, 1.0 - u_damp * u_dt);
    pos += vel * u_dt;
    life -= u_dt;

    if (u_mode == 2 && pos.y < 0.0 && vel.y < 0.0) {   // 喷泉地面反弹
        pos.y = 0.0;
        vel.y = -vel.y * 0.35;
        vel.xz *= 0.40;                  // 横向强衰减 + 砍寿命，防落地裙堆积
        life *= 0.50;
    }

    // ③ 重生：life 尽 / 出界 -> 按 hash(index, u_time) 在发射域重置
    if (life <= 0.0 || dot(pos, pos) > 2500.0) {
        spawn(pos, vel, life);
    }
    P[i] = vec4(pos, life);
    V[i] = vec4(vel, v4.w);
}
'''

# ------------------------------------------------------ 绘制着色器（330）--
# 裸 GL program：WindowBlock 换成两个 mat4 uniform（形态抄 particleward）。
DRAW_VERT = '''#version 330 core
in vec4 position;    // xyz + life（SSBO/PosBuf 同一块 buffer）
in vec4 velocity;    // xyz + seed（SSBO/VelBuf 同一块 buffer）
uniform mat4 projection;
uniform mat4 view;
uniform float u_point;
uniform float u_alpha;
uniform float u_cscale;   // 速度->染色尺度（按预设分档）
uniform int   u_sfade;    // 1 = alpha 随速度调制（湍流：快流成亮缕）
out vec4 v_col;
void main() {
    vec4 vp = view * vec4(position.xyz, 1.0);
    gl_Position = projection * vp;
    // 按速度模长染色：快暖慢冷
    float t = clamp(length(velocity.xyz) * u_cscale, 0.0, 1.0);
    vec3 cool = vec3(0.20, 0.45, 0.95);
    vec3 warm = vec3(1.00, 0.72, 0.25);
    vec3 hot  = vec3(1.00, 0.32, 0.12);
    vec3 col = (t < 0.5) ? mix(cool, warm, t * 2.0)
                         : mix(warm, hot, (t - 0.5) * 2.0);
    float fade = clamp(position.w * 0.5, 0.0, 1.0);   // 寿命将尽渐隐
    float sfade = (u_sfade == 1) ? mix(0.30, 1.0, t) : 1.0;
    v_col = vec4(col, u_alpha * (0.35 + 0.65 * fade) * sfade);
    float d = max(-vp.z, 0.2);
    gl_PointSize = clamp(u_point * 14.0 / d, 1.0, 14.0);
}
'''

DRAW_FRAG = '''#version 330 core
in vec4 v_col;
out vec4 frag;
void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    float a = exp(-3.0 * r2) * v_col.a;   // 高斯软边
    if (a < 0.004) discard;
    frag = vec4(v_col.rgb, a);
}
'''


def _compile(src, typ):
    """裸 GL 编译单个着色器（形态抄 particleward 的 ctypes 传参写法）。"""
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


def _link(vs_src, fs_src):
    prog = glCreateProgram()
    glAttachShader(prog, _compile(vs_src, GL_VERTEX_SHADER))
    glAttachShader(prog, _compile(fs_src, GL_FRAGMENT_SHADER))
    glLinkProgram(prog)
    ok = ctypes.c_int(0)
    glGetProgramiv(prog, GL_LINK_STATUS, ctypes.byref(ok))
    if not ok.value:
        log = ctypes.create_string_buffer(4096)
        glGetProgramInfoLog(prog, 4096, None, log)
        raise RuntimeError(f'link failed: {log.value!r}')
    return prog


# ------------------------------------------------------------ CPU 回退数学 --
def _uhash32(h):
    """numpy uint32 版 PCG 风格 hash（与 GLSL uhash 同式）。"""
    h = h.astype(np.uint32, copy=True)
    h ^= h >> np.uint32(16)
    h *= np.uint32(0x7feb352d)
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x846ca68b)
    h ^= h >> np.uint32(16)
    return h


class _ValueNoise:
    """向量化 3D value noise + curl（CPU 回退路径用，与 shader 同一套语义）。"""

    @staticmethod
    def _corner_hash(ci, dx, dy, dz):
        # 与 GLSL chash 相同：先各乘常量并按 2^32 回绕再异或
        t = (((ci[:, 0] + dx) * 73856093) ^ ((ci[:, 1] + dy) * 19349663)
             ^ ((ci[:, 2] + dz) * 83492791))
        return _uhash32(t).astype(np.float32) * (1.0 / 4294967296.0)

    @classmethod
    def noise(cls, q):
        q = q + 512.0
        c = np.floor(q)
        f = q - c
        f = f * f * (3.0 - 2.0 * f)
        ci = c.astype(np.int64)
        v00 = cls._corner_hash(ci, 0, 0, 0) + (cls._corner_hash(ci, 1, 0, 0)
                                               - cls._corner_hash(ci, 0, 0, 0)) * f[:, 0]
        v10 = cls._corner_hash(ci, 0, 1, 0) + (cls._corner_hash(ci, 1, 1, 0)
                                               - cls._corner_hash(ci, 0, 1, 0)) * f[:, 0]
        v01 = cls._corner_hash(ci, 0, 0, 1) + (cls._corner_hash(ci, 1, 0, 1)
                                               - cls._corner_hash(ci, 0, 0, 1)) * f[:, 0]
        v11 = cls._corner_hash(ci, 0, 1, 1) + (cls._corner_hash(ci, 1, 1, 1)
                                               - cls._corner_hash(ci, 0, 1, 1)) * f[:, 0]
        v0 = v00 + (v10 - v00) * f[:, 1]
        v1 = v01 + (v11 - v01) * f[:, 1]
        return (v0 + (v1 - v0) * f[:, 2]) * 2.0 - 1.0

    @classmethod
    def psi(cls, q):
        return np.stack([
            cls.noise(q),
            cls.noise(q + np.float32([31.416, -47.853, 12.793])),
            cls.noise(q + np.float32([-233.145, 88.23, -54.31]))], axis=1)

    @classmethod
    def curl(cls, q, e=0.55):
        ex = np.float32([e, 0, 0])
        ey = np.float32([0, e, 0])
        ez = np.float32([0, 0, e])
        cx = (cls.psi(q + ey)[:, 2] - cls.psi(q - ey)[:, 2]) \
           - (cls.psi(q + ez)[:, 1] - cls.psi(q - ez)[:, 1])
        cy = (cls.psi(q + ez)[:, 0] - cls.psi(q - ez)[:, 0]) \
           - (cls.psi(q + ex)[:, 2] - cls.psi(q - ex)[:, 2])
        cz = (cls.psi(q + ex)[:, 1] - cls.psi(q - ex)[:, 1]) \
           - (cls.psi(q + ey)[:, 0] - cls.psi(q - ey)[:, 0])
        return np.stack([cx, cy, cz], axis=1) / (2.0 * e)


# ------------------------------------------------------------------- app --
class Computeward(DemoApp):
    TITLE = 'computeward · 《洪流》（GPU 计算调度：百万粒子）'
    HELP = '1/2/3 preset  R reset  SPACE pause  drag/scroll camera  ESC quit'
    DURATION = 12.0
    SHOTS = ((3.5, 'docs/computeward_01.png'),
             (7.5, 'docs/computeward_02.png'),
             (11.0, 'docs/computeward_03.png'))
    FOV = 60.0

    # ------------------------------------------------------------ 建场 --
    def setup(self):
        self.dt = 1.0 / 60.0
        self.mode = 0
        self.reset_pending = True
        self._released = False

        # 能力探测（窗口已就绪）；模块缺失 / 探测不过 / --flag cpu 都走 CPU 回退
        det = {'available': False, 'gl_version': '?', 'reason': 'engine.compute 缺失'} \
            if not _HAS_COMPUTE else detect()
        self.gl_ver = (det['gl_version'] or '?')[:31]
        self.gpu = _HAS_COMPUTE and det['available'] \
            and 'cpu' not in self.args.flags
        if det['available'] and not self.gpu:
            print('[computeward] forced CPU fallback via --flag cpu')
        elif not det['available']:
            print(f"[computeward] GPU unavailable: {det['reason']}")

        # 绘制程序（裸 GL；两个 SSBO/回退 VBO 都供它的 attribute）
        self.draw_prog = _link(DRAW_VERT, DRAW_FRAG)
        self.dloc = {k: glGetUniformLocation(self.draw_prog, k.encode())
                     for k in ('projection', 'view', 'u_point', 'u_alpha',
                               'u_cscale', 'u_sfade')}

        self.count = (CLI_COUNT if self.gpu else COUNT_CPU) or \
            (COUNT_DEFAULT if self.gpu else COUNT_CPU)

        if self.gpu:
            try:
                self._build_gpu()
            except ComputeError as e:
                print(f'[computeward] GPU path failed ({e}); falling back to CPU')
                self.gpu = False
        if not self.gpu:
            self._build_cpu()

        self.vao = (ctypes.c_uint * 1)()
        glGenVertexArrays(1, self.vao)
        self._bind_attributes()

        print(f'[computeward] path={"GPU" if self.gpu else "CPU fallback"} '
              f'count={self.count:,} gl={self.gl_ver}')

        dist, ty, pitch = (MODE_CAM if self.gpu else CPU_MODE_CAM)[self.mode]
        self.cam = OrbitCamera(target=Vec3(0.0, ty, 0.0),
                               yaw=35.0, pitch=pitch, dist=dist)
        self.on_mouse_drag, self.on_mouse_scroll = orbit_camera_input(self, self.cam)

    def _build_gpu(self):
        """SSBO x2（带 numpy 初值）+ compute 程序。"""
        z = np.zeros((self.count, 4), dtype=np.float32)
        self.pos_buf = ComputeBuffer(z)          # xyz + life
        self.vel_buf = ComputeBuffer(z.copy())   # xyz + seed
        self.compute = ComputeShader(COMPUTE_SRC, name='computeward_sim')

    def _build_cpu(self):
        """回退：numpy 场 + 两个每帧 glBufferSubData 的 VBO。"""
        self.count = COUNT_CPU
        self.cpu_pos = np.zeros((self.count, 4), dtype=np.float32)
        self.cpu_vel = np.zeros((self.count, 4), dtype=np.float32)
        self.cpu_life = np.ones(self.count, dtype=np.float32)
        self.vbos = (ctypes.c_uint * 2)()
        glGenBuffers(2, self.vbos)
        for i in range(2):
            glBindBuffer(GL_ARRAY_BUFFER, self.vbos[i])
            glBufferData(GL_ARRAY_BUFFER, self.cpu_pos.nbytes, None,
                         GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _bind_attributes(self):
        """VAO：position/velocity 两根 vec4 属性。

        GPU 路径直接把 SSBO 绑成 GL_ARRAY_BUFFER 属性源（同一 buffer 对象，
        目标无关）；CPU 路径用自建 VBO。"""
        lp = glGetAttribLocation(self.draw_prog, b'position')
        lv = glGetAttribLocation(self.draw_prog, b'velocity')
        glBindVertexArray(self.vao[0])
        buf_pos = self.pos_buf.id if self.gpu else self.vbos[0]
        buf_vel = self.vel_buf.id if self.gpu else self.vbos[1]
        for buf, loc in ((buf_pos, lp), (buf_vel, lv)):
            glBindBuffer(GL_ARRAY_BUFFER, buf)
            if not self.gpu:            # 回退 VBO 在此分配存储
                glBufferData(GL_ARRAY_BUFFER, self.count * 16, None,
                             GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    # ---------------------------------------------------------- 重置 --
    def _reset_field(self):
        self.reset_pending = True
        if not self.gpu:
            rng = np.random.default_rng(20260701)
            self._cpu_spawn_slice(slice(None), rng)
            self.cpu_life[:] = self._life_tmp

    def _apply_cam(self, m):
        dist, ty, pitch = (MODE_CAM if self.gpu else CPU_MODE_CAM)[m]
        self.cam.set_target((0.0, ty, 0.0))
        self.cam.dist = dist

    # ---------------------------------------------------------- 交互 --
    def on_key(self, symbol, modifiers):
        if symbol in (key._1, key._2, key._3):      # 立即切预设并重置
            self._switch(int(symbol) - int(key._1))
        elif symbol == key.R:
            self._reset_field()
            self._apply_cam(self.mode)

    def _switch(self, m):
        self.mode = m % 3
        self._apply_cam(self.mode)
        self._reset_field()

    # ---------------------------------------------------------- 更新 --
    def update(self, dt):
        if self.paused:
            return
        self.dt = dt
        # 预设自动轮播（DURATION=12s 保证 time 循环后回落到 0 触发重置）
        m = int(self.time / PRESET_SEC) % 3
        if m != self.mode:
            self._switch(m)
        if not self.gpu:
            self._cpu_step(dt)

    # ------------------------------------------------- GPU 每帧（零回读）--
    def _gpu_step(self):
        prog = self.compute
        prog.use()
        self.pos_buf.bind(0)
        self.vel_buf.bind(1)
        prog.set('u_count', int(self.count))
        prog.set('u_time', float(self.time))
        prog.set('u_mode', int(self.mode))
        prog.set('u_damp', float(MODE_DAMP[self.mode]))
        if self.reset_pending:
            prog.set('u_reset', 1)
            prog.set('u_dt', 0.0)
            prog.dispatch(*prog.groups_for_n(self.count))
            barrier()
            self.reset_pending = False
        prog.set('u_reset', 0)
        prog.set('u_dt', float(self.dt))
        prog.dispatch(*prog.groups_for_n(self.count))
        barrier()
        prog.unuse()

    # ------------------------------------------------- CPU 每帧（回退）--
    def _cpu_spawn_slice(self, idx, rng):
        """按当前预设语义重播种（numpy 向量化，与 GLSL spawn 同式）。"""
        n = len(range(*idx.indices(self.count))) if isinstance(idx, slice) \
            else len(idx)
        if n == 0:
            return
        u1, u2, u3, u4, u5, u6, u7 = (rng.random(n) for _ in range(7))
        if self.mode == 0:
            pos = np.empty((n, 3), np.float32)
            ang = np.empty(n, np.float32)
            bulge = rng.random(n) < 0.12
            nb = int(bulge.sum())
            if nb:
                rad_b = 1.0 + 3.5 * rng.random(nb) * rng.random(nb)
                pos[bulge, 0] = np.cos(ang_b := rng.random(nb) * math.tau) * rad_b
                pos[bulge, 2] = np.sin(ang_b) * rad_b
                ang[bulge] = ang_b
            nd = n - nb
            if nd:
                rad_d = 3.0 + 16.0 * rng.random(nd) ** 0.8
                a_disc = rng.random(nd) * math.tau
                a_arm = np.floor(rng.random(nd) * 2.0) * math.pi \
                    + np.log(rad_d) * 1.7 + (rng.random(nd) - 0.5) * 0.35
                a = np.where(rng.random(nd) < 0.55, a_arm, a_disc)
                dmask = ~bulge
                pos[dmask, 0] = np.cos(a) * rad_d
                pos[dmask, 2] = np.sin(a) * rad_d
                ang[dmask] = a
            rad = np.hypot(pos[:, 0], pos[:, 2])
            pos[:, 1] = (rng.random(n) - 0.5) * 0.9 * np.exp(-rad * 0.10)
            vo = np.sqrt(60.0 / np.maximum(rad, 0.5))
            vel = np.stack([-np.sin(ang) * vo, (rng.random(n) - 0.5) * 0.3,
                            np.cos(ang) * vo], axis=1)
            life = 30.0 + rng.random(n) * 40.0
        elif self.mode == 1:
            pos = (np.stack([u1, u2, u3], axis=1) - 0.5) \
                * np.float32([12.0, 7.0, 12.0])
            vel = (np.stack([u4, u5, u6], axis=1) - 0.5) * 0.75
            life = 2.5 + u7 * 2.5
        else:
            jet = rng.integers(0, 6, n)
            ang = (jet / 6.0) * math.tau + (u1 - 0.5) * 0.06
            lat = np.stack([np.cos(ang), np.sin(ang)], axis=1) \
                * ((2.2 + u2 * 0.7)[:, None])
            pos = np.concatenate([lat * 0.1,
                                  np.full((n, 1), 0.05, np.float32)], axis=1)
            vel = np.concatenate([lat, ((8.0 + u3 * 3.0)[:, None])],
                                 axis=1)
            life = 2.8 + u4 * 1.2
        self.cpu_pos[idx, :3] = pos
        self.cpu_vel[idx, :3] = vel
        self._life_tmp = life.astype(np.float32)
        self.cpu_life[idx] = self._life_tmp

    def _cpu_step(self, dt):
        dt = min(dt, 1 / 30.0)      # 低帧率下钳步长，防积分飞散
        p3 = self.cpu_pos[:, :3]
        v3 = self.cpu_vel[:, :3]
        if self.mode == 0:
            r = np.maximum(np.hypot(p3[:, 0], p3[:, 2]), 0.4).astype(np.float32)
            radial = np.stack([p3[:, 0] / r, np.zeros_like(r), p3[:, 2] / r],
                              axis=1)
            tangent = np.stack([-radial[:, 2], np.zeros_like(r), radial[:, 0]],
                               axis=1)
            vo = np.sqrt(60.0 / r)
            vt = (v3 * tangent).sum(axis=1)
            force = -radial * (60.0 / (r * r))[:, None] \
                + tangent * ((vo - vt) * 0.8)[:, None] \
                + np.stack([np.zeros_like(r),
                            -p3[:, 1] * 2.2 - v3[:, 1] * 0.5,
                            np.zeros_like(r)], axis=1)
        elif self.mode == 1:
            q = p3 * 0.10 + np.float32([0.0, self.time * 0.15, 0.0])
            force = _ValueNoise.curl(q) * 55.0 - p3 * 0.15
        else:
            force = np.zeros_like(v3)
            force[:, 1] = -9.8
        v3 += force * np.float32(dt)
        v3 *= max(0.0, 1.0 - CPU_MODE_DAMP[self.mode] * dt)
        p3 += v3 * np.float32(dt)
        self.cpu_life -= dt
        if self.mode == 2:                          # 地面反弹
            m = (p3[:, 1] < 0.0) & (v3[:, 1] < 0.0)
            p3[m, 1] = 0.0
            v3[m, 1] *= -0.35
            v3[m, 0] *= 0.40
            v3[m, 2] *= 0.40
            self.cpu_life[m] *= 0.50
        dead = (self.cpu_life <= 0.0) | ((p3 * p3).sum(axis=1) > 2500.0)
        if dead.any():
            idx = np.nonzero(dead)[0]
            self._cpu_spawn_slice(idx, np.random.default_rng(
                int(self.time * 1000.0) + 7))

    def _cpu_upload(self):
        ptr = ctypes.POINTER(ctypes.c_float)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbos[0])
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.cpu_pos.nbytes,
                        self.cpu_pos.ctypes.data_as(ptr))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbos[1])
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.cpu_vel.nbytes,
                        self.cpu_vel.ctypes.data_as(ptr))
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    # ---------------------------------------------------------- 渲染 --
    def render(self):
        w, h = self.window.width, self.window.height
        glClearColor(0.02, 0.02, 0.05, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        projection = Mat4.perspective_projection(
            w / max(1, h), z_near=self.Z_NEAR, z_far=self.Z_FAR, fov=self.FOV)
        view = self.cam.view_matrix()
        self.window.projection = projection
        self.window.view = view

        if not self.paused:
            if self.gpu:
                self._gpu_step()
            else:
                self._cpu_upload()

        # 加色混合画点（画完还原，别污染基类 HUD）
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glUseProgram(self.draw_prog)
        f32 = np.asarray(list(projection) + list(view),
                         dtype=np.float32).ctypes.data_as(
                             ctypes.POINTER(ctypes.c_float))
        glUniformMatrix4fv(self.dloc['projection'], 1, GL_FALSE, f32)
        v16 = np.ascontiguousarray(np.asarray(list(view), dtype=np.float32))
        glUniformMatrix4fv(self.dloc['view'], 1, GL_FALSE,
                           v16.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        pt = MODE_POINT if self.gpu else CPU_MODE_POINT
        al = MODE_ALPHA if self.gpu else CPU_MODE_ALPHA
        glUniform1f(self.dloc['u_point'], pt[self.mode])
        glUniform1f(self.dloc['u_alpha'], al[self.mode])
        glUniform1f(self.dloc['u_cscale'], MODE_CSCALE[self.mode])
        glUniform1i(self.dloc['u_sfade'], 1 if self.mode == 1 else 0)
        glBindVertexArray(self.vao[0])
        glDrawArrays(GL_POINTS, 0, int(self.count))
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthMask(GL_TRUE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.window.projection = self.perspective

    # ------------------------------------------------------------ 状态 --
    def status_text(self):
        # HUD 用英文（基类 HUD 对中文逐帧 layout 开销大，见 particleward 实测）
        return (f'[{"GPU" if self.gpu else "CPU fallback"}] '
                f'{self.count:,} particles'
                f'  mode {self.mode + 1}/3 {MODES_EN[self.mode]}'
                f'  GL {self.gl_ver}'
                f'  {self.fps:5.1f} fps'
                f'{"  [PAUSED]" if self.paused else ""}')

    # ------------------------------------------------------------ 收尾 --
    def close(self):
        if not self._released:
            self._released = True
            if getattr(self, 'gpu', False):
                self.pos_buf.release()
                self.vel_buf.release()
                self.compute.release()
            else:
                if getattr(self, 'vbos', None) is not None:
                    glDeleteBuffers(2, self.vbos)
            if getattr(self, 'vao', None) is not None:
                glDeleteVertexArrays(1, self.vao)
            if getattr(self, 'draw_prog', 0):
                glDeleteProgram(self.draw_prog)
        super().close()


# ------------------------------------------------------------------ 入口 --
def main():
    """先把自有开关 --count 从 argv 摘出来（harness 的 parse_args 不认识它）。"""
    global CLI_COUNT
    argv = sys.argv[1:]
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == '--count' and i + 1 < len(argv):
            CLI_COUNT = int(argv[i + 1])
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    run(Computeward, rest)


if __name__ == '__main__':
    main()
