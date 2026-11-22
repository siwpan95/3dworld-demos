# -*- coding: utf-8 -*-
"""《谪仙 / Posterward》—— 复古未来主义动态海报（引擎能力拼装版）。

与初版（纯单文件 shader）的区别：**不再自包含**，直接消费 3dworld 引擎的
三条真实能力（0.7.4 起）——演示的意义从"一段 shader"变成"引擎能力在海报
形态上的组合证明"：

1. **引擎 GLSL 注入件** ``world3d.engine.sdf2d_glsl``：胶囊 SDF / smin /
   阿基米德螺线带（祥云）/ 边缘带 / 分层着色黄金参考 ``shadeLayer``，
   与 ``engine/sdf2d.py`` CPU 版同式同常数——shader 里不再手抄公式；
2. **匣境资产通道** ``world3d.engine.gltf.GLTF``：三件程序化金饰
   （环 / 绸带环 / 玉璧，由 ``scripts/make_posterward_glb.py`` 生成
   ``assets/posterward_ornaments.glb``）经真实 glTF 2.0 加载栈
   （``GLTF.load → build_scene_ex → Mesh``）进海报，PBR 材质
   ``PRESETS['gold']`` + ``LightEnvironment``（墨色天穹 / 暖阳）打光，
   与 gltfward《匣境》同一通道；
3. **引擎 CJK 字体** ``world3d.engine.fonts.resolve_cjk_font``：海报题签
   从抽象墨点换成真中文标题「谪仙」（初版的占位墨点已删）。

画面分层（每帧）：深墨背景+金尘 → 复古未来主义日轮（切缝 + 近晕色散）→
祥云螺线 → 飘带 → 仙女俯冲（分层 2D SDF，发髻/广袖/飘裙/金饰缘）→
厚涂后处理（fbm 笔触角 + 鬃毛脊线 + 亮度分档）→ **3D 金饰层**（PBR，
环绕人物缓旋）→ 海报版式（双线边框 / 朱印 / 暗角 / 颗粒）→ **真字题签**。

光照三教训（CPU/GLSL 双侧钉死，见 engine/sdf2d.py docstring）：rim 要
"朝光 × 近剪影"双条件；边缘带从内侧收紧（反转 smoothstep 会对一切内部点
恒为 1）；底光项防全黑剪影。沿用 crystalward 的独立二次验证。

用法::

    python examples/posterward.py
    python examples/posterward.py --smoke 3     # 冒烟（CI / verify_demos 会跑这个）
    python examples/posterward.py --shot        # 抓剧照到 docs/posterward_01..03.png
    python examples/posterward.py --flag hold   # 定格（观察造型 / 对图迭代）
    python examples/posterward.py --flag no3d      关 3D 金饰层（对照）
    python examples/posterward.py --flag notext    关真字题签（对照）
    python examples/posterward.py --flag nograin   关胶片颗粒
    python examples/posterward.py --flag noimpasto 关厚涂笔触（对照）

按键：SPACE 暂停   G 颗粒   I 厚涂   C 祥云   拖动/滚轮 转金饰视角   ESC 退出

资产许可：金饰 GLB 由本仓脚本程序化生成（无第三方素材），MIT。
调研尾注见 examples/README.md 对应行（螺旋 SDF / 厚涂笔触 / 域重复）。
"""
import math
import os

import pyglet
from pyglet.gl import (GL_DEPTH_TEST, GL_TRIANGLES, glEnable, glDisable,
                       glDepthMask)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key

from world3d.platform.app import DemoApp, make_fullscreen_quad, run, PROJECT_ROOT

# ---- 引擎能力 1：2D SDF GLSL 注入件（与 engine/sdf2d.py 同式同常数） ----
from world3d.engine.sdf2d_glsl import sdf2d_glsl                # noqa: E402
# ---- 引擎能力 2：匣境资产通道（真实 glTF 2.0 加载栈） ----
from world3d.engine.camera import OrbitCamera                   # noqa: E402
from world3d.engine.env import LightEnvironment                 # noqa: E402
from world3d.engine.gltf import GLTF                            # noqa: E402
from world3d.engine.material import PRESETS                     # noqa: E402
from world3d.engine.mathkit import mat4_from_trs, mat4_mul      # noqa: E402
from world3d.engine.mesh import Mesh                            # noqa: E402
from world3d.engine.shaders import get as get_shader            # noqa: E402
# ---- 引擎能力 3：CJK 字体解析（题签真字） ----
from world3d.engine.fonts import resolve_cjk_font               # noqa: E402

GLB_PATH = os.path.join(PROJECT_ROOT, 'assets', 'posterward_ornaments.glb')

VERTEX_SRC = '''#version 330 core
in vec3 position;
in vec4 colors;
out vec2 v_uv;
out vec3 v_tint;
void main() {
    v_uv = position.xy * 0.5 + 0.5;
    v_tint = colors.rgb;
    gl_Position = vec4(position.xy, 0.0, 1.0);
}'''

# ------------------------------------------------------------------
# 海报片元 = 引擎 GLSL 片段（s2_* 前缀）+ 宿主构图。注入件必须在前——
# 宿主的造型函数（dHair/dSkin/…）直接调用 s2_sdCap2 / s2_smin2 /
# s2_spiralBand2 / s2_shadeLayer，不再手抄公式。
# ------------------------------------------------------------------
FRAGMENT_HOST = '''in vec2 v_uv;
in vec3 v_tint;
out vec4 final_colors;

uniform float u_time;
uniform vec2  u_res;
uniform float u_grain;      // 胶片颗粒开关（G 键 / --flag nograin）
uniform float u_impasto;    // 厚涂笔触开关（I 键 / --flag noimpasto）
uniform float u_clouds;     // 祥云开关（C 键）

// ---------- 基础 ----------
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) { v += a * vnoise(p); p = p * 2.03 + 11.7; a *= 0.5; }
    return v;
}
vec2 rot2(vec2 p, float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c) * p;
}

// ---------- 飘带：动画折线段 + 沿程变宽（构图逻辑，宿主自有） ----------
float ribbon(vec2 p, float t, float seed) {
    float d = 1e5;
    vec2 prev = vec2(0.0);
    for (int i = 0; i < 5; i++) {
        float fi = float(i);
        float ph = seed * 7.3 + fi * 1.1;
        vec2 c = vec2(
            -0.9 + fi * 0.42 + 0.10 * sin(t * 0.9 + ph),
            -0.55 + fi * 0.30 + 0.16 * sin(t * 0.7 + ph * 1.7) + fi * fi * 0.05);
        if (i > 0) {
            float w = mix(0.045, 0.010, fi / 4.0);
            d = s2_smin2(d, s2_sdCap2(p, prev, c, w), 0.004);
        }
        prev = c;
    }
    return d;
}

// ---------- 祥云：引擎螺线带 + 头盘 + 伴飞小球 ----------
float cloud(vec2 p, float t) {
    p.x += 0.012 * t;                          // 慢速漂移
    float r = length(p);
    float sp = s2_spiralBand2(p, 0.016, 0.017);
    sp = max(sp, r - 0.155);                   // 只留最外两圈
    float head = length(p - vec2(0.16, 0.06)) - 0.048;
    float mate = length(p - vec2(-0.12, 0.10)) - 0.028;
    return s2_smin2(s2_smin2(sp, head, 0.010), mate, 0.008);
}

// ---------- 仙女（局部坐标：俯冲轴为 -Y，头在下） ----------
// 发（髻 + 贴着身轴向上飘的发流——俯冲时发丝向上；略偏左压在裙面上）
float dHair(vec2 p, float t) {
    float d = length(p - vec2(0.13, -0.44)) - 0.075;          // 髻
    d = s2_smin2(d, length(p - vec2(0.185, -0.36)) - 0.030, 0.012); // 髻顶珠
    float sway = 0.025 * sin(t * 1.6 + p.y * 5.0);
    d = s2_smin2(d, s2_sdCap2(p + vec2(sway, 0.0),
        vec2(0.02, -0.50), vec2(-0.02, -0.02), 0.042), 0.04);
    d = s2_smin2(d, s2_sdCap2(p + vec2(sway * 1.4, 0.0),
        vec2(-0.02, -0.02), vec2(0.02, 0.38), 0.026), 0.03);
    d = s2_smin2(d, s2_sdCap2(p + vec2(-sway, 0.0),
        vec2(0.10, -0.48), vec2(0.08, 0.10), 0.018), 0.03);
    return d;
}
// 肤（头 / 颈 / 躯干 / 双臂舒展 / 双腿拖曳）——关节用 smin 融合，硬 min 读成木偶
float dSkin(vec2 p) {
    float d = length(p - vec2(0.02, -0.58)) - 0.105;          // 头
    d = s2_smin2(d, s2_sdCap2(p, vec2(0.02, -0.48), vec2(0.045, -0.34), 0.038), 0.03);
    d = s2_smin2(d, s2_sdCap2(p, vec2(0.045, -0.36), vec2(0.075, -0.14), 0.105), 0.05);
    d = s2_smin2(d, s2_sdCap2(p, vec2(0.075, -0.16), vec2(0.095, -0.02), 0.070), 0.04);
    // 左臂（舒展弧线：肩→肘微垂→手外扬；肘部夹角钝化去"昆虫足"感）
    d = s2_smin2(d, s2_sdCap2(p, vec2(-0.04, -0.30), vec2(-0.34, -0.26), 0.052), 0.035);
    d = s2_smin2(d, s2_sdCap2(p, vec2(-0.34, -0.26), vec2(-0.60, -0.40), 0.040), 0.030);
    d = s2_smin2(d, length(p - vec2(-0.63, -0.42)) - 0.030, 0.02);        // 手
    // 右臂
    d = s2_smin2(d, s2_sdCap2(p, vec2(0.14, -0.30), vec2(0.42, -0.22), 0.050), 0.035);
    d = s2_smin2(d, s2_sdCap2(p, vec2(0.42, -0.22), vec2(0.66, -0.36), 0.038), 0.030);
    d = s2_smin2(d, length(p - vec2(0.69, -0.38)) - 0.028, 0.02);         // 手
    // 腿（上拖，膝部微屈，脚尖绷直收尾）
    d = s2_smin2(d, s2_sdCap2(p, vec2(0.10, -0.03), vec2(0.17, 0.30), 0.062), 0.04);
    d = s2_smin2(d, s2_sdCap2(p, vec2(0.17, 0.30), vec2(0.24, 0.58), 0.040), 0.030);
    d = s2_smin2(d, s2_sdCap2(p, vec2(0.24, 0.58), vec2(0.27, 0.72), 0.026), 0.02);
    d = s2_smin2(d, s2_sdCap2(p, vec2(0.12, -0.01), vec2(0.28, 0.28), 0.058), 0.04);
    d = s2_smin2(d, s2_sdCap2(p, vec2(0.28, 0.28), vec2(0.40, 0.60), 0.036), 0.030);
    d = s2_smin2(d, s2_sdCap2(p, vec2(0.40, 0.60), vec2(0.44, 0.74), 0.024), 0.02);
    return d;
}
// 裙（腰际下的大形 + 随风的波浪缘；半径必须大幅压过身体剪影，
// 否则绯橙面料全被后画的皮肤盖住、只剩描金边——初版踩过）
float dDress(vec2 p, float t) {
    vec2 q = p;
    q.x += 0.030 * sin(q.y * 13.0 + t * 2.2);                 // 褶皱随风
    float d = s2_sdCap2(q, vec2(0.085, -0.12), vec2(0.17, 0.30), 0.235);
    d = s2_smin2(d, s2_sdCap2(q, vec2(0.17, 0.28), vec2(0.32, 0.62), 0.170), 0.02);
    d = s2_smin2(d, s2_sdCap2(q, vec2(0.10, -0.10), vec2(0.44, 0.88), 0.080), 0.07);
    d = s2_smin2(d, s2_sdCap2(q, vec2(0.05, -0.18), vec2(-0.34, 0.28), 0.070), 0.07);
    d = s2_smin2(d, s2_sdCap2(q, vec2(0.14, 0.30), vec2(-0.02, 0.72), 0.055), 0.06);
    return d;
}
// 广袖（沿双臂的外扩软形，肘后翻飞——宽大柔软，同样压过手臂剪影）
float dSleeve(vec2 p, float t) {
    vec2 q = p;
    q.y += 0.025 * sin(q.x * 11.0 + t * 2.6);                 // 袖缘摆动
    float d = s2_sdCap2(q, vec2(-0.04, -0.30), vec2(-0.36, -0.22), 0.150);
    d = s2_smin2(d, s2_sdCap2(q, vec2(-0.36, -0.22), vec2(-0.64, -0.38), 0.090), 0.04);
    d = s2_smin2(d, s2_sdCap2(q, vec2(-0.28, -0.32), vec2(-0.60, 0.04), 0.070), 0.05);
    d = s2_smin2(d, s2_sdCap2(q, vec2(0.14, -0.30), vec2(0.44, -0.18), 0.140), 0.02);
    d = s2_smin2(d, s2_sdCap2(q, vec2(0.44, -0.18), vec2(0.70, -0.34), 0.085), 0.04);
    d = s2_smin2(d, s2_sdCap2(q, vec2(0.36, -0.26), vec2(0.70, 0.06), 0.065), 0.05);
    return d;
}
// SDF 梯度（伪法线）——GLSL 没有函数指针，逐层显式写
vec2 gradDress(vec2 p, float t) {
    float e = 0.008;
    return normalize(vec2(dDress(p + vec2(e, 0.0), t) - dDress(p - vec2(e, 0.0), t),
                          dDress(p + vec2(0.0, e), t) - dDress(p - vec2(0.0, e), t)) + 1e-6);
}
vec2 gradSleeve(vec2 p, float t) {
    float e = 0.008;
    return normalize(vec2(dSleeve(p + vec2(e, 0.0), t) - dSleeve(p - vec2(e, 0.0), t),
                          dSleeve(p + vec2(0.0, e), t) - dSleeve(p - vec2(0.0, e), t)) + 1e-6);
}
vec2 gradHair(vec2 p, float t) {
    float e = 0.006;
    return normalize(vec2(dHair(p + vec2(e, 0.0), t) - dHair(p - vec2(e, 0.0), t),
                          dHair(p + vec2(0.0, e), t) - dHair(p - vec2(0.0, e), t)) + 1e-6);
}
vec2 gradSkin(vec2 p) {
    float e = 0.008;
    return normalize(vec2(dSkin(p + vec2(e, 0.0)) - dSkin(p - vec2(e, 0.0)),
                          dSkin(p + vec2(0.0, e)) - dSkin(p - vec2(0.0, e))) + 1e-6);
}

// ---------- 分层着色：薄壳包住引擎黄金参考 s2_shadeLayer ----------
// 常数即《谪仙》三版迭代收敛值（key 0.30+0.42 / fill 0.16 / 边缘带 -0.055..-0.006）
vec3 shade(float d, vec2 n, vec3 base, vec3 rimCol, float rimAmt, float luma) {
    return s2_shadeLayer(d, n, base, rimCol, rimAmt, luma,
                         vec2(-0.55, -0.75), vec2(0.45, 0.75),
                         0.30, 0.42, 0.16);
}

void main() {
    // --- 以短边归一化的中心坐标 ---
    vec2 p = (v_uv - 0.5) * vec2(u_res.x / u_res.y, 1.0) * 2.0;
    float t = u_time;

    // --- 缓慢推镜（海报的呼吸感）---
    float zoom = 1.0 + 0.05 * sin(t * 0.21);
    p /= zoom;
    p.y -= 0.02 * sin(t * 0.13);

    // --- 1) 深墨背景 ---
    vec3 col = mix(vec3(0.035, 0.022, 0.045), vec3(0.10, 0.045, 0.02),
                   smoothstep(-1.1, 1.1, p.y));
    col *= 0.85 + 0.15 * fbm(p * 2.2);

    // 金尘
    float dust = hash21(floor(p * 42.0) + floor(t * 0.5) * 0.017);
    col += step(0.9965, dust) * vec3(1.0, 0.75, 0.4) * 0.55;

    // --- 2) 复古未来主义日轮（右上偏置 = 不对称构图） ---
    vec2 sp = p - vec2(0.62, 0.34);
    float sd = length(sp);
    float sunR = 0.52;
    // 环纹：下半横向切缝（上密下疏，经典 slat sun 的中式化）
    float slat = 1.0;
    if (sp.y < 0.0) {
        float g = smoothstep(0.0, sunR, -sp.y);               // 越往下缝越宽
        slat = smoothstep(0.42 - 0.36 * g, 0.48 - 0.36 * g,
                          fract(sd * 9.0 + g * 0.5));
        slat = mix(slat, 1.0, step(0.98, fract(sd * 9.0)));
    }
    float sunMask = smoothstep(sunR + 0.004, sunR - 0.004, sd) * slat;
    // 光晕：贴着日轮的两层窄晕（晕圈半径失控会把整幅洗成土黄——首版踩过）
    float ha = smoothstep(sunR * 1.38, sunR * 0.99, sd);
    col += vec3(0.60, 0.20, 0.05) * ha * 0.28;                        // 近晕
    col += vec3(0.40, 0.16, 0.04) * smoothstep(sunR * 1.85, sunR, sd) * 0.10; // 远晕
    col.r += 0.06 * smoothstep(sunR * 1.42, sunR * 1.02, sd);         // 近晕色散
    col.b += 0.04 * smoothstep(sunR * 1.30, sunR * 0.92, sd);
    // 日轮本体：外圈金环 + 内盘暖橙
    float ringG = smoothstep(0.030, 0.006, abs(sd - (sunR - 0.045)));
    col += vec3(1.0, 0.72, 0.28) * ringG * 0.9;
    col += vec3(1.0, 0.55, 0.16) * sunMask * (0.55 + 0.25 * fbm(sp * 6.0 + t * 0.3));
    // 内嵌回纹盘（抽象：同心短弧）
    float arcs = smoothstep(0.9, 1.0, sin(sd * 26.0 + atan(sp.y, sp.x) * 6.0));
    col += vec3(1.0, 0.85, 0.5) * arcs * sunMask * 0.10;

    // --- 3) 祥云（两簇压在日轮边缘读成剪影，一簇在左下） ---
    if (u_clouds > 0.5) {
        float dc  = cloud(p * 1.10 - vec2(0.42, 0.10), t * 0.8) / 1.10;
        float dc2 = cloud(p * 1.30 - vec2(0.80, 0.72), -t * 0.6) / 1.30;
        float dc3 = cloud(p * 0.85 - vec2(-0.95, -0.80), t * 0.4) / 0.85;
        float cd = s2_smin2(s2_smin2(dc, dc2, 0.004), dc3, 0.004);
        // 云体：比背景略亮的暖灰剪影 + 金边（纯黑会融进深墨背景）
        vec3 cloudCol = vec3(0.14, 0.07, 0.09) * (0.8 + 0.5 * fbm(p * 5.0 + t * 0.1));
        col = mix(col, cloudCol, smoothstep(0.006, -0.004, cd));
        float edge = smoothstep(0.016, 0.002, abs(cd));
        col += vec3(1.0, 0.62, 0.20) * edge * 0.38;
        float cr = smoothstep(0.006, -0.002, cd);
        col += vec3(0.9, 0.55, 0.15) * cr * 0.12 * fbm(p * 8.0 + t);
    }

    // --- 4) 飘带（两条后景 + 一条前景） ---
    float rb1 = ribbon(rot2(p - vec2(0.10, 0.30), -0.30), t, 1.0);
    float rb2 = ribbon(rot2(p - vec2(-0.25, 0.05), 0.18), t * 1.13, 2.3);
    float rb3 = ribbon(rot2(p - vec2(-0.05, -0.35), -0.10), t * 0.87, 4.1);
    // 远端收细后读成"乱线"横穿画面——淡出半径收紧
    float rbfade = smoothstep(1.15, 0.55, length(p));
    vec3 rb1c = mix(vec3(1.0, 0.50, 0.12), vec3(0.85, 0.16, 0.05),
                    0.5 + 0.5 * sin(rb1 * 60.0));
    col = mix(col, rb1c, smoothstep(0.010, -0.005, rb1) * 0.9 * rbfade);
    col = mix(col, vec3(0.95, 0.32, 0.08),
              smoothstep(0.008, -0.004, rb2) * 0.8 * rbfade);

    // --- 5) 仙女（整体旋转 = 俯冲轴线倾斜；左下偏置 = 不对称构图） ---
    vec2 q = rot2(p - vec2(-0.18, -0.05), 0.32) * 1.30;

    float dd = dDress(q, t);
    float dsv = dSleeve(q, t);
    float dh = dHair(q, t);
    float dsk = dSkin(q);

    // 厚涂光照只在造型近旁才算梯度（省掉 4 次全量 SDF 求值的 90% 空转）。
    // 注意这里用 min 而不是 smin：k=0 的 smin 会除零（engine kit 的语义边界）。
    vec3 dressCol = col, sleeveCol = col, hairCol = col, skinCol = col;
    if (min(min(dd, dsv), min(dh, dsk)) < 0.12) {
        vec2 nd = gradDress(q, t);
        vec2 nh = gradHair(q, t);
        vec2 ns = gradSkin(q);
        vec2 nsv = gradSleeve(q, t);

        // 裙：高对比绯橙（压暗保饱和）+ 金织纹
        vec3 dressBase = mix(vec3(0.30, 0.040, 0.012), vec3(0.78, 0.22, 0.04),
                             0.5 + 0.5 * nd.y);
        float weave = sin(q.x * 90.0) * sin(q.y * 90.0);      // 编织纹理
        dressBase *= 0.88 + 0.12 * weave;
        // 金缎光泽（fbm 频率太低会读成迷彩斑块——提频 + 收阈值）
        float bands = smoothstep(0.58, 0.82, fbm(q * 16.0 + t * 0.15));
        dressBase = mix(dressBase, vec3(0.80, 0.55, 0.14), bands * 0.30);
        // 广袖：柔软、比裙略亮
        vec3 sleeveBase = mix(vec3(0.42, 0.09, 0.025), vec3(0.92, 0.40, 0.10),
                              0.5 + 0.5 * nsv.y);
        sleeveBase *= 0.90 + 0.10 * weave;
        float fl = smoothstep(0.60, 0.85, fbm(q * 12.0 - t * 0.2));
        sleeveBase = mix(sleeveBase, vec3(0.92, 0.68, 0.28), fl * 0.26);

        dressCol = shade(dd, nd, dressBase, vec3(1.0, 0.62, 0.22), 0.75, 0.85);
        sleeveCol = shade(dsv, nsv, sleeveBase, vec3(1.0, 0.68, 0.28), 0.70, 0.92);
        hairCol = shade(dh, nh, vec3(0.06, 0.045, 0.075),
                        vec3(0.85, 0.70, 0.95), 0.35, 1.0);
        // 肤：冷冽白皙（压一档亮度，别抢面料的戏）+ 橙金 rim（日轮逆光）
        skinCol = shade(dsk, ns, vec3(0.82, 0.77, 0.75),
                        vec3(1.0, 0.55, 0.18), 1.10, 0.94);
    }
    col = mix(col, dressCol, smoothstep(0.004, -0.004, dd));
    col = mix(col, sleeveCol, smoothstep(0.004, -0.004, dsv));
    // 金饰缘：裙/袖边一圈金线
    float trim = smoothstep(0.012, 0.002, abs(min(dd, dsv)));
    col += vec3(1.0, 0.80, 0.35) * trim * 0.45;
    col = mix(col, hairCol, smoothstep(0.003, -0.003, dh));
    col = mix(col, skinCol, smoothstep(0.003, -0.003, dsk));

    // 金属饰品闪点（发髻珠 / 腰饰 / 袖扣）—— 十字星闪光
    float glint = 0.0;
    for (int i = 0; i < 3; i++) {
        vec2 gp = q - vec2(0.185 + 0.14 * float(i), -0.36 + 0.30 * float(i));
        float tw = pow(max(0.0, sin(t * 3.0 + float(i) * 2.1)), 6.0);
        float star = pow(max(0.0, 1.0 - abs(gp.x) * 90.0), 3.0)
                   + pow(max(0.0, 1.0 - abs(gp.y) * 90.0), 3.0);
        glint += tw * star * step(0.012, length(gp));
    }
    col += vec3(1.0, 0.85, 0.5) * glint * 0.8;

    // --- 6) 前景飘带（半透明压在人物之上 = 空间深度） ---
    col = mix(col, vec3(1.0, 0.50, 0.12),
              smoothstep(0.005, -0.003, rb3) * 0.35 * rbfade);

    // --- 7) 厚涂后处理 ---
    if (u_impasto > 0.5) {
        // fbm 驱动笔触角 → 各向异性格子 → 鬃毛脊线 + 单元明度漂移
        // （脊线频率太低会读成"木纹"——提频 + 降幅）
        float ang = fbm(p * 2.6) * 6.283;
        vec2 r = rot2(p, ang);
        vec2 cell = vec2(13.0, 34.0);
        vec2 id = floor(r * cell);
        float cid = hash21(id);
        float ridge = 0.965 + 0.035 * sin(r.y * cell.y * 18.85 + cid * 6.283);
        col *= ridge;
        col *= 0.96 + 0.08 * cid;
        // 亮度分档（thick paint 的"色阶感"）
        float lum = dot(col, vec3(0.299, 0.587, 0.114));
        float lq = (floor(lum * 5.0) + smoothstep(0.2, 0.8, fract(lum * 5.0))) / 5.0;
        col *= mix(1.0, lq / max(lum, 1e-3), 0.22);
    }

    // --- 8) 海报版式：双线边框 + 朱印（题签让位给引擎真字，见 Python 层） ---
    vec2 b1 = abs(p) - vec2(0.94 * u_res.x / u_res.y, 0.92);
    float frame1 = smoothstep(0.006, 0.002, abs(max(b1.x, b1.y)));
    vec2 b2 = abs(p) - vec2(0.905 * u_res.x / u_res.y, 0.885);
    float frame2 = smoothstep(0.004, 0.0012, abs(max(b2.x, b2.y)));
    col = mix(col, vec3(0.85, 0.60, 0.25), frame1 * 0.8 + frame2 * 0.5);
    // 朱印（右下红章 + 白文）
    vec2 se = p - vec2(0.72 * u_res.x / u_res.y, -0.70);
    float seal = smoothstep(0.052, 0.046, max(abs(se.x) - 0.008, abs(se.y)));
    col = mix(col, vec3(0.62, 0.10, 0.08), seal * 0.85);
    float strokes = 0.0;
    strokes += smoothstep(0.014, 0.006, abs(se.y - 0.012)) * smoothstep(0.036, 0.030, abs(se.x));
    strokes += smoothstep(0.014, 0.006, abs(se.x - 0.014)) * smoothstep(0.030, 0.024, abs(se.y));
    strokes += smoothstep(0.010, 0.004, length(se - vec2(-0.016, -0.016)));
    col = mix(col, vec3(0.95, 0.88, 0.80), strokes * seal * 0.9);

    // --- 9) 暗角 + 颗粒 + 色调映射 ---
    float vig = smoothstep(1.55, 0.35, length(v_uv - 0.5) * 1.75);
    col *= mix(0.42, 1.0, vig);
    col += (hash21(v_uv * u_res + fract(t) * 61.7) - 0.5) * 0.055 * u_grain;
    col = col / (col + vec3(0.82));
    col = pow(max(col, 0.0), vec3(0.88));

    final_colors = vec4(col * v_tint, 1.0);
}'''

#: 引擎 GLSL 片段（s2_ 前缀）拼在宿主之前——宿主造型函数直接消费 s2_*。
#: ⚠️ `#version` 必须是**整个源码的第一行**，所以版本行单独加在装配最顶端
#: （把引擎片段拼在 `#version` 之后的宿主源码前面，是本 demo 的装配顺序）。
FRAGMENT_SRC = ('#version 330 core\n'
                + sdf2d_glsl('s2_') + FRAGMENT_HOST)


def _quat_axis(axis, ang):
    """轴角 → 引擎四元数 (w,x,y,z)（axis 归一化）。"""
    lx = math.sqrt(sum(c * c for c in axis)) or 1.0
    s = math.sin(ang / 2.0)
    return (math.cos(ang / 2.0),
            axis[0] / lx * s, axis[1] / lx * s, axis[2] / lx * s)


class PosterwardApp(DemoApp):
    TITLE = 'posterward · 谪仙 (复古未来主义动态海报 · 引擎能力拼装版)'
    HELP = ('SPACE 暂停   G 颗粒   I 厚涂   C 祥云   '
            '拖动/滚轮 转金饰视角   ESC 退出')
    DURATION = 24.0
    HUD = True
    SHOTS = ((3.0, 'docs/posterward_01.png'),
             (9.5, 'docs/posterward_02.png'),
             (16.0, 'docs/posterward_03.png'))

    def setup(self):
        glDisable(GL_DEPTH_TEST)                    # 海报层是纯 2D 全屏 pass
        flags = self.args.flags
        self.grain = 0.0 if 'nograin' in flags else 1.0
        self.impasto = 0.0 if 'noimpasto' in flags else 1.0
        self.clouds = 0.0 if 'noclouds' in flags else 1.0
        self.show3d = 'no3d' not in flags
        self.showtext = 'notext' not in flags
        if 'hold' in flags:                         # 定格（观察造型 / 对图迭代）
            self.paused = True

        # ---- 海报底（全屏片元着色器，消费引擎 sdf2d_glsl 片段） ----
        self.program = ShaderProgram(Shader(VERTEX_SRC, 'vertex'),
                                     Shader(FRAGMENT_SRC, 'fragment'))
        self.quad = make_fullscreen_quad(self.program)

        # ---- 引擎能力 2：匣境通道载入金饰 GLB ----
        self.glb_name = os.path.basename(GLB_PATH)
        self.tri_count = 0
        self.meshes = []
        self.env = None
        self.pbr = None
        if self.show3d:
            self.env = LightEnvironment(
                sun_dir=(0.45, 0.75, 0.32), sun_color=(1.0, 0.82, 0.55),
                sun_intensity=3.4, ibl_intensity=1.05,
                sky_zenith=(0.10, 0.055, 0.10),           # 墨色天穹（留一点紫，别全黑）
                sky_horizon=(0.90, 0.42, 0.10),           # 暖橙地平（配日轮）
                sky_ground=(0.16, 0.07, 0.05)).bake()
            self.pbr = get_shader('pbr')
            gltf = GLTF.load(GLB_PATH)
            _scene, pending, _nmap = gltf.build_scene_ex()
            for _node, md, _mi in pending:
                mat = PRESETS['gold'].clone(name='posterward_gold')
                src = gltf.material(_mi)
                mat.base_color = src.base_color          # 资产自带的金 (0.92,0.70,0.28)
                mat.metallic = min(src.metallic, 0.92)   # 无贴图全金属易读成白镜面
                mat.roughness = max(src.roughness, 0.22)
                # 极轻暖自发光：暗部保住"金"的读感（纯 IBL 在墨色天穹下读成深铜）
                mat.emissive = (0.13, 0.075, 0.015)
                self.meshes.append(Mesh(md, self.pbr, mat))
            self.tri_count = sum(m.data.triangle_count for m in self.meshes)

        # ---- 相机：世界单位与海报坐标对齐（z=0 处半高恰为 1.0） ----
        # 引擎 yaw 约定：eye = target + dist·(cos·cos(yaw), sin(pitch), cos·sin(yaw))
        # —— yaw=0 的相机在 +X 轴、**不是** +Z（踩着一次，金饰整体转了 90°）；
        # 正对海报平面要 yaw=90°，pitch=0 需显式放开 pitch_range（默认地面视角 5..89）。
        # fov=30° → dist = 1/tan(15°)，世界 (x, y, 0) 即海报坐标 p=(x, y)。
        self.cam_dist = 1.0 / math.tan(math.radians(15.0))
        self.cam = OrbitCamera(target=(0.0, 0.0, 0.0), yaw=90.0, pitch=0.0,
                               dist=self.cam_dist,
                               pitch_range=(-20.0, 20.0),
                               dist_range=(self.cam_dist * 0.6,
                                           self.cam_dist * 1.6),
                               fov=30.0, near=0.1, far=30.0)
        self.on_mouse_drag, self.on_mouse_scroll = self._orbit_input()

        # ---- 引擎能力 3：真字题签（竖排「谪仙」+ 底部小注） ----
        self.font_name = resolve_cjk_font()
        self.labels = []
        if self.showtext:
            w, h = self.window.width, self.window.height
            gold = (235, 204, 140, 225)
            dim = (200, 168, 110, 170)
            common = {'font_name': self.font_name, 'anchor_x': 'center',
                      'anchor_y': 'center', 'multiline': False}
            self.labels = [
                pyglet.text.Label('谪', x=0.10 * w, y=0.82 * h,
                                  font_size=h * 0.055, color=gold, **common),
                pyglet.text.Label('仙', x=0.10 * w, y=0.73 * h,
                                  font_size=h * 0.055, color=gold, **common),
                pyglet.text.Label('复古未来主义 · 动态海报', x=0.30 * w, y=0.065 * h,
                                  font_size=h * 0.022, color=dim, **common),
            ]

    def _orbit_input(self):
        """鼠标拖动/滚轮 → 金饰视角（与 orbit_camera_input 同款签名）。"""

        def on_drag(x, y, dx, dy, buttons, modifiers):
            self.cam.orbit(dx, dy)

        def on_scroll(x, y, scroll_x, scroll_y):
            self.cam.zoom(scroll_y)

        return on_drag, on_scroll

    # ------------------------------------------------------------ 每帧 --
    def _animate(self):
        """金饰编排：正环绕人物公转（前） / 绸带环压轴心（后） / 玉璧伴日轮。"""
        if len(self.meshes) < 3:
            return
        t = self.time
        ring, silk, bisk = self.meshes[0], self.meshes[1], self.meshes[2]
        # 1) 正环：绕仙女公转 + 自旋（贴人物俯冲轴的小环）
        ang = t * 0.5
        m = mat4_from_trs(
            (-0.16 + 0.46 * math.cos(ang),
             -0.02 + 0.34 * math.sin(ang),
             0.20 * math.sin(t * 0.8)),
            _quat_axis((0.4, 1.0, 0.2), t * 1.4), (0.12, 0.12, 0.12))
        ring.set_model(m)
        # 2) 绸带环：压在人物上半身之后当"佛光背环"，慢转
        #    （压在腰上会读成呼啦圈——首版踩过；抬到胸口高度 + 放大 + 后移）
        tilt = mat4_from_trs((0.0, 0.0, 0.0), _quat_axis((1.0, 0.0, 0.0), 0.55),
                             (1.0, 1.0, 1.0))
        spin = mat4_from_trs((-0.14, 0.16, -0.55),
                             _quat_axis((0.0, 1.0, 0.0), t * 0.45),
                             (0.40, 0.40, 0.40))
        silk.set_model(mat4_mul(spin, tilt))
        # 3) 玉璧：日轮左缘，绕画面轴缓旋 + 侧倾一点让璧孔读得出来
        bisk.set_model(mat4_mul(
            mat4_from_trs((0.22, 0.50, 0.20),
                          _quat_axis((0.0, 0.0, 1.0), t * 0.35),
                          (0.15, 0.15, 0.15)),
            mat4_from_trs((0.0, 0.0, 0.0),
                          _quat_axis((1.0, 0.0, 0.0), 0.32),
                          (1.0, 1.0, 1.0))))

    def update(self, dt):
        if self.paused:
            return
        self._animate()

    def render(self):
        w, h = self.window.width, self.window.height
        # ---- 1) 海报底（不写深度，给 3D 层让路） ----
        glDepthMask(False)
        prog = self.program
        prog.use()
        prog['u_time'] = self.time
        prog['u_res'] = (float(w), float(h))
        prog['u_grain'] = self.grain
        prog['u_impasto'] = self.impasto
        prog['u_clouds'] = self.clouds
        self.quad.draw(GL_TRIANGLES)
        glDepthMask(True)

        # ---- 2) 3D 金饰层（PBR，真实 glTF 资产） ----
        if self.show3d and self.meshes:
            glEnable(GL_DEPTH_TEST)
            aspect = w / max(1, h)
            self.window.view = self.cam.view_Mat4()
            self.window.projection = self.cam.projection(aspect)
            pbr = self.pbr
            pbr.use()
            start = self.env.apply(pbr, 0)
            # ⚠️ 用 cam.eye 而不是 cam.position：OrbitCamera 的 `.position` 是
            # 基类 Camera 的残留属性（恒为默认 (0,2,8)），不随 yaw/pitch/dist
            # 更新——喂错 uCamPos 会让所有高光算到错误的视点上（本 demo 实测）。
            pbr['uCamPos'] = Vec3(*self.cam.eye)
            # 补一盏暖点光（贴近镜头前上方）：墨色天穹下纯 IBL 读成深铜，
            # 金饰需要一盏"摄影棚补光"才立得起来（Render 支持 4 盏，用 1 盏）
            pbr['uPointCount'] = 1
            pbr['uPointPos0'] = Vec3(0.35, 0.60, 2.6)
            pbr['uPointColor0'] = Vec3(2.2, 1.45, 0.72)
            for m in self.meshes:
                m.draw(unit_start=start)
            glDisable(GL_DEPTH_TEST)

        # ---- 3) 真字题签（2D 正交投影；HUD 由基类自管投影） ----
        if self.labels:
            self.window.projection = Mat4.orthogonal_projection(
                0, w, 0, h, -255, 255)
            self.window.view = Mat4()
            for lab in self.labels:
                lab.draw()

    def on_key(self, symbol, modifiers):
        k = pyglet.window.key
        if symbol == k.G:
            self.grain = 0.0 if self.grain else 1.0
        elif symbol == k.I:
            self.impasto = 0.0 if self.impasto else 1.0
        elif symbol == k.C:
            self.clouds = 0.0 if self.clouds else 1.0

    def status_text(self):
        flags = []
        if not self.grain:
            flags.append('no-grain')
        if not self.impasto:
            flags.append('no-impasto')
        if not self.clouds:
            flags.append('no-clouds')
        if not self.show3d:
            flags.append('no-3d')
        if not self.showtext:
            flags.append('no-text')
        suffix = ('  [' + ','.join(flags) + ']') if flags else ''
        engine_bit = (f'GLB {self.glb_name} 三角 {self.tri_count}'
                      if self.show3d else 'GLB off')
        return (f'{engine_bit} · {self.font_name or "default-font"} · '
                f't={self.time:6.2f}s / {self.DURATION:.0f}s   '
                f'{self.fps:5.1f} fps{suffix}'
                f'{"  [PAUSED]" if self.paused else ""}')


if __name__ == '__main__':
    run(PosterwardApp)
