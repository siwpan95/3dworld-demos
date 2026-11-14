# -*- coding: utf-8 -*-
"""《晶极 / Crystalward》—— 晶体的极致色彩与形态。

一个全屏片元着色器 + SDF 光线步进的**光谱渲染**演示：零几何、零贴图、零素材，
画面里的每一种颜色都来自 400~700nm 可见光谱的逐波长积分，而不是调色板插值。

> **能力落点（2026-11-14 反哺）**：本演示按仓惯例是**自包含快照**，能力本体在
> 引擎仓 `siwpan/3dworld` 的 `world3d/engine/spectrum.py`（CPU 黄金参考，零 GL）
> 与 `world3d/engine/spectral_glsl.py`（可注入 GLSL 片段）。本文件里的
> ``pgauss`` / ``cieXYZ`` / ``xyz2rgb`` / ``band`` 与引擎片段**逐字同式同常数**，
> 由引擎仓 ``tests/test_spectrum.py``（11 项）钉住；改公式时两边必须同步。
> 之所以不直接 import：本仓 venv 装的是 PyPI 发行版，引入未发版模块会让
> 65 个 demo 的运行环境分叉（与仓库 README「剥离时快照」的分工一致）。

## 为什么值得单独一个 demo

既有 ``raymarch_sdf.py`` 演示"用 SDF 画形状"，``material_lab.py`` 演示"程序化
PBR 纹理"，两者都是 **RGB 三通道** 的近似。本 demo 补的是另一条完全不同的路：
把颜色当作**光谱**来算。于是调研到的矿物学 / 宝石学事实可以直接写成公式：

| 现象 | 本 demo 里的公式 | 调研出处 |
|------|------------------|----------|
| 体色（选择性吸收） | Beer-Lambert ``T(λ)=exp(-a(λ)·L)``，L 由晶体**内部二次步进**实测 | 河北省地矿局 / 科普中国「矿物颜色成因」 |
| Cr³⁺ 晶体场致色 | 双吸收带 550+410nm → 红宝石；620+430nm → 祖母绿（**同种离子、不同配位场**） | 矿晶知识课堂第 18 期 / 科普中国 |
| Fe²⁺→Ti⁴⁺ 电荷迁移 | 580nm 宽强吸收带 → 蓝宝石；带宽远大于 d-d 跃迁 | 同上 |
| F 心色心 | 555nm 单峰；``u_anneal`` 加热使色心复原、颜色褪去 | 河北省地矿局「紫色萤石 / 烟晶」 |
| 氧化膜薄膜干涉（铋） | ``Δ = 2·n·d·cosθ_t``，``I(λ) ∝ 1+cos(2πΔ/λ + π)``（单次 π 相移） | 宝石晕彩效应成因机理综述 / JGG 2025 |
| 欧泊变彩（Bragg–Snell） | ``λ_peak = 2d·sqrt(n_eff² − sin²θ)``，微球 150~350nm | iScience 2021 / 成都理工《宝石学中缺陷研究现状与展望》 |
| 色散火彩（钻石） | Cauchy ``n(λ)=n_D + B·(589²/λ² − 1)``，逐波长独立折射出射 | 钻石 n≈2.417、色散 0.044、临界角 24.4° |
| 多色性 / 双折射 | 两束偏振光走不同光程 → 相位差干涉，吸收谱随光轴方向插值 | 维基「多色性」/ 百度百科 |
| CIE 1931 → sRGB | Wyman/Sloan/Shirley (JCGT 2013) 分段高斯拟合 CMF + XYZ→sRGB 矩阵 | 见 docs/crystalward_调研与致色机理.md |

**形态侧**同样来自调研，且刻意区分了两个常被混淆的概念：**晶系**（内部原子
对称性，固定）与**晶习**（外部生长形态，随冷却速率 / 过饱和度 / 空间而变）。
8 种形态覆盖等轴晶系的立方 / 八面 / 菱形十二面体、六方柱+锥头、板状双晶、
针状放射晶簇、漏斗晶（hopper，边缘长得比面心快）与枝晶（扩散限制生长）。
另设 ``T`` 双晶开关、``R`` 换生长种子。

## 用法

    python examples/crystalward.py                    # 48s 八幕自动演出
    python examples/crystalward.py --smoke 3
    python examples/crystalward.py --shot            # 5 张剧照到 docs/
    python examples/crystalward.py --bench 3        # 真实帧率（vsync 关）
    python examples/crystalward.py --size 960x540    # 掉帧时降采样
    python examples/crystalward.py --flag hold       # 关自动演出，手动把玩
    python examples/crystalward.py --flag shape=6,mech=3   # 直接指定铋漏斗晶 + 氧化膜

按键：1~8 形态 / Q,E 致色机理 / D 色散 / B 双折射 / U 紫外灯 / T 双晶 /
      V 切换 Cr³⁺ 配位场（红宝石↔祖母绿）/ [ ] 调氧化膜厚(nm) / - = 调色散 /
      R 换种子（色心退火复位）/ A 自动演出 / 拖动旋转 / 滚轮推拉 /
      SPACE 暂停 / ESC 退出。

## 实现注意（沿用本仓踩过的坑）

- 顶点着色器必须声明并使用 ``in vec3 position; in vec4 colors;``（pyglet 2.1.16
  按名字校验属性，声明了却不用会被 GLSL 优化掉）。
- **每个 uniform 都要在片元里真读到**，否则 ``prog['u_xxx']=`` 会抛
  ``ShaderException: A Uniform with the name ... was not found``。
- 全屏 pass 前 ``glDisable(GL_DEPTH_TEST)``；控制台输出一律 ASCII。
- SDF 由多个 ``min`` 并集拼成、并非处处精确，主步进一律用 0.8 倍保守步长，
  否则 hopper / 枝晶的薄边会被跨过去出现破洞。
- 着色是「一次光谱循环同时算吸收 / 色散 / 干涉」，不是三趟后期叠加：
  每个波长有自己的折射方向与自己的干涉相位，这也是火彩与假色能同时成立的原因。
"""
import math

import pyglet
from pyglet.gl import GL_DEPTH_TEST, GL_TRIANGLES, glDisable
from pyglet.graphics.shader import Shader, ShaderProgram

from world3d.platform.app import DemoApp, make_fullscreen_quad, run

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

FRAGMENT_SRC = '''#version 330 core
in vec2 v_uv;
in vec3 v_tint;
out vec4 final_colors;

uniform vec2  u_res;
uniform float u_time;      // 演出时间（秒，已按 DURATION 取模）
uniform vec3  u_eye;       // 相机位置
uniform vec3  u_target;    // 注视点
uniform int   u_shape;     // 0..7 形态
uniform int   u_mech;      // 0..7 致色机理
uniform float u_grow;      // 0..1 晶体生长（每幕开头 1.2s 从 0 长到 1）
uniform float u_rot;       // 晶体自转（弧度）
uniform float u_film;      // 氧化膜 / 微球层厚度，单位 nm
uniform float u_ior;       // 钠 D 线折射率
uniform float u_disp;      // 色散系数 B（Cauchy）
uniform float u_disp_on;   // 色散开关
uniform float u_twin;      // 双晶开关
uniform float u_uv;        // 紫外灯强度
uniform float u_anneal;    // 色心退火进度（1 = 完全褪色）
uniform float u_site;      // Cr3+ 配位场：0 刚玉(红宝石) / 1 绿柱石(祖母绿)
uniform float u_bire;      // 双折射 / 偏光干涉开关
uniform float u_seed;      // 生长种子（膜厚图与背景星点）
uniform float u_samples;   // 光谱采样数（--flag low 降到 8）
uniform float u_auto;      // 自动演出标志（背景呼吸）

const float PI = 3.14159265;
const int   MAX_SAMPLES = 16;
const int   MAX_STEPS   = 84;
const int   INNER_STEPS = 24;

mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float band(float w, float mu, float sigma) {
    float t = (w - mu) / sigma;
    return exp(-0.5 * t * t);
}

// ------------------------------------------------------------- CIE 1931 --
// Wyman, Sloan & Shirley, "Simple Analytic Approximations to the CIE XYZ
// Color Matching Functions", JCGT 2(2), 2013: 分段高斯拟合，避免查表纹理。
float pgauss(float x, float mu, float s1, float s2) {
    float t = (x - mu) * (x < mu ? 1.0 / s1 : 1.0 / s2);
    return exp(-0.5 * t * t);
}
vec3 cieXYZ(float w) {
    float x = 1.056 * pgauss(w, 599.8, 37.9, 31.0)
            + 0.362 * pgauss(w, 442.0, 16.0, 26.7)
            - 0.065 * pgauss(w, 501.1, 20.4, 26.2);
    float y = 0.821 * pgauss(w, 568.8, 46.9, 40.5)
            + 0.286 * pgauss(w, 530.9, 16.3, 31.1);
    float z = 1.217 * pgauss(w, 437.0, 11.8, 36.0)
            + 0.681 * pgauss(w, 459.0, 26.0, 13.8);
    return vec3(x, y, z);
}
vec3 xyz2rgb(vec3 c) {
    return vec3( 3.2406 * c.x - 1.5372 * c.y - 0.4986 * c.z,
                -0.9689 * c.x + 1.8758 * c.y + 0.0415 * c.z,
                 0.0557 * c.x - 0.2040 * c.y + 1.0570 * c.z);
}

// ---------------------------------------------------------------- SDF --
float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}
float sdOcta(vec3 p, float s) {
    p = abs(p);
    return (p.x + p.y + p.z - s) * 0.57735027;
}
float sdRhombicDodeca(vec3 p, float s) {
    vec3 q = abs(p);
    return (max(max(q.x + q.y, q.y + q.z), q.x + q.z) - s) * 0.70710678;
}
// 六边形在 xz 平面，y 为柱轴
float sdHexPrismY(vec3 p, vec2 h) {
    vec3 q = abs(p);
    float hex = max(q.x * 0.8660254 + q.z * 0.5, q.z) - h.x;
    vec2 d = vec2(hex, q.y - h.y);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
float sdSegment(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}
// iq 的精确 box frame：12 条棱组成的方框架，漏斗晶的阶梯就靠它嵌套
float sdBoxFrame(vec3 p, vec3 b, float e) {
    p = abs(p) - b;
    vec3 q = abs(p + e) - e;
    return min(min(
        length(max(vec3(p.x, q.y, q.z), 0.0)) + min(max(p.x, max(q.y, q.z)), 0.0),
        length(max(vec3(q.x, p.y, q.z), 0.0)) + min(max(q.x, max(p.y, q.z)), 0.0)),
        length(max(vec3(q.x, q.y, p.z), 0.0)) + min(max(q.x, max(q.y, p.z)), 0.0));
}

float sdBismuthHopper(vec3 p) {
    float d = 1e5;
    for (int i = 0; i < 5; i++) {
        float fi = float(i);
        float s = 0.94 - fi * 0.165;
        vec3 q = p - vec3(0.0, -fi * 0.085, 0.0);
        q.xz = rot(fi * 0.10 + u_seed * 0.15) * q.xz;
        // 边缘长得比面心快 → 层层内缩下沉的空心台阶
        d = min(d, sdBoxFrame(q, vec3(s), 0.072));
    }
    // 内核补一点体积，避免从某些角度看穿成纯线框
    return min(d, sdBox(p, vec3(0.16)) - 0.02);
}

float sdDendrite(vec3 p) {
    float bound = length(p) - 1.75;
    if (bound > 0.35) return bound;         // 远场提前剔除：48 段线段的并集很贵
    float d = sdSegment(p, vec3(0.0, -1.25, 0.0), vec3(0.0, 1.25, 0.0), 0.048);
    for (int i = 0; i < 6; i++) {
        float fi = float(i);
        float y = -1.00 + fi * 0.36;
        float len = 0.86 - fi * 0.10;
        for (int k = 0; k < 4; k++) {
            float a = float(k) * 1.5707963 + fi * 0.42;
            vec3 dir = normalize(vec3(cos(a), 0.62, sin(a)));
            vec3 base = vec3(0.0, y, 0.0);
            vec3 tip = base + dir * len;
            d = min(d, sdSegment(p, base, tip, 0.030 - fi * 0.002));
            // 二级分枝：过饱和度进一步升高时长出的树状结构
            vec3 mid = base + dir * len * 0.62;
            float a2 = a + 0.62;
            vec3 d2 = normalize(vec3(cos(a2), 0.5, sin(a2)));
            d = min(d, sdSegment(p, mid, mid + d2 * len * 0.42, 0.019));
        }
    }
    return d;
}

float sdAcicular(vec3 p) {
    float d = sdBox(p, vec3(0.52, 0.11, 0.52)) - 0.02;
    for (int i = 0; i < 9; i++) {
        float fi = float(i);
        float a = fi * 2.3999632 + u_seed * 0.4;      // 黄金角放射
        float tilt = 0.28 + 0.42 * fract(fi * 0.618);
        vec3 dir = normalize(vec3(cos(a) * tilt, 1.0, sin(a) * tilt));
        float len = 0.80 + 0.55 * fract(fi * 0.37 + 0.11);
        vec3 base = vec3(cos(a) * 0.13, 0.04, sin(a) * 0.13);
        d = min(d, sdSegment(p, base, base + dir * len, 0.042));
    }
    return d;
}

float shapeSDF(vec3 q) {
    float d;
    if (u_shape == 0) {                       // 等轴晶系 · 立方（萤石 / 黄铁矿）
        d = sdBox(q, vec3(0.86)) - 0.035;
    } else if (u_shape == 1) {                // 等轴晶系 · 八面体（金刚石 / 磁铁矿）
        d = sdOcta(q, 1.42);
    } else if (u_shape == 2) {                // 等轴晶系 · 菱形十二面体（石榴石）
        d = sdRhombicDodeca(q, 1.18);
    } else if (u_shape == 3) {                // 六方 / 三方 · 六方柱 + 双锥头（石英 / 绿柱石）
        vec3 a = abs(q);
        float r = max(a.x * 0.8660254 + a.z * 0.5, a.z);
        float prism = sdHexPrismY(q, vec2(0.72, 0.78));
        // 双锥与柱体取并集（max 求交只会把柱体削矮，长不出晶头）
        float cone = max(a.y - 1.30 + r * 1.02, r - 0.72);
        d = min(prism, cone);
    } else if (u_shape == 4) {                // 单斜 · 板状 / 燕尾双晶（石膏 / 重晶石）
        vec3 b = q; b.xz = rot(0.34) * b.xz;
        d = sdBox(b, vec3(0.92, 0.135, 0.56)) - 0.03;
        if (u_twin > 0.5) {
            vec3 c = q - vec3(0.30, 0.0, 0.0); c.xz = rot(-0.34) * c.xz;
            d = min(d, sdBox(c, vec3(0.92, 0.135, 0.56)) - 0.03);
        }
    } else if (u_shape == 5) {                // 斜方 · 针状放射晶簇（辉锑矿 / 钠沸石）
        d = sdAcicular(q);
    } else if (u_shape == 6) {                // 骨架生长 · 漏斗晶（铋 / 石盐）
        d = sdBismuthHopper(q);
    } else {                                  // 扩散限制生长 · 枝晶（自然铜 / 锰氧化物）
        d = sdDendrite(q);
    }
    return d;
}

// 返回 (距离, 台阶/面哈希)：后者用来给氧化膜画"厚度地图"
vec2 map(vec3 p) {
    float g = max(u_grow, 0.06);
    vec3 q = p / g;
    q.xz = rot(u_rot) * q.xz;
    q.xy = rot(u_rot * 0.31) * q.xy;
    float d = shapeSDF(q);
    if (u_twin > 0.5 && u_shape != 4 && u_shape != 5 && u_shape != 7) {
        vec3 m = q; m.x = 0.62 - m.x;         // 依对称面连生的双晶
        d = min(d, shapeSDF(m));
    }
    float rnd = hash21(floor(q.xy * 3.0) + floor(q.z * 3.0) + u_seed * 7.0);
    return vec2(d * g, rnd);
}

vec3 calcNormal(vec3 p) {
    vec2 e = vec2(0.0018, 0.0);
    return normalize(vec3(
        map(p + e.xyy).x - map(p - e.xyy).x,
        map(p + e.yxy).x - map(p - e.yxy).x,
        map(p + e.yyx).x - map(p - e.yyx).x));
}

float march(vec3 ro, vec3 rd, out vec3 hit) {
    float t = 0.0;
    for (int i = 0; i < MAX_STEPS; i++) {
        hit = ro + rd * t;
        float d = map(hit).x;
        if (d < 0.0018) return t;
        t += max(d * 0.80, 0.006);            // 保守步长：并集 SDF 非精确
        if (t > 9.0) break;
    }
    return -1.0;
}

// 晶体内部的真实光程：Beer-Lambert 的 L 不是拍脑袋常数
float insidePath(vec3 p, vec3 dir) {
    float t = 0.02;
    for (int i = 0; i < INNER_STEPS; i++) {
        float d = -map(p + dir * t).x;
        if (d < 0.005) break;
        t += clamp(d * 0.80, 0.012, 0.22);
        if (t > 3.5) break;
    }
    return t;
}

// ------------------------------------------------------------- 环境光 --
// 火彩只在"小而亮的点光源"下最明显（钻石光学调研结论），所以环境刻意
// 做成两颗高亮点 + 一条灯带 + 极弱渐变的暗展厅，而不是大面积柔光。
float envLum(vec3 d) {
    vec3 L1 = normalize(vec3(0.55, 0.82, 0.34));
    vec3 L2 = normalize(vec3(-0.70, 0.36, -0.42));
    // 基础照度 + 宽瓣 + 极窄点光：三层缺一不可。
    // 只有窄点光时（第一版）折射出射方向几乎采不到光，整颗晶体黑成剪影；
    // 宽瓣负责基础亮度与色散的可读性，极窄点光才是火彩里那几束彩色闪光。
    float v = 0.42;
    v += 1.05 * pow(max(dot(d, L1), 0.0), 22.0);
    v += 0.62 * pow(max(dot(d, L1), 0.0), 7.0);
    v += 3.40 * pow(max(dot(d, L1), 0.0), 190.0);
    v += 0.78 * pow(max(dot(d, L2), 0.0), 26.0);
    v += 2.10 * pow(max(dot(d, L2), 0.0), 260.0);
    v += 0.30 * pow(max(1.0 - abs(d.y * 3.2 - 0.55), 0.0), 18.0) * step(0.0, d.x);
    v += 0.14 * smoothstep(-1.0, 1.0, d.y);
    return v;
}

// ------------------------------------------------------- 致色机理谱 --
// a(λ)：单位光程的吸收系数。每条曲线对应一种矿物学致色机制。
float absorption(float w, vec3 rd) {
    float a = 0.012;
    if (u_mech == 0) {                        // 晶体场 d-d 跃迁（Cr3+）
        if (u_site < 0.5)                    // 刚玉配位场 → 红宝石
            a += 1.85 * band(w, 550.0, 58.0) + 1.05 * band(w, 410.0, 45.0);
        else                                  // 绿柱石配位场 → 祖母绿
            a += 1.55 * band(w, 620.0, 72.0) + 1.35 * band(w, 430.0, 48.0);
    } else if (u_mech == 1) {                // Fe2+->Ti4+ 电荷迁移（蓝宝石）
        a += 2.70 * band(w, 580.0, 88.0) + 0.42 * band(w, 430.0, 62.0);
    } else if (u_mech == 2) {                // F 心 / [AlO4]4+ 色心（萤石、烟晶）
        a += 3.10 * (1.0 - u_anneal) * band(w, 555.0, 62.0) + 0.035;
    } else if (u_mech == 3) {                // 金属 + 氧化膜：体色中性，颜色在反射里
        a += 0.26;
    } else if (u_mech == 4) {                // 欧泊：乳白体色，颜色在布拉格衍射里
        a += 0.16 + 0.30 * band(w, 625.0, 95.0);
    } else if (u_mech == 5) {                // 钻石：无色，颜色全靠色散
        a += 0.010;
    } else if (u_mech == 6) {                // 多色性：吸收谱随光轴方向换挡
        float k = 0.5 + 0.5 * dot(rd, normalize(vec3(0.32, 0.90, 0.22)));
        a += mix(1.95 * band(w, 595.0, 62.0), 2.00 * band(w, 470.0, 68.0), k);
    } else {                                  // 光致发光：无色基质 + UV 激发
        a += 0.022;
    }
    return a;
}

// 反射侧的光谱调制：薄膜干涉 / 布拉格变彩 / 荧光发射
float interfaceMod(float w, float deltaNm, float lambdaBragg, float pathL) {
    float m = 1.0;
    if (u_mech == 3) {                        // Δ = 2·n·d·cosθ_t；一次 π 相移
        float phase = 2.0 * PI * deltaNm / w + PI;
        m = 0.18 + 1.55 * (0.5 + 0.5 * cos(phase));
    } else if (u_mech == 4) {                // λ_peak = 2d·sqrt(neff² − sin²θ)
        float pk = exp(-pow((w - lambdaBragg) / 34.0, 2.0));
        m = 0.22 + 4.20 * pk;
    } else if (u_mech == 7) {                // UV 激发 → 465nm 蓝色发射带
        m += u_uv * 4.2 * exp(-pow((w - 465.0) / 34.0, 2.0));
    }
    // 双折射：o/e 两束偏振光的光程差 → 正交偏光下的干涉色
    // 相位差正比于实际内部光程 pathL（不是视角余弦），所以厚的地方色序更高
    if (u_bire > 0.5) {
        float ph = 2.0 * PI * (0.055 + u_disp * 0.8) * pathL * 7200.0 / w;
        m *= 0.62 + 0.72 * (0.5 + 0.5 * cos(ph));
    }
    return m;
}

// ------------------------------------------------------------- 着色 --
vec3 shadeCrystal(vec3 p, vec3 n, vec3 rd, float rnd) {
    vec3 L1 = normalize(vec3(0.55, 0.82, 0.34));
    float ct = clamp(dot(-rd, n), 0.0, 1.0);
    float fres = 0.045 + 0.955 * pow(1.0 - ct, 5.0);
    vec3 rdir = reflect(rd, n);
    // 晶面朝向带来漫反射层次：否则一颗立方体会读成平的黑纸片
    float diff = 0.52 + 0.78 * max(dot(n, L1), 0.0);
    float envR = (0.48 + envLum(rdir) * 0.72) * diff;

    vec3 tdir = refract(rd, n, 1.0 / u_ior);
    if (dot(tdir, tdir) < 1e-4) tdir = rdir;
    float pathL = insidePath(p + tdir * 0.006, tdir);
    vec3 pOut = p + tdir * pathL;
    vec3 nOut = calcNormal(pOut);
    // 内部全反射回程：钻石的"亮度"主要来自这一项
    float bounce = envLum(reflect(tdir, nOut)) * 0.42;

    // 生长纹 / 色带：吸收系数沿空间轻微起伏（幻影水晶、色带电气石）
    float veins = 0.5 + 0.5 * sin(dot(p, vec3(7.0, 4.0, 5.0))
                                  + 2.6 * sin(p.y * 8.0));
    float zone = 0.82 + 0.36 * veins;

    // 氧化膜厚度地图：不同台阶暴露在空气里的时间不同，膜厚也就不同
    float filmNm = u_film * (0.62 + 0.76 * rnd);
    float sin2t = 1.0 - ct * ct;
    float ctFilm = sqrt(max(0.02, 1.0 - sin2t / (1.92 * 1.92)));   // Bi2O3 膜 n≈1.92
    float deltaNm = 2.0 * 1.92 * filmNm * ctFilm;
    float lambdaBragg = 2.0 * u_film * 0.86 *
                        sqrt(max(0.06, 1.45 * 1.45 - sin2t));       // n_eff≈1.45
    // 反射是无色的：只有"假色"机理（薄膜 / 变彩 / 荧光）才给它抬基底，
    // 否则 30% 的白反射会把所有体色冲成粉彩（第二版剧照的教训）。
    bool structural = (u_mech == 3 || u_mech == 4 || u_mech == 7);
    float transScale = (u_mech == 3) ? 0.05 : ((u_mech == 4) ? 0.14 : 1.0);
    float fw = clamp(fres * 1.15 + (structural ? 0.46 : 0.045), 0.0, 1.0);
    float body = 0.72 + 0.40 * diff;
    // 钻石靠"暗底 + 彩色闪光"读火彩：基础照度压低、方向响应放大
    float baseT = (u_mech == 5) ? 0.34 : 0.62;
    float dirT = (u_mech == 5) ? 2.20 : 1.45;

    vec3 sum = vec3(0.0);
    float count = 0.0;
    for (int i = 0; i < MAX_SAMPLES; i++) {
        if (float(i) >= u_samples) break;
        float w = 400.0 + 300.0 * float(i) / max(1.0, u_samples - 1.0);
        // Cauchy 色散：每个波长有自己的折射率，因此有自己的出射方向
        float nl = u_ior + u_disp * u_disp_on * (589.0 * 589.0 / (w * w) - 1.0);
        vec3 outDir = refract(tdir, -nOut, nl);
        float envT;
        if (dot(outDir, outDir) < 1e-4) {
            envT = (0.55 + bounce) * body;    // 全反射：光没出去，回到晶体里
        } else {
            envT = (baseT + envLum(outDir) * dirT + bounce * 0.30) * body;
        }
        // 钻石幕：高折射率 + 24.4° 临界角使大半内部光线全反射，一次出射
        // 拿不到随 λ 变化的方向 ⇒ 火彩全白。这里补一条"回程出射"：把
        // Cauchy 的 n(λ) 差按演示增益放大成切向偏角，再采同一组点光源。
        float slit = 0.0;
        if (u_mech == 5) {
            vec3 tang = cross(n, rdir);
            if (dot(tang, tang) < 1e-6) tang = cross(n, vec3(0.0, 1.0, 0.0));
            tang = normalize(tang);
            float spread = (nl - u_ior) * 7.0;
            vec3 fireDir = normalize(rdir + tang * spread);
            float fire = envLum(fireDir);
            // 显式谱-角映射：把 Cauchy 的 n(λ) 差当成棱镜角色散。
            // 关键是"**单道**随空间扫动的窄缝"：固定多道缝会让很多波长同时
            // 通过，光谱一积分又回到白。每个像素只放行 1~2 个波长才有饱和度。
            float ph = (nl - u_ior) * 60.0
                     + 0.45 * dot(n, vec3(0.7, 0.2, -0.6));
            // 双频叠加：单频 sin 会画出等距平行条纹（像衍射光栅），
            // 两组不同方向/频率的相位和才让色斑弯曲、疏密自然。
            float sweep = 0.5
                + 0.28 * sin(dot(p, vec3(4.1, 6.3, 3.7)) * 1.15 + u_rot * 2.0)
                + 0.22 * sin(dot(p, vec3(-2.7, 3.1, 5.2)) * 1.9 - u_rot * 1.3);
            float slitCenter = -0.30 + 2.9 * sweep;
            slit = exp(-pow((ph - slitCenter) / 0.30, 2.0))
                 + 0.32 * exp(-pow((ph - slitCenter - 1.55) / 0.32, 2.0));
            float glint = pow(1.0 - ct, 1.5) *
                          (0.45 + 0.55 * max(dot(n, L1), 0.0));
            // 白基底必须近乎为零：所有波长等量叠加就是白，会直接吃掉火彩
            envT = (0.02 + fire * 0.05
                    + slit * (5.0 + 14.0 * glint)
                    + bounce * 0.04) * body;
        }
        float T = exp(-absorption(w, rd) * pathL * 2.70 * zone);
        float im = interfaceMod(w, deltaNm, lambdaBragg, pathL);
        vec3 basis = xyz2rgb(cieXYZ(w));
        // 高折射率下斜面的 Fresnel 接近 1，无色镜面反射会盖掉火彩；
        // 钻石幕让反射也随缝走，只留很薄一层白高光。
        float refl = fw * envR * im * ((u_mech == 5) ? (0.10 + 0.70 * slit) : 1.0);
        sum += basis * (T * envT * transScale * (1.0 - fw)
                        + refl
                        + u_uv * 0.22 * band(w, 400.0, 24.0));
        count += 1.0;
    }
    vec3 col = sum / max(1.0, count) * 2.9;
    // 面本身的菲涅尔白光：只留很薄一层，别把体色洗白
    col += vec3(0.9, 0.95, 1.0) * pow(1.0 - ct, 8.0) * 0.20;
    return max(col, vec3(0.0));
}

vec3 background(vec2 uv, vec3 rd) {
    vec3 col = vec3(0.010, 0.014, 0.026);
    col += vec3(0.020, 0.032, 0.062) * smoothstep(-1.0, 1.0, rd.y);
    // 主体背后的柔光晕，把晶体从纯黑背景里托出来
    col += vec3(0.026, 0.040, 0.086) *
           pow(max(0.0, 1.0 - length(uv - 0.5) * 2.0), 3.0);
    // 自动演出时背景轻微呼吸，静止把玩时保持恒定
    col *= 1.0 + 0.06 * u_auto * sin(u_time * 0.7);
    vec2 cell = uv * vec2(96.0, 54.0);
    float h = hash21(floor(cell) + u_seed * 13.0);
    float star = step(0.9955, h) *
                 pow(max(0.0, 1.0 - length(fract(cell) - 0.5) * 2.0), 9.0);
    col += star * vec3(0.55, 0.68, 1.0) * (0.6 + 0.4 * sin(u_time * 2.0 + h * 40.0));
    // 底部可见光谱尺：用的就是本 demo 自己的 CIE 转换，把"颜色模型"亮出来。
    // 抬到 HUD 底条（30px）之上，别和帮助行叠在一起。
    float by = 0.105, bw = 0.022;
    float d = abs(uv.y - by);
    if (d < bw) {
        float w = 400.0 + 300.0 * uv.x;
        vec3 spec = max(xyz2rgb(cieXYZ(w)), vec3(0.0));
        float a = smoothstep(bw, bw * 0.35, d);
        col = mix(col, spec * 0.85, a);
    }
    // 暗角
    col *= mix(0.55, 1.0, smoothstep(1.5, 0.25, length(uv - 0.5) * 1.7));
    return col;
}

void main() {
    vec2 uv = v_uv;
    vec2 p = (uv - 0.5) * vec2(u_res.x / u_res.y, 1.0) * 2.0;
    vec3 f = normalize(u_target - u_eye);
    vec3 r = normalize(cross(f, vec3(0.0, 1.0, 0.0)));
    vec3 up = cross(r, f);
    vec3 rd = normalize(p.x * r + p.y * up + 1.75 * f);

    vec3 col = background(uv, rd);
    vec3 hit;
    float t = march(u_eye, rd, hit);
    if (t > 0.0) {
        vec3 n = calcNormal(hit);
        float rnd = map(hit).y;
        vec3 c = shadeCrystal(hit, n, rd, rnd);
        col = c;
        // 边缘轮廓光，把晶面的转折读出来
        float rim = pow(1.0 - clamp(dot(-rd, n), 0.0, 1.0), 3.0);
        col += vec3(0.14, 0.18, 0.28) * rim * 0.55;
    }
    // ACES 近似 + gamma
    col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.90));
    final_colors = vec4(col * v_tint, 1.0);
}'''

# ASCII only: 控制台 / HUD 在中文 Windows 代码页下会乱码
SHAPE_NAMES = (
    'CUBIC {100} fluorite', 'OCTAHEDRAL {111} diamond',
    'DODECAHEDRAL {110} garnet', 'HEXAGONAL prism quartz',
    'TABULAR swallow-twin gypsum', 'ACICULAR radiating stibnite',
    'HOPPER bismuth skeleton', 'DENDRITIC native copper',
)
MECH_NAMES = (
    'CRYSTAL FIELD Cr3+', 'IVCT Fe2+/Ti4+', 'F-CENTER color center',
    'THIN FILM Bi2O3', 'BRAGG opal', 'DISPERSION diamond fire',
    'PLEOCHROISM cordierite', 'PHOTOLUMINESCENCE UV',
)

# 八幕自动演出：(形态, 机理, 膜厚 nm, IOR, 色散 B, Cr 配位场, UV)
ACTS = (
    (0, 2, 420.0, 1.434, 0.010, 0.0, 0.0),   # 紫色萤石：F 心
    (1, 5, 380.0, 2.417, 0.044, 0.0, 0.0),   # 钻石八面体：色散火彩
    (3, 0, 400.0, 1.577, 0.013, 1.0, 0.0),   # 祖母绿：Cr3+ 在绿柱石场
    (6, 3, 300.0, 1.900, 0.010, 0.0, 0.0),   # 铋漏斗晶：一级干涉色最饱和
    (2, 1, 410.0, 1.770, 0.018, 0.0, 0.0),   # 蓝宝石色（借用石榴石形态对照）
    (5, 6, 430.0, 1.550, 0.016, 0.0, 0.0),   # 针状晶簇：多色性
    (7, 4, 230.0, 1.450, 0.012, 0.0, 0.0),   # 枝晶 + 欧泊变彩
    (4, 7, 400.0, 1.520, 0.014, 0.0, 1.0),   # 板状双晶 + UV 荧光
)
ACT_LEN = 6.0
DURATION = ACT_LEN * len(ACTS)


def _mech_defaults(mech):
    """手动模式给指定机理配一套合理的 (膜厚, IOR, 色散, 配位场, UV)。"""
    for _shape, m, film, ior, disp, site, uv in ACTS:
        if m == mech:
            return film, ior, disp, site, uv
    return 420.0, 1.52, 0.014, 0.0, 0.0


class Crystalward(DemoApp):
    """晶体的极致色彩与形态：按波长积分的 SDF 光谱渲染。"""

    TITLE = '晶极 / Crystalward · 晶体的极致色彩与形态'
    HELP = ('1-8 形态  Q/E 致色  D 色散  B 双折射  U 紫外  T 双晶  V 配位场  '
            '[ ] 膜厚  - = 色散  R 种子  A 自动  拖动/滚轮  SPACE 暂停  ESC 退出')
    DURATION = DURATION
    HUD = True
    # 五张剧照刻意覆盖五种互不相同的色彩机制与形态
    SHOTS = ((4.0, 'docs/crystalward_01.png'),    # 紫色萤石 F 心 · 立方
             (10.0, 'docs/crystalward_02.png'),   # 钻石色散火彩 · 八面体
             (14.0, 'docs/crystalward_03.png'),   # 祖母绿 Cr3+ · 六方柱锥头
             (22.0, 'docs/crystalward_04.png'),   # 铋氧化膜干涉 · 漏斗晶
             (40.0, 'docs/crystalward_05.png'))   # 欧泊变彩 · 枝晶

    def setup(self):
        glDisable(GL_DEPTH_TEST)
        self.program = ShaderProgram(Shader(VERTEX_SRC, 'vertex'),
                                     Shader(FRAGMENT_SRC, 'fragment'))
        self.quad = make_fullscreen_quad(self.program)
        self.shape = 0
        self.mech = 2
        self.grow = 1.0
        self.rot = 0.0
        self.film = 420.0
        self.ior = 1.434
        self.disp = 0.010
        self.site = 0.0
        self.uv_lamp = 0.0
        self.auto = True
        self.disp_on = 1.0
        self.bire = 0.0
        self.twin = 0.0
        self.anneal = 0.0
        self.seed = 1.0
        self.act_i = -1
        self.yaw = 0.62
        self.pitch = 0.28
        self.dist = 3.55
        self.samples = 14.0

        flags = self.args.flags
        self.auto = 'hold' not in flags
        self.samples = 8.0 if 'low' in flags else 14.0
        # 先铺演出起始幕，再让 --flag 覆盖：顺序反了的话 `shape=` / `mech=`
        # 会被起始幕、以及随后每帧的 update 无声吞掉（第一版就踩了这个）。
        self._apply_act(0.0, force=True)
        for fl in flags:
            if fl.startswith('shape='):
                try:
                    self.shape = max(0, min(7, int(fl.split('=', 1)[1])))
                except ValueError:
                    pass
            elif fl.startswith('mech='):
                try:
                    self.mech = max(0, min(7, int(fl.split('=', 1)[1])))
                except ValueError:
                    pass
        if not self.auto:
            (self.film, self.ior, self.disp,
             self.site, self.uv_lamp) = _mech_defaults(self.mech)
            self.disp_on = 1.0 if self.mech in (1, 5, 6, 7) else 0.55
            self.bire = 1.0 if self.mech in (6, 7) else 0.0
            self.twin = 1.0 if self.shape == 4 else 0.0
            self.act_i = -1
            self.grow = 1.0

    def _apply_act(self, t, force=False):
        """按演出时间取幕；每幕开头 1.2s 让晶体从 0 长到 1。"""
        idx = int(t // ACT_LEN) % len(ACTS)
        if not force and idx == self.act_i:
            local = t - idx * ACT_LEN
            self.grow = min(1.0, local / 1.2)
            # 色心幕里演示「加热褪色」：退火进度随幕内时间推进
            self.anneal = min(1.0, max(0.0, (local - 2.2) / 3.0)) if self.mech == 2 else 0.0
            return
        self.act_i = idx
        shape, mech, film, ior, disp, site, uv = ACTS[idx]
        self.shape, self.mech = shape, mech
        self.film, self.ior, self.disp = film, ior, disp
        self.site, self.uv_lamp = site, uv
        self.disp_on = 1.0 if mech in (1, 5, 6, 7) else 0.55
        self.twin = 1.0 if shape in (4,) else 0.0
        self.bire = 1.0 if mech in (6, 7) else 0.0
        self.seed = 1.0 + float(idx) * 0.37
        local = t - idx * ACT_LEN
        self.grow = min(1.0, local / 1.2)
        self.anneal = 0.0

    def update(self, dt):
        if self.auto:
            self._apply_act(self.time)
            self.yaw += dt * 0.16
        self.rot += dt * 0.11

    def on_key(self, symbol, modifiers):
        k = pyglet.window.key
        if k._1 <= symbol <= k._8:
            self.shape = symbol - k._1
            self.auto = False
            self.grow = 1.0
        elif symbol == k.Q:
            self.mech = (self.mech - 1) % len(MECH_NAMES)
            self.auto = False
        elif symbol == k.E:
            self.mech = (self.mech + 1) % len(MECH_NAMES)
            self.auto = False
        elif symbol == k.D:
            self.disp_on = 0.0 if self.disp_on else 1.0
        elif symbol == k.B:
            self.bire = 0.0 if self.bire else 1.0
        elif symbol == k.U:
            self.uv_lamp = 0.0 if self.uv_lamp else 1.0
        elif symbol == k.T:
            self.twin = 0.0 if self.twin else 1.0
        elif symbol == k.V:
            self.site = 0.0 if self.site else 1.0
        elif symbol == k.A:
            self.auto = not self.auto
            self.act_i = -1
        elif symbol == k.R:
            self.seed += 1.37
            self.anneal = 0.0
        elif symbol == k.BRACKETLEFT:
            self.film = max(120.0, self.film - 30.0)
        elif symbol == k.BRACKETRIGHT:
            self.film = min(760.0, self.film + 30.0)
        elif symbol in (k.MINUS, k.UNDERSCORE):
            self.disp = max(0.0, self.disp - 0.004)
        elif symbol in (k.EQUAL, k.PLUS):
            self.disp = min(0.12, self.disp + 0.004)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.yaw += dx * 0.006
        self.pitch = max(-1.15, min(1.15, self.pitch + dy * 0.005))

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.dist = max(2.35, min(6.0, self.dist - scroll_y * 0.25))

    def render(self):
        w, h = self.window.width, self.window.height
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        eye = (self.dist * cp * sy, self.dist * sp, self.dist * cp * cy)
        prog = self.program
        prog.use()
        values = {
            'u_res': (float(w), float(h)),
            'u_time': self.time,
            'u_eye': eye,
            'u_target': (0.0, 0.0, 0.0),
            'u_shape': int(self.shape),
            'u_mech': int(self.mech),
            'u_grow': self.grow,
            'u_rot': self.rot,
            'u_film': self.film,
            'u_ior': self.ior,
            'u_disp': self.disp,
            'u_disp_on': self.disp_on,
            'u_twin': self.twin,
            'u_uv': self.uv_lamp,
            'u_anneal': self.anneal,
            'u_site': self.site,
            'u_bire': self.bire,
            'u_seed': self.seed,
            'u_samples': self.samples,
            'u_auto': 1.0 if self.auto else 0.0,
        }
        for name, value in values.items():
            prog[name] = value
        self.quad.draw(GL_TRIANGLES)

    def status_text(self):
        # 要短：HUD 标题在左、状态在右，超宽时基类会自动把标题藏起来
        return (f'{("A" + str(self.act_i + 1) + "/8") if self.auto else "MAN":<6} '
                f'{SHAPE_NAMES[self.shape][:9]:<9} '
                f'{MECH_NAMES[self.mech][:10]:<10} '
                f'n{self.ior:.2f} B{self.disp:.3f} d{self.film:.0f}nm '
                f't{self.time:4.1f}s {self.fps:4.0f}fps'
                f'{" P" if self.paused else ""}')


if __name__ == '__main__':
    run(Crystalward)
