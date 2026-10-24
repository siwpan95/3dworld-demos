# -*- coding: utf-8 -*-
"""《胞格 / Cellward》—— 元胞自动机实验场（141 条规则实时跑）。

**为什么值得单独一个 demo**
----------------------------
``world3d.logic.ca`` 是引擎里规则数量最多的一块能力：``RULES`` 注册表里
**实测 141 条**规则，横跨四个类别 ——

* ``life``（63 条）：Conway 生命游戏与它的整个家族（HighLife / Day&Night /
  Maze / Coral / Replicator / 各种 LTL…）；
* ``continuous``（58 条）：连续场规则 —— Gray-Scott 反应扩散的十几种形态
  （斑点/迷宫/蠕虫/有丝分裂/滑翔子）、Lenia（人造生命）、SmoothLife、
  BZ 反应、Wolfram 一维规则；
* ``multi_state``（9 条）：多态规则 —— Brian's Brain、Wireworld（电路）、
  沙堆自组织临界、森林火灾、晶体生长；
* ``complex``（11 条）：Cyclic CA、Langton's Loop、Snowflake、Hodgepodge。

``tests/test_ca.py`` 覆盖了这些规则的演化正确性，但**演示仓里零个 demo 用过它**。
本 demo 是把这块能力真正"看见"的那一端：不写任何规则，全部从 ``RULES`` 取。

实现要点
--------
* 网格状态每一代写进一张 ``GL_NEAREST`` 纹理，全屏四边形采样 ——
  CPU 只做「演化 + 上传」，画面 100% 由 GPU 采样。
* **固定步长累加器**：``SPEED`` 是"每秒多少代"，不是"每帧多少步"。
  于是 ``--shot 6`` 时正好是第 180 代 —— 截图可复现，换台机器也一样。
* 网格分辨率按规则代价分档（见 ``TIERS``）：Lenia 的卷积核比生命游戏贵
  两个数量级，同分辨率下会直接把帧率打到个位数。

用法::

    python examples/cellward.py                    正常运行
    python examples/cellward.py --smoke 3          冒烟
    python examples/cellward.py --shot             抓剧照到 docs/cellward_01/02.png
    python examples/cellward.py --flag coral       起手就切到 coral 规则
    python examples/cellward.py --flag lenia       起手 Lenia（会自动降分辨率）

按键：1-4 换类别   [ ] 上/下一条规则   N 随机规则   R 重新播种
      ，/. 调代数速度   G 换网格分辨率   P 换配色   S 单步   SPACE 暂停   ESC 退出
"""
import pyglet
import time

from pyglet.gl import (GL_DEPTH_TEST, GL_NEAREST, GL_TRIANGLES,
                       glDisable, glEnable)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4
from pyglet.window import key

# ⚠️ 不要再加 sys.path 引导：引擎由 `pip install 3dworld` 提供
# （见 requirements.txt 与 docs/依赖与版本.md）。
from world3d.logic.ca import (CellularAutomaton, list_rules,               # noqa: E402
                              list_rules_by_category)
from world3d.platform.app import (DemoApp, make_fullscreen_quad,           # noqa: E402
                                  run)

VERTEX_SRC = '''#version 330 core
in vec3 position;
in vec4 colors;
out vec2 v_uv;
void main() {
    v_uv = position.xy * 0.5 + 0.5;
    gl_Position = vec4(position.xy, 0.0, 1.0);
}'''

FRAGMENT_SRC = '''#version 330 core
in vec2 v_uv;
out vec4 final_colors;

uniform sampler2D u_state;   // R = 状态值, B = 余晖（刚死的格子）
uniform vec2 u_grid;         // 网格 (cols, rows)
uniform vec2 u_view;         // 画面像素
uniform vec3 u_c0, u_c1, u_c2, u_c3;
uniform float u_scan;        // 像素栅格线强度
uniform float u_binary;      // 1 = 二态规则

void main() {
    // --- 保持网格宽高比：按短边做信箱式留边，而不是把方格拉成矩形 ---
    float ga = u_grid.x / u_grid.y;
    float va = u_view.x / u_view.y;
    vec2 uv = v_uv;
    vec2 span = vec2(1.0, 1.0);
    if (va > ga) { span.x = ga / va; } else { span.y = va / ga; }
    vec2 q = (v_uv - 0.5) / span + 0.5;
    if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) {
        final_colors = vec4(0.016, 0.018, 0.024, 1.0);
        return;
    }

    vec4 s = texture(u_state, q);
    // 状态值已经归一化在 R 通道（离散规则 0/1，多态 0..k，连续场 0..255）
    float t = clamp(s.r, 0.0, 1.0);
    // 二态规则（生命游戏家族）只有 0/1 两级：直接喂调色板只会得到"黑 + 白"，
    // 整屏灰扑扑。改成"底色 → 第三档色"，把高亮留给余晖。
    if (u_binary > 0.5) { t = s.r * 0.66; }

    // --- 四段渐变（配色切换只改这四个端点） ---
    vec3 col;
    if (t < 0.34)      col = mix(u_c0, u_c1, t / 0.34);
    else if (t < 0.67) col = mix(u_c1, u_c2, (t - 0.34) / 0.33);
    else               col = mix(u_c2, u_c3, (t - 0.67) / 0.33);

    // --- 余晖：刚死的格子留一点冷光，生命游戏才有"呼吸" ---
    col += u_c2 * s.b * 0.30;

    // --- 像素栅格：格子够少时才画细线，免得变成一层网 ---
    vec2 g = q * u_grid;
    float cells = u_grid.x * u_grid.y;
    float line = smoothstep(0.0, 0.06, min(fract(g.x), fract(g.y)));
    col *= mix(1.0, 0.55 + 0.45 * line, u_scan * (1.0 - smoothstep(4000.0, 20000.0, cells)));

    // --- 暗角 ---
    vec2 d = v_uv - 0.5;
    col *= 1.0 - 0.55 * dot(d, d);
    final_colors = vec4(col, 1.0);
}'''

#: 自动轮播的门面规则（按可用性过滤；``--flag hold`` 关掉轮播）
SHOWCASE = ('maze', 'cyclic_6', 'gray_scott_coral', 'hodgepodge',
            'brians_brain', 'mazectric', 'wireworld', 'coral', 'lenia')
CYCLE_SEC = 8.0

PALETTES = (
    # 冷焰（默认）：近黑 → 深蓝 → 青 → 米白
    ((0.02, 0.03, 0.06), (0.05, 0.22, 0.45), (0.15, 0.72, 0.80), (0.92, 0.97, 1.00)),
    # 岩浆：黑 → 暗红 → 橙 → 亮黄
    ((0.04, 0.02, 0.02), (0.32, 0.05, 0.04), (0.85, 0.30, 0.06), (1.00, 0.92, 0.62)),
    # 苔原：墨绿 → 青苔 → 黄绿 → 象牙
    ((0.02, 0.04, 0.03), (0.06, 0.24, 0.16), (0.45, 0.72, 0.28), (0.94, 0.97, 0.86)),
    # 幽灵：近黑 → 紫 → 洋红 → 白
    ((0.03, 0.02, 0.05), (0.22, 0.08, 0.42), (0.72, 0.20, 0.66), (1.00, 0.94, 1.00)),
)

#: 网格分辨率档位（cols, rows）。连续场规则每个元胞要跑卷积核，比离散规则贵
#: 一到两个数量级（实测 Lenia 在 120×180 上单代 480ms），所以按代价分档。
TIERS = ((72, 48), (96, 64), (144, 96), (192, 128), (240, 160))
TIER_NAMES = ('XS', 'S', 'M', 'L', 'XL')

#: 名字前缀命中这些的规则走最小档
HEAVY = ('lenia', 'smooth_life', 'larger_than_life', 'ltl_', 'lt_l')


def tier_for(rule_name, category):
    if rule_name.startswith(HEAVY):
        return 0
    if category == 'continuous':
        return 1
    return 2


class Cellward(DemoApp):
    TITLE = 'CELLward - cellular automata lab (logic/ca, 141 rules)'
    HELP = ('1-4 category  [ ] rule  N random  R reseed  , . speed  '
            'G grid  P palette  S step  SPACE pause  ESC')
    DURATION = None
    SHOTS = ((5.0, 'docs/cellward_01.png'), (13.0, 'docs/cellward_02.png'))

    SPEEDS = (5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 240.0)
    SHOT_SPEED = 3               # SHOTS 时刻依赖速度：固定 30 代/秒，--shot 可复现

    def setup(self):
        glDisable(GL_DEPTH_TEST)
        self.program = ShaderProgram(Shader(VERTEX_SRC, 'vertex'),
                                     Shader(FRAGMENT_SRC, 'fragment'))
        self.quad = make_fullscreen_quad(self.program)
        self.by_cat = list_rules_by_category()
        self.cats = ('life', 'continuous', 'multi_state', 'complex')
        self.palette = 0
        self.speed_i = self.SHOT_SPEED
        self.acc = 0.0
        self.gen = 0
        self.step_ms = 0.0
        self.scan = 1.0
        self.seed = 20261024
        self.tier_override = None
        self.cycle = 'hold' not in self.args.flags
        self._cyc_i = 0

        want = None
        for f in self.args.flags:
            if f in ('palette1', 'palette2', 'palette3'):
                self.palette = int(f[-1])
            elif f == 'noscan':
                self.scan = 0.0
            elif f.startswith('seed'):
                try:
                    self.seed = int(f[4:])
                except ValueError:
                    pass
            elif f in list_rules():
                want = f

        self.cat_i = 0
        if want:
            for i, c in enumerate(self.cats):
                if want in self.by_cat.get(c, ()):
                    self.cat_i = i
                    break
            self.rule = want
        else:
            allr = set(list_rules())
            self.showcase = [r for r in SHOWCASE if r in allr] or sorted(allr)[:1]
            self.rule = self.showcase[0]
        self._build()

    # ------------------------------------------------------------ 构建 --
    def _build(self):
        cat = self.cats[self.cat_i]
        tier = self.tier_override
        if tier is None:
            tier = tier_for(self.rule, cat)
        cols, rows = TIERS[tier]
        self.tier = tier
        self.cols, self.rows = cols, rows
        self.auto = CellularAutomaton(rows, cols, rule_name=self.rule,
                                      initial_density=0.30, seed=self.seed)
        self.tex = pyglet.image.Texture.create(
            cols, rows, min_filter=GL_NEAREST, mag_filter=GL_NEAREST)
        self._buf = bytearray(cols * rows * 4)
        self._img = pyglet.image.ImageData(cols, rows, 'RGBA', bytes(self._buf))
        self.acc = 0.0
        self.gen = 0
        self._dirty = True
        self.upload()

    # ------------------------------------------------------------ 上传 --
    def upload(self):
        """把 ``grid`` 写成 RGBA（R=状态, B=余晖），一次 blit 到纹理。

        两个坑都在这里兜住：
        1. **状态要缩放到 0..255**。离散规则的状态是 0/1/2…，直接塞进 8 位通道
           会被 GL 归一化成 1/255 ≈ 0.004 —— 满屏黑，能看见的全是余晖通道
           （第一版就是这样：活细胞黑着、死细胞发亮）。
        2. 逐行用 ``bytes(map(LUT.__getitem__, row))`` 而不是逐像素写 Python int：
           144×96 的逐像素版本实测 ~9ms，查表切片版稳定在 1ms 内。
        """
        rows, cols = self.rows, self.cols
        rule = self.auto.rule
        if getattr(rule, 'has_float_state', False):
            lut = list(range(256))                 # 连续场本身就是 0..255
        else:
            denom = max(1, self.auto.num_states - 1)
            lut = [min(255, int(round(i * 255.0 / denom))) for i in range(256)]
        tl = max(1, self.auto.trail_length)
        tlut = [min(255, int(v * 255.0 / tl)) for v in range(64)]
        buf = self._buf
        grid, trail = self.auto.grid, self.auto.trail
        has_trail = any(any(r) for r in trail)
        get = lut.__getitem__
        tget = tlut.__getitem__
        for y in range(rows):
            base = (rows - 1 - y) * cols * 4      # ImageData 原点在左下，翻一次
            buf[base:base + cols * 4:4] = bytes(map(get, grid[y]))
            if has_trail:
                buf[base + 2:base + cols * 4:4] = bytes(
                    map(tget, (min(63, v) for v in trail[y])))
        self._img.set_data('RGBA', cols * 4, bytes(buf))
        self.tex.blit_into(self._img, 0, 0, 0)
        self._dirty = False

    # ------------------------------------------------------------ 演化 --
    def advance(self):
        t0 = time.perf_counter()
        self.auto.step()
        self.gen += 1
        self.step_ms = (time.perf_counter() - t0) * 1000.0
        self._dirty = True

    def update(self, dt):
        if self.paused:
            return
        if self.cycle and getattr(self, 'showcase', None):
            i = int(self.time / CYCLE_SEC)
            if i != self._cyc_i:
                self._cyc_i = i
                self.rule = self.showcase[i % len(self.showcase)]
                for c_i, c in enumerate(self.cats):
                    if self.rule in self.by_cat.get(c, ()):
                        self.cat_i = c_i
                        break
                self.tier_override = None
                self._build()
        self.acc += dt * self.SPEEDS[self.speed_i]
        budget = 0
        while self.acc >= 1.0 and budget < 8:     # 慢规则单帧最多 8 代，防卡死
            self.acc -= 1.0
            self.advance()
            budget += 1
        if self.acc > 4.0:
            self.acc = 4.0                        # 掉帧后别攒一屁股债
        if self._dirty:
            self.upload()

    # ------------------------------------------------------------ 交互 --
    def on_key(self, symbol, modifiers):
        k = key
        cat = self.cats[self.cat_i]
        rules = self.by_cat.get(cat, [])
        if symbol in (k._1, k._2, k._3, k._4):
            self.cat_i = symbol - k._1
            cat = self.cats[self.cat_i]
            rules = self.by_cat.get(cat, [])
            self.rule = rules[0]
            self.tier_override = None
            self._build()
        elif symbol in (k.BRACKETLEFT, k.BRACKETRIGHT):
            if rules:
                i = rules.index(self.rule) if self.rule in rules else 0
                self.rule = rules[(i + (1 if symbol == k.BRACKETRIGHT else -1))
                                  % len(rules)]
                self.tier_override = None
                self._build()
        elif symbol == k.N:
            import random
            self.rule = random.Random(self.seed + self.gen).choice(list_rules())
            for i, c in enumerate(self.cats):
                if self.rule in self.by_cat.get(c, ()):
                    self.cat_i = i
                    break
            self.tier_override = None
            self._build()
        elif symbol == k.R:
            self.auto.randomize(0.30)
            self.gen = 0
            self._dirty = True
            self.upload()
        elif symbol == k.COMMA:
            self.speed_i = max(0, self.speed_i - 1)
        elif symbol == k.PERIOD:
            self.speed_i = min(len(self.SPEEDS) - 1, self.speed_i + 1)
        elif symbol == k.G:
            self.tier_override = (self.tier + 1) % len(TIERS)
            self._build()
        elif symbol == k.P:
            self.palette = (self.palette + 1) % len(PALETTES)
        elif symbol == k.S:
            self.advance()
            self.upload()

    def status_text(self):
        cat = self.cats[self.cat_i]
        return (f'{self.rule} [{cat}]  {self.cols}x{self.rows} '
                f'{TIER_NAMES[self.tier]}  gen {self.gen}'
                f'  {self.SPEEDS[self.speed_i]:.0f}/s  step {self.step_ms:5.1f}ms'
                f'  pal {self.palette + 1}/{len(PALETTES)}  {self.fps:5.1f} fps'
                f'{"  [PAUSED]" if self.paused else ""}')

    # ------------------------------------------------------------ 渲染 --
    def render(self):
        w, h = self.window.width, self.window.height
        self.window.projection = Mat4.orthogonal_projection(0, w, 0, h, -1, 1)
        self.window.view = Mat4()
        prog = self.program
        prog.use()
        prog['u_grid'] = (float(self.cols), float(self.rows))
        prog['u_view'] = (float(w), float(h))
        c = PALETTES[self.palette]
        prog['u_c0'], prog['u_c1'], prog['u_c2'], prog['u_c3'] = c
        prog['u_scan'] = self.scan
        prog['u_binary'] = 1.0 if self.auto.num_states <= 2 else 0.0
        self.tex.bind(0)
        prog['u_state'] = 0
        self.quad.draw(GL_TRIANGLES)
        glEnable(GL_DEPTH_TEST)
        self.window.projection = self.perspective


if __name__ == '__main__':
    run(Cellward)
