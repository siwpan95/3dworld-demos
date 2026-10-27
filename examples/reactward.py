# -*- coding: utf-8 -*-
"""《噬纹 / Reactward》—— Gray-Scott 反应扩散（creative coding 缺口之一：图灵斑图）。

两种虚拟化学物质 U（蓝）与 V（红）在平面上扩散并反应：

    U + 2V → 3V        （V 吃掉 U 繁殖自己）
    U → P              （U 缓慢衰变）

离散化后每步::

    u' = u + dt·(Du·∇²u − u·v² + F·(1−u))
    v' = v + dt·(Dv·∇²v + u·v² − (F+k)·v)

仅有的两个控制参数 **F（喂入率）与 k（清除率）** 决定斑图形态——珊瑚 / 迷宫 /
蠕虫 / 泡沫 / 孤子，全部来自同一对方程（Turing 1952 形态发生论的教科书演示）。

实现：**GPU ping-pong**——两份 RGBA16F 状态纹理（引擎 `engine.targets.RenderTarget`，
R=U G=V），每帧 12 个子步在 fragment shader 里做 3×3 拉普拉斯；边界 REPEAT（无限
平铺）。**零素材**：初始场由初始化着色器现算（种子方块 + 哈希噪声），调色是 IQ
cosine palette。鼠标按住可向场内注入 V（"喂一口"，斑图会绕开/吞掉它）。

用法::

    python examples/reactward.py                 正常运行（预设自动轮播）
    python examples/reactward.py --smoke 2       冒烟
    python examples/reactward.py --shot          抓帧到 docs/ 后关闭

按键：1-5 切预设   SPACE 暂停   R 重置种子   左键拖动注入 V   ESC 退出
"""
import math
import sys

from pyglet.gl import (GL_BLEND, GL_FALSE, GL_FLOAT, GL_FRAMEBUFFER, GL_REPEAT,
                       GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
                       GL_TRIANGLES, glBindFramebuffer, glBindTexture,
                       glDisable, glTexParameteri, glViewport)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.window import key

from world3d.engine.targets import RenderTarget                      # noqa: E402
from world3d.platform.app import DemoApp, run                        # noqa: E402

SIZE = 512                           # 场分辨率（正方形）
SUBSTEPS = 12                        # 每帧模拟子步数
DT = 1.0                             # 每子步的时间步长

# (名称, F, k, Du, Dv, palette 相位) —— F/k 是 Pearson 经典参数域的实测点
PRESETS = (
    ('coral',   0.0545, 0.0620, 0.210, 0.105, 0.0),
    ('maze',    0.0290, 0.0570, 0.160, 0.080, 2.2),
    ('worms',   0.0780, 0.0610, 0.180, 0.090, 3.6),
    ('bubbles', 0.0980, 0.0570, 0.190, 0.095, 4.6),
    ('uskate',  0.0620, 0.0609, 0.205, 0.100, 5.5),
)
PRESET_SECS = 9.0                    # 预设自动轮播周期（保证 --shot 两帧不同）


# ------------------------------------------------------------ 着色器 --
QUAD_VERT = '''#version 330 core
in vec2 position;
in vec2 texcoord;
out vec2 v_uv;
void main() {
    v_uv = texcoord;
    gl_Position = vec4(position, 0.0, 1.0);
}'''

SIM_FRAG = '''#version 330 core
in vec2 v_uv;
out vec4 out_color;
uniform sampler2D u_state;
uniform vec2 u_texel;                 // 1/SIZE
uniform float u_F, u_K, u_du, u_dv;
void main() {
    // 3x3 拉普拉斯：中心 -1、正交邻 0.2、对角 0.05（权重和为 0，边界由 REPEAT 无限平铺）
    vec2 t = u_texel;
    float u = texture(u_state, v_uv).r;
    float v = texture(u_state, v_uv).g;
    float lu = -u, lv = -v;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            if (i == 0 && j == 0) continue;
            float w = (i == 0 || j == 0) ? 0.2 : 0.05;
            vec4 s = texture(u_state, v_uv + vec2(i, j) * t);
            lu += w * s.r;
            lv += w * s.g;
        }
    }
    float uvv = u * v * v;
    float du = u_du * lu - uvv + u_F * (1.0 - u);
    float dv = u_dv * lv + uvv - (u_F + u_K) * v;
    out_color = vec4(clamp(u + du, 0.0, 1.0), clamp(v + dv, 0.0, 1.0), 0.0, 1.0);
}'''

INIT_FRAG = '''#version 330 core
in vec2 v_uv;
out vec4 out_color;
uniform float u_seed;
float hash(vec2 p) {
    return fract(sin(dot(p + u_seed, vec2(127.1, 311.7))) * 43758.5453);
}
void main() {
    float u = 1.0;
    float v = 0.0;
    // 场中央撒 5 个种子方块（V=1 的"营养斑"），四周哈希散点打破对称
    vec2 c = v_uv - 0.5;
    if (max(abs(c.x), abs(c.y)) < 0.04) v = 1.0;
    for (int k = 0; k < 5; k++) {
        float a = 6.2831853 * float(k) / 5.0 + u_seed;
        vec2 d = vec2(cos(a), sin(a)) * 0.22;
        if (max(abs(c.x - d.x), abs(c.y - d.y)) < 0.02) v = 1.0;
    }
    if (hash(floor(v_uv * 90.0)) > 0.985) v = max(v, 0.6);
    out_color = vec4(u, v, 0.0, 1.0);
}'''

BRUSH_FRAG = '''#version 330 core
in vec2 v_uv;
out vec4 out_color;
uniform sampler2D u_state;
uniform vec2 u_brush;                 // 鼠标位置（UV）
void main() {
    vec4 s = texture(u_state, v_uv);
    float d = length(v_uv - u_brush);
    float m = smoothstep(0.030, 0.010, d);      // 软边圆刷
    s.g = mix(s.g, 1.0, m);
    out_color = s;
}'''

DISPLAY_FRAG = '''#version 330 core
in vec2 v_uv;
out vec4 out_color;
uniform sampler2D u_state;
uniform vec3 u_pa, u_pb, u_pc, u_pd;  // IQ cosine palette 系数
void main() {
    float v = texture(u_state, v_uv).g;
    float t = clamp(v * 1.6, 0.0, 1.0);
    vec3 col = u_pa + u_pb * cos(6.2831853 * (u_pc * t + u_pd));
    // 高浓度处提亮一点，让"活性区"从底色里浮出来
    col += vec3(0.10, 0.06, 0.02) * smoothstep(0.55, 1.0, v);
    out_color = vec4(col, 1.0);
}'''


def _bind(tex, unit):
    """pyglet 2.1 的 Texture 没有 .use(unit)——手动绑纹理单元。"""
    from pyglet.gl import GL_TEXTURE0, glActiveTexture
    glActiveTexture(GL_TEXTURE0 + unit)
    glBindTexture(GL_TEXTURE_2D, tex.id)


def _quad(prog):
    """全屏两三角（NDC），供所有 pass 复用。"""
    return prog.vertex_list(
        6, GL_TRIANGLES,
        position=('f', (-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1)),
        texcoord=('f', (0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1)))


class Reactward(DemoApp):
    TITLE = 'REACTward - Gray-Scott reaction-diffusion (Turing patterns)'
    HELP = ('1-5 presets  drag: inject V  R reseed  SPACE pause  ESC')
    SHOTS = ((5.0, 'docs/reactward_01.png'), (16.0, 'docs/reactward_02.png'))

    def setup(self):
        self._rt = [RenderTarget(SIZE, SIZE, depth=False),
                    RenderTarget(SIZE, SIZE, depth=False)]
        for rt in self._rt:
            for tex in rt.color:
                glBindTexture(GL_TEXTURE_2D, tex.id)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        self.sim = ShaderProgram(Shader(QUAD_VERT, 'vertex'),
                                 Shader(SIM_FRAG, 'fragment'))
        self.init_prog = ShaderProgram(Shader(QUAD_VERT, 'vertex'),
                                       Shader(INIT_FRAG, 'fragment'))
        self.brush = ShaderProgram(Shader(QUAD_VERT, 'vertex'),
                                   Shader(BRUSH_FRAG, 'fragment'))
        self.disp = ShaderProgram(Shader(QUAD_VERT, 'vertex'),
                                  Shader(DISPLAY_FRAG, 'fragment'))
        self.quads = {p: _quad(p) for p in
                      (self.sim, self.init_prog, self.brush, self.disp)}
        self.preset_i = 0
        self.cur = 0                        # ping-pong 指针（cur 是最新状态）
        self._brush_uv = None               # 待注入位置（UV）
        self._seed = 1.0
        self._reseed()

    # ------------------------------------------------------------ 场操作 --
    def _reseed(self):
        self._seed = (self._seed * 1.6180339 + 7.3) % 97.0
        with self._rt[0]:
            self.init_prog.use()
            self.init_prog['u_seed'] = self._seed
            self.quads[self.init_prog].draw(GL_TRIANGLES)
        self.cur = 0

    def _preset(self):
        return PRESETS[self.preset_i]

    def _step_sim(self):
        src, dst = self._rt[self.cur], self._rt[1 - self.cur]
        name, F, K, du, dv, _ph = self._preset()
        with dst:
            self.sim.use()
            self.sim['u_state'] = 0
            _bind(src.color[0], 0)
            self.sim['u_texel'] = (1.0 / SIZE, 1.0 / SIZE)
            self.sim['u_F'] = F
            self.sim['u_K'] = K
            self.sim['u_du'] = du
            self.sim['u_dv'] = dv
            self.quads[self.sim].draw(GL_TRIANGLES)
        self.cur = 1 - self.cur

    def _apply_brush(self):
        if self._brush_uv is None:
            return
        src, dst = self._rt[self.cur], self._rt[1 - self.cur]
        with dst:
            self.brush.use()
            self.brush['u_state'] = 0
            _bind(src.color[0], 0)
            self.brush['u_brush'] = self._brush_uv
            self.quads[self.brush].draw(GL_TRIANGLES)
        self.cur = 1 - self.cur
        self._brush_uv = None

    # ------------------------------------------------------------ 主循环 --
    def update(self, dt):
        if self.paused:
            return
        # 预设自动轮播：换 F/k（同一场继续演化，形态"变形"而非重置——更好看）
        self.preset_i = int(self.time / PRESET_SECS) % len(PRESETS)
        self._apply_brush()
        for _ in range(SUBSTEPS):
            self._step_sim()

    def render(self):
        w, h = self.window.width, self.window.height
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, w, h)
        glDisable(GL_BLEND)
        glDisable(self._depth_false())      # 2D 全屏，无需深度
        state = self._rt[self.cur]
        self.disp.use()
        self.disp['u_state'] = 0
        _bind(state.color[0], 0)
        ph = self._preset()[5]
        self.disp['u_pa'] = (0.52, 0.42, 0.36)
        self.disp['u_pb'] = (0.48, 0.42, 0.36)
        self.disp['u_pc'] = (1.0, 1.0, 1.0)
        self.disp['u_pd'] = (0.00 + ph, 0.18 + ph, 0.36 + ph)
        self.quads[self.disp].draw(GL_TRIANGLES)

    def _depth_false(self):
        # 兼容 helper：GL_DEPTH_TEST 常量在此处引用，避免文件头堆积
        from pyglet.gl import GL_DEPTH_TEST
        return GL_DEPTH_TEST

    def status_text(self):
        name = self._preset()[0]
        brush = 'inject' if self._brush_uv else '-'
        return (f'Gray-Scott  preset {self.preset_i + 1} ({name})   '
                f'{self.time:6.1f}s   substeps x{SUBSTEPS}   brush {brush}')

    # ------------------------------------------------------------ 输入 --
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self._brush_uv = (x / max(1, self.window.width),
                          y / max(1, self.window.height))

    def on_mouse_press(self, x, y, button, modifiers):
        self._brush_uv = (x / max(1, self.window.width),
                          y / max(1, self.window.height))

    def on_key(self, symbol, modifiers):
        if key._1 <= symbol <= key._5:
            self.preset_i = symbol - key._1
        elif symbol == key.R:
            self._reseed()


if __name__ == '__main__':
    sys.exit(run(Reactward))
