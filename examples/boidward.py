# -*- coding: utf-8 -*-
"""《群迹 / Boidward》—— logic.boids 的 Flock/Boid 与三力预设首秀（实例化渲染）。

**为什么单独一个 demo**
-----------------------
``world3d.logic.boids`` 是纯数学层（只依赖 ``pyglet.math.Vec3``，不碰 OpenGL），
此前只有 ``examples/boids.py`` 这一个消费者，且它把 boid 画成"薄片三角翼"。
本 demo 换个更直白的视角：**把每只 boid 画成一个小圆锥**（锥尖指向速度方向），
于是"一群体朝同一方向飞"在画面上就是"一片锥尖整齐指向同一点"——群集的运动学
比薄片更易读；同时把**着色**做成可切换的两种语义，把三力"看得见"：

* **速度着色（默认）**：慢=冷蓝、快=暖橙，整群颜色趋同即"群体同步"。
* **行为着色（按 C 切换）**：逐 boid 现场重算分离/对齐/聚合三力的相对强度，
  按 (分离, 对齐, 聚合) = (红, 绿, 蓝) 归一化成顶点色——哪根力主导，鸟就偏哪种色。
  这直接把 Reynolds 三力"画"了出来，无需改引擎。

**捕食者（三力的放大镜）**
--------------------------
默认有一只**自动绕圈巡航的捕食者**（权重远大于三力，见 ``DEFAULT_WEIGHTS``），
半径内的 boid 被推开——你会看到"炸群 → 重新聚拢"的循环，分离力在这一刻
主导（行为着色下整片变红）。左键点击地面可把捕食者钉到鼠标落点（鼠标即吸引/
排斥源），按 D 整体开关捕食者，对比有无捕食者时群体的差异。

**本 demo 消费的引擎 API**
--------------------------
* ``world3d.logic.boids.Flock``：构造即 ``reset()``；``step(dt)`` 推进；
  ``set_preset(name)`` 套用 ``PRESETS`` 里的权重+速度上限+涡旋；
  ``set_predator(Vec3|None)`` 增减捕食者；``positions()/velocities()``
  返回平坦坐标列表（省去建对象的开销，渲染层直接打包实例属性）。
* ``world3d.logic.boids.PRESETS / PRESET_LABELS / PRESET_MAX_SPEED /
  PRESET_TURBULENCE / DOMAIN``：P 键在 ``PRESETS`` 的各档间循环（顺序由
  字典 ``keys()`` 决定），``DOMAIN`` 给出密度调好的域（bounds=9 / 邻域 3.0）。
* ``world3d.logic.camera.OrbitCamera``：斜视俯瞰鸟群，鼠标拖动/滚轮接
  ``orbit_camera_input``。
* 渲染走 ``world3d.platform.app.DemoApp`` + 原生 **实例化** 绘制（一只圆锥几何
  只上传一次，位置/速度/颜色是逐实例属性，一次 ``draw`` 画完整个群）。

**HUD（status_text）读的是**
----------------------------
``Flock`` 的 ``count``、``mean_speed()``（平均速率）、当前 ``PRESET_LABELS``
预设名，外加捕食者开关与着色模式——没有读任何内部数组，全是公共 API。

**控制键**
----------
P 切预设   A/Z 增减数量   C 切着色(速度/三力)   D 捕食者开关
左键 钉捕食者到地面   拖动/滚轮 相机   SPACE 暂停   ESC 退出

**复现性**：固定 ``SEED``，捕食者轨迹由确定性 ``self.time`` 驱动，同种子逐位一致。

用法::

    python examples/boidward.py                正常运行
    python examples/boidward.py --smoke 2      冒烟
    python examples/boidward.py --shot         抓剧照到 docs/boidward_01/02.png
"""
import math

import pyglet
from pyglet.gl import (GL_BLEND, GL_DEPTH_TEST, GL_LINES, GL_ONE_MINUS_SRC_ALPHA,
                       GL_SRC_ALPHA, GL_TRIANGLES, glBlendFunc, glDisable,
                       glEnable)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Vec3

from world3d.logic.boids import (DOMAIN, Flock, PRESETS, PRESET_LABELS)
from world3d.logic.camera import OrbitCamera
from world3d.platform.app import DemoApp, orbit_camera_input, run

SEED = 20261006                 # 固定种子：同种子 + 同 dt 序列 => 逐位可复现
INITIAL_COUNT = 300            # 群规模：DOMAIN 密度下 300 只已是整齐的一大群
STEP_COUNT = 40                # A/Z 每次增减的步长
MIN_COUNT, MAX_COUNT = 20, 500

# --------------------------------------------------------------- 圆锥几何 --
def cone_geometry(seg=12, r=0.30, L=0.85):
    """一只朝向 +Z 的小圆锥（锥尖在 +Z，底圈在 z=0），局部坐标、单位尺寸。

    拆成 side（锥面）+ base（底盖）两类三角形；逐三角形写法线，底盖法线 -Z，
    锥面法线按 (面中点·法线) 的符号校正为向外——避免半边锥面被 lambert 涂黑。
    顶点色统一白 (1,1,1,1)，真正着色由逐实例的 ``i_col`` 决定（乘以 base 白）。
    """
    pos, nrm, cols = [], [], []
    apex = (0.0, 0.0, L)
    center = (0.0, 0.0, 0.0)
    ring = [(r * math.cos(2 * math.pi * j / seg),
             r * math.sin(2 * math.pi * j / seg), 0.0) for j in range(seg)]

    def emit(p, n):
        pos.extend(p)
        nrm.extend(n)
        cols.extend((1.0, 1.0, 1.0, 1.0))

    for j in range(seg):
        p1, p2 = ring[j], ring[(j + 1) % seg]
        # 锥面法线 = normalize(cross(apex->p1, apex->p2))，按中点方向校正朝外
        e1 = (p1[0] - apex[0], p1[1] - apex[1], p1[2] - apex[2])
        e2 = (p2[0] - apex[0], p2[1] - apex[1], p2[2] - apex[2])
        nx = e1[1] * e2[2] - e1[2] * e2[1]
        ny = e1[2] * e2[0] - e1[0] * e2[2]
        nz = e1[0] * e2[1] - e1[1] * e2[0]
        ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        nx, ny, nz = nx / ln, ny / ln, nz / ln
        mx = (apex[0] + p1[0] + p2[0]) / 3.0
        my = (apex[1] + p1[1] + p2[1]) / 3.0
        mz = (apex[2] + p1[2] + p2[2]) / 3.0
        if nx * mx + ny * my + nz * mz < 0.0:     # 法线朝里就翻外
            nx, ny, nz = -nx, -ny, -nz
        emit(apex, (nx, ny, nz))
        emit(p2, (nx, ny, nz))
        emit(p1, (nx, ny, nz))
    for j in range(seg):                          # 底盖：-Z
        p1, p2 = ring[j], ring[(j + 1) % seg]
        emit(center, (0.0, 0.0, -1.0))
        emit(p1, (0.0, 0.0, -1.0))
        emit(p2, (0.0, 0.0, -1.0))
    return pos, nrm, cols


def ring_geometry(radius=1.15, segments=40):
    """三个正交圆环（捕食者准星）：比实心球省顶点，也更"准星"。"""
    verts, cols = [], []
    for axis in range(3):
        for k in range(segments):
            a0 = 2.0 * math.pi * k / segments
            a1 = 2.0 * math.pi * (k + 1) / segments
            for a in (a0, a1):
                ca, sa = math.cos(a), math.sin(a)
                if axis == 0:
                    verts.extend((0.0, radius * ca, radius * sa))
                elif axis == 1:
                    verts.extend((radius * ca, 0.0, radius * sa))
                else:
                    verts.extend((radius * ca, radius * sa, 0.0))
                cols.extend((1.0, 0.36, 0.26, 1.0))
    return verts, cols


# ---------------------------------------------------------------- 着色器 --
# 实例化顶点着色器：由逐实例的 i_vel 现场构造正交基 (right, up, fwd)，
# 把局部圆锥旋到航向、平移到 i_pos。基准值 fwd*0.001 防与参考上向量平行的退化。
BIRD_VERT = '''#version 330 core
in vec3 position;
in vec3 normals;
in vec4 colors;
in vec3 i_pos;
in vec3 i_vel;
in vec4 i_col;
out vec3 v_nrm;
out vec4 v_col;
uniform WindowBlock { mat4 projection; mat4 view; } window;

void main() {
    vec3 fwd = normalize(i_vel);
    vec3 ref = vec3(0.0, 1.0, 0.0) + fwd * 0.001;
    vec3 right = normalize(cross(ref, fwd));
    vec3 up = cross(fwd, right);
    mat3 basis = mat3(right, up, fwd);
    vec3 world = i_pos + basis * position;
    v_nrm = normalize(basis * normals);
    v_col = colors * i_col;          // base 白 (1,1,1,1)，实际色来自 i_col
    gl_Position = window.projection * window.view * vec4(world, 1.0);
}'''

BIRD_FRAG = '''#version 330 core
in vec3 v_nrm;
in vec4 v_col;
out vec4 final_colors;
uniform vec3 u_sun;
void main() {
    vec3 n = normalize(v_nrm);
    float lam = max(dot(n, u_sun), 0.0);
    vec3 col = v_col.rgb * (0.40 + 0.75 * lam);
    final_colors = vec4(col, 1.0);
}'''

SOLID_VERT = '''#version 330 core
in vec3 position;
in vec4 colors;
out vec4 v_col;
uniform WindowBlock { mat4 projection; mat4 view; } window;
uniform vec3 u_offset;
void main() {
    v_col = colors;
    gl_Position = window.projection * window.view * vec4(position + u_offset, 1.0);
}'''

SOLID_FRAG = '''#version 330 core
in vec4 v_col;
out vec4 final_colors;
void main() { final_colors = v_col; }'''

# 背景竖向渐变：NDC 全屏四边形，z=w ⇒ 最远，绘制前关深度，给薄片鸟一个明暗参照。
BG_VERT = '''#version 330 core
in vec3 position;
in vec4 colors;
out float v_t;
void main() {
    v_t = position.y * 0.5 + 0.5;
    gl_Position = vec4(position.xy, 1.0, 1.0);
}'''

BG_FRAG = '''#version 330 core
in float v_t;
out vec4 final_colors;
void main() {
    vec3 bottom = vec3(0.020, 0.030, 0.058);
    vec3 top = vec3(0.105, 0.135, 0.225);
    final_colors = vec4(mix(bottom, top, v_t), 1.0);
}'''


class Boidward(DemoApp):
    TITLE = 'BOIDward - logic.boids Flock (cone + instance + 3-force color)'
    HELP = ('P preset  A/Z count  C color(speed/force)  D predator  '
            'leftclick pin  drag/scroll  SPACE pause  ESC 退出')
    DURATION = None
    FOV = 55.0
    Z_NEAR, Z_FAR = 0.5, 400.0
    # 抓帧时刻：群自组织到同向需几秒，之后长期维持"整群同向平移"。
    SHOTS = ((6.0, 'docs/boidward_01.png'),
             (18.0, 'docs/boidward_02.png'))

    def setup(self):
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)

        self.preset = 'tight'
        self.color_mode = 'speed'      # 'speed' | 'force'
        self.pred_on = True            # 默认有捕食者：三力效果最直观
        self.pred_manual = None        # 左键钉定点（覆盖自动巡航）
        self.pred_pos = None
        self._frame = 0
        self._color_cache = None

        self.camera = OrbitCamera(target=Vec3(0, 0, 0), yaw=34.0, pitch=26.0,
                                 dist=18.0)
        self.on_mouse_drag, self.on_mouse_scroll = orbit_camera_input(self, self.camera)
        self.window.push_handlers(on_mouse_press=self.on_mouse_press)

        self.bird_prog = ShaderProgram(Shader(BIRD_VERT, 'vertex'),
                                       Shader(BIRD_FRAG, 'fragment'))
        self.solid_prog = ShaderProgram(Shader(SOLID_VERT, 'vertex'),
                                        Shader(SOLID_FRAG, 'fragment'))
        self.bg_prog = ShaderProgram(Shader(BG_VERT, 'vertex'),
                                     Shader(BG_FRAG, 'fragment'))

        self.base_pos, self.base_nrm, self.base_col = cone_geometry()

        ring_pos, ring_col = ring_geometry()
        self.ring_vlist = self.solid_prog.vertex_list(
            len(ring_pos) // 3, GL_LINES,
            position=('f', ring_pos), colors=('f', ring_col))

        bg_pos = (-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0,
                  -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0)
        self.bg_vlist = self.bg_prog.vertex_list(
            6, GL_TRIANGLES, position=('f', bg_pos), colors=('f', [1.0] * 24))

        self.vlist = None
        self.instances = []
        self._build_flock(INITIAL_COUNT)

    # ------------------------------------------------------------ 群构建 --
    def _build_flock(self, count):
        """建一个 Flock 并重建实例化顶点列表（数量变化时调用）。"""
        self.count = int(count)
        # DOMAIN 给出密度调好的域（bounds=9 / 邻域 3.0 / 格 3.0 / 撒点 4.0），
        # **密度决定观感**：同样 300 只，域收到 9 后群体才真像"一群"。
        self.flock = Flock(count=self.count, seed=SEED, **DOMAIN)
        self.flock.set_preset(self.preset)

        if self.vlist is not None:
            self.vlist.delete()
        n = len(self.base_pos) // 3
        zero3 = [0.0] * (n * 3)
        zero4 = [0.0] * (n * 4)
        # 坑位（来自 examples/boids.py 的实测）：实例属性必须给初始数据，
        # 长度按"基础几何顶点数 × 分量"算，否则会被静默丢属性。
        self.vlist = self.bird_prog.vertex_list_instanced(
            n, GL_TRIANGLES,
            instance_attributes=['i_pos', 'i_vel', 'i_col'],
            position=('f', self.base_pos),
            normals=('f', self.base_nrm),
            colors=('f', self.base_col),
            i_pos=('f', list(zero3)),
            i_vel=('f', list(zero3)),
            i_col=('f', list(zero4)))
        xs, ys, zs = self.flock.positions()
        vx, vy, vz = self.flock.velocities()
        self.instances = []
        for i in range(self.count):
            self.instances.append(self.vlist.add_instance(
                i_pos=(xs[i], ys[i], zs[i]),
                i_vel=(vx[i], vy[i], vz[i]),
                i_col=(1.0, 1.0, 1.0, 1.0)))
        self._color_cache = None

    # ------------------------------------------------------------- 着色 --
    def _speed_colors(self):
        """速度 → 冷蓝(慢)~暖橙(快)。整群颜色趋同即"群体同步"。"""
        xs, ys, zs = self.flock.positions()
        vx, vy, vz = self.flock.velocities()
        ms = self.flock.max_speed or 1.0
        out = []
        for i in range(self.count):
            sp = math.sqrt(vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i])
            k = min(1.0, sp / ms)
            out.append((0.25 + 0.75 * k, 0.55 * (1.0 - k) + 0.15,
                        0.95 * (1.0 - k) + 0.10, 1.0))
        return out

    def _force_colors(self):
        """逐 boid 现场重算三力相对强度，按 (分离, 对齐, 聚合)=(红,绿,蓝) 归一化。

        O(n²) 仅在本模式下跑，且 update 里做了 3 帧一次的缓存，不会拖帧。
        注意：这里的邻域参数取自 Flock 的公共字段 neighbour_radius / sep_radius，
        与引擎内部一致，所以"哪种力主导"的判定是可信的。
        """
        xs, ys, zs = self.flock.positions()
        vx, vy, vz = self.flock.velocities()
        rn2 = self.flock.neighbour_radius ** 2
        rs2 = self.flock.sep_radius ** 2
        out = []
        n = self.count
        for i in range(n):
            pxi, pyi, pzi = xs[i], ys[i], zs[i]
            vxi, vyi, vzi = vx[i], vy[i], vz[i]
            sax = say = saz = 0.0
            aax = aay = aaz = 0.0
            cax = cay = caz = 0.0
            ns = 0
            for j in range(n):
                if j == i:
                    continue
                dx = xs[j] - pxi
                dy = ys[j] - pyi
                dz = zs[j] - pzi
                d2 = dx * dx + dy * dy + dz * dz
                if d2 > rn2 or d2 < 1e-12:
                    continue
                aax += vx[j]; aay += vy[j]; aaz += vz[j]
                cax += xs[j]; cay += ys[j]; caz += zs[j]
                ns += 1
                if d2 < rs2:
                    inv = 1.0 / math.sqrt(d2)
                    sax -= dx * inv; say -= dy * inv; saz -= dz * inv
            if ns:
                inv = 1.0 / ns
                aax = aax * inv - vxi; aay = aay * inv - vyi; aaz = aaz * inv - vzi
                cax = cax * inv - pxi; cay = cay * inv - pyi; caz = caz * inv - pzi
            al = math.sqrt(aax * aax + aay * aay + aaz * aaz)
            cl = math.sqrt(cax * cax + cay * cay + caz * caz)
            sl = math.sqrt(sax * sax + say * say + saz * saz)
            tot = sl + al + cl + 1e-6
            out.append((sl / tot, al / tot, cl / tot, 1.0))
        return out

    # ------------------------------------------------------------ 更新 --
    def update(self, dt):
        if self.paused:
            return

        # 捕食者：默认绕圈巡航；左键钉定点优先；D 关掉则清掉。
        if self.pred_on:
            if self.pred_manual is not None:
                pos = self.pred_manual
            else:
                ang = self.time * 0.7
                b = self.flock.bounds * 0.5
                pos = Vec3(math.cos(ang) * b, 0.0, math.sin(ang * 1.3) * b)
            self.flock.set_predator(pos)
            self.pred_pos = pos
        else:
            self.flock.set_predator(None)
            self.pred_pos = None

        self.flock.step(dt)

        # 着色：速度模式每帧算（便宜）；三力模式 3 帧缓存一次。
        self._frame += 1
        if self.color_mode == 'force':
            if self._color_cache is None or self._frame % 3 == 0:
                self._color_cache = self._force_colors()
            cols = self._color_cache
        else:
            cols = self._speed_colors()

        xs, ys, zs = self.flock.positions()
        vx, vy, vz = self.flock.velocities()
        for i in range(self.count):
            inst = self.instances[i]
            inst.i_pos = (xs[i], ys[i], zs[i])
            inst.i_vel = (vx[i], vy[i], vz[i])
            inst.i_col = cols[i]

    # ------------------------------------------------------------ 渲染 --
    def render(self):
        glDisable(GL_DEPTH_TEST)
        self.bg_prog.use()
        self.bg_vlist.draw(GL_TRIANGLES)
        glEnable(GL_DEPTH_TEST)

        self.window.view = self.camera.view_matrix()

        if self.pred_on and self.pred_pos is not None:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            self.solid_prog.use()
            self.solid_prog['u_offset'] = self.pred_pos
            self.ring_vlist.draw(GL_LINES)
            glDisable(GL_BLEND)

        self.bird_prog.use()
        self.bird_prog['u_sun'] = (0.4, 0.8, 0.4)
        self.vlist.draw(GL_TRIANGLES)

    # ------------------------------------------------------------ 交互 --
    def on_key(self, symbol, modifiers):
        k = pyglet.window.key
        if symbol == k.P:
            order = list(PRESETS.keys())
            idx = (order.index(self.preset) + 1) % len(order)
            self.preset = order[idx]
            self.flock.set_preset(self.preset)
        elif symbol == k.A:
            self._change_count(+STEP_COUNT)
        elif symbol == k.Z:
            self._change_count(-STEP_COUNT)
        elif symbol == k.C:
            self.color_mode = 'force' if self.color_mode == 'speed' else 'speed'
            self._color_cache = None
        elif symbol == k.D:
            self.pred_on = not self.pred_on
            if not self.pred_on:
                self.pred_manual = None

    def _change_count(self, delta):
        n = max(MIN_COUNT, min(MAX_COUNT, self.count + delta))
        if n != self.count:
            self._build_flock(n)

    def on_mouse_press(self, x, y, buttons, modifiers):
        """左键：把捕食者钉到点击射线与 y=0 平面的交点（鼠标即吸引/排斥源）。

        射线由视图矩阵反投影得到——视图是刚体变换，逆 = 旋转转置 + 平移，
        纯算术即可，不必求一般 4×4 逆矩阵（照抄 examples/boids.py 的成熟写法）。
        """
        if not (buttons & pyglet.window.mouse.LEFT):
            return
        w, h = self.window.width, self.window.height
        ndc_x = 2.0 * x / max(1, w) - 1.0
        ndc_y = 2.0 * y / max(1, h) - 1.0
        aspect = w / float(max(1, h))
        tan_half = math.tan(math.radians(self.FOV) * 0.5)
        dx = ndc_x * tan_half * aspect
        dy = ndc_y * tan_half
        dz = -1.0
        m = self.camera.view_matrix()
        right = (m[0], m[4], m[8])
        up = (m[1], m[5], m[9])
        fwd = (-m[2], -m[6], -m[10])
        dirx = right[0] * dx + up[0] * dy + fwd[0] * (-dz)
        diry = right[1] * dx + up[1] * dy + fwd[1] * (-dz)
        dirz = right[2] * dx + up[2] * dy + fwd[2] * (-dz)
        ln = math.sqrt(dirx * dirx + diry * diry + dirz * dirz) or 1.0
        dirx, diry, dirz = dirx / ln, diry / ln, dirz / ln
        eye = self.camera.eye
        if abs(diry) < 1e-6:
            return
        t = -eye.y / diry
        if t <= 0.0:
            return
        b = self.flock.bounds
        pos = Vec3(max(-b, min(b, eye.x + dirx * t)),
                   max(-b, min(b, eye.y + diry * t)),
                   max(-b, min(b, eye.z + dirz * t)))
        self.pred_manual = pos
        self.pred_on = True

    # -------------------------------------------------------------- HUD --
    def status_text(self):
        avg = self.flock.mean_speed()
        pred = ('ON' if self.pred_on else 'off')
        return (f'{PRESET_LABELS[self.preset]}  boids={self.count}  '
                f'avgSpeed={avg:5.2f}  predator={pred}  '
                f'color={self.color_mode}  '
                f't={self.time:5.1f}s  {self.fps:5.1f} fps'
                f'{"  [PAUSED]" if self.paused else ""}')


if __name__ == '__main__':
    run(Boidward)
