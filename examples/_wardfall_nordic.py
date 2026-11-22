# -*- coding: utf-8 -*-
"""《城邦 / Wardfall》的诺德组件库（私有模块，仿 examples/_harness.py 约定）。

只放**不依赖 GL 上下文**的纯几何/状态函数：地形、松林、长屋、龙语墙、
城堡、广场、城门楼、龙 mesh 与绕城位姿、飘雪、OST。主文件 wardfall.py
在 `setup()` 离线调用这些函数，把静态件并入同一个 MeshBuilder（保持全城
单 draw call），把龙/雪各自建成独立 vertex_list。

与 nordic_saga.py 的关系：龙 mesh 直接移植自 ``world3d.logic.nordic_scene``
的 ``_dragon_*_mesh``（只依赖 ``world3d.engine.geometry`` 的 tube/box/lathe/
merge_meshes，这些返回**已非索引**的 MeshData，可直接喂 vertex_list）；
龙的位姿沿**绕城圆形轨迹**自定义，扑翼/淡入复用 ``world3d.logic.nordic_saga``
的纯函数。
"""

import math
import random

from world3d.engine.geometry import (MeshData, box, sphere, tube, lathe,
                                     merge_meshes)

_TAU = 2.0 * math.pi


# =============================================================== 地形 ===
def _smoothstep(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


def terrain_height(x, z):
    """三级台地（白漫城式：内城最高、外城最低）+ 低幅正弦起伏 + 中心压平。

    半径阈值按 wardfall 把城墙半径映射到世界 ~70 来定：内城 plateau < r1，
    中城 ring < r2，之外是郊区/松林（更低、并向外缓坡）。建筑/路面/城墙/树
    的 base 都采样它，于是全城依山而建、城墙相隔的层次自然出现。
    """
    r = math.hypot(x, z)
    H_OUT, H_MID, H_IN = -0.8, 0.6, 2.2
    r1, r2 = 22.0, 48.0
    h = H_OUT
    h = max(h, H_OUT + (H_MID - H_OUT) * _smoothstep(r2, r2 - 8.0, r))
    h = max(h, H_OUT + (H_IN - H_OUT) * _smoothstep(r1, r1 - 8.0, r))
    flat = _smoothstep(12.0, 0.0, r)                 # 中心 12m 压平
    ripple = (math.sin(x * 0.12) * math.sin(z * 0.10) * 0.18
              + math.sin(x * 0.27 + z * 0.20) * 0.08) * (1.0 - flat)
    outer = max(0.0, r - 70.0) * 0.012               # 远郊缓坡，避免地平线穿出
    return h + ripple + outer * 0.4


def _terrain_normal(x, z, e=1.5):
    hx = terrain_height(x + e, z) - terrain_height(x - e, z)
    hz = terrain_height(x, z + e) - terrain_height(x, z - e)
    n = (-hx, 2.0 * e, -hz)
    l = math.hypot(*n) or 1.0
    return (n[0] / l, n[1] / l, n[2] / l)


def _face_normal(a, b, c):
    e1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    e2 = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    n = (e1[1] * e2[2] - e1[2] * e2[1],
         e1[2] * e2[0] - e1[0] * e2[2],
         e1[0] * e2[1] - e1[1] * e2[0])
    l = math.hypot(*n) or 1.0
    return (n[0] / l, n[1] / l, n[2] / l)


def build_terrain(mb, R_world=900.0, step=12.0):
    """细分地形网格（kind=0），逐顶点采样 terrain_height。并入主 mb。"""
    n = int(R_world / step)
    col = (0.26, 0.30, 0.19, 1.0)
    uv_t = step / 2.4
    for i in range(-n, n):
        for j in range(-n, n):
            x0, z0 = i * step, j * step
            x1, z1 = x0 + step, z0 + step
            y00 = terrain_height(x0, z0)
            y10 = terrain_height(x1, z0)
            y11 = terrain_height(x1, z1)
            y01 = terrain_height(x0, z1)
            cx, cz = (x0 + x1) * 0.5, (z0 + z1) * 0.5
            nrm = _terrain_normal(cx, cz)
            mb.quad((x0, y00, z0), (x1, y10, z0), (x1, y11, z1), (x0, y01, z1),
                    nrm, col, ((0, 0), (uv_t, 0), (uv_t, uv_t), (0, uv_t)), 0.0)


# --- 沿地形铺路的薄板：折线按 2m 细分，每段 base 采样地形 + 固定抬升 ---
def _segment_box(mb, a, b, thickness, y0, y1, col, kind):
    dx, dz = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dz)
    if L < 1e-6:
        return
    nx, nz = -dz / L * thickness * 0.5, dx / L * thickness * 0.5
    poly = [(a[0] + nx, a[1] + nz), (b[0] + nx, b[1] + nz),
            (b[0] - nx, b[1] - nz), (a[0] - nx, a[1] - nz)]
    _extrude(mb, poly, y0, y1, col, col, kind, kind, uv_tile=3.0)


def segment_box_terrain(mb, a, b, thickness, col, kind, step=2.0):
    ax, az = a
    bx, bz = b
    dx, dz = bx - ax, bz - az
    L = math.hypot(dx, dz)
    if L < 1e-6:
        return
    segs = max(1, int(L / step))
    for i in range(segs):
        t0 = i / segs
        t1 = (i + 1) / segs
        p0 = (ax + dx * t0, az + dz * t0)
        p1 = (ax + dx * t1, az + dz * t1)
        base = (terrain_height(p0[0], p0[1]) + terrain_height(p1[0], p1[1])) * 0.5 + 0.06
        _segment_box(mb, p0, p1, thickness, base, base + 0.14, col, kind)


# ============================================================ 基础挤出 ===
def _extrude(mb, poly, y0, y1, col_top, col_side, kind_top, kind_side,
             uv_tile=2.4):
    """从 wardfall 搬来的耳切挤出（顶 + 侧，无底），poly 为 CCW (x,z)。"""
    pts = _ensure_ccw(poly)
    n = len(pts)
    for a, b, c in _triangulate(pts):
        mb.tri((a[0], y1, a[1]), (b[0], y1, b[1]), (c[0], y1, c[1]),
               (0.0, 1.0, 0.0), col_top,
               ((a[0] / uv_tile, a[1] / uv_tile),
                (b[0] / uv_tile, b[1] / uv_tile),
                (c[0] / uv_tile, c[1] / uv_tile)), kind_top)
    for i in range(n):
        ax, az = pts[i]
        bx, bz = pts[(i + 1) % n]
        dx, dz = bx - ax, bz - az
        L = math.hypot(dx, dz)
        if L < 1e-6:
            continue
        nx, nz = dz / L, -dx / L
        u1 = L / uv_tile
        v01 = (y1 - y0) / uv_tile
        mb.quad((ax, y0, az), (bx, y0, bz), (bx, y1, bz), (ax, y1, az),
                (nx, 0.0, nz), col_side,
                ((0.0, 0.0), (u1, 0.0), (u1, v01), (0.0, v01)), kind_side)


def _ensure_ccw(poly):
    s = 0.0
    n = len(poly)
    for i in range(n):
        x0, z0 = poly[i]
        x1, z1 = poly[(i + 1) % n]
        s += x0 * z1 - x1 * z0
    return list(poly) if s >= 0 else list(reversed(poly))


def _signed_area(poly):
    s = 0.0
    n = len(poly)
    for i in range(n):
        x0, z0 = poly[i]
        x1, z1 = poly[(i + 1) % n]
        s += x0 * z1 - x1 * z0
    return s * 0.5


def _triangulate(poly):
    pts = _ensure_ccw(poly)
    idx = list(range(len(pts)))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 100000:
        guard += 1
        clipped = False
        for i in range(len(idx)):
            a = idx[(i - 1) % len(idx)]
            b = idx[i]
            c = idx[(i + 1) % len(idx)]
            ax, az = pts[a]
            bx, bz = pts[b]
            cx, cz = pts[c]
            cross = (bx - ax) * (cz - az) - (bz - az) * (cx - ax)
            if cross <= 1e-9:
                continue
            inside = False
            for j in idx:
                if j in (a, b, c):
                    continue
                px, pz = pts[j]
                w0 = (bx - ax) * (pz - az) - (bz - az) * (px - ax)
                w1 = (cx - bx) * (pz - bz) - (cz - bz) * (px - bx)
                w2 = (ax - cx) * (pz - cz) - (az - cz) * (px - cx)
                if w0 > 1e-9 and w1 > 1e-9 and w2 > 1e-9:
                    inside = True
                    break
            if not inside:
                tris.append((a, b, c))
                idx.pop(i)
                clipped = True
                break
        if not clipped:
            break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    elif len(idx) > 3:
        for i in range(1, len(idx) - 1):
            tris.append((idx[0], idx[i], idx[i + 1]))
    return [(pts[a], pts[b], pts[c]) for a, b, c in tris]


def _extrude_rect(mb, cx, cz, yaw, hx, hz, y0, y1, col, kind):
    c, s = math.cos(yaw), math.sin(yaw)
    corners = ((-hx, -hz), (hx, -hz), (hx, hz), (-hx, hz))
    poly = [(cx + lx * c - lz * s, cz + lx * s + lz * c)
            for lx, lz in corners]
    _extrude(mb, poly, y0, y1, col, col, kind, kind, uv_tile=2.4)


def _grid_surface(mb, nu, nv, fn, col, kind):
    pts = [fn(i, j) for i in range(nu + 1) for j in range(nv + 1)]
    for i in range(nu):
        for j in range(nv):
            a = pts[i * (nv + 1) + j]
            b = pts[(i + 1) * (nv + 1) + j]
            c = pts[(i + 1) * (nv + 1) + (j + 1)]
            d = pts[i * (nv + 1) + (j + 1)]
            nrm = _face_normal(a, b, c)
            mb.quad(a, b, c, d, nrm, col,
                    ((0, 0), (1, 0), (1, 1), (0, 1)), kind)


def _cone(mb, cx, cz, y0, r, h, col, kind, segs=7):
    ring = [(cx + r * math.cos(i / segs * _TAU),
             cz + r * math.sin(i / segs * _TAU)) for i in range(segs)]
    apex = (cx, y0 + h, cz)
    for i in range(segs):
        a = ring[i]
        b = ring[(i + 1) % segs]
        nrm = _face_normal((a[0], y0, a[1]), (b[0], y0, b[1]), apex)
        mb.tri((a[0], y0, a[1]), (b[0], y0, b[1]), apex, nrm, col,
               ((0, 0), (0, 0), (0, 0)), kind)


# ============================================================ 松林 ===
def build_pine_forest(mb, seed, inner=78.0, outer=210.0, count=240):
    """城外松林（诺德省遍布松林）。kind=7 干 / kind=8 针叶。返回树数。"""
    rng = random.Random(seed * 131 + 7)
    TRUNK = (0.34, 0.22, 0.13, 1.0)
    NEEDLE = (0.09, 0.24, 0.12, 1.0)
    n = 0
    tries = 0
    while n < count and tries < count * 4:
        tries += 1
        ang = rng.random() * _TAU
        rad = inner + rng.random() * (outer - inner)
        x = math.cos(ang) * rad
        z = math.sin(ang) * rad
        if rad < inner:
            continue
        base = terrain_height(x, z)
        th = rng.uniform(2.0, 4.5)
        _extrude_rect(mb, x, z, 0.0, 0.35, 0.35, base, base + th, TRUNK, 7.0)
        y0 = base + th
        for k in range(3):
            rr = 2.2 - k * 0.5
            hh = 2.6 - k * 0.4
            _cone(mb, x, z, y0 + k * 1.4, rr, hh, NEEDLE, 8.0)
        n += 1
    return n


# ========================================================== 诺德长屋 ===
WOOD = (0.42, 0.28, 0.17, 1.0)
ROOF_C = (0.50, 0.34, 0.22, 1.0)
HEARTH = (1.0, 0.55, 0.22, 1.0)


def build_longhouse(mb, x, z, yaw, length=15.0, width=7.0, wall_h=3.2):
    """船形屋顶长屋（Jorrvaskr 的倒扣战船屋顶）+ 中央火塘。

    屋顶用半圆柱壳，但其"拱高"沿屋长变化：两端翘起、中间平缓 —— 于是长屋
    看起来像一艘扣在地上的维京战船。返回火塘世界坐标（供点光用）。
    """
    base = terrain_height(x, z)
    _extrude_rect(mb, x, z, yaw, length / 2, width / 2, base, base + wall_h,
                  WOOD, 7.0)
    c, s = math.cos(yaw), math.sin(yaw)
    nu, nv = 16, 6

    def fn(i, j):
        u = i / nu
        v = j / nv
        lx = (u - 0.5) * length
        lz = (v - 0.5) * width
        cross = math.sqrt(max(0.0, 1.0 - (2.0 * v - 1.0) ** 2))
        roof_h = 1.6 + 1.9 * (0.5 - 0.5 * math.cos(2.0 * math.pi * u))
        r = roof_h * cross
        y = base + wall_h + r
        wx = x + lx * c - lz * s
        wz = z + lx * s + lz * c
        return (wx, y, wz)

    _grid_surface(mb, nu, nv, fn, ROOF_C, 4.0)
    # 火塘（kind=14 发光块）
    _extrude_rect(mb, x, z, yaw, 0.6, 0.6, base + 0.2, base + 1.4, HEARTH, 14.0)
    return (x, base + 1.0, z)


# ========================================================== 龙语墙 ===
RUNE = (0.45, 0.50, 0.62, 1.0)


def build_rune_wall(mb, x, z, yaw, length=16.0, height=5.0):
    """刻满龙语的古墙（kind=11，emissive 由 u_rune 驱动）。返回墙心坐标。"""
    base = terrain_height(x, z)
    _extrude_rect(mb, x, z, yaw, length / 2, 0.6, base, base + height, RUNE, 11.0)
    return (x, base + height * 0.5, z)


# ============================================================ 城堡 ===
STONE = (0.60, 0.58, 0.54, 1.0)


def build_citadel(mb, x, z, h=14.0):
    """云顶区主堡：中央高石塔 + 四角塔楼（kind=9）。返回中心坐标。"""
    base = terrain_height(x, z)
    _extrude_rect(mb, x, z, 0.4, 5.0, 5.0, base, base + h, STONE, 9.0)
    for ang in (0.0, 90.0, 180.0, 270.0):
        a = math.radians(ang)
        tx = x + math.cos(a) * 5.5
        tz = z + math.sin(a) * 5.5
        tb = terrain_height(tx, tz)
        _extrude_rect(mb, tx, tz, 0.0, 1.6, 1.6, tb, tb + h + 3.0, STONE, 9.0)
    return (x, base + h * 0.6, z)


# ============================================================ 广场 ===
PLAZA_C = (0.55, 0.52, 0.47, 1.0)


def build_plaza(mb, x, z, R=10.0):
    """中央广场石板（kind=10）+ 四座市集棚。返回中心坐标。"""
    base = terrain_height(x, z)
    _extrude_rect(mb, x, z, 0.0, R, R, base + 0.02, base + 0.18, PLAZA_C, 10.0)
    for k in range(4):
        a = math.pi * 0.5 * k + 0.4
        sx = x + math.cos(a) * (R * 0.6)
        sz = z + math.sin(a) * (R * 0.6)
        _extrude_rect(mb, sx, sz, a, 1.6, 1.6, base + 0.18, base + 2.4, WOOD, 7.0)
        _extrude_rect(mb, sx, sz, a, 1.9, 1.9, base + 2.4, base + 2.9, ROOF_C, 4.0)
    return (x, base + 0.1, z)


# ========================================================== 城门楼 ===
GATE_C = (0.58, 0.56, 0.52, 1.0)


def build_gatehouses(mb, gate_pts):
    """遍历 wall['gates']，每座建双塔 + 门楣（kind=12）。返回门数。"""
    n = 0
    for (gx, gz) in gate_pts:
        base = terrain_height(gx, gz)
        for off in (-3.0, 3.0):
            tx = gx + off
            tz = gz
            tb = terrain_height(tx, tz)
            _extrude_rect(mb, tx, tz, 0.0, 1.4, 1.4, tb, tb + 7.0, GATE_C, 12.0)
        _extrude_rect(mb, gx, gz, 0.0, 3.4, 1.2, base + 5.0, base + 6.2,
                      GATE_C, 12.0)
        n += 1
    return n


# ============================================================== 龙 ===
def _rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def _rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]


def _trans(x, y, z):
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]


def _mul(a, b):
    out = [0.0] * 16
    for c in range(4):
        for r in range(4):
            out[c * 4 + r] = (a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1]
                              + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3])
    return out


def _uv_scale(md, su, sv):
    uvs = md.uvs
    for i in range(0, len(uvs), 2):
        uvs[i] *= su
        uvs[i + 1] *= sv
    return md


def _place(md, x, y, z, rz=0.0, rx=0.0):
    if rx:
        md.transform(_rot_x(rx))
    if rz:
        md.transform(_rot_z(rz))
    md.transform(_trans(x, y, z))
    return md


_DRAGON_SPINE = (
    (0.00, 0.06, 2.78), (0.00, 0.30, 2.40), (0.03, 0.66, 1.82),
    (0.05, 0.84, 1.14), (0.04, 0.78, 0.42), (0.00, 0.66, -0.36),
    (-0.04, 0.74, -1.10), (-0.02, 0.98, -1.86), (0.03, 1.30, -2.60),
    (0.06, 1.62, -3.30), (0.04, 1.88, -3.96), (0.00, 2.02, -4.55),
)
_DRAGON_RADII = (0.085, 0.145, 0.20, 0.25, 0.33, 0.39, 0.37, 0.31,
                 0.245, 0.175, 0.105, 0.035)


def dragon_body_mesh():
    """低多边形飞龙躯体（局部 +z 朝前）：管扫掠躯干 + 箱式头颚 + 后掠双角。"""
    body = tube(list(_DRAGON_SPINE), list(_DRAGON_RADII), radial=10,
                caps=True, name='dragon_body')
    _uv_scale(body, 3.0, 2.0)
    parts = [body]
    skull = box(0.46, 0.40, 0.56)
    _place(skull, 0.0, 0.30, 2.78)
    parts.append(skull)
    snout = box(0.30, 0.235, 0.68)
    _place(snout, 0.0, 0.235, 3.22)
    parts.append(snout)
    jaw = box(0.28, 0.115, 0.56)
    _place(jaw, 0.0, 0.075, 3.10)
    parts.append(jaw)
    for sx in (-1.0, 1.0):
        horn = lathe([(0.052, 0.0), (0.070, 0.14), (0.028, 0.62),
                      (0.001, 0.86)], segments=6)
        horn.transform(_rot_x(118.0))
        _place(horn, sx * 0.155, 0.50, 2.62)
        parts.append(horn)
    return merge_meshes(parts)


def dragon_wing_mesh(side=1.0):
    """单侧翼（枢轴在原点，+x 展开）。side=-1 时镜像到 -x。"""
    p = ((0.00, 0.00, 0.00), (2.55, 1.30, 0.60), (2.95, 0.80, -0.12),
         (2.25, 0.30, -1.00), (1.18, 0.04, -1.30), (0.25, -0.03, -0.74))
    tris = ((0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5))
    md = MeshData([], [], [])
    for t in tris:
        for q in (t, tuple(reversed(t))):
            for idx in q:
                vx, vy, vz = p[idx]
                md.positions += [vx * side, vy, vz * side]
                md.uvs += [vx * 0.33 * side, vz * 0.33]
    md.compute_normals()
    bone = tube([(0.0, 0.0, 0.0), (1.25, 0.62, 0.30), (2.55, 1.30, 0.60)],
                0.075, 6)
    return merge_meshes([md, bone])


def dragon_eye_mesh():
    return sphere(0.07, 8, 6)


def dragon_fire_mesh():
    md = lathe([(0.0, 0.0), (0.34, 0.22), (0.27, 1.15), (0.12, 2.45),
                (0.001, 3.30)], segments=10)
    md.transform(_rot_x(-90.0))
    return md


def dragon_city_pose(t):
    """龙绕城圆形轨迹的位姿。朝向/扑翼复用 nordic_saga 纯函数，淡入自定。

    返回 (pos(x,y,z), yaw(rad), wing(deg), alpha)。

    注意：nordic_saga 的 ``dragon_alpha`` 是为 60s 电影机位设计的（开场 21s
    龙都隐在远空），Wardfall 的龙是常驻 headline，故这里改成开场 2.5s 淡入后
    常显，避免用户等半天看不见龙。
    """
    from world3d.logic import nordic_saga as NS
    R = 64.0
    omega = 2.0 * math.pi / 42.0
    a = t * omega
    hy = 24.0 + 3.0 * math.sin(t * 0.13)
    x = math.cos(a) * R
    z = math.sin(a) * R
    yaw = -a                                   # +z 朝切线方向
    alpha = min(1.0, t / 2.5)                  # 开场短淡入，之后常显
    return ((x, hy, z), yaw, NS.wing_angle(t), alpha)


# ============================================================== 飘雪 ===
def _hash01(*args):
    s = math.sin(sum(a * (61.7 + 29.3 * k) for k, a in enumerate(args, 1))) \
        * 43758.5453
    return s - math.floor(s)


class SnowField:
    """CPU 更新的飘雪（billboard 菱形片，position[:] 直写缓冲）。

    端口自 world3d.logic.nordic_scene.Snowfall，但改成**跟随相机**：雪场以
    ``half``(hx,hy,hz) 为半幅，flake 存的是相对中心的局部坐标，每帧把中心挪到
    相机 XZ 上，于是镜头无论怎么拉远推近，雪都落在视野里。自带的极简雪 shader
    只吃 position（外加 u_alpha 控深浅）。
    """

    def __init__(self, program, count=1500, half=(160.0, 75.0, 160.0),
                 size=0.55, speed=7.0, seed=12345):
        self.program = program
        self.half = half
        self.size = size
        self.t = 0.0
        hx, hy, hz = half
        self.flakes = []
        for i in range(count):
            self.flakes.append([
                (2.0 * _hash01(i, 1, seed) - 1.0) * hx,
                _hash01(i, 2, seed + 5) * hy,
                (2.0 * _hash01(i, 3, seed + 9) - 1.0) * hz,
                speed * (0.7 + 0.5 * _hash01(i, 4, seed + 13)),
                _TAU * _hash01(i, 5, seed + 17)])
        n = count * 6
        self._flat = [0.0] * (n * 3)
        self.vlist = program.vertex_list(n, 0x0004, position=('f', self._flat))
        self.count = count
        self.cx = 0.0
        self.cz = 0.0

    def update(self, dt, cam_eye, cam_fwd):
        self.t += dt
        t = self.t
        self.cx, self.cz = cam_eye.x, cam_eye.z      # 雪场中心跟随相机 XZ
        fx, fy, fz = cam_fwd.x, cam_fwd.y, cam_fwd.z
        ln = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
        fx, fy, fz = fx / ln, fy / ln, fz / ln
        if abs(fy) > 0.94:
            fx, fy, fz = 0.0, 0.0, -1.0
        rx, ry, rz = -fz, 0.0, fx
        rl = math.sqrt(rx * rx + rz * rz) or 1.0
        rx, rz = rx / rl, rz / rl
        ux = ry * fz - rz * fy
        uy = rz * fx - rx * fz
        uz = rx * fy - ry * fx
        ul = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
        ux, uy, uz = ux / ul, uy / ul, uz / ul
        hx, hy, hz = self.half
        span_y = hy
        s = self.size
        flat = self._flat
        o = 0
        for fl in self.flakes:
            cx = self.cx + fl[0] + math.sin(_TAU * t / 15.0 + fl[4]) * 1.35
            cy = (fl[1] - fl[3] * t) % span_y
            cz = self.cz + fl[2] + math.sin(_TAU * t / 20.0 + fl[4] * 1.7) * 0.9
            ax, ay, az = cx + (rx + ux) * s, cy + uy * s, cz + (rz + uz) * s
            bx, by, bz = cx + (ux - rx) * s, cy + uy * s, cz + (uz - rz) * s
            ex, ey, ez = cx - (rx + ux) * s, cy - uy * s, cz - (rz + uz) * s
            dx, dy, dz = cx + (rx - ux) * s, cy - uy * s, cz + (rz - uz) * s
            for vx, vy, vz in ((ax, ay, az), (bx, by, bz), (ex, ey, ez),
                              (ax, ay, az), (ex, ey, ez), (dx, dy, dz)):
                flat[o] = vx
                flat[o + 1] = vy
                flat[o + 2] = vz
                o += 3
        self.vlist.position[:] = flat


# ============================================================== 音频 ===
def nordic_ost(duration=60.0):
    """合成北欧 OST（离线渲染一次后走缓存）。失败不阻断视觉。"""
    try:
        from world3d.logic.nordic_audio import ensure_ost
        from world3d.engine.audio import TimelineAudio
        wav = ensure_ost(duration=duration)
        return TimelineAudio(wav, loop=True, volume=0.8)
    except Exception as e:                       # noqa: BLE001
        print(f'[audio] nordic_ost init failed: {e}')
        return None


def wardfall_ost(duration=60.0):
    """Wardfall 多段情绪配乐：冬境(ambient) → 巨龙(eerie) → 城邦(band) →
    盛夏(ambient)。直接复用融合系统的情绪段（fusion_audio.build_mood_ost），
    失败则回退到 nordic_ost。"""
    try:
        from world3d.logic.fusion_audio import ensure_wardfall_ost
        from world3d.engine.audio import TimelineAudio
        wav = ensure_wardfall_ost(duration=duration)
        return TimelineAudio(wav, loop=True, volume=0.8)
    except Exception as e:                       # noqa: BLE001
        print(f'[audio] wardfall_ost failed, fallback nordic_ost: {e}')
        return nordic_ost(duration)
