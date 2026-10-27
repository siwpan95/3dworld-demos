# -*- coding: utf-8 -*-
"""《渊境 / Brogueward》—— 用 3dworld 引擎复刻 Brogue 的地牢爬行。

Brogue（Brian Walker, 2009；Brogue: Community Edition, AGPLv3）被广泛认为
是最接近原版 Rogue 的现代传统类 Rogue：设计简单、玩法复杂——容易学习，
难以精通。本 demo 把它的核心循环搬进 3D 方块地牢。

借鉴记账（铁律 10）
------------------
- **机制参考**（非代码移植）：Brogue: Community Edition 源码
  <https://github.com/broguece/BrogueCE>（AGPLv3）与官方 wiki
  <https://brogue.wiki/wiki/Quickstart>。借的是**玩法机制**：26 层取
  Amulet 回地表、物品驱动成长（无经验等级）、走进即交互、FOV 记忆、
  火把光照衰减、饥饿、陷阱/毒气/火焰的环境战术、回合制自动战斗。
- **未复制**任何 C 源代码、数据表或美术资源；怪物/物品数值为本作
  独立调整（梯度仿照原版手感，数值全部重定）。3D 化的视觉表达
  （方块地牢/几何怪物）是本项目原创演绎。
- 因此本文件以 MIT（随演示仓）发布是安全的：AGPL 管代码衍生物，
  不管玩法机制本身。

引擎消费
--------
- ``logic.dungeon``：地牢骨架（随机房间 + L 走廊）；
- ``logic.fov``（本作新增反哺引擎）：视野投射 + 光照衰减；
- ``core.pathing_core``：怪物追击 A*；
- ``engine.audio.SfxPlayer``：即发音效（WAV 由本文件 numpy 合成并缓存）；
- 渲染为自建 unlit 顶点色管线（地形每格变色的 FOV 光照无法走引擎
  Mesh 的 Material uniform，见 TerrainMesh 注释）。

玩法范围（26 层完整闭环）
    10 种怪物 / 23 种物品 / 3 种陷阱 / 火·毒气·水·草 / 饥饿 /
    门（挡视野、bump 开启、可烧毁）与机关宝藏室 vault（封门+守卫+稀有战利品）/
    魅惑药剂 + 魅惑盟友 AI（友军自动接敌）/ 武器攻击模式（矛 reach 突刺 /
    战斧 sweep 横扫）/ 法杖（充能制、自动索敌：火焰/闪电链/迟缓）+
    戒指（被动：再生/潜行/防护）/ 录像回放（--flag record 记账，
    --flag replay=PATH 确定性复演并对账终局）。
"""
from __future__ import annotations

import math
import os
import random
import sys
import wave

import numpy as np
import pyglet

from world3d.core.pathing_core import PathingService
from world3d.logic.dungeon import generate_dungeon
from world3d.platform.app import DemoApp, run
from pyglet import gl, shapes, text as pgtext
from pyglet.math import Mat4, Vec3

# FOV 走引擎件（logic.fov，2026-10-27 新增反哺）；0.5.0 wheel 尚未携带，
# 缺则用本文件内置的同款 shadowcasting（引擎 0.6.0 发版后自动切换）。
try:
    from world3d.logic.fov import (compute_fov, light_level,
                                   visible_through)
    _FOV_FROM_ENGINE = True
except ImportError:                                 # pragma: no cover
    _FOV_FROM_ENGINE = False

    def visible_through(opaque, x0, y0, x1, y1, max_dist=64):
        """Bresenham 逐格视线判定（与引擎 logic.fov 同款语义）。"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if dx + dy > max_dist:
            return False
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        err = dx - dy
        x, y = x0, y0
        while (x, y) != (x1, y1):
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            if (x, y) != (x1, y1) and opaque(x, y):
                return False
        return True

    def _cast_light(visible, opaque, cx, cy, row, start, end, radius_sq,
                    xx, xy, yx, yy):
        if start < end:
            return
        radius = int(radius_sq ** 0.5)
        for j in range(row, radius + 1):
            dx, dy = -j - 1, -j
            blocked = False
            new_start = 0.0
            while dx <= 0:
                dx += 1
                x = cx + dx * xx + dy * xy
                y = cy + dx * yx + dy * yy
                l_slope = (dx - 0.5) / (dy + 0.5)
                r_slope = (dx + 0.5) / (dy - 0.5)
                if start < r_slope:
                    continue
                if end > l_slope:
                    break
                sq = dx * dx + dy * dy
                if sq <= radius_sq:
                    visible[(x, y)] = sq
                if blocked:
                    if not opaque(x, y):
                        blocked = False
                        start = new_start
                        continue
                    new_start = r_slope
                    continue
                if opaque(x, y) and j < radius:
                    blocked = True
                    _cast_light(visible, opaque, cx, cy, j + 1, start,
                                l_slope, radius_sq, xx, xy, yx, yy)
                    new_start = r_slope
            if blocked:
                break

    _MULT = (
        (1, 0, 0, -1, -1, 0, 0, 1),
        (0, 1, -1, 0, 0, -1, 1, 0),
        (0, 1, 1, 0, 0, -1, -1, 0),
        (1, 0, 0, 1, -1, 0, 0, -1),
    )

    def compute_fov(opaque, origin, radius):
        ox, oy = origin
        radius_sq = radius * radius
        visible = {(ox, oy): 0}
        for octant in range(8):
            _cast_light(visible, opaque, ox, oy, 1, 1.0, 0.0, radius_sq,
                        _MULT[0][octant], _MULT[1][octant],
                        _MULT[2][octant], _MULT[3][octant])
        return visible

    def light_level(dist, radius):
        if dist >= radius:
            return 0.0
        v = 1.0 - (dist / radius) ** 2
        return v ** 1.5 if v > 0.0 else 0.0

# =========================================================== 常量与色板 ==

MAP_W, MAP_H = 79, 29          # Brogue 地图口径（DCOLS x DROWS）
DEPTH_MAX = 26
FOV_RADIUS = 7.5               # 火把可视半径
TILE = 1.0                     # 世界坐标下每格边长
DEMO_STEP = 0.38               # 演示模式每步间隔（秒）

# 地形 ID
T_WALL, T_FLOOR, T_WATER, T_GRASS = 0, 1, 2, 3
T_STAIRS_DOWN, T_STAIRS_UP = 4, 5
T_RUBBLE = 6                   # 纯装饰
T_DOOR_C, T_DOOR_O = 7, 8      # 关门（挡视野/bump 开/可烧）/ 开门

# 陷阱 ID
TRAP_NONE, TRAP_DART, TRAP_GAS, TRAP_PIT = 0, 1, 2, 3

# 楼层色调：浅层冷石蓝 → 深层暗红（Brogue「越深越暗越暖」的口味）
_TINT_A = (1.05, 1.00, 0.90)   # depth 1（暖石色）
_TINT_B = (1.10, 0.86, 0.78)   # depth 26


def depth_tint(depth):
    """楼层色调插值：返回 (r, g, b) 乘子。"""
    t = min(1.0, (depth - 1) / max(1.0, DEPTH_MAX - 1))
    return tuple(a + (b - a) * t for a, b in zip(_TINT_A, _TINT_B))


# 基础地形色（会被 depth_tint 与光照再乘）
# 墙顶刻意压暗：读图层级「暗紫=墙 / 亮=地板 / 黑=未知」一目了然
TERRAIN_COLORS = {
    T_WALL: (0.155, 0.15, 0.175),
    T_FLOOR: (0.58, 0.555, 0.51),
    T_WATER: (0.16, 0.34, 0.62),
    T_GRASS: (0.24, 0.46, 0.22),
    T_STAIRS_DOWN: (0.30, 0.85, 0.95),
    T_STAIRS_UP: (0.85, 0.85, 0.30),
    T_RUBBLE: (0.42, 0.40, 0.38),
    T_DOOR_C: (0.55, 0.36, 0.18),
    T_DOOR_O: (0.40, 0.28, 0.15),
}

# ============================================================= 数据表 ==
# 怪物表（数值为本作独立调整；梯度仿 Brogue 手感）
#   name, shape, color, hp, defense, accuracy, dmg(lo,hi), speed(%),
#   depth(lo,hi), regen(回合/1hp, 0=不回复), ranged(火息等, None)
MONSTERS = [
    dict(name='rat',     shape='rat',     color=(0.55, 0.55, 0.58), hp=6,  df=0,  ac=4,  dmg=(1, 3),   spd=100, dr=(1, 3),  regen=0,   ranged=None),
    dict(name='kobold',  shape='kobold',  color=(0.42, 0.62, 0.40), hp=8,  df=1,  ac=5,  dmg=(1, 4),   spd=100, dr=(1, 4),  regen=0,   ranged=None),
    dict(name='jackal',  shape='rat',     color=(0.70, 0.55, 0.30), hp=7,  df=1,  ac=6,  dmg=(2, 4),   spd=150, dr=(2, 6),  regen=0,   ranged=None),
    dict(name='goblin',  shape='goblin',  color=(0.45, 0.66, 0.38), hp=11, df=2,  ac=6,  dmg=(2, 6),   spd=100, dr=(3, 8),  regen=0,   ranged=None),
    dict(name='ogre',    shape='ogre',    color=(0.72, 0.50, 0.36), hp=26, df=3,  ac=10, dmg=(6, 12),  spd=50,  dr=(7, 14), regen=0,   ranged=None),
    dict(name='wraith',  shape='wraith',  color=(0.55, 0.55, 0.80), hp=15, df=6,  ac=12, dmg=(4, 9),   spd=100, dr=(9, 18), regen=0,   ranged=None),
    dict(name='troll',   shape='ogre',    color=(0.35, 0.58, 0.42), hp=34, df=5,  ac=11, dmg=(8, 14),  spd=100, dr=(11, 20), regen=8,  ranged=None),
    dict(name='vampire', shape='goblin',  color=(0.60, 0.20, 0.28), hp=42, df=8,  ac=14, dmg=(10, 18), spd=100, dr=(14, 23), regen=0,  ranged=None),
    dict(name='dragon',  shape='dragon',  color=(0.75, 0.30, 0.16), hp=62, df=10, ac=16, dmg=(12, 22), spd=100, dr=(18, 26), regen=10, ranged=('fire', 5, 6, 12)),
    dict(name='warden',  shape='wraith',  color=(0.85, 0.78, 0.35), hp=85, df=12, ac=18, dmg=(14, 22), spd=100, dr=(20, 26), regen=0,  ranged=('drain', 6, 6, 10)),
]

# 物品表（同上：独立数值）
#   kind: food/potion/scroll/weapon/armor/gold/amulet
ITEMS = {
    'ration':    dict(kind='food',   name='food ration',      color=(0.78, 0.60, 0.30), food=900),
    'mango':     dict(kind='food',   name='mango',            color=(0.90, 0.55, 0.15), food=250, heal=3),
    'healing':   dict(kind='potion', name='healing potion',   color=(0.85, 0.20, 0.25), heal=25),
    'life':      dict(kind='potion', name='life potion',      color=(0.95, 0.30, 0.60), maxhp=10),
    'strength':  dict(kind='potion', name='strength potion',  color=(0.30, 0.45, 0.90), strength=1),
    'charm':     dict(kind='potion', name='charm potion',     color=(0.95, 0.45, 0.75), charm=1),
    'enchant':   dict(kind='scroll', name='scroll of enchantment', color=(0.90, 0.88, 0.70), enchant=1),
    'teleport':  dict(kind='scroll', name='scroll of teleport',    color=(0.70, 0.70, 0.90), tele=True),
    'dagger':    dict(kind='weapon', name='dagger',           color=(0.75, 0.75, 0.80), dmg=(1, 3), acc=3, req=12),
    'sword':     dict(kind='weapon', name='short sword',      color=(0.82, 0.82, 0.88), dmg=(2, 5), acc=4, req=14),
    'spear':     dict(kind='weapon', name='spear',            color=(0.65, 0.72, 0.55), dmg=(2, 6), acc=4, req=15, reach=2),
    'war axe':   dict(kind='weapon', name='war axe',          color=(0.70, 0.62, 0.55), dmg=(4, 9), acc=5, req=18, sweep=1),
    'fire staff':      dict(kind='staff', name='staff of fire',       color=(0.95, 0.45, 0.15), effect='fire', dmg=(8, 14),  srange=6),
    'lightning staff': dict(kind='staff', name='staff of lightning',  color=(0.45, 0.65, 0.95), effect='bolt', dmg=(10, 16), srange=7),
    'slowing staff':   dict(kind='staff', name='staff of slowing',    color=(0.60, 0.85, 0.60), effect='slow', dmg=(0, 0),   srange=6),
    'regen ring':      dict(kind='ring', name='ring of regeneration', color=(0.95, 0.55, 0.60), effect='regen'),
    'stealth ring':    dict(kind='ring', name='ring of stealth',      color=(0.55, 0.55, 0.70), effect='stealth'),
    'protection ring': dict(kind='ring', name='ring of protection',   color=(0.85, 0.85, 0.90), effect='protect'),
    'leather':   dict(kind='armor',  name='leather armor',    color=(0.60, 0.42, 0.28), df=1, req=13),
    'scale':     dict(kind='armor',  name='scale armor',      color=(0.52, 0.55, 0.60), df=3, req=16),
    'plate':     dict(kind='armor',  name='plate armor',      color=(0.68, 0.70, 0.75), df=5, req=20),
    'gold':      dict(kind='gold',   name='gold',             color=(0.95, 0.80, 0.20), amount=(15, 60)),
    'amulet':    dict(kind='amulet', name='Amulet of Yendor', color=(0.60, 0.95, 1.00)),
}
FLOOR_ITEMS = {                       # 深度 → 物品权重表
    'always': [('ration', 10), ('gold', 16), ('healing', 8), ('mango', 5)],
    'mid':    [('strength', 4), ('enchant', 6), ('dagger', 3), ('leather', 3),
               ('teleport', 3), ('spear', 2), ('charm', 3),
               ('fire staff', 2), ('slowing staff', 2),
               ('stealth ring', 1), ('protection ring', 1)],
    'deep':   [('sword', 3), ('scale', 3), ('life', 3), ('war axe', 2),
               ('plate', 2), ('lightning staff', 2), ('regen ring', 1)],
}

# vault 宝藏室稀有战利品池（封门 + 守卫的回报）
VAULT_LOOT = ['strength', 'life', 'enchant', 'plate', 'war axe', 'spear',
              'fire staff', 'lightning staff', 'slowing staff',
              'regen ring', 'stealth ring', 'protection ring']

TRAP_NAMES = {TRAP_DART: 'dart trap', TRAP_GAS: 'poison gas trap',
              TRAP_PIT: 'trap door'}


def monsters_for_depth(depth, rng):
    """按深度抽取一种怪物类型（浅层权重高浅怪，深层反之）。"""
    pool = []
    for i, m in enumerate(MONSTERS):
        lo, hi = m['dr']
        if not (lo <= depth <= hi):
            continue
        w = 6 - min(5, abs(depth - (lo + hi) // 2) // 3)
        pool += [i] * max(1, w)
    return MONSTERS[rng.choice(pool)] if pool else MONSTERS[0]


def floor_item_for_depth(depth, rng):
    w = list(FLOOR_ITEMS['always'])
    if depth >= 3:
        w += FLOOR_ITEMS['mid']
    if depth >= 9:
        w += FLOOR_ITEMS['deep']
    if depth == DEPTH_MAX:
        w += [('amulet', 40)]
    total = sum(x[1] for x in w)
    r = rng.randrange(total)
    for key, wt in w:
        r -= wt
        if r < 0:
            return key
    return 'gold'


# ============================================================ 几何工具 ==
def cube_tris(x, y, z, sx, sy, sz, top_bias=1.0):
    """单位立方体三角汤（36 顶点），中心在 (x,y,z)、半尺寸 (sx,sy,sz)。

    顶面亮度加权由调用方颜色处理；这里只给几何。
    """
    x0, x1 = x - sx, x + sx
    y0, y1 = y, y + sy * 2          # y0 是"落地"底面
    z0, z1 = z - sz, z + sz
    v = []
    for quad in (
        ((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)),   # 顶
        ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)),   # 前(+z)
        ((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)),   # 后(-z)
        ((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1)),   # 右(+x)
        ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)),   # 左(-x)
    ):
        a, b, c, d = quad
        v += [a, b, c, a, c, d]
    return v


def pyramid_tris(x, y, z, sx, sy, sz):
    """四棱锥（尖顶）：6 面 ×3 顶点 = 18 顶点。"""
    x0, x1 = x - sx, x + sx
    y0, y1 = y, y + sy * 2
    z0, z1 = z - sz, z + sz
    apex = (x, y1 + sy * 0.6, z)
    quads = [((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))]
    tris = [quads[0][0], quads[0][1], quads[0][2],
            quads[0][0], quads[0][2], quads[0][3]]
    base = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]
    tris += [base[0], base[1], apex] + [base[1], base[2], apex]
    tris += [base[2], base[3], apex] + [base[3], base[0], apex]
    return tris


def shade_quad_color(color, face_i):
    """面朝向亮度（伪光照）：顶面最亮、前后次之、左右最暗。"""
    k = (1.0, 0.82, 0.74, 0.68, 0.68)[face_i] if face_i < 5 else 1.0
    return (color[0] * k, color[1] * k, color[2] * k)


# ============================================================ 关卡生成 ==
class Level:
    """一层地牢：地形、物品、怪物、陷阱、气体与火焰场。"""

    def __init__(self, depth, run_seed):
        self.depth = depth
        self.rng = random.Random((run_seed * 7919 + depth * 104729) & 0x7fffffff)
        d = generate_dungeon(MAP_W, MAP_H, rooms=9, seed=self.rng.randrange(1 << 30))
        grid = d['grid']                      # 1=地板 0=墙
        rooms = d['rooms']
        self.terrain = [[T_WALL] * MAP_W for _ in range(MAP_H)]
        self.trap = [[TRAP_NONE] * MAP_W for _ in range(MAP_H)]
        self.gas = [[0.0] * MAP_W for _ in range(MAP_H)]      # 毒气浓度
        self.fire = [[0] * MAP_W for _ in range(MAP_H)]       # 剩余燃烧回合
        self.items = {}                       # (x, y) -> [item dict]
        self.monsters = []                    # Monster list
        self.explored = [[False] * MAP_W for _ in range(MAP_H)]
        self.turns = 0
        self.spawn_cooldown = 60

        # ---- 地形铺开 + 装饰 ----
        for y in range(MAP_H):
            for x in range(MAP_W):
                if grid[y][x]:
                    self.terrain[y][x] = T_FLOOR
        for room in rooms[1:]:
            if self.rng.random() < 0.55:
                self._patch(room, T_GRASS if self.rng.random() < 0.6 else T_WATER)
        for room in rooms:
            if self.rng.random() < 0.35:
                x = self.rng.randrange(room['x'], room['x'] + room['w'])
                y = self.rng.randrange(room['y'], room['y'] + room['h'])
                if self.terrain[y][x] == T_FLOOR:
                    self.terrain[y][x] = T_RUBBLE

        # ---- 楼梯与入口 ----
        first = rooms[0]
        self.up = (first['x'] + first['w'] // 2, first['y'] + first['h'] // 2)
        self.player_start = self.up
        # depth 1 的 up 楼梯 = 地表出口（拿 Amulet 后回这里胜利）
        self.terrain[self.up[1]][self.up[0]] = T_STAIRS_UP
        far = max(rooms[1:] or rooms[1:],
                  key=lambda r: (r['x'] - self.up[0]) ** 2 + (r['y'] - self.up[1]) ** 2)
        self.down = (far['x'] + far['w'] // 2, far['y'] + far['h'] // 2)
        if depth < DEPTH_MAX:
            self.terrain[self.down[1]][self.down[0]] = T_STAIRS_DOWN
        self.has_amulet = False

        # ---- 物品 ----
        n_items = self.rng.randint(3, 5)
        for _ in range(n_items):
            pos = self._rand_floor(min_dist=self.up, dist=6)
            if pos:
                key = floor_item_for_depth(depth, self.rng)
                it = dict(ITEMS[key])
                it['key'] = key
                if it['kind'] == 'gold':
                    it['amount'] = self.rng.randint(*it['amount'])
                if it['kind'] == 'staff':
                    it['charges'] = self.rng.randint(3, 6)
                self.items.setdefault(pos, []).append(it)

        # ---- 怪物 ----
        n_mon = min(3 + depth // 4, 7)
        for _ in range(n_mon):
            pos = self._rand_floor(min_dist=self.up, dist=10)
            if not pos:
                continue
            self.monsters.append(Monster.spawn(monsters_for_depth(depth, self.rng), pos))

        # ---- 陷阱 ----
        if depth >= 2:
            for _ in range(self.rng.randint(2, 4)):
                pos = self._rand_floor(min_dist=self.up, dist=4)
                if pos:
                    self.trap[pos[1]][pos[0]] = self.rng.choice(
                        (TRAP_DART, TRAP_GAS, TRAP_PIT))

        # ---- 门（房间咽喉点，挡视野、bump 开、可烧毁）----
        self._place_doors(rooms)
        # ---- 机关宝藏室 vault（封门 + 守卫 + 稀有战利品）----
        self.vault_cells = set()
        if depth >= 2:
            self._place_vault(rooms)

    def _door_candidates(self, room):
        """房间外圈与走廊相接的咽喉格（两侧垂直方向是墙）。"""
        out = []
        for x in range(room['x'] - 1, room['x'] + room['w'] + 1):
            for y in (room['y'] - 1, room['y'] + room['h']):
                if 0 < x < MAP_W - 1 and 0 < y < MAP_H - 1 \
                        and self.terrain[y][x] == T_FLOOR:
                    out.append((x, y))
        for y in range(room['y'] - 1, room['y'] + room['h'] + 1):
            for x in (room['x'] - 1, room['x'] + room['w']):
                if 0 < x < MAP_W - 1 and 0 < y < MAP_H - 1 \
                        and self.terrain[y][x] == T_FLOOR:
                    out.append((x, y))
        res = []
        for (x, y) in out:
            # 真·1 宽咽喉：一轴两侧是墙 + 另一轴两侧通透（才能挡住视线）
            ns = (not self.walkable(x, y - 1) and not self.walkable(x, y + 1)
                  and self.walkable(x - 1, y) and self.walkable(x + 1, y))
            ew = (not self.walkable(x - 1, y) and not self.walkable(x + 1, y)
                  and self.walkable(x, y - 1) and self.walkable(x, y + 1))
            if ns or ew:
                res.append((x, y))
        return res

    def _place_doors(self, rooms):
        """咽喉点概率放门（出生房不设，防开局堵死）。"""
        n = 0
        for ri, room in enumerate(rooms):
            if ri == 0 or n >= 6:
                continue
            for (x, y) in self._door_candidates(room):
                if self.terrain[y][x] == T_FLOOR and self.rng.random() < 0.5:
                    self.terrain[y][x] = T_DOOR_C
                    n += 1

    def _place_vault(self, rooms):
        """宝藏室：随机房间所有咽喉封门 + 两件稀有物 + 金 + 守卫。"""
        def has(r, c):
            return (r['x'] <= c[0] < r['x'] + r['w']
                    and r['y'] <= c[1] < r['y'] + r['h'])
        cands = [r for r in rooms[1:]
                 if not has(r, self.down) and not has(r, self.up)]
        if not cands:
            return
        room = self.rng.choice(cands)
        for (x, y) in self._door_candidates(room):
            self.terrain[y][x] = T_DOOR_C
        cx, cy = room['x'] + room['w'] // 2, room['y'] + room['h'] // 2
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if self.terrain[cy + dy][cx + dx] == T_FLOOR:
                    self.vault_cells.add((cx + dx, cy + dy))
        spots = [(cx - 1, cy - 1), (cx + 1, cy - 1), (cx - 1, cy + 1),
                 (cx + 1, cy + 1), (cx, cy - 1), (cx, cy + 1)]
        self.rng.shuffle(spots)
        for key in self.rng.sample(VAULT_LOOT, k=2):
            it = dict(ITEMS[key])
            it['key'] = key
            if it['kind'] == 'staff':
                it['charges'] = self.rng.randint(4, 8)
            self.items.setdefault(spots.pop(), []).append(it)
        gd = dict(ITEMS['gold'])
        gd['key'] = 'gold'
        gd['amount'] = self.rng.randint(40, 130)
        self.items.setdefault(spots.pop(), []).append(gd)
        # 守卫：宝物旁必驻一只（当前深度池）
        self.monsters.append(
            Monster.spawn(monsters_for_depth(self.depth, self.rng), spots.pop()))

    def _patch(self, room, kind):
        """房间内随机小片水/草（Brogue 地形点缀）。"""
        cx = room['x'] + room['w'] // 2 + self.rng.randint(-1, 1)
        cy = room['y'] + room['h'] // 2 + self.rng.randint(-1, 1)
        r = self.rng.randint(1, 2)
        for y in range(max(1, cy - r), min(MAP_H - 1, cy + r + 1)):
            for x in range(max(1, cx - r), min(MAP_W - 1, cx + r + 1)):
                if self.terrain[y][x] == T_FLOOR and self.rng.random() < 0.85:
                    self.terrain[y][x] = kind

    def _rand_floor(self, min_dist, dist):
        for _ in range(60):
            x = self.rng.randrange(1, MAP_W - 1)
            y = self.rng.randrange(1, MAP_H - 1)
            if (self.terrain[y][x] == T_FLOOR
                    and (x - min_dist[0]) ** 2 + (y - min_dist[1]) ** 2 >= dist * dist
                    and (x, y) not in self.items):
                return (x, y)
        return None

    def walkable(self, x, y):
        if not (0 <= x < MAP_W and 0 <= y < MAP_H):
            return False
        t = self.terrain[y][x]
        return t != T_WALL and t != T_DOOR_C

    def transparent(self, x, y):
        return self.walkable(x, y)

    def opaque(self, x, y):
        """挡光语义（logic.fov 的 opaque 回调：True = 挡光）。"""
        return not self.walkable(x, y)

    def passable(self, x, y):
        """寻路语义：关门可通过（bump 开门耗一步）。"""
        if not (0 <= x < MAP_W and 0 <= y < MAP_H):
            return False
        return self.terrain[y][x] != T_WALL

    def move_cost(self, x, y):
        """回合消耗（水格双倍——Brogue 的水减速）。"""
        return 2 if self.terrain[y][x] == T_WATER else 1


# ================================================================ 实体 ==
class Monster:
    def __init__(self, tpl, pos):
        self.tpl = tpl
        self.name = tpl['name']
        self.hp = self.max_hp = tpl['hp']
        self.x, self.y = pos
        self.awake = False
        self.regen_tick = 0
        self.ranged_tick = 0
        self.slow_t = 0                 # 迟缓法杖剩余回合
        self.charm_t = 0                # 魅惑剩余回合（>0 = 友军）
        # 渲染插值
        self.rx, self.ry = float(pos[0]), float(pos[1])
        self.px, self.py = float(pos[0]), float(pos[1])
        self.anim_t = 1.0
        self.bump = 0.0                 # 攻击前冲量

    @classmethod
    def spawn(cls, tpl, pos):
        return cls(tpl, pos)

    def die(self):
        pass


class Player:
    def __init__(self, pos):
        self.x, self.y = pos
        self.rx, self.ry = float(pos[0]), float(pos[1])
        self.px, self.py = float(pos[0]), float(pos[1])
        self.anim_t = 1.0
        self.bump = 0.0
        self.max_hp = 30
        self.hp = 30
        self.strength = 13
        self.weapon = None              # item dict 或 None
        self.weapon_ench = 0
        self.armor = None
        self.armor_ench = 0
        self.ring = None                # 已装备戒指（item dict）
        self.staff_charges = {}         # key -> 剩余充能（聚合池）
        self.inventory = {}             # key -> count
        self.hunger = 1300
        self.gold = 0
        self.has_amulet = False
        self.regen_tick = 0
        self.lumen_depth = 0

    # ---- 派生数值 ----
    def attack_acc(self):
        acc = 8
        if self.weapon:
            acc += self.weapon['acc'] + self.weapon_ench * 2
        if self.strength < (self.weapon or {}).get('req', 0):
            acc -= 3
        return acc

    def damage_range(self):
        lo, hi = 1, 3
        if self.weapon:
            lo += self.weapon['dmg'][0] + self.weapon_ench
            hi += self.weapon['dmg'][1] + self.weapon_ench
        return (max(1, lo), max(2, hi))

    def defense(self):
        df = 0
        if self.armor:
            df += self.armor['df'] + self.armor_ench
        if self.strength < (self.armor or {}).get('req', 0):
            df = max(0, df - 2)
        if self.has_ring('protect'):
            df += 2
        return df

    def has_ring(self, effect):
        return self.ring is not None and self.ring.get('effect') == effect


# ================================================================ 游戏 ==
class Game:
    """回合制核心：状态与规则，不碰 GL。"""

    def __init__(self, seed=None):
        self.seed = seed if seed is not None else random.randrange(1 << 30)
        self.rng = random.Random(self.seed)
        # 演示 AI 专用 RNG 流（独立于游戏 rng——回放不重跑 AI，
        # 游戏状态只由"动作"消耗 rng，录像复演才可能逐位一致）
        self.ai_rng = random.Random((self.seed * 31337 + 17) & 0x7fffffff)
        self.depth = 1
        self.level = Level(self.depth, self.seed)
        self.player = Player(self.level.player_start)
        self.fov = {}
        self.messages = []              # (text, color)
        self.turn = 0
        self.state = 'play'             # play / dead / won
        self.score = 0
        self.amulet_warning = False
        self.on_level_changed = None    # App 层挂钩：换层时重建地形网格
        from cordis import Context
        self._pather = PathingService(Context(), {'count': 8})
        self.update_fov()
        self.msg('You descend into the Realm of Yendor.', (0.85, 0.90, 1.0))
        self.msg('Reach depth 26, take the Amulet, and climb back out.', (0.7, 0.75, 0.85))

    # ---------------------------------------------------------- 消息 --
    def msg(self, text, color=(0.8, 0.84, 0.9)):
        self.messages.append((text, color))
        if len(self.messages) > 40:
            self.messages.pop(0)

    # ---------------------------------------------------------- FOV --
    def update_fov(self):
        lv = self.level
        px, py = self.player.x, self.player.y
        # ⚠ opaque 回调语义是「True = 挡光」——传 lv.opaque 而不是
        # lv.transparent（MVP 轮曾传反：地板挡光、墙透光，视野缩成
        # 贴身 9 格，直到门复检才现形）。
        self.fov = compute_fov(lv.opaque, (px, py), int(FOV_RADIUS))
        # 直线遮挡复检：slope 窗口 shadowcasting 会从墙角斜缝「绕射」，
        # 把 T 字口正后方格（关门后 1 格）也标亮——Bresenham 复检剔除。
        # 墙/门格自身是射线端点，仍保留（看得见墙、看不见墙后）。
        self.fov = {(x, y): d for (x, y), d in self.fov.items()
                    if visible_through(lv.opaque, px, py, x, y)}
        for (x, y) in self.fov:
            lv.explored[y][x] = True

    def visible(self, x, y):
        return (x, y) in self.fov

    # ---------------------------------------------------------- 移动 --
    def try_move(self, dx, dy):
        if self.state != 'play':
            return
        p = self.player
        lv = self.level
        nx, ny = p.x + dx, p.y + dy
        if dx and dy:                   # 对角穿墙角禁止（Brogue 规则）
            if not (lv.passable(p.x + dx, p.y) and lv.passable(p.x, p.y + dy)):
                self.msg('The way is blocked.')
                return
        target = next((m for m in lv.monsters if (m.x, m.y) == (nx, ny)
                       and m.charm_t <= 0), None)
        if target:
            self.player_attack(target)
            self.end_turn()
            return
        occupant = self.monster_at(nx, ny)
        if occupant:
            self.msg(f'Your charmed {occupant.name} is in the way.')
            return
        if lv.terrain[ny][nx] == T_DOOR_C:
            lv.terrain[ny][nx] = T_DOOR_O
            self.msg('You push the door open.', (0.75, 0.6, 0.4))
            sfx('door')
            self.end_turn()
            return
        if not lv.passable(nx, ny):
            self.msg('A stone wall bars the way.')
            return
        # 矛的 reach：正交直线两格外有敌、中间通透 → 隔位突刺（不移动）
        wp = p.weapon or {}
        if wp.get('reach', 1) >= 2 and (dx == 0 or dy == 0):
            mid_free = lv.transparent(nx, ny) and occupant is None
            far = (next((m for m in lv.monsters
                         if (m.x, m.y) == (p.x + dx * 2, p.y + dy * 2)
                         and m.charm_t <= 0), None) if mid_free else None)
            if far:
                self.player_attack(far)
                self.end_turn()
                return
        depth_before = self.depth
        p.x, p.y = nx, ny
        self.on_enter_tile()
        if self.depth != depth_before or self.state != 'play':
            return                              # 换层/死亡已在楼格结算完
        self.end_turn(cost=lv.move_cost(nx, ny))

    def on_enter_tile(self):
        p, lv = self.player, self.level
        t = lv.terrain[p.y][p.x]
        # 楼梯
        if t == T_STAIRS_DOWN:
            self.descend()
            return
        if t == T_STAIRS_UP:
            self.ascend()
            return
        # 物品自动拾取
        stack = lv.items.get((p.x, p.y))
        if stack:
            it = stack[0]
            self.pick_up(it)
            stack.pop(0)
            if not stack:
                del lv.items[(p.x, p.y)]
        # 陷阱
        tr = lv.trap[p.y][p.x]
        if tr != TRAP_NONE:
            lv.trap[p.y][p.x] = TRAP_NONE
            self.trigger_trap(tr)

    def trigger_trap(self, tr):
        p = self.player
        if tr == TRAP_DART:
            dmg = self.rng.randint(4, 8)
            p.hp -= dmg
            self.msg(f'A dart springs out! (-{dmg} HP)', (1.0, 0.55, 0.45))
            sfx('hit')
        elif tr == TRAP_GAS:
            self.msg('A cloud of green gas erupts!', (0.5, 0.95, 0.4))
            lv = self.level
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    x, y = p.x + dx, p.y + dy
                    if lv.walkable(x, y) and abs(dx) + abs(dy) <= 2:
                        lv.gas[y][x] = min(1.6, lv.gas[y][x] + 1.0)
            sfx('gas')
        elif tr == TRAP_PIT:
            self.msg('The floor opens beneath you!', (0.9, 0.6, 0.3))
            self.descend(fall=True)
        self.check_death()

    def descend(self, fall=False):
        if fall:
            self.player.hp -= self.rng.randint(2, 5)
            self.msg('You fall through the trap door!', (1.0, 0.55, 0.45))
        self.depth += 1
        self.level = Level(self.depth, self.seed)
        self.player.x, self.player.y = self.level.player_start
        self.player.rx, self.player.ry = float(self.player.x), float(self.player.y)
        self.update_fov()
        if self.depth == DEPTH_MAX:
            self.msg('A cold light burns at the heart of this place...', (0.7, 0.9, 1.0))
        elif self.depth == 1:
            self.msg('You surface into pale daylight.', (1.0, 0.95, 0.8))
        else:
            self.msg(f'You descend to depth {self.depth}.', (0.75, 0.8, 0.9))
        sfx('stairs')
        if self.on_level_changed:
            self.on_level_changed()
        self.check_death()

    def ascend(self):
        if self.depth == 1:
            if self.player.has_amulet:
                self.state = 'won'
                self.score = self.player.gold + DEPTH_MAX * 100
                self.msg('*** You escape with the Amulet of Yendor! ***', (1.0, 0.9, 0.4))
                sfx('win')
            else:
                self.msg('Sunlight blazes above — but you came for the Amulet.', (0.8, 0.8, 0.85))
            return
        self.depth -= 1
        self.level = Level(self.depth, self.seed)
        self.player.x, self.player.y = self.level.player_start
        self.player.rx, self.player.ry = float(self.player.x), float(self.player.y)
        self.update_fov()
        self.msg(f'You climb to depth {self.depth}.', (0.75, 0.8, 0.9))
        sfx('stairs')
        if self.on_level_changed:
            self.on_level_changed()

    # ---------------------------------------------------------- 拾取 --
    def pick_up(self, it):
        p, kind = self.player, it['kind']
        if kind == 'gold':
            p.gold += it['amount']
            self.msg(f'You pick up {it["amount"]} gold.', (0.95, 0.85, 0.35))
            sfx('gold')
            return
        if kind == 'amulet':
            p.has_amulet = True
            self.level.has_amulet = True
            self.msg('*** You hold the Amulet of Yendor! Now climb to the surface! ***',
                     (0.6, 0.95, 1.0))
            sfx('win')
            return
        if kind == 'weapon':
            old = p.weapon
            if old and old['dmg'] >= it['dmg'] and old['acc'] >= it['acc']:
                p.inventory[it['key']] = p.inventory.get(it['key'], 0) + 1
                self.msg(f'You stash the {it["name"]}.')
                return
            p.weapon, p.weapon_ench = it, 0
            self.msg(f'You wield the {it["name"]}.', (0.9, 0.85, 0.6))
            sfx('pickup')
            return
        if kind == 'armor':
            old = p.armor
            if old and old['df'] >= it['df']:
                p.inventory[it['key']] = p.inventory.get(it['key'], 0) + 1
                self.msg(f'You stash the {it["name"]}.')
                return
            p.armor, p.armor_ench = it, 0
            self.msg(f'You wear the {it["name"]}.', (0.9, 0.85, 0.6))
            sfx('pickup')
            return
        if kind == 'staff':
            p.staff_charges[it['key']] = p.staff_charges.get(it['key'], 0) \
                + it.get('charges', 4)
            p.inventory[it['key']] = p.inventory.get(it['key'], 0) + 1
            self.msg(f'You take the {it["name"]}.', (0.7, 0.8, 0.95))
            sfx('pickup')
            return
        if kind == 'ring':
            if p.ring is None:
                p.ring = it
                self.msg(f'You slip on the {it["name"]}.', (0.85, 0.7, 0.95))
            else:
                p.inventory[it['key']] = p.inventory.get(it['key'], 0) + 1
                self.msg(f'You stash the {it["name"]}.')
            sfx('pickup')
            return
        p.inventory[it['key']] = p.inventory.get(it['key'], 0) + 1
        self.msg(f'You found {it["name"]}.', (0.75, 0.85, 0.75))
        sfx('pickup')

    # ---------------------------------------------------------- 物品使用 --
    def use_slot(self, slot):
        if self.state != 'play':
            return
        keys = [k for k, n in self.player.inventory.items() if n > 0]
        if slot >= len(keys):
            return
        key = keys[slot]
        p = self.player
        it = dict(ITEMS[key])
        p.inventory[key] -= 1
        if p.inventory[key] <= 0:
            del p.inventory[key]
        kind = it['kind']
        if kind == 'food':
            p.hunger = min(2000, p.hunger + it['food'])
            if it.get('heal'):
                p.hp = min(p.max_hp, p.hp + it['heal'])
            self.msg(f'You eat the {it["name"]}.', (0.85, 0.75, 0.5))
            sfx('eat')
        elif kind == 'potion':
            if 'heal' in it:
                p.hp = min(p.max_hp, p.hp + it['heal'])
                self.msg(f'A warm glow knits your wounds (+{it["heal"]}).', (0.95, 0.5, 0.55))
                sfx('heal')
            elif 'maxhp' in it:
                p.max_hp += it['maxhp']
                p.hp += it['maxhp']
                self.msg('You feel your life force deepen!', (0.95, 0.5, 0.8))
                sfx('heal')
            elif 'strength' in it:
                p.strength += it['strength']
                self.msg('You feel mightier!', (0.5, 0.6, 1.0))
                sfx('heal')
            elif 'charm' in it:
                self.charm_nearest()
        elif kind == 'staff':
            p.inventory[key] = p.inventory.get(key, 0) + 1      # 数量只随碎裂减少
            self.fire_staff(key)
        elif kind == 'ring':
            old = p.ring
            p.ring = it
            if old is not None:
                p.inventory[old['key']] = p.inventory.get(old['key'], 0) + 1
            self.msg(f'You now wear the {it["name"]}.', (0.85, 0.7, 0.95))
            sfx('magic')
        elif kind == 'scroll':
            if it.get('enchant'):
                if p.weapon:
                    p.weapon_ench += 1
                    self.msg(f'Your {p.weapon["name"]} gleams (+{p.weapon_ench}).', (0.95, 0.9, 0.5))
                elif p.armor:
                    p.armor_ench += 1
                    self.msg(f'Your {p.armor["name"]} hardens (+{p.armor_ench}).', (0.95, 0.9, 0.5))
                else:
                    p.inventory[key] = p.inventory.get(key, 0) + 1
                    self.msg('Nothing here to enchant.')
                sfx('magic')
            elif it.get('tele'):
                lv = self.level
                pos = None
                for _ in range(80):
                    x = self.rng.randrange(1, MAP_W - 1)
                    y = self.rng.randrange(1, MAP_H - 1)
                    if lv.terrain[y][x] not in (T_WALL,):
                        pos = (x, y)
                        break
                if pos:
                    p.x, p.y = pos
                    self.update_fov()
                    self.msg('The world folds around you.', (0.7, 0.7, 0.95))
                    sfx('magic')
        self.end_turn()

    # ---------------------------------------------------- 施法/魅惑 --
    def charm_nearest(self):
        """魅惑药剂：视野内最近敌人倒戈 40 回合。"""
        best, bd = None, 1 << 30
        for m in self.level.monsters:
            if m.charm_t > 0:
                continue
            d = self.fov.get((m.x, m.y))
            if d is not None and d <= 64 and d < bd:
                best, bd = m, d
        if best:
            best.charm_t = 40
            best.awake = True
            self.msg(f'The {best.name} falls under your charm!', (0.95, 0.55, 0.85))
            sfx('charm')
        else:
            self.msg('You feel a wave of misplaced benevolence.', (0.9, 0.6, 0.8))

    def fire_staff(self, key=None):
        """法杖自动索敌：视野内最近的敌人（Z = 第一把有充能的法杖）。

        充能语义统一在这里：命中目标才扣 1 充能（空放不扣），
        扣到 0 法杖碎裂、移出背包。use_slot 与 Z 键都走本入口。
        """
        p = self.player
        if key is None:
            key = next((k for k, n in p.inventory.items()
                        if n > 0 and ITEMS[k]['kind'] == 'staff'
                        and p.staff_charges.get(k, 0) > 0), None)
            if key is None:
                self.msg('You have no charged staff.')
                return
        if p.staff_charges.get(key, 0) <= 0:
            self.msg('The staff is out of charge.')
            return
        tpl = ITEMS[key]
        rng_sq = tpl['srange'] ** 2
        best, bd = None, 1 << 30
        for m in self.level.monsters:
            if m.charm_t > 0:
                continue
            d = self.fov.get((m.x, m.y))
            if d is not None and d <= rng_sq and d < bd:
                best, bd = m, d
        if best is None:
            self.msg('No target in sight for the staff.')
            return
        # ---- 命中即扣充能 ----
        p.staff_charges[key] -= 1
        crumbled = p.staff_charges[key] <= 0
        if crumbled:
            p.inventory.pop(key, None)
            self.msg(f'The {tpl["name"]} crumbles to dust.', (0.6, 0.6, 0.6))
        eff = tpl['effect']
        best.awake = True
        second = None
        if eff == 'slow':
            best.slow_t = 20
            self.msg(f'The {best.name} slows to a crawl.', (0.6, 0.9, 0.6))
            sfx('zap')
        else:
            dmg = self.rng.randint(*tpl['dmg'])
            best.hp -= dmg
            if eff == 'fire':
                lv = self.level
                lv.fire[best.y][best.x] = max(lv.fire[best.y][best.x], 3)
                self.msg(f'A gout of flame sears the {best.name} for {dmg}!',
                         (1.0, 0.55, 0.25))
                sfx('ranged')
            else:                           # bolt：闪电 + 链式跳一个
                self.msg(f'Lightning arcs into the {best.name} for {dmg}!',
                         (0.6, 0.8, 1.0))
                sfx('zap')
                sd = 1 << 30
                for m in self.level.monsters:
                    if m is best or m.charm_t > 0:
                        continue
                    d2 = (m.x - best.x) ** 2 + (m.y - best.y) ** 2
                    if 0 < d2 <= 16 and self.visible(m.x, m.y) and d2 < sd:
                        second, sd = m, d2
                if second:
                    half = max(1, dmg // 2)
                    second.hp -= half
                    second.awake = True
                    self.msg(f'The arc jumps to the {second.name} for {half}!',
                             (0.6, 0.8, 1.0))
        if best.hp <= 0:
            self.kill_monster(best)
        if second is not None and second.hp <= 0:
            self.kill_monster(second)

    # ---------------------------------------------------------- 战斗 --
    def player_attack(self, m, primary=True):
        p = self.player
        hit = self.rng.randint(0, 20) + p.attack_acc() > 10 + self.level_rnd_def(m)
        p.bump = 1.0
        if hit:
            lo, hi = p.damage_range()
            dmg = self.rng.randint(lo, hi)
            m.hp -= dmg
            m.awake = True
            self.msg(f'You hit the {m.name} for {dmg}.', (0.9, 0.9, 0.7))
            sfx('hit')
            if m.hp <= 0:
                self.kill_monster(m)
        else:
            self.msg(f'You miss the {m.name}.', (0.6, 0.62, 0.68))
        # 战斧横扫：主目标之外，正交相邻的敌人各再判一刀
        if primary and (p.weapon or {}).get('sweep'):
            for o in list(self.level.monsters):
                if o is m or o.charm_t > 0:
                    continue
                if abs(o.x - p.x) + abs(o.y - p.y) == 1:
                    self.player_attack(o, primary=False)

    def level_rnd_def(self, m):
        return m.tpl['df']

    def monster_attack(self, m):
        p = self.player
        hit = self.rng.randint(0, 20) + m.tpl['ac'] > 10 + p.defense()
        m.bump = 1.0
        if hit:
            lo, hi = m.tpl['dmg']
            dmg = self.rng.randint(lo, hi)
            p.hp -= dmg
            self.msg(f'The {m.name} hits you for {dmg}!', (1.0, 0.45, 0.4))
            sfx('hurt')
        else:
            self.msg(f'The {m.name} misses you.')
        self.check_death()

    def kill_monster(self, m):
        self.level.monsters.remove(m)
        self.msg(f'The {m.name} dies.', (0.75, 0.8, 0.75))
        sfx('kill')
        # 尸体掉落：低概率留食物
        if self.rng.random() < 0.12:
            key = 'ration'
            it = dict(ITEMS[key])
            it['key'] = key
            self.level.items.setdefault((m.x, m.y), []).append(it)

    def check_death(self):
        if self.player.hp <= 0 and self.state == 'play':
            self.state = 'dead'
            self.score = self.player.gold + self.depth * 60
            self.msg('*** You die... ***', (1.0, 0.3, 0.3))
            sfx('death')

    # ---------------------------------------------------------- 回合 --
    def end_turn(self, cost=1):
        if self.state != 'play':
            return
        for _ in range(cost):
            self.turn += 1
            self.level.turns += 1
            self.monster_phase()
            if self.state != 'play':
                return
            self.environment_phase()
            self.hunger_phase()

    # ---- 怪物回合 ----
    def monster_phase(self):
        lv = self.level
        p = self.player
        for m in list(lv.monsters):
            if m.hp <= 0:               # 本回合内已被友军/环境击杀
                continue
            spd = m.tpl['spd']
            if m.slow_t > 0:
                m.slow_t -= 1
                spd = 50
            moves = 2 if spd >= 150 else (1 if spd >= 100 else (self.turn % 2 == 0))
            if not moves:
                continue
            if not m.awake:
                # 惊醒判定：距离 + 视线（潜行戒压缩察觉半径 7→4）
                d2 = (m.x - p.x) ** 2 + (m.y - p.y) ** 2
                if d2 <= (16 if p.has_ring('stealth') else 49) and \
                        compute_fov(lv.opaque, (m.x, m.y), 7).get((p.x, p.y)) is not None:
                    m.awake = True
                    if self.visible(m.x, m.y):
                        self.msg(f'The {m.name} notices you!', (0.95, 0.75, 0.5))
                continue
            if m.charm_t > 0:
                # 魅惑盟友：扑向最近的敌对怪物；到期倒戈回敌营
                m.charm_t -= 1
                if m.charm_t <= 0:
                    m.awake = True
                    if self.visible(m.x, m.y):
                        self.msg(f'The {m.name} shakes off your charm!', (0.95, 0.6, 0.6))
                    continue
                foes = [o for o in lv.monsters if o is not m and o.charm_t <= 0]
                if foes:
                    tgt = min(foes, key=lambda o: (o.x - m.x) ** 2 + (o.y - m.y) ** 2)
                    if (tgt.x - m.x) ** 2 + (tgt.y - m.y) ** 2 <= 2:
                        hit = self.rng.randint(0, 20) + m.tpl['ac'] > 10 + tgt.tpl['df']
                        m.bump = 1.0
                        if hit:
                            dmg = self.rng.randint(*m.tpl['dmg'])
                            tgt.hp -= dmg
                            tgt.awake = True
                            if self.visible(m.x, m.y):
                                self.msg(f'Your {m.name} hits the {tgt.name} for {dmg}.',
                                         (0.95, 0.8, 0.5))
                            if tgt.hp <= 0:
                                self.kill_monster(tgt)
                        elif self.visible(m.x, m.y):
                            self.msg(f'Your {m.name} misses the {tgt.name}.')
                    else:
                        self.chase_toward(m, (tgt.x, tgt.y))
                continue
            dist2 = (m.x - p.x) ** 2 + (m.y - p.y) ** 2
            if dist2 == 1:
                self.monster_attack(m)
                continue
            # 远程（dragon 火息 / warden 魔印），每 3 回合一次
            rngd = m.tpl['ranged']
            if rngd and dist2 <= rngd[1] ** 2:
                m.ranged_tick += 1
                if m.ranged_tick >= 3:
                    m.ranged_tick = 0
                    kind, _, lo, hi = rngd
                    dmg = self.rng.randint(lo, hi)
                    p.hp -= dmg
                    self.msg(f'The {m.name} strikes from afar for {dmg}!', (1.0, 0.5, 0.3))
                    sfx('ranged')
                    if kind == 'fire':      # 火息点燃地面
                        lv.fire[p.y][p.x] = max(lv.fire[p.y][p.x], 2)
                    self.check_death()
                    if self.state != 'play':
                        return
                    continue
            # A* 追击
            self.chase_toward(m, (p.x, p.y))

    def chase_toward(self, m, goal):
        """A* 追向 goal；关门按可通过寻路，下一格是门则先开门（耗此回合）。"""
        lv = self.level
        path = self._pather.find_path((m.x, m.y), goal, lv.passable)
        if path and len(path) >= 2:
            nx, ny = path[1]
            if lv.terrain[ny][nx] == T_DOOR_C:
                lv.terrain[ny][nx] = T_DOOR_O
                if self.visible(nx, ny):
                    self.msg(f'The {m.name} pushes the door open.', (0.7, 0.6, 0.45))
                    sfx('door')
                return
            if goal != (nx, ny) and not any(
                    (o.x, o.y) == (nx, ny) for o in lv.monsters):
                m.px, m.py = float(m.x), float(m.y)
                m.x, m.y = nx, ny
                m.anim_t = 0.0
                return
        # A* 不可达 → 贪心逼近（不可踩上 goal 本格）
        gx, gy = goal
        dx = (gx > m.x) - (gx < m.x)
        dy = (gy > m.y) - (gy < m.y)
        for ndx, ndy in ((dx, dy), (dx, 0), (0, dy)):
            nx, ny = m.x + ndx, m.y + ndy
            if (nx, ny) != goal and lv.walkable(nx, ny) and not any(
                    (o.x, o.y) == (nx, ny) for o in lv.monsters):
                m.px, m.py = float(m.x), float(m.y)
                m.x, m.y = nx, ny
                m.anim_t = 0.0
                return

    # ---- 环境回合（气体扩散 / 火焰蔓延 / 刷新）----
    def environment_phase(self):
        lv = self.level
        p = self.player
        # 毒气：扩散 + 衰减 + 伤害
        new_gas = [[0.0] * MAP_W for _ in range(MAP_H)]
        for y in range(MAP_H):
            for x in range(MAP_W):
                g = lv.gas[y][x]
                if g <= 0.02:
                    continue
                out = g * 0.10
                stay = g * 0.80
                spread = g * 0.10
                new_gas[y][x] += stay
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if lv.walkable(nx, ny):
                        new_gas[ny][nx] += spread
                    else:
                        new_gas[y][x] += spread
                lv.gas[y][x] = 0.0                  # 全部重算进 new_gas
        lv.gas = new_gas
        if lv.gas[p.y][p.x] > 0.15:
            dmg = max(1, int(lv.gas[p.y][p.x] * 3))
            p.hp -= dmg
            if self.turn % 4 == 0:
                self.msg('The gas burns your lungs!', (0.5, 0.95, 0.4))
        for m in list(lv.monsters):
            if lv.gas[m.y][m.x] > 0.15 and self.rng.random() < 0.4:
                m.hp -= max(1, int(lv.gas[m.y][m.x] * 2))
                if m.hp <= 0:
                    m.awake = True
                    self.kill_monster(m)
        # 火焰：蔓延与熄灭
        burning = [(x, y) for y in range(MAP_H) for x in range(MAP_W)
                   if lv.fire[y][x] > 0]
        for x, y in burning:
            lv.fire[y][x] -= 1
            if lv.fire[y][x] <= 0:
                continue
            # 引燃邻居（草易燃、水不可燃、门板可烧穿）
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if lv.terrain[ny][nx] in (T_DOOR_C, T_DOOR_O) \
                        and self.rng.random() < 0.5:
                    lv.terrain[ny][nx] = T_FLOOR
                    if self.visible(nx, ny):
                        self.msg('A door burns through!', (1.0, 0.6, 0.3))
                if (lv.terrain[ny][nx] == T_GRASS and lv.fire[ny][nx] == 0
                        and self.rng.random() < 0.45):
                    lv.fire[ny][nx] = 3
            if lv.terrain[y][x] == T_GRASS and self.rng.random() < 0.3:
                lv.terrain[y][x] = T_FLOOR          # 草烧成焦土
        if lv.fire[p.y][p.x] > 0:
            dmg = self.rng.randint(3, 6)
            p.hp -= dmg
            if self.turn % 3 == 0:
                self.msg('You are burning!', (1.0, 0.5, 0.2))
        for m in list(lv.monsters):
            if lv.fire[m.y][m.x] > 0:
                m.hp -= self.rng.randint(2, 5)
                if m.hp <= 0:
                    self.kill_monster(m)
        if p.hp <= 0:
            self.check_death()
        # 怪物刷新（清不空一层——Brogue 的重要压力设计）
        lv.spawn_cooldown -= 1
        n = len(lv.monsters)
        cap = 8
        rate = 25 if p.has_amulet else 90
        if lv.spawn_cooldown <= 0 and n < cap:
            lv.spawn_cooldown = rate
            pos = None
            for _ in range(40):
                x = self.rng.randrange(1, MAP_W - 1)
                y = self.rng.randrange(1, MAP_H - 1)
                if lv.walkable(x, y) and (x - p.x) ** 2 + (y - p.y) ** 2 >= 144:
                    pos = (x, y)
                    break
            if pos:
                lv.monsters.append(Monster.spawn(monsters_for_depth(lv.depth, self.rng), pos))

    # ---- 饥饿回合 ----
    def hunger_phase(self):
        p = self.player
        p.hunger -= 1
        if p.hunger == 120:
            self.msg('You are getting hungry.', (0.95, 0.8, 0.4))
        elif p.hunger == 40:
            self.msg('You are starving!', (1.0, 0.5, 0.3))
        elif p.hunger <= 0:
            p.regen_tick += 1
            if p.regen_tick >= 8:
                p.regen_tick = 0
                p.hp -= 1
                self.check_death()
        else:
            # 自然回复（再生戒：间隔 24 → 12）
            if p.hp < p.max_hp:
                p.regen_tick += 1
                if p.regen_tick >= (12 if p.has_ring('regen') else 24):
                    p.regen_tick = 0
                    p.hp += 1

    # ---------------------------------------------------- 等待一回合 --
    def wait(self):
        if self.state == 'play':
            self.msg('You wait.')
            self.end_turn()

    # -------------------------------------------------------- 渲染辅助 --
    def monster_at(self, x, y):
        return next((m for m in self.level.monsters if (m.x, m.y) == (x, y)), None)


# ---------------------------------------------------------------- 音效 --
def _write_wav(path, samples, sr=22050):
    """numpy float[-1,1] → 16bit WAV（标准库 wave，免 scipy）。"""
    data = (np.clip(samples, -1.0, 1.0) * 32000).astype('<i2').tobytes()
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data)


def _tone(dur, f0, f1, sr=22050, decay=6.0, noise=0.0):
    n = int(sr * dur)
    t = np.arange(n) / sr
    freq = f0 + (f1 - f0) * (t / max(1e-6, dur))
    phase = 2 * np.pi * np.cumsum(freq) / sr
    s = np.sin(phase)
    if noise > 0:
        s = s * (1 - noise) + np.random.default_rng(7).standard_normal(n) * noise
    env = np.exp(-decay * t)
    return s * env * 0.8


SFX_SPECS = {
    'hit':    lambda: _tone(0.14, 220, 110, decay=18, noise=0.4),
    'hurt':   lambda: _tone(0.20, 160, 70, decay=12, noise=0.5),
    'kill':   lambda: _tone(0.30, 300, 60, decay=9, noise=0.3),
    'pickup': lambda: _tone(0.16, 520, 780, decay=10),
    'gold':   lambda: _tone(0.20, 880, 1320, decay=12),
    'heal':   lambda: _tone(0.35, 440, 660, decay=6),
    'magic':  lambda: _tone(0.30, 300, 900, decay=7, noise=0.1),
    'stairs': lambda: _tone(0.28, 200, 90, decay=8),
    'gas':    lambda: _tone(0.45, 90, 60, decay=5, noise=0.8),
    'ranged': lambda: _tone(0.25, 500, 150, decay=10, noise=0.3),
    'eat':    lambda: _tone(0.18, 180, 140, decay=9, noise=0.5),
    'door':   lambda: _tone(0.18, 90, 140, decay=8, noise=0.25),
    'zap':    lambda: _tone(0.16, 900, 200, decay=14, noise=0.6),
    'charm':  lambda: _tone(0.30, 500, 750, decay=6, noise=0.05),
    'death':  lambda: _tone(0.9, 220, 40, decay=3, noise=0.4),
    'win':    lambda: _tone(0.8, 440, 880, decay=3),
}


def ensure_sfx(sfx_dir):
    """确保 SFX WAV 存在（首次运行合成，生成物不入库）。"""
    os.makedirs(sfx_dir, exist_ok=True)
    for name, make in SFX_SPECS.items():
        p = os.path.join(sfx_dir, f'{name}.wav')
        if not os.path.exists(p):
            _write_wav(p, make())


_SFX = None


def sfx(name):
    global _SFX
    if _SFX is None:
        try:
            from world3d.engine.audio import SfxPlayer
            d = os.path.join(PROJECT_ROOT, 'assets', 'brogueward_sfx')
            ensure_sfx(d)
            _SFX = SfxPlayer(d, names=tuple(SFX_SPECS), gain=0.5)
        except Exception as e:                              # noqa: BLE001
            print(f'[sfx] init failed: {e}', flush=True)
            _SFX = False
    if _SFX:
        _SFX.play(name)


PROJECT_ROOT = os.environ.get('WORLD3D_PROJECT_ROOT') or os.getcwd()

# ================================================================ 渲染 ==
# 引擎 Mesh 管线不走顶点色（颜色在 Material uniform），而 FOV 光照要
# 每格变色 → 自建 unlit 顶点色管线（harness 同款 WindowBlock UBO）。

_VERT = '''#version 330 core
in vec3 position;
in vec4 colors;
uniform WindowBlock { mat4 projection; mat4 view; } window;
uniform mat4 u_model;
out vec4 v_col;
void main() {
    v_col = colors;
    gl_Position = window.projection * window.view * u_model * vec4(position, 1.0);
}'''

_FRAG = '''#version 330 core
in vec4 v_col;
out vec4 final_color;
void main() { final_color = v_col; }'''


def _make_program():
    from pyglet.graphics.shader import Shader, ShaderProgram
    return ShaderProgram(Shader(_VERT, 'vertex'), Shader(_FRAG, 'fragment'))


class TerrainMesh:
    """静态地形合并网格 + 每回合 FOV 顶点色刷新。

    每格顶点索引登记在 ``cell_verts``，光照变化只重写颜色数组——
    位置/索引终生不变（Brogue 的地形语义）。
    """

    def __init__(self, program, game):
        self.program = program
        self.game = game
        self._build()
        self.update_colors()

    # ---- 几何构建 ----
    def _build(self):
        lv = self.game.level
        pos, col, idx = [], [], []
        self.cell_verts = {}                 # (x,y) -> (start, count)

        def quad(p0, p1, p2, p3, rgb, alpha=1.0):
            """加一个四边形（4 顶点 + 6 索引），返回起始顶点号。"""
            start = len(pos) // 3
            for p in (p0, p1, p2, p3):
                pos.extend((p[0], p[1], p[2]))
                col.extend((rgb[0], rgb[1], rgb[2], alpha))
            idx.extend((start, start + 1, start + 2,
                        start, start + 2, start + 3))
            return start

        for y in range(MAP_H):
            for x in range(MAP_W):
                t = lv.terrain[y][x]
                wx, wz = x * TILE, y * TILE
                base = TERRAIN_COLORS[t]
                if t == T_WALL:
                    top = (wx + TILE, 0.7, wz + TILE)
                    quad((wx, 0.7, wz), (wx, 0.7, wz + TILE), top,
                         (wx + TILE, 0.7, wz), base)
                    # 暴露侧面（邻格是地板才画）；侧面受光更亮，立体感
                    if lv.walkable(x, y + 1):        # -z 侧朝向观察者
                        quad((wx, 0.0, wz + TILE), (wx + TILE, 0.0, wz + TILE),
                             (wx + TILE, 0.7, wz + TILE), (wx, 0.7, wz + TILE),
                             tuple(min(1.0, c * 2.6) for c in base))
                    if lv.walkable(x, y - 1):
                        quad((wx + TILE, 0.0, wz), (wx, 0.0, wz),
                             (wx, 0.7, wz), (wx + TILE, 0.7, wz),
                             tuple(min(1.0, c * 2.6) for c in base))
                    if lv.walkable(x - 1, y):
                        quad((wx, 0.0, wz), (wx, 0.0, wz + TILE),
                             (wx, 0.7, wz + TILE), (wx, 0.7, wz),
                             tuple(min(1.0, c * 2.2) for c in base))
                    if lv.walkable(x + 1, y):
                        quad((wx + TILE, 0.0, wz + TILE), (wx + TILE, 0.0, wz),
                             (wx + TILE, 0.7, wz), (wx + TILE, 0.7, wz + TILE),
                             tuple(min(1.0, c * 2.2) for c in base))
                elif t == T_DOOR_C:
                    # 关门：半高门板（顶面注册进光照格，侧沿画在四周）
                    ch = 0.5
                    top = (wx + TILE, ch, wz + TILE)
                    s = quad((wx, ch, wz), (wx, ch, wz + TILE), top,
                             (wx + TILE, ch, wz), base)
                    self.cell_verts[(x, y)] = (s, 4)
                    lo_side = tuple(min(1.0, c * 2.6) for c in base)
                    quad((wx, 0.0, wz + TILE), (wx + TILE, 0.0, wz + TILE),
                         (wx + TILE, ch, wz + TILE), (wx, ch, wz + TILE), lo_side)
                    quad((wx + TILE, 0.0, wz), (wx, 0.0, wz),
                         (wx, ch, wz), (wx + TILE, ch, wz), lo_side)
                    quad((wx, 0.0, wz), (wx, 0.0, wz + TILE),
                         (wx, ch, wz + TILE), (wx, ch, wz), lo_side)
                    quad((wx + TILE, 0.0, wz + TILE), (wx + TILE, 0.0, wz),
                         (wx + TILE, ch, wz), (wx + TILE, ch, wz + TILE), lo_side)
                else:
                    h = 0.16 if t == T_WATER else 0.0
                    c = base
                    if t == T_FLOOR:
                        # 棋盘微差，增强格子读感
                        k = 1.0 if (x + y) % 2 == 0 else 0.94
                        c = tuple(v * k for v in base)
                    start = quad((wx, h, wz), (wx, h, wz + TILE),
                                 (wx + TILE, h, wz + TILE), (wx + TILE, h, wz), c)
                    self.cell_verts[(x, y)] = (start, 4)
        self.n_vert = len(pos) // 3
        self._pos = pos
        self.vlist = self.program.vertex_list_indexed(
            self.n_vert, gl.GL_TRIANGLES, idx,
            position=('f', pos), colors=('f', col))

    # ---- FOV 光照 ----
    def update_colors(self):
        """按 可见(火把衰减)/记忆(暗)/未知(黑) 三态重写顶点色。"""
        lv = self.game.level
        fov = self.game.fov
        tint = depth_tint(lv.depth)
        col = self.vlist.colors
        for (x, y), (start, n) in self.cell_verts.items():
            t = lv.terrain[y][x]
            base = TERRAIN_COLORS[t]
            if (x, y) in lv.vault_cells:    # 宝藏室地板金染
                base = (min(1.0, base[0] + 0.25), min(1.0, base[1] + 0.16),
                        min(1.0, base[2] * 0.5 + 0.05))
            if (x, y) in fov:
                d = fov[(x, y)]
                lit = 0.32 + 0.68 * light_level(math.sqrt(d), FOV_RADIUS)
            elif lv.explored[y][x]:
                lit = 0.10                   # 记忆态：远暗于可见区下限
            else:
                lit = 0.0
            r = min(1.0, base[0] * tint[0]) * lit
            g = min(1.0, base[1] * tint[1]) * lit
            b = min(1.0, base[2] * tint[2]) * lit
            for i in range(start, start + n):
                col[i * 4] = r
                col[i * 4 + 1] = g
                col[i * 4 + 2] = b
                col[i * 4 + 3] = 1.0

    def refresh(self):
        """回合后刷新：pyglet attribute 赋值需要扁平 list（实测）。"""
        self.update_colors()
        self.vlist.colors[:] = self.vlist.colors[:]

    def draw(self):
        self.program['u_model'] = Mat4()
        self.vlist.draw(gl.GL_TRIANGLES)

    def delete(self):
        self.vlist.delete()


class DynamicBatch:
    """每帧 CPU 变换的实体批次（玩家/怪物/物品/火/气）。

    帧内 numpy 组装三角汤 → 单 vertex_list 单 draw call；
    回合制实体 <60 个，CPU 变换成本可忽略（flowward 5000 粒子同量级验证）。
    ⚠ pyglet attribute 只接受**等长**整体赋值（切片赋值要求同长），所以
    vertex_list 按「实际需要的顶点数」建（首次/扩容时重建），每帧只传
    count 顶点；``draw(count)`` 只画前缀。半透明（气/火）走第二个批次、
    深度只读。
    """

    CAP = 24000                                    # numpy 预留上限（保护）

    def __init__(self, program):
        self.program = program
        self.vlist = None
        self.capacity = 0
        self._pos = np.zeros((self.CAP, 3), np.float32)
        self._col = np.zeros((self.CAP, 4), np.float32)
        self.count = 0

    def begin(self):
        self.count = 0

    def add_tris(self, verts, color, alpha=1.0):
        n = len(verts)
        if self.count + n > self.CAP:
            return
        sl = slice(self.count, self.count + n)
        arr = np.asarray(verts, np.float32).reshape(n, 3)
        self._pos[sl] = arr
        self._col[sl, 0] = color[0]
        self._col[sl, 1] = color[1]
        self._col[sl, 2] = color[2]
        self._col[sl, 3] = alpha
        self.count += n

    def add_cube(self, x, y, z, sx, sy, sz, color, alpha=1.0):
        verts = cube_tris(x, y, z, sx, sy, sz)
        n = len(verts)
        if self.count + n > self.CAP:
            return
        sl = slice(self.count, self.count + n)
        self._pos[sl] = np.asarray(verts, np.float32).reshape(n, 3)
        # 面朝向亮度：顶面 6 顶点 ×1.0，前后 0.78，左右 0.66（顶点序见 cube_tris）
        k = np.empty(n, np.float32)
        k[:] = 0.70
        k[0:6] = 1.0
        k[6:18] = 0.80
        self._col[sl, 0] = np.minimum(1.0, color[0] * k)
        self._col[sl, 1] = np.minimum(1.0, color[1] * k)
        self._col[sl, 2] = np.minimum(1.0, color[2] * k)
        self._col[sl, 3] = alpha
        self.count += n

    def add_pyramid(self, x, y, z, sx, sy, sz, color, alpha=1.0):
        verts = pyramid_tris(x, y, z, sx, sy, sz)
        n = len(verts)
        if self.count + n > self.CAP:
            return
        sl = slice(self.count, self.count + n)
        self._pos[sl] = np.asarray(verts, np.float32).reshape(n, 3)
        k = np.empty(n, np.float32)
        k[:] = 0.78
        k[0:6] = 1.0
        self._col[sl, 0] = np.minimum(1.0, color[0] * k)
        self._col[sl, 1] = np.minimum(1.0, color[1] * k)
        self._col[sl, 2] = np.minimum(1.0, color[2] * k)
        self._col[sl, 3] = alpha
        self.count += n

    def end(self):
        """上传并绘制（capacity 恒等于 count：实体数变化才重建 vertex_list，
        其余帧等长原地赋值——pyglet attribute 切片赋值要求同长）。"""
        if not self.count:
            return
        if self.vlist is None or self.count != self.capacity:
            if self.vlist is not None:
                self.vlist.delete()
            self.capacity = self.count
            self.vlist = self.program.vertex_list(
                self.capacity, gl.GL_TRIANGLES,
                position=('f', self._pos[:self.count].ravel().tolist()),
                colors=('f', self._col[:self.count].ravel().tolist()))
        else:
            self.vlist.position[:] = self._pos[:self.count].ravel().tolist()
            self.vlist.colors[:] = self._col[:self.count].ravel().tolist()
        self.program['u_model'] = Mat4()
        self.vlist.draw(gl.GL_TRIANGLES)

    def delete(self):
        if self.vlist is not None:
            self.vlist.delete()
            self.vlist = None


# ---- 实体形状参数 ----
MONSTER_SHAPE = {
    'rat':    (0.20, 0.16, 0.26),      # 半尺寸 sx, sy, sz（扁长）
    'kobold': (0.26, 0.36, 0.22),
    'goblin': (0.28, 0.42, 0.24),
    'ogre':   (0.40, 0.55, 0.34),
    'wraith': (0.30, 0.50, 0.26),
    'dragon': (0.46, 0.50, 0.60),
}

ITEM_SHAPE = {
    'food':   (0.16, 0.12, 0.16, 0.10),      # x, y(悬浮), z, 半高
    'potion': (0.11, 0.12, 0.11, 0.18),
    'scroll': (0.16, 0.10, 0.12, 0.03),
    'weapon': (0.05, 0.12, 0.05, 0.30),
    'armor':  (0.20, 0.12, 0.16, 0.14),
    'staff':  (0.05, 0.14, 0.05, 0.36),
    'ring':   (0.08, 0.12, 0.08, 0.05),
    'gold':   (0.09, 0.14, 0.09, 0.06),
    'amulet': (0.16, 0.30, 0.16, 0.20),
}


def _world(x, y):
    """格坐标 → 世界中心。"""
    return (x * TILE + TILE / 2, y * TILE + TILE / 2)


class GameHud:
    """游戏 HUD：左上状态块 + 底部消息 + 结局大字（2D overlay）。"""

    def __init__(self):
        from world3d.engine.fonts import resolve_cjk_font
        kw = {}
        cjk = resolve_cjk_font()
        if cjk:
            kw['font_name'] = cjk
        self.batch = pyglet.graphics.Batch()
        self.kw = kw
        bg = (10, 12, 16, 140)
        self.panel = shapes.Rectangle(10, 0, 240, 150, color=bg, batch=self.batch)
        self.panel.anchor_y = 0
        self.msg_bg = shapes.Rectangle(0, 0, 10, 96, color=bg, batch=self.batch)
        self.stat = pgtext.Label('', font_size=11, x=18, y=0, anchor_y='top',
                                 width=226, multiline=True, color=(225, 228, 235, 255),
                                 batch=self.batch, **kw)
        self.msgs = [pgtext.Label('', font_size=11, x=14, y=18 + i * 20,
                                  anchor_y='bottom', color=(205, 210, 220, 255),
                                  batch=self.batch, **kw) for i in range(4)]
        self.big = pgtext.Label('', font_size=30, weight='bold', x=0, y=0,
                                anchor_x='center', anchor_y='center',
                                color=(255, 220, 120, 255), batch=self.batch, **kw)

    def layout(self, w, h):
        # ⚠ shapes.Rectangle.position 只接受 (x, y)（text.Label 才收三元组）
        self.panel.position = (10, h - 226)
        self.panel.width, self.panel.height = 250, 192
        self.stat.position = (18, h - 42, 0)
        self.msg_bg.position = (0, 88)
        self.msg_bg.width, self.msg_bg.height = w, 92
        self.big.position = (w / 2, h / 2, 0)

    def update(self, game):
        p = game.player
        hunger = 'full' if p.hunger > 600 else (
            'hungry' if p.hunger > 120 else 'STARVING' if p.hunger > 0 else 'fainting')
        weap = (p.weapon['name'] + (f' +{p.weapon_ench}' if p.weapon_ench else '')
                if p.weapon else 'bare hands')
        arm = (p.armor['name'] + (f' +{p.armor_ench}' if p.armor_ench else '')
               if p.armor else 'no armor')
        staffs = '  '.join(
            f'{ITEMS[k]["name"].replace("staff of ", "st.")} '
            f'x{p.staff_charges.get(k, 0)}'
            for k, n in p.inventory.items()
            if n > 0 and ITEMS[k]['kind'] == 'staff')
        ring = p.ring['name'] if p.ring else 'no ring'
        am = '  ★Amulet' if p.has_amulet else ''
        self.stat.text = (f'HP  {max(0, p.hp)}/{p.max_hp}\n'
                          f'Str {p.strength}   Depth {game.depth}/{DEPTH_MAX}\n'
                          f'Hunger {hunger}   Gold {p.gold}\n'
                          f'{weap}\n{arm}{am}\n'
                          f'{staffs if staffs else "no staff"}\nRing: {ring}')
        for i, lab in enumerate(self.msgs):
            idx = len(game.messages) - 4 + i
            if 0 <= idx < len(game.messages):
                txt, col = game.messages[idx]
                lab.text = '> ' + txt
                lab.color = (int(col[0] * 255), int(col[1] * 255),
                             int(col[2] * 255), 235)
            else:
                lab.text = ''
        if game.state == 'dead':
            self.big.text = f'YOU DIED   score {game.score}'
            self.big.color = (255, 90, 80, 255)
        elif game.state == 'won':
            self.big.text = f'*** VICTORY ***   score {game.score}'
            self.big.color = (255, 220, 120, 255)
        else:
            self.big.text = ''

    def draw(self):
        self.batch.draw()


# ================================================================ App ==
class BroguewardApp(DemoApp):
    TITLE = 'brogueward · 《渊境》——3dworld 复刻 Brogue（26 层取 Amulet 回地表）'
    HELP = ('移动 HJKLYUBN/方向键/小键盘   1-8 使用物品   Z 施法   SPACE 等待   '
            'R 重开   拖拽旋转   滚轮缩放   ESC 退出'
            '   （--flag record 录像 / replay=PATH 回放）')
    SHOTS = ((3.0, 'docs/brogueward_01.png'),
             (14.0, 'docs/brogueward_02.png'),
             (26.0, 'docs/brogueward_03.png'))

    def __init__(self, width=1280, height=720, args=None):
        self._seed = None
        self._demo = True                   # smoke/shot/bench 下自动演示
        self._record = False
        self._replay = None                 # 回放文件内容（dict）
        if args is not None:
            for f in args.flags:
                if f.startswith('seed='):
                    self._seed = int(f.split('=', 1)[1])
                if f == 'play':
                    self._demo = False
                if f == 'record':
                    self._record = True
                if f.startswith('replay='):
                    import json
                    with open(f.split('=', 1)[1], 'r', encoding='utf-8') as fh:
                        self._replay = json.load(fh)
        self.game = None
        self.cam_yaw = 180.0
        self.cam_pitch = 72.0               # 高俯角（Brogue 的 top-down 口味）
        self.cam_dist = 21.0
        self.cam_target = (0.0, 0.0, 0.0)
        self._demo_clock = 0.0
        self._zoom_all = False
        super().__init__(width, height, args)

    # ------------------------------------------------------------ setup --
    def setup(self):
        self.program = _make_program()
        self.new_run(self._seed)
        self.batch_dyn = DynamicBatch(self.program)
        self.batch_fx = DynamicBatch(self.program)
        self.hud_game = GameHud()
        self.hud_game.layout(self.window.width, self.window.height)
        # 录像/回放状态（在 HUD 就绪后初始化）
        self._rec_runs = []
        self._rec_run = None
        if self._record:
            self._rec_start(self.game.seed)
        if self._replay is not None:
            self._demo = False
            self._rp = {'runs': list(self._replay.get('runs', [])), 'idx': -1,
                        'pos': 0, 'actions': [], 'expected': None,
                        'report': [], 'done': False}
            self._replay_next_run()

    def new_run(self, seed=None):
        if hasattr(self, 'terrain') and self.terrain is not None:
            self.terrain.delete()
        self.game = Game(seed)
        self.game.on_level_changed = self._rebuild_terrain
        self.terrain = TerrainMesh(self.program, self.game)
        wx, wz = _world(self.game.player.x, self.game.player.y)
        self.cam_target = (wx, 0.5, wz)

    # ------------------------------------------------------- 录像/回放 --
    def _rec_start(self, seed):
        self._rec_run = {'seed': seed, 'actions': []}

    def _rec_close(self):
        if self._rec_run is not None:
            g = self.game
            if g.state != 'play':
                self._rec_run['result'] = {
                    'state': g.state, 'turn': g.turn, 'depth': g.depth,
                    'hp': max(0, g.player.hp), 'score': g.score,
                    'gold': g.player.gold}
            self._rec_runs.append(self._rec_run)
            self._rec_run = None

    def _rec_save(self):
        if not self._record:
            return
        import json
        d = os.path.join(PROJECT_ROOT, 'assets', 'brogueward_runs')
        os.makedirs(d, exist_ok=True)
        runs = list(self._rec_runs) + ([self._rec_run] if self._rec_run else [])
        path = os.path.join(d, f'record_{self.game.seed}.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'runs': runs}, fh, ensure_ascii=False)
        print(f'[record] saved {path} ({len(runs)} run(s))', flush=True)

    def do(self, act):
        """统一动作入口：键盘 / 演示 AI / 回放全部从这里走（录像在此记账）。"""
        g = self.game
        kind = act[0]
        if kind == 'new':
            self.new_run(act[1] if len(act) > 1 else None)
            if self._record:
                self._rec_close()
                self._rec_start(self.game.seed)
        elif kind == 'move':
            g.try_move(act[1], act[2])
        elif kind == 'wait':
            g.wait()
        elif kind == 'use':
            g.use_slot(act[1])
        elif kind == 'fire':
            g.fire_staff()
        if self._record and kind != 'new' and self._rec_run is not None:
            self._rec_run['actions'].append(act)
        self._after_turn()
        if self._record and g.state != 'play' and self._rec_run is not None:
            self._rec_close()
            self._rec_save()

    def _replay_next_run(self):
        rp = self._rp
        rp['idx'] += 1
        if rp['idx'] >= len(rp['runs']):
            rp['done'] = True
            for idx, verdict in rp['report']:
                print(f'[replay] run#{idx}: {verdict}', flush=True)
            n_ok = sum(1 for _, v in rp['report'] if v == 'MATCH')
            print(f'[replay] {n_ok}/{len(rp["report"])} runs match', flush=True)
            return
        run = rp['runs'][rp['idx']]
        rp['actions'] = run.get('actions', [])
        rp['pos'] = 0
        rp['expected'] = run.get('result')
        self.do(['new', run.get('seed')])

    def _replay_step(self):
        """回放节拍：按 DEMO_STEP 逐动作复演，终局对账后进下一 run。"""
        rp = self._rp
        if rp['pos'] < len(rp['actions']):
            act = rp['actions'][rp['pos']]
            rp['pos'] += 1
            self.do(act)
            if self.game.state != 'play':
                g = self.game
                got = {'state': g.state, 'turn': g.turn, 'depth': g.depth,
                       'hp': max(0, g.player.hp), 'score': g.score,
                       'gold': g.player.gold}
                exp = rp.get('expected')
                if exp is None:
                    rp['report'].append((rp['idx'], 'n/a (no recorded result)'))
                else:
                    ok = all(got.get(k) == v for k, v in exp.items())
                    rp['report'].append(
                        (rp['idx'], 'MATCH' if ok else f'MISMATCH {got} != {exp}'))
                self._replay_next_run()
        else:
            # 动作耗尽仍未终局（录制端中途退出的残局）
            rp['report'].append((rp['idx'], 'n/a (actions exhausted)'))
            self._replay_next_run()

    def _rebuild_terrain(self):
        """换层回调：删旧地形网格、按新 level 重建（光照/几何同步）。"""
        if self.terrain is not None:
            self.terrain.delete()
        self.terrain = TerrainMesh(self.program, self.game)

    # ------------------------------------------------------------ 回合驱 --
    def _after_turn(self):
        self.game.update_fov()
        self.terrain.refresh()
        self.hud_game.update(self.game)

    def _demo_step(self):
        """演示 AI：逻辑在模块级 demo_action()（纯函数，回放共用同一套）。"""
        self.do(demo_action(self.game))

    # ------------------------------------------------------------ update --
    def update(self, dt):
        g = self.game
        # 实体动画插值
        for m in g.level.monsters:
            m.anim_t = min(1.0, m.anim_t + dt * 5.0)
            m.bump = max(0.0, m.bump - dt * 4.0)
            m.rx = (m.px + (m.x - m.px) * m.anim_t)
            m.ry = (m.py + (m.y - m.py) * m.anim_t)
        p = g.player
        p.anim_t = min(1.0, p.anim_t + dt * 5.0)
        p.bump = max(0.0, p.bump - dt * 4.0)
        p.rx = p.px + (p.x - p.px) * p.anim_t
        p.ry = p.py + (p.y - p.py) * p.anim_t
        # 相机跟随
        wx, wz = _world(p.x, p.y)
        self.cam_target = (wx, 0.5, wz)
        # 演示步进 / 回放步进
        if self._replay is not None:
            if not self._rp['done']:
                self._demo_clock += dt
                if self._demo_clock >= DEMO_STEP:
                    self._demo_clock = 0.0
                    self._replay_step()
        elif self._demo:
            self._demo_clock += dt
            if self._demo_clock >= DEMO_STEP:
                self._demo_clock = 0.0
                self._demo_step()

    # ------------------------------------------------------------ render --
    def render(self):
        g = self.game
        # 相机
        yaw = math.radians(self.cam_yaw)
        pitch = math.radians(self.cam_pitch)
        if self._zoom_all:
            tgt = (MAP_W * TILE / 2, 0.0, MAP_H * TILE / 2)
            dist = 46.0
        else:
            tgt = self.cam_target
            dist = self.cam_dist
        eye = (tgt[0] + dist * math.cos(pitch) * math.cos(yaw),
               tgt[1] + dist * math.sin(pitch),
               tgt[2] + dist * math.cos(pitch) * math.sin(yaw))
        self.window.view = Mat4.look_at(Vec3(*eye), Vec3(*tgt), Vec3(0.0, 1.0, 0.0))

        self.program.use()
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.terrain.draw()
        self.batch_dyn.begin()
        self._draw_entities(g, self.batch_dyn, opaque=True)
        self.batch_dyn.end()
        # 半透明特效（气/火）：blend 开 + 深度只读
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDepthMask(gl.GL_FALSE)
        self.batch_fx.begin()
        self._draw_entities(g, self.batch_fx, opaque=False)
        self.batch_fx.end()
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)

    def _draw_entities(self, g, batch, opaque):
        lv = g.level
        t = self.time
        if opaque:
            # ---- 物品 ----
            for (x, y), stack in lv.items.items():
                if not lv.explored[y][x]:
                    continue
                wx, wz = _world(x, y)
                lit = 0.35 if (x, y) not in g.fov else 1.0
                for it in stack:
                    sx, hy, sz, hh = ITEM_SHAPE[it['kind']]
                    col = tuple(min(1.0, c * lit + 0.05) for c in it['color'])
                    if it['kind'] in ('gold', 'amulet'):
                        bob = 0.10 * math.sin(t * 3.0 + x * 1.7)
                        batch.add_pyramid(wx, hy + bob, wz, sx, hh, sz, col)
                    else:
                        batch.add_cube(wx, hy, wz, sx, hh, sz, col)
            # ---- 玩家 ----
            p = g.player
            wx, wz = _world(p.x, p.y)
            ex = wx + (p.rx - p.x) * TILE
            ez = wz + (p.ry - p.y) * TILE
            kcol = (0.95, 0.85, 0.65)                  # 冒险家暖白
            if p.hp <= p.max_hp * 0.3:
                kcol = (0.95, 0.45, 0.35)
            batch.add_cube(ex, 0.0, ez, 0.22, 0.30, 0.18, kcol)
            batch.add_pyramid(ex, 0.60, ez, 0.15, 0.12, 0.15, (0.98, 0.92, 0.78))
            if p.has_amulet:
                batch.add_pyramid(ex, 0.85 + 0.08 * math.sin(t * 4), ez,
                                  0.10, 0.10, 0.10, (0.6, 0.95, 1.0))
            # ---- 怪物 ----
            for m in lv.monsters:
                if not lv.explored[m.y][m.x]:
                    continue
                lit = 0.30 if (m.x, m.y) not in g.fov else 1.0
                ex = _world(m.x, m.y)[0] + (m.rx - m.x) * TILE
                ez = _world(m.x, m.y)[1] + (m.ry - m.y) * TILE
                sx, sy, sz = MONSTER_SHAPE[m.tpl['shape']]
                col = tuple(min(1.0, c * lit) for c in m.tpl['color'])
                if m.tpl['shape'] == 'wraith':
                    batch.add_pyramid(ex, 0.0, ez, sx, sy, sz, col,
                                      alpha=0.95 if opaque else 0.8)
                else:
                    batch.add_cube(ex, 0.0, ez, sx, sy, sz, col)
                if m.charm_t > 0:       # 魅惑盟友：头顶金色小锥
                    batch.add_pyramid(ex, sy + 0.28 + 0.05 * math.sin(t * 4),
                                      ez, 0.09, 0.09, 0.09, (1.0, 0.85, 0.3))
        else:
            # ---- 火 ----
            for y in range(MAP_H):
                for x in range(MAP_W):
                    f = lv.fire[y][x]
                    if f > 0:
                        wx, wz = _world(x, y)
                        flick = 0.5 + 0.5 * math.sin(t * 13.0 + x * 3.1 + y * 1.3)
                        h = 0.25 + 0.35 * flick
                        batch.add_cube(wx, 0.0, wz, 0.28, h, 0.28,
                                       (1.0, 0.45 + 0.3 * flick, 0.15),
                                       alpha=0.55 + 0.25 * flick)
            # ---- 毒气 ----
            for y in range(MAP_H):
                for x in range(MAP_W):
                    gc = lv.gas[y][x]
                    if gc > 0.04:
                        wx, wz = _world(x, y)
                        h = 0.2 + min(0.8, gc) * 0.9
                        drift = 0.05 * math.sin(t * 2.0 + x * 2.0 + y)
                        batch.add_cube(wx + drift, 0.02, wz, 0.45, h, 0.45,
                                       (0.35, 0.9, 0.3),
                                       alpha=min(0.55, 0.2 + gc * 0.25))

    # ---------------------------------------------------------- 输入 --
    def _on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.SPACE:
            # 回合制语义：SPACE = 等待一回合（覆盖基类"暂停"）
            self.do(['wait'])
            return pyglet.event.EVENT_HANDLED
        return super()._on_key_press(symbol, modifiers)

    def on_key(self, symbol, modifiers):
        k = pyglet.window.key
        g = self.game
        if self._replay is not None:        # 回放模式不接玩家输入
            return
        if symbol == k.R:
            self.do(['new', None])
            return
        if g.state != 'play':
            if symbol == k.ENTER:
                self.do(['new', None])
            return
        moves = {
            k.H: (-1, 0), k.J: (0, 1), k.K: (0, -1), k.L: (1, 0),
            k.Y: (-1, -1), k.U: (1, -1), k.B: (-1, 1), k.N: (1, 1),
            k.LEFT: (-1, 0), k.RIGHT: (1, 0), k.UP: (0, -1), k.DOWN: (0, 1),
            k.NUM_1: (-1, 1), k.NUM_2: (0, 1), k.NUM_3: (1, 1),
            k.NUM_4: (-1, 0), k.NUM_6: (1, 0), k.NUM_7: (-1, -1),
            k.NUM_8: (0, -1), k.NUM_9: (1, -1), k.NUM_5: (0, 0),
        }
        if symbol in moves:
            dx, dy = moves[symbol]
            if (dx, dy) == (0, 0):
                self.do(['wait'])
            else:
                self.do(['move', dx, dy])
            return
        if symbol == k.Z:
            self.do(['fire'])
            return
        # 数字键（主键盘）使用背包
        slot = {k._1: 0, k._2: 1, k._3: 2, k._4: 3, k._5: 4,
                k._6: 5, k._7: 6, k._8: 7}.get(symbol)
        if slot is not None:
            self.do(['use', slot])

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.cam_yaw -= dx * 0.4
        self.cam_pitch = min(85.0, max(15.0, self.cam_pitch + dy * 0.3))

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.cam_dist = min(40.0, max(7.0, self.cam_dist - scroll_y * 1.5))
        self._zoom_all = False

    # ------------------------------------------------------------ HUD --
    def _draw_hud(self):
        super()._draw_hud()
        w, h = self.window.width, self.window.height
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDisable(gl.GL_DEPTH_TEST)
        self.window.projection = Mat4.orthogonal_projection(0, w, 0, h, -255, 255)
        self.window.view = Mat4()
        self.hud_game.draw()
        self.window.projection = self.perspective
        gl.glEnable(gl.GL_DEPTH_TEST)

    def status_text(self):
        g = self.game
        if self._replay is not None:
            mode = 'replay'
        else:
            mode = 'demo' if self._demo else 'play'
        st = {'play': '', 'dead': '  [DEAD - R to restart]',
              'won': '  [VICTORY!]'}.get(g.state, '')
        return (f'depth {g.depth}/{DEPTH_MAX}  turn {g.turn}  {mode}{st}   '
                f'{self.fps:5.1f} fps')


def demo_action(g):
    """演示 AI 的下一个动作（纯函数：只依赖 game 状态与 g.rng）。

    键盘、演示、回放共用同一动作词汇表（move/wait/use/fire/new），
    录像把动作序列记账成 JSON，回放按序复演即可逐帧对上。
    """
    if g.state != 'play':
        return ['new', None]
    p = g.player
    lv = g.level
    adj = next((m for m in lv.monsters
                if abs(m.x - p.x) <= 1 and abs(m.y - p.y) <= 1), None)
    if adj:
        if adj.charm_t > 0:
            return ['wait']                 # 友军挡路：等它打完
        dx, dy = _toward(p, adj)
        return ['move', dx, dy]
    # 有充能的法杖 + 视野内有敌人 → 概率施法（AI 用独立 ai_rng 流，
    # 不消耗游戏 rng——回放时不重跑 AI，两种子流互不干扰）
    if g.ai_rng.random() < 0.3:
        key = next((k for k, n in p.inventory.items()
                    if n > 0 and ITEMS[k]['kind'] == 'staff'
                    and p.staff_charges.get(k, 0) > 0), None)
        if key:
            rsq = ITEMS[key]['srange'] ** 2
            if any(m.charm_t <= 0 and g.fov.get((m.x, m.y), 1 << 30) <= rsq
                   for m in lv.monsters):
                return ['fire']
    if lv.turns < 10 and lv.depth < DEPTH_MAX:
        # 探索期：随机走（可走方向里挑一个）
        opts = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (dx or dy) and lv.walkable(p.x + dx, p.y + dy)
                and not (dx and dy
                         and not (lv.walkable(p.x + dx, p.y)
                                  and lv.walkable(p.x, p.y + dy)))]
        if opts:
            dx, dy = g.ai_rng.choice(opts)
            return ['move', dx, dy]
    goal = lv.down if not p.has_amulet else lv.up
    nxt = _bfs_next(lv, (p.x, p.y), goal)
    if nxt:
        return ['move', nxt[0] - p.x, nxt[1] - p.y]
    return ['wait']


def _toward(p, m):
    """从 p 指向 m 的攻击步（对角优先）。"""
    dx = (m.x > p.x) - (m.x < p.x)
    dy = (m.y > p.y) - (m.y < p.y)
    return (dx, dy)


def _bfs_next(lv, start, goal):
    """BFS 最短路的第一步（演示 AI 用；地图小，整层重算无压力）。"""
    if start == goal:
        return None
    from collections import deque
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = cur[0] + dx, cur[1] + dy
            if (nx, ny) in prev or not lv.passable(nx, ny):
                continue
            if dx and dy:                       # 同款对角穿角禁止
                if not (lv.walkable(cur[0] + dx, cur[1])
                        and lv.walkable(cur[0], cur[1] + dy)):
                    continue
            prev[(nx, ny)] = cur
            q.append((nx, ny))
    if goal not in prev:
        return None
    node = goal
    while prev[node] != start and prev[node] is not None:
        node = prev[node]
    return node if prev[node] == start else None


if __name__ == '__main__':
    sys.exit(run(BroguewardApp))

