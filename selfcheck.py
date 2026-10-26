#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3dworld-demos 无头自检：程序化地形 + 体素面剔除（纯逻辑，不开窗口）。

为什么不用常规包导入
--------------------
本仓是 3dworld 引擎的演示集：`examples/*.py` 在 import 期就会拉 pyglet 2 的
`pyglet.graphics.shader` / `pyglet.math.Vec3`，演示能力本体在同级引擎仓
同级引擎仓（默认 ``../3dworld``）的 ``world3d.logic.*``。本机系统 Python 的 pyglet 是
1.5.31，而 ``world3d/__init__.py`` 会 ``from world3d.engine import *``（引擎门面
用到 pyglet 2 的 Quaternion），所以常规 ``import world3d.logic.terrain`` 在包入口
就会炸，更别说无窗口运行。

因此自检走「纯逻辑模块」路线：按文件路径直接加载
``world3d/logic/terrain.py`` 与 ``world3d/logic/voxel.py``（这两个模块只 import
math/array，正是引擎仓 tests 直接单测的对象），既不碰 pyglet，也不初始化 GUI。

验证点
------
1. 地形：同 (坐标, seed) 高程逐位可复现、有明显起伏；分层配色是 [0,1] 内的
   RGB（深水偏蓝、雪线更亮）；emit_row 逐行发射的顶点/颜色数量正确。
2. 体素：高度图可复现且落在 1..MAXH；面剔除后三角形数明显少于不剔除；
   分段网格化（z0/z1 追加）与整块网格化逐位一致（渐进重建契约）。
"""
from __future__ import annotations

import array
import importlib.util
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True          # 不在项目/引擎仓里留 .pyc
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent


def _locate_engine_logic() -> Path:
    """定位 world3d 的 logic 目录：优先 pip 装好的引擎包，回退同级引擎源码树。

    演示仓 3859c66 之后不再做 sys.path 引导，引擎一律由 pip 提供
    （``pip install -r requirements.txt``）；只有本机同时 checkout 了引擎仓
    （默认与演示仓同级）时才走源码树回退。

    这里用 ``find_spec`` 只定位、**不导入**：``import world3d`` 会执行包入口
    （拉起 pyglet 2 的 Quaternion），在 pyglet 1.x 环境里直接炸 —— 正是本自检
    要绕开的那条路。
    """
    spec = importlib.util.find_spec("world3d")
    if spec is not None and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            candidate = Path(loc).resolve() / "logic"
            if candidate.is_dir():
                return candidate
    src = ROOT.parent / "3dworld" / "world3d" / "logic"
    check(src.is_dir(),
          f"找不到引擎逻辑目录：既没有已安装的 world3d，也没有 {src}"
          "（先 pip install -r requirements.txt）")
    return src


class CheckError(RuntimeError):
    pass


def check(condition, message):
    if not condition:
        raise CheckError(message)


ENGINE_LOGIC = _locate_engine_logic()


def load_logic(module_name: str, filename: str):
    """按文件加载引擎纯逻辑模块（绕过会 import 渲染引擎的 world3d 包入口）。"""
    path = ENGINE_LOGIC / filename
    check(path.is_file(), f"找不到引擎逻辑模块：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    check(spec is not None and spec.loader is not None, f"无法为 {path} 建立导入器")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def check_terrain(terrain):
    # --- 高程：确定性 + 起伏 ---
    h1 = terrain.height_at(3.0, 7.0, 11)
    h2 = terrain.height_at(3.0, 7.0, 11)
    check(h1 == h2, f"地形高程不确定：同 (坐标, seed) 得到 {h1!r} / {h2!r}")
    check(math.isfinite(h1), f"地形高程不是有限数：{h1!r}")

    samples = [terrain.height_at(x, y * 3.7, 11)
               for x in range(-200, 201, 40)
               for y in range(-50, 51, 20)]
    check(max(samples) - min(samples) > 1.0, "地形高程缺少起伏（整片几乎同高）")
    check(all(-60.0 < h < 120.0 for h in samples),
          "地形高程超出合理区间：%r..%r" % (min(samples), max(samples)))

    # --- 配色：RGB 分量域内 + 深水偏蓝 / 雪线更亮 ---
    water = terrain.terrain_color(-5.0, 0.10, 0.60, 0.50)
    snow = terrain.terrain_color(32.0, 0.10, 0.90, 0.50)
    for label, color in (("深水", water), ("雪线", snow)):
        check(len(color) == 3, f"{label}配色应为 RGB 三元组，实际 {color!r}")
        check(all(0.0 <= c <= 1.0 for c in color),
              f"{label}配色分量越界：{color!r}")
    check(water[2] > water[0], f"深水配色应偏蓝，实际 {water!r}")
    check(sum(snow) > sum(water), f"雪线配色应比深水更亮：{snow!r} vs {water!r}")

    # --- emit_row：一个四边形 -> 6 顶点（18 个坐标 float + 18 个 RGB float）---
    heights = [1.0, 2.0, 3.0, 4.0]
    colors = [(0.1, 0.2, 0.3), (0.2, 0.3, 0.4), (0.3, 0.4, 0.5), (0.4, 0.5, 0.6)]
    pos: list[float] = []
    col: list[float] = []
    terrain.emit_row(pos, col, heights, colors, n=2, step=1.0, ox=0.0, oz=0.0, j=0)
    check(len(pos) == 18, f"emit_row 顶点坐标数量错误：{len(pos)}（应为 18）")
    check(len(col) == 18, f"emit_row 顶点颜色数量错误：{len(col)}（应为 18）")

    return h1, water, snow


def check_voxel(voxel):
    # --- 高度图：规模、取值范围、可复现 ---
    heights = voxel.build_heights(7)
    check(len(heights) == voxel.W * voxel.W,
          f"高度图规模错误：{len(heights)}（应为 {voxel.W * voxel.W}）")
    check(min(heights) >= 1 and max(heights) <= voxel.MAXH,
          f"高度图越界：{min(heights)}..{max(heights)}（应在 1..{voxel.MAXH}）")
    check(heights == voxel.build_heights(7), "体素高度图不确定：同 seed 两次结果不同")

    # --- 面剔除：三角形数下降，且缓冲长度与面数严格对应 ---
    pos_cull, col_cull, faces_cull = voxel.build_mesh(heights, 7, True)
    pos_full, col_full, faces_full = voxel.build_mesh(heights, 7, False)
    check(faces_cull > 0 and faces_full > 0, "网格化没有产出任何面")
    check(faces_full > faces_cull,
          f"面剔除无效：剔除 {faces_cull} 面，未剔除 {faces_full} 面")
    check(len(pos_cull) == faces_cull * 6 * 3,
          f"剔除网格坐标数不符：{len(pos_cull)} != {faces_cull * 6 * 3}")
    check(len(col_cull) == faces_cull * 6 * 4,
          f"剔除网格颜色数不符：{len(col_cull)} != {faces_cull * 6 * 4}")
    check(len(pos_full) == faces_full * 6 * 3,
          f"全量网格坐标数不符：{len(pos_full)} != {faces_full * 6 * 3}")
    check(faces_full / faces_cull > 1.5,
          f"剔除率异常：{faces_full}/{faces_cull} = {faces_full / faces_cull:.2f}")

    # --- 分段网格化 == 整块网格化（渐进重建的逐位契约）---
    seg_pos = array.array("f")
    seg_col = array.array("f")
    step = 7
    for z0 in range(0, voxel.W, step):
        voxel.build_mesh(heights, 7, True, seg_pos, seg_col, z0, min(voxel.W, z0 + step))
    check(len(seg_pos) == len(pos_cull) and list(seg_pos) == list(pos_cull),
          "分段重建的顶点坐标与整块重建不一致")
    check(len(seg_col) == len(col_cull) and list(seg_col) == list(col_cull),
          "分段重建的顶点颜色与整块重建不一致")

    return len(heights), faces_cull, faces_full


def main():
    check(ENGINE_LOGIC.is_dir(), f"找不到同级引擎仓的逻辑目录：{ENGINE_LOGIC}")
    terrain = load_logic("_selfcheck_terrain", "terrain.py")
    voxel = load_logic("_selfcheck_voxel", "voxel.py")

    h1, water, snow = check_terrain(terrain)
    height_count, faces_cull, faces_full = check_voxel(voxel)

    print(
        "selfcheck ok: terrain height determinism (height_at=%.2f) + layered colors; "
        "voxel heights + face culling %d->%d faces + segmented mesh bitwise equal (%d columns)"
        % (h1, faces_full, faces_cull, height_count)
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - 自检入口需要把任何失败转成 FAIL + 退出码 1
        print(f"FAIL: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
