# -*- coding: utf-8 -*-
"""生成《谪仙 / Posterward》的金饰资产：assets/posterward_ornaments.glb。

三个静态 PBR 金饰节点（环 / 绸带环 / 玉璧），供 posterward 经引擎真实
glTF 2.0 加载栈（``world3d.engine.gltf.GLTF`` → ``build_scene_ex`` →
``Mesh``）消费——即 gltfward《匣境》的同一通道，资产由本脚本程序化生成、
可复现（产物入库，生成器也入库）。

生成::

    python scripts/make_posterward_glb.py        # 幂等，覆盖写

glB 结构：JSON chunk（UTF-8，4 字节对齐补空格）+ BIN chunk（4 字节对齐补
\\0）；每 mesh 一个 PRIMITIVE（POSITION + NORMAL + indices，uint32），
共享一个金属度 1.0 的 gold 材质（无贴图槽，与引擎"材质参数直读"口径一致）。
"""
from __future__ import annotations

import json
import math
import os
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'assets', 'posterward_ornaments.glb')


def torus(ring_r: float, tube_r: float, seg_ring: int, seg_tube: int,
          scale=(1.0, 1.0, 1.0)) -> tuple[list[float], list[float], list[int]]:
    """环面网格：中心环在 XZ 平面（y 为环轴），法线朝外。

    scale 用于把圆环压成绸带环 / 扁玉璧（非均匀缩放后法线按逆转置修正——
    对纯缩放可用 1/s 分量近似，这里直接逐顶点重算解析法线：环面参数化下
    法线方向与缩放无关的结论不成立，故用数值法：沿两参数方向的叉积）。
    """
    positions, normals, indices = [], [], []
    sx, sy, sz = scale

    def pt(i: int, j: int) -> tuple[float, float, float]:
        u = 2 * math.pi * i / seg_ring
        v = 2 * math.pi * j / seg_tube
        cu, su = math.cos(u), math.sin(u)
        # 圆环在 XZ 平面：环向 (cu, 0, su)，管向沿 (cv*cu, sv, cv*su)；
        # 非均匀缩放直接作用在各分量上（绸带环 = X 拉长 + Y 压扁）
        x = (ring_r * cu) * sx + tube_r * math.cos(v) * cu * sx
        y = tube_r * math.sin(v) * sy
        z = (ring_r * su) * sz + tube_r * math.cos(v) * su * sz
        return x, y, z

    def normal(i: int, j: int) -> tuple[float, float, float]:
        # 数值法：∂p/∂u × ∂p/∂v（解析式对非均匀缩放繁琐，二阶差分足够）
        e = 0.01
        i0 = (i + seg_ring - 1) % seg_ring
        i2 = (i + 1) % seg_ring
        j0 = (j + seg_tube - 1) % seg_tube
        j2 = (j + 1) % seg_tube
        pu = [a - b for a, b in zip(pt(i2, j), pt(i0, j))]
        pv = [a - b for a, b in zip(pt(i, j2), pt(i, j0))]
        n = (pu[1] * pv[2] - pu[2] * pv[1],
             pu[2] * pv[0] - pu[0] * pv[2],
             pu[0] * pv[1] - pu[1] * pv[0])
        ln = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) or 1.0
        return n[0] / ln, n[1] / ln, n[2] / ln

    for i in range(seg_ring):
        for j in range(seg_tube):
            x, y, z = pt(i, j)
            positions += [x, y, z]
            nx, ny, nz = normal(i, j)
            normals += [nx, ny, nz]
    for i in range(seg_ring):
        for j in range(seg_tube):
            a = i * seg_tube + j
            b = ((i + 1) % seg_ring) * seg_tube + j
            c = ((i + 1) % seg_ring) * seg_tube + (j + 1) % seg_tube
            d = i * seg_tube + (j + 1) % seg_tube
            indices += [a, b, d, b, c, d]
    return positions, normals, indices


def build_meshes() -> list[tuple[str, list, list, list]]:
    """三个金饰：正环 / 绸带环（X 向拉长压扁）/ 玉璧（扁厚环）。"""
    return [
        ('ring', *torus(1.0, 0.10, 64, 16)),
        ('ribbon', *torus(1.0, 0.05, 64, 12, scale=(1.45, 0.55, 1.0))),
        ('bisk', *torus(0.9, 0.30, 56, 14, scale=(1.0, 1.0, 0.18))),
    ]


def write_glb(path: str) -> None:
    meshes = build_meshes()
    # 二进制段：逐 mesh 写 positions/normals/indices
    bin_data = bytearray()
    buffer_views, accessors = [], []
    mesh_defs = []

    def add_view(data: bytes, target: int) -> int:
        pad = (4 - len(bin_data) % 4) % 4
        bin_data.extend(b'\x00' * pad)
        idx = len(buffer_views)
        buffer_views.append({'buffer': 0, 'byteOffset': len(bin_data),
                             'byteLength': len(data), 'target': target})
        bin_data.extend(data)
        return idx

    for name, pos, nor, idx in meshes:
        pv = add_view(struct.pack(f'<{len(pos)}f', *pos), 34962)
        pa = len(accessors)
        accessors.append({
            'bufferView': pv, 'componentType': 5126, 'count': len(pos) // 3,
            'type': 'VEC3',
            'min': [min(pos[k::3]) for k in range(3)],
            'max': [max(pos[k::3]) for k in range(3)]})
        nv = add_view(struct.pack(f'<{len(nor)}f', *nor), 34962)
        na = len(accessors)
        accessors.append({'bufferView': nv, 'componentType': 5126,
                          'count': len(nor) // 3, 'type': 'VEC3'})
        iv = add_view(struct.pack(f'<{len(idx)}I', *idx), 34963)
        ia = len(accessors)
        accessors.append({'bufferView': iv, 'componentType': 5125,
                          'count': len(idx), 'type': 'SCALAR'})
        mesh_defs.append({
            'name': name,
            'primitives': [{'attributes': {'POSITION': pa, 'NORMAL': na},
                            'indices': ia, 'material': 0}]})

    gltf = {
        'asset': {'version': '2.0',
                  'generator': '3dworld-demos/scripts/make_posterward_glb.py'},
        'scene': 0,
        'scenes': [{'nodes': [0, 1, 2]}],
        'nodes': [{'name': n, 'mesh': i} for i, (n, *_r) in enumerate(meshes)],
        'meshes': mesh_defs,
        'materials': [{
            'name': 'gold',
            'pbrMetallicRoughness': {
                'baseColorFactor': [0.92, 0.70, 0.28, 1.0],
                'metallicFactor': 1.0,
                'roughnessFactor': 0.25},
        }],
        'buffers': [{'byteLength': len(bin_data)}],
        'bufferViews': buffer_views,
        'accessors': accessors,
    }

    json_bytes = json.dumps(gltf, ensure_ascii=False,
                            separators=(',', ':')).encode('utf-8')
    pad_j = (4 - len(json_bytes) % 4) % 4
    json_bytes += b' ' * pad_j
    pad_b = (4 - len(bin_data) % 4) % 4
    bin_data.extend(b'\x00' * pad_b)

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, total))          # glTF 头
        f.write(struct.pack('<II', len(json_bytes), 0x4E4F534A))    # JSON chunk
        f.write(json_bytes)
        f.write(struct.pack('<II', len(bin_data), 0x004E4942))      # BIN chunk
        f.write(bytes(bin_data))
    return total


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    total = write_glb(OUT)
    size = os.path.getsize(OUT)
    print(f'[make_posterward_glb] wrote {OUT} ({size} bytes, glb total {total})')
    # 自校验：结构能被引擎加载栈吃下（无 GL 依赖的部分）
    try:
        from world3d.engine.gltf import GLTF
        g = GLTF.load(OUT)
        meshes = g.build_scene_ex()
        n_prim = sum(len(m['primitives']) for m in g.json['meshes'])
        print(f'[make_posterward_glb] engine GLTF.load ok: '
              f'{len(g.json["meshes"])} meshes / {n_prim} primitives / '
              f'pending={len(meshes[1])}')
    except Exception as exc:                                # noqa: BLE001
        print(f'[make_posterward_glb] engine load check SKIPPED: {exc}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
