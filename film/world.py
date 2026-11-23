# -*- coding: utf-8 -*-
"""《造界》世界层 —— 把 12 幕拼成一个连续空间。

为什么不再用 concat
------------------
`concat.py` 是把 12 个 mp4 用 xfade 粘起来：段间是**硬切/淡入**，世界不连续。
这一层改成：12 个区域的**场景节点合并进同一个 scene**，相机走一条**连续路径**，
段间用"飞行过渡"（4 秒）连起来 —— 观众看到的是"一路走下去"，不是 12 段分镜。

区域布局
--------
沿 **-Z 轴**线性排列，间距 300m（第 I 幕在 z=0，第 XII 幕在 z=-3300）。
300m 的依据：区域自身尺寸 60~160m，300m 保证不重叠；4 秒过渡 ≈ 75 m/s，
在雾里看起来像"高速穿行"而不是瞬移。

时长
----
`Σ各幕 duration + 11 × 过渡`。各幕 duration 之和 733s，加 11×4=44 → **777s**。

用法::

    python film/world.py                 # -> film/scenes/world.scene.json + world.shot.json
    python film/world.py --check
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILM_DIR = HERE
ROOT = os.path.dirname(FILM_DIR)

# (幕 id, 各幕 duration)。顺序即世界里的推进顺序。
REGIONS = (
    ('01_void', 24.0),
    ('02_terrain', 66.0),
    ('03_city', 80.0),
    ('04_machine', 74.0),
    ('05_crystal', 64.0),
    ('06_cave', 64.0),
    ('07_swarm', 58.0),
    ('08_body', 58.0),
    ('09_field', 54.0),
    ('10_void', 80.0, '10_void2'),   # 文件名为 10_void2（早期避与 01_void 重名）
    ('11_loop', 64.0),
    ('12_finale', 47.0),
)

SPACING = 800.0          # 区域间距（m）：必须 > 相机 far（400），否则会看到邻区结构
BLEND = 6.0              # 段间飞行过渡（s）；与 score.py 的 WORLD_BLEND 必须一致

SCENE_OUT = os.path.join(FILM_DIR, 'scenes', 'world.scene.json')
SHOT_OUT = os.path.join(FILM_DIR, 'shots', 'world.shot.json')

#: 世界级环境（12 区共用一个 —— 这也顺带解决了跨幕调色不一致）
WORLD_ENV = {
    'sun_dir': [-0.42, 0.34, 0.84],
    'sun_color': [1.00, 0.82, 0.62],
    'sun_intensity': 1.35,
    'sky_zenith': [0.045, 0.060, 0.105],
    'sky_horizon': [0.30, 0.26, 0.24],
    'sky_ground': [0.055, 0.050, 0.045],
    'ibl_intensity': 0.42,
    'sky_intensity': 0.45,
    'fog_density': 0.0075,
    'fog_falloff': 0.024,
    'fog_color': [0.30, 0.27, 0.25],
}


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _shift_nodes(node, dz):
    """把一棵节点树整体沿 Z 平移。"""
    # 注意：**没写 translation 的节点也要平移**（默认 0,0,0）。
    # 首版写成  —— 于是所有省略 translation 的节点（环、流线、种子球）
    # 全留在原点，第 I 幕里能看到 09/11 区的东西。
    tr = node.get('translation') or [0.0, 0.0, 0.0]
    node['translation'] = [tr[0], tr[1], round(tr[2] + dz, 3)]
    for cd in node.get('children', ()):
        _shift_nodes(cd, dz)


def _channel_values(shot, prop):
    """取某条相机通道的 (times, values)。

    values 统一成"每项都是 list"：位置/朝向是向量，fov 是标量 —— 标量包成
    `[v]`，这样下游不用区分两种形状（首版就栽在这：fov 是 float，
    `list(v)` 直接 TypeError）。
    """
    for ch in shot.get('channels', ()):
        if ch.get('node') == '$camera' and ch.get('prop') == prop:
            vals = []
            for v in ch['values']:
                vals.append([float(v)] if isinstance(v, (int, float)) else list(v))
            return list(ch['times']), vals
    return [], []


def build_world():
    children = []
    cam_t, cam_t_tgt, cam_pos, cam_tgt, cam_fov = [], [], [], [], []
    cursor = 0.0
    meta = []

    # 先全部读进来（过渡要用下一段的起点）
    loaded = []
    for idx, entry in enumerate(REGIONS):
        name, dur = entry[0], entry[1]
        stem = entry[2] if len(entry) > 2 else name
        sp = os.path.join(FILM_DIR, 'scenes', f'{stem}.scene.json')
        hp = os.path.join(FILM_DIR, 'shots', f'{stem}.shot.json')
        if not (os.path.exists(sp) and os.path.exists(hp)):
            raise SystemExit(f'缺资产: {name}')
        loaded.append((name, dur, idx * -SPACING, _load(sp), _load(hp)))

    for k, (name, dur, z0, scene, shot) in enumerate(loaded):
        # --- 场景节点：平移后并入世界 ---
        kids = scene.get('root', {}).get('children', ())
        if not kids:
            # 第 I 幕的 scene 是**空 children**（它原本靠 ParticleSystem 撑场面），
            # 而世界层没有粒子 —— 补一颗种子球，与第 XII 幕同构（首尾闭环）。
            kids = [{
                'name': f'{name}_seed',
                'translation': [0.0, 0.0, 0.0],
                'mesh': {'builder': 'sphere',
                         'args': {'radius': 1.5, 'segments': 24, 'rings': 16,
                                  'color': [1.0, 0.97, 0.90]}},
                'material': {'preset': 'marble',
                             'base_color': [1.0, 0.97, 0.90],
                             'roughness': 0.4, 'metallic': 0.0,
                             'emissive': [2.2, 1.9, 1.5]},
            }]
        for cd in kids:
            _shift_nodes(cd, z0)
            children.append(cd)

        # --- 相机：段内沿用原运镜，平移到世界坐标 ---
        t_pos, v_pos = _channel_values(shot, 'position')
        t_tgt, v_tgt = _channel_values(shot, 'target')
        t_fov, v_fov = _channel_values(shot, 'fov')
        dur_actual = dur
        for t, v in zip(t_pos, v_pos):
            # 归一化到 [0,1] 再映射到本段时长（原 shot 的 duration 可能略有出入）
            u = t / (t_pos[-1] or 1.0)
            cam_t.append(round(cursor + u * dur_actual, 3))
            cam_pos.append([round(v[0], 2), round(v[1], 2), round(v[2] + z0, 2)])
        for t, v in zip(t_tgt, v_tgt):
            u = t / (t_tgt[-1] or 1.0)
            cam_t_tgt.append(round(cursor + u * dur_actual, 3))
            cam_tgt.append([round(v[0], 2), round(v[1], 2), round(v[2] + z0, 2)])
        for i, (t, v) in enumerate(zip(t_fov, v_fov)):
            pass
        # fov：每段起止各一帧，取该段首末值
        fov_a = v_fov[0][0] if v_fov else 50.0
        fov_b = v_fov[-1][0] if v_fov else 50.0
        cam_fov.append((cam_t[0] if k == 0 else cursor, fov_a))
        cam_fov.append((cursor + dur_actual, fov_b))

        meta.append({'name': name, 't0': round(cursor, 2),
                     't1': round(cursor + dur_actual, 2), 'z0': round(z0, 1)})
        cursor += dur_actual

        # --- 段间飞行过渡：从本段终点插到下一段起点 ---
        if k + 1 < len(loaded):
            nxt = loaded[k + 1]
            nz0 = nxt[2]
            npos0 = None
            for t, v in zip(*_channel_values(nxt[4], 'position')):
                npos0 = [v[0], v[1], v[2] + nz0]
                break
            a = cam_pos[-1]
            b = npos0 or a
            for j in range(1, 3):
                u = j / 3.0
                cam_t.append(round(cursor + u * BLEND, 3))
                cam_t_tgt.append(round(cursor + u * BLEND, 3))
                cam_pos.append([round(a[0] + (b[0] - a[0]) * u, 2),
                                round(a[1] + (b[1] - a[1]) * u + 6.0 * (1 - abs(2 * u - 1)), 2),
                                round(a[2] + (b[2] - a[2]) * u, 2)])
                cam_tgt.append(list(cam_tgt[-1]))
                # 过渡期 fov 拉宽一点（像高速穿行时的视野扩张）
                cam_fov.append((round(cursor + u * BLEND, 3), round(fov_b + 8.0, 1)))
            cam_t.append(round(cursor + BLEND, 3))
            cam_pos.append(list(cam_pos[-1]))
            cam_t_tgt.append(round(cursor + BLEND, 3))
            cam_tgt.append([round(cam_tgt[-1][0], 2), round(cam_tgt[-1][1], 2),
                            round(cam_tgt[-1][2], 2)])
            cursor += BLEND

    total = cursor
    return {
        'name': 'world',
        'environment': WORLD_ENV,
        'root': {'name': 'root', 'children': children},
    }, {
        'version': 1, 'name': 'world', 'duration': round(total, 3),
        'scene': 'scenes/world.scene.json',
        'acts': [{'t': m['t0'], 'label': m['name']} for m in meta],
        'notes': (f'12 区沿 -Z 排列（间距 {SPACING}m），段间 {BLEND}s 飞行过渡；'
                  f'相机由各段原运镜重采样后串成一条路径。'),
        'channels': [
            {'node': '$camera', 'prop': 'position', 'interpolation': 'LINEAR',
             'times': cam_t, 'values': cam_pos},
            {'node': '$camera', 'prop': 'target', 'interpolation': 'LINEAR',
             'times': cam_t_tgt, 'values': cam_tgt},
            {'node': '$camera', 'prop': 'fov', 'interpolation': 'LINEAR',
             'times': [t for t, _ in cam_fov],
             'values': [v for _, v in cam_fov]},
        ],
    }, meta, total


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args(argv)
    scene, shot, meta, total = build_world()
    n = len(scene['root']['children'])
    print(f'[world] {len(meta)} 区 / {n} 节点 / {len(shot["channels"][0]["times"])} 关键帧 / '
          f'{total:.1f}s')
    print(f'[world] 首区 {meta[0]["name"]} @ z={meta[0]["z0"]}，'
          f'末区 {meta[-1]["name"]} @ z={meta[-1]["z0"]}')
    if args.check:
        print('WORLD_CHECK_OK')
        return 0
    for path, data in ((SCENE_OUT, scene), (SHOT_OUT, shot)):
        with open(path, 'w', encoding='utf-8', newline='') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
            f.write(chr(10))
        with open(path, encoding='utf-8') as f:
            json.load(f)
        print(f'[world] 写入 {path} ({os.path.getsize(path) / 1024:.1f} KB)')
    print('WORLD_OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
