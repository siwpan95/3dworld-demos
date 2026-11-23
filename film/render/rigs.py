# -*- coding: utf-8 -*-
"""骨架驱动层 —— 把 `logic` 层的关节角逐帧写进 scene 的节点树。

为什么要单独一层
----------------
`scene JSON` 表达不了"逐帧求值的关节"。前几幕的 `passes.py` 只有两种模式：
"建好对象、每帧 draw"（粒子、水面）。机械臂不行 —— 它每帧的关节角由 IK 解出，
要写回节点的四元数、并让 `world_matrix` 冒泡重算。

为什么不用 `logic/skeleton`
--------------------------
`Node` 本身就是一棵带层级变换的树（children / translation / 四元数 rotation /
`world_matrix` 累乘），**场景节点即骨骼**。再引一套平行的 `Skeleton` 只会让
同一棵关节树存在两份（本仓已有三处"两套骨架"的教训）。

关节约定（与 `logic/choreography.build_arm` 一致，改名要同步）
----------------------------------------------------------
turret 绕 **Y**（方位）、seg1/seg2 绕 **Z**（俯仰）、grip 不转。
`arm_ik()` 返回 `(turret_yaw, seg1_pitch, seg2_pitch)`，单位度。
"""
import math

from pyglet.math import Vec3

from world3d.logic.choreography import arm_ik

#: 关节名后缀（与 `film/gen/machine.py` 的节点命名绑定）
JOINTS = ('_turret', '_seg1', '_seg2', '_grip')


def quat_axis(axis, deg):
    """绕单位轴转 ``deg`` 度的四元数 (w, x, y, z) —— 与 Node.rotation 同格式。"""
    h = math.radians(deg) * 0.5
    c, s = math.cos(h), math.sin(h)
    return (c, axis[0] * s, axis[1] * s, axis[2] * s)


class ArmRig:
    """一条机械臂：持有关节节点 + 底座位置，按目标点求 IK。"""

    def __init__(self, turret, seg1, seg2, grip):
        self.turret = turret
        self.seg1 = seg1
        self.seg2 = seg2
        self.grip = grip
        # turret 的 translation 是相对 root 的，因此就是世界底座位置
        self.base = Vec3(*turret.translation)

    def update(self, target):
        """解 IK 并写回关节旋转；返回末端误差（米），供调用方断言可达性。"""
        yaw, a1, a2 = arm_ik(target, base=self.base)
        self.turret.rotation = quat_axis((0.0, 1.0, 0.0), yaw)
        self.seg1.rotation = quat_axis((0.0, 0.0, 1.0), a1)
        self.seg2.rotation = quat_axis((0.0, 0.0, 1.0), a2)
        for n in (self.turret, self.seg1, self.seg2):
            if hasattr(n, 'mark_dirty'):
                n.mark_dirty()
        return (Vec3(*self.seg2.world_matrix_world_tip()) - target).length \
            if hasattr(self.seg2, 'world_matrix_world_tip') else 0.0


class MachineRig:
    """一幕的骨架集合：逐帧给每条臂喂目标点。

    目标点轨迹是**程序化**的：每条臂在自身底座上方划一段椭圆，
    相位错开 90°，看起来像四台机器在各自巡检、偶尔同时朝中心伸。
    """

    def __init__(self, scene, count=4, prefix='arm'):
        self.arms = []
        for i in range(count):
            nodes = {}
            for suffix in JOINTS:
                nodes[suffix] = self._find(scene, f'{prefix}{i}{suffix}')
            if any(v is None for v in nodes.values()):
                raise RuntimeError(f'骨骼节点缺失: {prefix}{i}* -> '
                                   f'{[k for k, v in nodes.items() if v is None]}')
            self.arms.append(ArmRig(nodes['_turret'], nodes['_seg1'],
                                  nodes['_seg2'], nodes['_grip']))

    @staticmethod
    def _find(scene, name):
        root = getattr(scene, 'root', None)
        if root is None:
            return None
        for n in root.traverse():
            if getattr(n, 'name', None) == name:
                return n
        return None

    def update(self, t):
        """按时间推进四条臂。纯函数式（只依赖 t），保证成片确定性。"""
        for i, arm in enumerate(self.arms):
            ph = t * 0.42 + i * (math.pi * 0.5)
            # 椭圆巡检：半径 9~11、高度 4~8，绕着各自的底座转
            r = 9.6 + 1.4 * math.sin(t * 0.31 + i)
            target = Vec3(arm.base.x + math.cos(ph) * r,
                          arm.base.y + 6.0 + 2.2 * math.sin(t * 0.57 + i * 1.7),
                          arm.base.z + math.sin(ph) * r)
            arm.update(target)


class SwarmRig:
    """群集驱动：`logic/boids.Flock` 是纯 CPU 模拟，这里把它映射到节点位置。

    与粒子的相位策略一致：**按固定 dt 补步**（`round(t*fps)` 决定"应该已经走了
    多少步"），而不是用墙钟时间 —— 同 t 必然同状态，probe 跳帧也能复现。

    捕获者（predator）会在中途出现：它把群"推"成一个旋涡，是这一幕唯一的
    戏剧动作。位置同样只依赖 t。
    """

    def __init__(self, scene, count=80, bounds=26.0, seed=20261123, fps=30.0):
        from world3d.logic.boids import Flock
        self.flock = Flock(count=count, bounds=bounds, seed=seed)
        self.nodes = []
        for i in range(count):
            node = None
            for n in scene.root.traverse():
                if getattr(n, 'name', None) == f'boid{i}':
                    node = n
                    break
            self.nodes.append(node)
        self.fps = float(fps)
        self._steps = 0
        self._dt = 1.0 / self.fps

    def update(self, t):
        target = int(round(float(t) * self.fps))
        while self._steps < target:
            # 捕食者按时间绕群转圈，把 boid 推开形成旋涡
            ph = self._steps * self._dt * 0.55
            self.flock.set_predator((math.cos(ph) * 16.0,
                                     5.0 + 6.0 * math.sin(ph * 0.7),
                                     math.sin(ph) * 16.0))
            self.flock.step(self._dt)
            self._steps += 1
        for node, b in zip(self.nodes, self.flock.boids):
            if node is None:
                continue
            node.translation = (b.pos.x, b.pos.y, b.pos.z)
