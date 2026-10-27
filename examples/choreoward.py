# -*- coding: utf-8 -*-
"""五幕编排剧场 —— ``logic.choreography`` 官方宿主（WorldApp）的放映厅。

本文件是"引擎能力演示"：演出逻辑 100% 来自引擎，demo 只做 CLI 壳
（--smoke/--shot/--bench）+ 标题。此前 choreography 只有单测覆盖
（断言 IK 落点在宝石/台座 0.9 单位内、pose(0)==pose(48)），没有可看的
演示——本 demo 补上（"未消费引擎能力清零"的最后两块之一）。

消费的引擎能力：
- ``logic/choreography.py``  48s 五幕无缝循环：升起 / 环形波 / 机械臂取放 /
                             螺旋 / 归位。机械臂三关节角度由两连杆逆运动学
                             解出（余弦定理 + 方位角，不是手调）；相机走
                             17 帧电影运镜脚本；所有周期项整除 DURATION，
                             pose(0)==pose(48) 硬约束 ⇒ 循环无跳帧。
- ``logic/animation.py``     Channel/Sampler/Timeline：translation/rotation/
                             scale/tint/glow 五类通道一次烘焙，逐帧线性采样。
- ``logic/skeleton.py``      Skeleton 节点树 + build_arm：父子矩阵继承算出
                             夹爪世界矩阵；``carry_weight(t)`` 把宝石的
                             "静置姿态"混合到"夹爪姿态"——宝石是真的被
                             机械臂带着走。
- ``logic/playback.py``      PlaybackController：播放/暂停/循环/跳幕。
- ``presentation/app.py``    WorldApp 官方装配：时间轴采样 → T·R·S 矩阵 +
                             tint/glow 材质 → 渲染；自带幕字幕 HUD 与
                             时间轴面板；assets/gem.obj 存在时自动挂载宝石。

操作（WorldApp 内建）：SPACE 暂停  L 循环  C 电影机位/自由轨道
                      N/P 跳幕  R 重播  -/= 速度  拖动/滚轮 自由视角
                      ESC 退出

用法::

    python examples/choreoward.py            正常放映（48s 循环）
    python examples/choreoward.py --smoke 2  冒烟（SEC 秒后自动关闭）
    python examples/choreoward.py --shot     抓帧到 docs/ 后关闭
    python examples/choreoward.py --bench 5  关 vsync 基准并输出帧统计
"""
import argparse
import sys

import pyglet

from world3d.presentation.app import WorldApp

TITLE = '3dworld · 五幕编排剧场 — 逆运动学机械臂取放（48s 无缝循环）'
SHOTS = (
    (6.5, 'docs/choreo_01_rise.png'),      # 第一幕：环形立方体错峰升起
    (15.0, 'docs/choreo_02_wave.png'),     # 第二幕：径向+垂直行波扫过环带
    (26.5, 'docs/choreo_03_pick.png'),     # 第三幕：放件特写（宝石落座台座）
    (37.5, 'docs/choreo_04_spiral.png'),   # 第四幕：螺旋柱收拢完成 + 核心脉冲
                                           # （37.0 收拢到位，36.0 还半散着）
    (44.0, 'docs/choreo_05_return.png'),   # 第五幕：拉远升高，回落归位
)


class _FrameTimer:
    """--bench 帧计时：量 on_draw 链之间的间隔（vsync 关后即真实吞吐）。

    与 platform/app.py 的 _sample_frame_time 同款语义：前 1.0s 丢弃预热
    （着色器编译 / 首帧上传，实测首帧能到百毫秒级），p1 / 最慢帧才反映
    "稳态卡不卡"。WorldApp/KKApp 自管窗口，这里以 handler 形式挂在
    窗口的 on_draw 链上（返回 None 不阻断既有绘制）。
    """

    def __init__(self, window):
        self._now = None
        self._start = None
        self.ms = []
        self._warm = False
        window.push_handlers(on_draw=self._mark)

    def _mark(self):
        import time as _t
        now = _t.perf_counter()
        if self._start is None:
            self._start = now
        if self._now is not None and self._warm:
            self.ms.append((now - self._now) * 1000.0)
        self._now = now
        if not self._warm and now - self._start > 1.0:
            self._warm = True

    def report(self):
        """返回 (frames, span_s, mean, median, p1, worst_ms, jank_ratio)。"""
        ms = sorted(self.ms)
        if not ms:
            return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        span = sum(ms) / 1000.0
        mean = len(ms) / span
        median = 1000.0 / ms[len(ms) // 2]
        p1 = 1000.0 / ms[min(len(ms) - 1, int(len(ms) * 0.99))]
        jank = sum(1 for m in ms if m > 20.0) / len(ms)
        return (len(ms), span, mean, median, p1, ms[-1], jank)


def main(argv=None):
    ap = argparse.ArgumentParser(description=TITLE)
    ap.add_argument('--smoke', type=float, default=None, metavar='SEC',
                    help='run SEC seconds, then close (CI smoke)')
    ap.add_argument('--shot', action='store_true',
                    help='capture SHOTS to docs/ then close')
    ap.add_argument('--bench', type=float, default=None, metavar='SEC',
                    help='benchmark SEC seconds without vsync, print stats')
    args = ap.parse_args(argv)

    app = WorldApp()
    app.renderer.window.set_caption(TITLE)   # pyglet 2.1: caption 只读，走 setter 方法

    if args.shot:
        # 官方跳帧抓帧：seek → 采样 → 渲染 → 抓图，1 秒内跑完全部剧照
        for t, path in SHOTS:
            app.capture_at(t, path)
        app.close()
    else:
        timer = None
        if args.bench is not None:
            # 基准模式：窗口是 WorldRenderer 自建的，只能后置关 vsync；
            # interval=0 让重绘"能多快就多快"（否则量的还是定时器节流）
            app.renderer.window.set_vsync(False)   # pyglet 2.1: vsync 只读，走 setter 方法
            timer = _FrameTimer(app.renderer.window)
            pyglet.clock.schedule_once(lambda dt: app.close(), args.bench + 0.5)
            print(f'[bench] vsync=off, running {args.bench}s ...')
        elif args.smoke is not None:
            print(f'[smoke] auto-close in {args.smoke}s')
            pyglet.clock.schedule_once(lambda dt: app.close(), args.smoke)
        pyglet.app.run(interval=0 if args.bench is not None else 1 / 60)
        if timer is not None:
            n, span, mean, median, p1, worst, jank = timer.report()
            print(f'[bench] frames={n} span={span:.2f}s '
                  f'fps mean={mean:.1f} median={median:.1f} p1={p1:.1f} '
                  f'worst_frame_ms={worst:.2f} jank_gt20ms={jank * 100:.1f}%')

    print('EXIT_OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
