# -*- coding: utf-8 -*-
""".kkrieger 致敬放映厅 —— ``presentation.kkrieger_app`` 官方装配的 CLI 壳。

本文件是"引擎能力演示"：场景/编排/配乐 100% 来自引擎，demo 只做 CLI 壳
（--smoke/--shot/--bench）+ 标题。此前 kkrieger 家族只有单测覆盖，没有
可看的演示——本 demo 补上（"未消费引擎能力清零"的最后两块之二）。

消费的引擎能力：
- ``logic/kkrieger.py``      150s 六幕无缝循环时间轴：走廊 / 大厅 / 核心 /
                             撤离 / 突入 / 圣殿。三道舱门开闭时刻按镜头
                             过门时刻反推（关早了镜头穿门板）；第五幕
                             高速突入另有一组速开门时刻。
- ``presentation/kkrieger_scene.py``
                             KKRenderer：零素材程序化场景——金属板/灯带/
                             舱门/管道/货箱/立柱/旋转环/风扇/无人机全由
                             代码即时生成；面板颜色用确定性哈希抖动模拟
                             "程序化纹理"；圣殿的 lathe 凹槽柱 / torus
                             发光嵌环 / (2,3) torus_knot 反应堆 / tube
                             导管全部是新几何类型。静态几何合并单 draw
                             call，动画对象逐帧 model+材质驱动。
- ``logic/kkrieger_audio.py`` 程序化配乐：纯 Python 离线渲染（约 2.2 万个
                             音符事件），曲式与动画共用同一套时间常量，
                             音画同步是"编译期"对齐；WAV 缓存于
                             assets/kkrieger_ost_150s.wav（首渲约 25s，
                             之后秒过）。
- ``engine/audio.py``        TimelineAudio 播放 + 播头漂移校正。
- ``logic/playback.py`` / ``presentation/hud.py`` / ``timeline_ui.py``
                             播放控制、幕字幕、时间轴面板。

操作（KKApp 内建）：SPACE 暂停  R 重播  N/P 下一幕/上一幕  ←/→ 快退快进
                      L 循环  M 静音  ESC 退出

用法::

    python examples/kkriegerward.py            正常放映（150s 循环 + 配乐）
    python examples/kkriegerward.py --smoke 2  冒烟（SEC 秒后自动关闭）
    python examples/kkriegerward.py --shot     抓帧到 docs/ 后关闭
    python examples/kkriegerward.py --bench 5  关 vsync 基准并输出帧统计
"""
import argparse
import sys

import pyglet

from world3d.presentation.kkrieger_app import KKApp   # noqa: E402  (模块级设 dpi_scaling='real')

TITLE = ('3dworld · .kkrieger 致敬 — 程序化走廊/大厅/圣殿漫游'
         '（150s 六幕 + 程序化 OST）')
SHOTS = (
    (6.0, 'docs/kk_01_corridor.png'),      # 第一幕：走廊，首道舱门已开
    (30.0, 'docs/kk_02_atrium.png'),       # 第二幕：大厅环绕纪念碑
    (55.0, 'docs/kk_03_overload.png'),     # 第三幕：过载——环加速核心脉冲
    (80.0, 'docs/kk_04_retreat.png'),      # 第四幕：撤离，灯带逐段熄灭
    (105.0, 'docs/kk_05_dash.png'),        # 第五幕：高速突入过舱门
    (130.0, 'docs/kk_06_sanctum.png'),     # 第六幕：圣殿环绕反应堆
    (138.8, 'docs/kk_07_ignition.png'),    # 第六幕：点火前兆（避开发光爆闪帧）
)


class _FrameTimer:
    """--bench 帧计时：量 on_draw 链之间的间隔（vsync 关后即真实吞吐）。

    与 platform/app.py 的 _sample_frame_time 同款语义：前 1.0s 丢弃预热，
    p1 / 最慢帧才反映"稳态卡不卡"。KKApp 自管窗口，以 handler 形式挂在
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

    # KKApp 构造即建窗/烘焙时间轴/初始化程序化配乐（缓存缺失时会现场渲染）
    app = KKApp(shots=list(SHOTS) if args.shot else None)
    app.renderer.window.set_caption(TITLE)   # pyglet 2.1: caption 只读，走 setter 方法

    if args.shot:
        # 官方跳帧抓帧：seek → 采样 → 渲染 → 抓图，1 秒内跑完全部剧照
        app.capture_shots()
        app.close()
    else:
        timer = None
        if args.bench is not None:
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
