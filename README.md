# 3dworld-demos — 3dworld 引擎的演示程序集

> 仓库：<https://gitee.com/siwpan/3dworld-demos> ｜ 许可：MIT（见 [LICENSE](LICENSE)）
> 引擎本体在独立仓 [siwpan/3dworld](https://gitee.com/siwpan/3dworld)（PyPI 包名 `3dworld`）。

本仓是 **3dworld 三维引擎的演示程序集**：39 个单文件演示 + 两条主线演出入口
（48 秒五幕演出 / 150 秒六幕走廊漫游）+ 成片录制 / exe 打包 / 批量验收工具。

2026-10-19 从引擎仓剥离：**演示所代表的能力全部保留在引擎内**（`world3d.logic` /
`world3d.engine` / `world3d.games` / `world3d.presentation`），本仓只放"可运行的
演示形态"。能力落点对照表见引擎仓 README「能力清单」一节。

## 安装与运行

```bat
:: ① 建虚拟环境并按本仓依赖清单安装（引擎在 PyPI，包名 3dworld / 导入名 world3d）
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

:: ② 指向本仓的配置树与产物目录（截图落点 / cordis 配置树都在本仓）
env.bat                     :: 或手工：
set WORLD3D_PROJECT_ROOT=%CD%
set WORLD3D_CONFIGS=%CD%\configs

:: ③ 跑演示
python examples\demoscene_tunnel.py           正常运行
python examples\demoscene_tunnel.py --smoke   冒烟（N 秒后自动关闭）
python main.py                                48 秒五幕演出
python kkrieger_demo.py                       150 秒六幕走廊漫游
```

> `WORLD3D_PROJECT_ROOT` / `WORLD3D_CONFIGS` 两个环境变量是必须的：pip 安装的引擎
> 推导不出本仓路径（`--shot` 截图与 cordis 配置树都靠它们定位）。`env.bat` 一键设好。

**依赖方式已于 2026-10-22 切换**：`examples/*.py` 里不再有任何 `sys.path` 引导
（旧的那句指向演示仓父目录，剥离后已失效、会让 39 个 demo 全部
`ModuleNotFoundError`），一律由 `pip install 3dworld` 提供。三份清单的分工、
例外（`mountain.py` 复用兄弟模块的噪声原语）与验证证据见
[docs/依赖与版本.md](docs/依赖与版本.md)。

## 目录

```
examples/        39 个单文件演示与工具（详见 examples/README.md：统一命令行 + 踩坑清单）
main.py          48 秒五幕演出入口（能力在引擎 presentation/app.py）
kkrieger_demo.py 150 秒六幕走廊漫游入口（能力在引擎 presentation/kkrieger_app.py）
pyglet_3d_demo*.py  历史 pyglet 1.x 兼容件（与 cocos2d 共存环境）
configs/         cordis 配置树（玩法内核的挂载声明，与引擎仓同源）
assets/          演示素材（程序化 glTF / OST / SFX；生成物不入库）
docs/            演示截图（--shot 产物）+ 场景样例 JSON + 依赖与版本 + 分发指南
experiments/     实验产物（bloodmusic OST 对照等）
requirements.txt        运行依赖（3dworld 范围版本 + 可选 zhuangtaiji）
requirements.lock.txt   精确锁版快照（截图对照 / 发版归档 / CI）
requirements-dev.txt    门禁 ruff 与打包 PyInstaller
scripts/
  verify_demos.py   批量验收（冒烟 + 截图 + 退出码）
  record_all.py     成片批量录制（ffmpeg）
  build_exe.py      PyInstaller 单文件打包
  gen_sfx.py        SFX 生成（bloodmusic 预设 → WAV）
  new_demo.py       新演示脚手架
recordings/      成片 MP4（本地产物，不入库）
dist/            exe 与发行版包（本地产物，不入库）
```

## 工具

```bat
python scripts\verify_demos.py                全部演示冒烟 + 截图验收
python scripts\verify_demos.py --bench 3       真实帧率（vsync 与重绘节流全关）
python scripts\record_all.py --list            查看可录制条目
python scripts\build_exe.py --pack             打包精选 6 个 + 组装发行版
```

分发（成片 / exe / Gitee release）的完整手册见 [docs/分发指南.md](docs/分发指南.md)。

## 与引擎的版本关系

演示对引擎版本**不锁死**：`requirements.txt` 里写 `3dworld>=0.3.0,<0.4`（0.3.x 都接受），
需要逐位可复现时用 `requirements.lock.txt`。完整口径见
[docs/依赖与版本.md](docs/依赖与版本.md)。
其中 terrain / voxel / particles / 隧道 / SDF 五项演示保持**自包含**（其能力已上提
引擎 `logic/terrain.py` `logic/voxel.py` `engine/particles.py` `engine/shaders.py`，
演示代码为剥离时快照）；其余演示本就是引擎模块的薄壳。

## 许可

MIT，见 [LICENSE](LICENSE)。
