# 《造界 / Worldwright》—— 3dworld 全能力视频项目

> 这是一部**原创短片工程**，不是演示录屏合集。
> 场景、模型、运镜、粒子、配乐全部由本目录产出；`examples/` 只作为能力清单的索引。

## 它要证明什么

12 幕 / 720 秒 / 1080p30，把 67 个 demo 代表的**引擎能力**在一个自洽世界里真实消费一次。
验收口径是五层可证伪的链路（依赖 → 建模 → 演出 → 渲染 → 成片），任何一层断了
就只算"演示集回放"。完整计划见 `docs/视频项目计划.md`（若未随仓，见工作区同名文件）。

## 目录

```
project.json            工程清单：舞台尺寸 / 帧率 / 幕表
scenes/*.scene.json     声明式场景（scene_build schema，可 diff、可 scene_viewer 热重载）
shots/*.shot.json       演出关键帧（Shot schema，可被导演台回放微调）
render/passes.py        渲染趟：一帧 = 场景 → 粒子 → Bloom（顺序是工程决策，注释写理由）
render/probe.py         单帧探针：把一个 t 渲成 PNG，镜头迭代压到秒级
render/render_film.py   逐帧离屏 → ffmpeg 管道（确定性 dt=1/fps）
out/                    成片 MP4（可再生物，不入库）
```

## 怎么跑

```bat
:: 环境（引擎推导不出本仓路径）
env.bat

:: 单帧探针（推荐先跑这个：改一版镜头 3 秒就能看到）
python film\render\probe.py --act 01_void --t 0,8,16,24 --size 960x540

:: 确定性检查（两次从头跑到第 120 帧，比像素 sha256）
python film\render\render_film.py --act 01_void --check-determinism 120 --size 640x360

:: 出片
python film\render\render_film.py --act 01_void --size 1920x1080 --crf 18
```

## 三条踩出来的硬约束（改代码前必读）

1. **相机必须用 `FlyCamera`**。`ShotPlayer._apply_camera` 只写回 `position` / `target` / `fov`，
   `yaw/pitch/dist` 会被它直接 `return False`；而 `OrbitCamera` 从 0.7.4 起对 `position`
   赋值**无效**（眼位只由 yaw/pitch/dist 决定）。两者相乘 = 用 OrbitCamera 接通道会静默不动。
2. **`ParticleSystem.set_uniforms()` 不绑定 program**。漏了 `prog.use()` 会得到"满屏没有粒子"，
   而且**不报错**（写进未绑定的 program 只是静默丢值）。必须先 `use()` 再设 uniform。
3. **粒子相位是累加的**（`advance` 只改速率、不改已积累相位）。所以渲染**必须从 t=0
   顺序推进**，跳帧不可复现；"同 t 渲两次"不能证明整链确定性，只有"两次都从头跑到第 N 帧"能。

## 进度

| 幕 | 状态 | 证据 |
|---|---|---|
| 01 空无 | **已出片** | 24.00s / 960x540 / 30fps / yuv420p；确定性 sha256 一致 |
| 02–12 | 待建 | 见计划书 §3.2 |

## 与引擎的关系

引擎（`3dworld`）是只读依赖，但本项目发现的引擎缺口会**反哺**回引擎仓，
每处改动带证据与测试。已反哺：

- `0.7.7` `shot_path`：截图落点可被 `WORLD3D_DOCS_DIR` 重定向（此前是死变量）
- `0.7.7` `Renderer.render(pre_post=...)`：场景内自定义绘制的唯一插入点
