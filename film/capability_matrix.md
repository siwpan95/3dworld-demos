# 能力矩阵 —— 67 个 demo → 12 幕

> **本文件由 `film/gen/matrix.py` 生成，不要手改。**
> 归属写成数据、覆盖用集合差集对账：缺 / 多 / 重三种错都会让脚本非零退出。

- 幕数 **12**，各幕时长合计 **733.0s**，扣掉 11 处 xfade 1.2s 后成片 **719.8s**
- demo 总数 **67**，已归属 **67**，未归属 **0**

| # | 幕 | 时长 | demo 数 | demo（= 能力载体） | 引擎能力落点 |
|---|---|---:|---:|---|---|
| 1 | `01_void` 空无 | 24s | 2 | `particles_gpu`, `particleward` | engine/particles.py（GPU 实例化）、postfx Bloom |
| 2 | `02_terrain` 地脉 | 66s | 7 | `terrain_flyover`, `terrainward`, `cdlod_terrain`, `water_waves`, `volumetric_fog`, `godhand`, `demoscene_tunnel` | logic/terrain、terrain_lod、terrain_clipmap、logic/waves、engine/volumetrics、core/terrain_core、shaders→tunnel |
| 3 | `03_city` 城邦 | 80s | 12 | `wardfall`, `lindenwood`, `collapseward`, `texward`, `sandward`, `material_lab`, `shadow_map`, `crowdward`, `blockworld`, `minecraft_style`, `tower_defense`, `scene_walker` | logic/town、logic/lsystem、logic/wfc、engine/proctex、engine/texproj、engine/texture、engine/soft_shadow、behavior_tree + pathing_core、logic/blocks、games/tower_defense、adaptive_music |
| 4 | `04_machine` 机械 | 74s | 9 | `choreoward`, `boxward`, `brickward`, `eventward`, `fsmward`, `skinned_creature`, `ozzward`, `gltfward`, `kkriegerward` | logic/skeleton、logic/animation、anim/ozz_lite、engine/skin、engine/gltf、engine/orchestration.Bus、engine/fsm、logic/choreography |
| 5 | `05_crystal` 晶体 | 64s | 5 | `crystalward`, `posterward`, `raymarch_sdf`, `stillhall`, `dof_focus` | engine/spectrum、spectral_glsl、sdf2d、sdf2d_glsl、shaders→raymarch、engine/dof、games/hall |
| 6 | `06_cave` 洞窟 | 64s | 6 | `dungeonward`, `cellward`, `voxel_terrain`, `brogueward`, `rayhold`, `liminall` | logic/dungeon、logic/ca、logic/voxel、logic/fov、engine/particles、shaders→tunnel、logic/liminall、games/fps |
| 7 | `07_swarm` 法则·群 | 58s | 4 | `boids`, `boidward`, `swarmward`, `pulsefall` | logic/boids、engine/particles、games/storm、games/pulse、engine/audio |
| 8 | `08_body` 法则·体 | 58s | 3 | `rigidward`, `box2dward`, `wboit_demo` | physics/jolt_lite、physics/box2d_lite、engine/oit |
| 9 | `09_field` 法则·场 | 54s | 6 | `pointcloud_edl`, `gpu_cull_field`, `computeward`, `curveward`, `flowward`, `reactward` | engine/pointcloud、engine/gpu_cull、engine/compute、logic/curves、engine/framegraph |
| 10 | `10_void` 虚空 | 80s | 6 | `voidward`, `solar_system`, `nordic_saga`, `kid_a_walker_3d`, `mountain`, `motion_blur` | logic/skeleton（轨道）、games/void、logic/nordic_saga、logic/kid_a、engine/motionblur、adaptive_music |
| 11 | `11_loop` 回环 | 64s | 1 | `csm_shadow` | engine/csm（另复用 mountain 的气象、lindenwood 的季节、volumetrics） |
| 12 | `12_finale` 收束 | 47s | 6 | `engine_showcase`, `frame_graph`, `replayward`, `director`, `scene_viewer`, `audioward` | renderer(PBR/IBL) + postfx + framegraph + presentation/film + logic/replay + presentation/timeline_ui |

## 跨幕复用（不重复计入归属）

- `mountain`：X 幕主用（远景孤峰），XI 幕复用其气象（雨雪）
- `lindenwood`：III 幕主用（L-System 植被），XI 幕复用其季节色
- `volumetric_fog`：II 幕主用（高度雾），XI 幕复用（光柱随光向变化）
- `csm_shadow`：XI 幕主用（级联光向变化），XII 幕复用（全 pass 叠加）
- `liminall`：VI 幕主用（后室走廊），X 幕复用（梦境插叙）

## 判定口径

- **"有归属"≠"能播"**：本表只回答"这个能力在哪一幕被消费"。
  能否播由 `verify_demos.py` 的全量冒烟与每幕的 `probe.py` 帧证据回答。
- **主归属唯一**：一个 demo 只记在它贡献最大的那一幕，复用写在上面单独一节。
  否则"67 全归属"这个数会因为跨幕复用而虚高。
