# 《晶极 / Crystalward》调研与致色机理记账

> 调研日期：2026-11-14 ｜ 对应演示：`examples/crystalward.py`
> 记账口径（治理铁律「借鉴必记账」）：**来源 + 许可 + 借用范围**，三者缺一不可。
> 本 demo 只借用**科学事实、术语与公开的物理/数学公式**，未复制任何来源的代码、
> 图片、数据表或成段文案；全部 GLSL / Python 为本项目独立实现。

## 1. 调研结论：晶体的"颜色"不是一类现象，而是至少五类机制

矿物学通行的分类把颜色分成**自色 / 他色 / 假色**（体色是否由自身成分决定），
而致色的物理成因通常列为五种。本 demo 把其中四种直接写进了着色器的吸收谱 `a(λ)`，
第五种（物理光学致色）拆成薄膜干涉与布拉格衍射两条独立实现。

| # | 致色机制 | 关键事实（用于 shader 建模） | 本 demo 落点 |
|---|----------|------------------------------|--------------|
| 1 | 过渡金属离子 d-d / f-f 跃迁（色素离子 Ti/V/Cr/Mn/Fe/Co/Ni/Cu、稀土） | 未填满 d/f 亚层的电子在轨道间跃迁，选择性吸收特定波段；**同一 Cr³⁺ 在刚玉配位场里呈红、在绿柱石配位场里呈绿**——配位环境改变跃迁能量 | `u_mech=0`，`u_site` 切两套吸收带 |
| 2 | 离子间电荷迁移（IVCT，Fe²⁺↔Ti⁴⁺、Fe²⁺↔Fe³⁺） | 吸收强度远大于 d-d 跃迁，故颜色浓烈；蓝宝石的蓝来自 Fe-Ti 电荷转移；加热/辐照可改变 Fe 价态，使绿柱石在海蓝 / 金色之间转换 | `u_mech=1`，580nm 宽带 + 高吸收系数 |
| 3 | 晶格缺陷色心（F 心、`[AlO4]4+`） | 阴离子空位俘获电子形成 F 心；紫萤石、烟晶均属此类；**加热使结构复原、颜色变淡** | `u_mech=2`，`u_anneal` 退火褪色动画 |
| 4 | 能带间跃迁 | 禁带宽度决定吸收边；窄带隙矿物不透明并呈金属光泽 | 未单独成幕（与色心幕共用"强吸收"视觉族） |
| 5 | 物理光学致色（假色）：锖色 / 晕色 / 变彩 | 干涉、衍射、散射造成，与成分无关；**不是多色性**（多色性来自两束偏振光的差异吸收） | `u_mech=3` 薄膜干涉、`u_mech=4` 布拉格变彩 |

**术语澄清（调研里最容易被误用的一点）**：多色性 / 变彩 / 变色效应是三件事。
多色性取决于**观察方向**（晶体各向异性 → 两束偏振光差异吸收，一轴晶二色性、
二轴晶三色性）；变彩取决于**观察角度**（欧泊微球的布拉格衍射）；变色效应取决于
**光源光谱**（亚历山大变石）。本 demo 的 `u_mech=6` 只实现第一种。

## 2. 关键公式（全部在片元着色器里按波长求值）

| 公式 | 形式 | 来源 |
|------|------|------|
| 选择性吸收 / 体色 | `T(λ) = exp(−a(λ)·L)`，`L` 由晶体内部二次 SDF 步进实测 | 矿物颜色成因科普（河北省地矿局、科普中国） |
| 薄膜干涉光程差 | `Δ = 2·n·d·cosθ_t = 2d·√(n₁² − n₂²sin²θ₁)`；单次 π 相移时 `Δ = (k+½)λ` 相长 | 《宝石晕彩效应的成因机理综述》、《宝石与宝石学》2025 巴林左旗晕彩石榴石；另核 sciencecalculators.org / nrtechstudio 的通用薄膜公式 |
| 干涉强度 | `I = I₁ + I₂ + 2√(I₁I₂)·cos(2πΔ/λ)` | 《宝石与宝石学》2025 式 (1) |
| 欧泊 Bragg–Snell | `mλ = 2d·√(n_eff² − sin²θ)`；`n_eff = √(n_sphere²·f + n_medium²·(1−f))`，理想密堆 `f≈0.7405` | iScience 24, 102121 (2021) 式 (2)；成都理工《宝石学中缺陷研究现状与展望》2025 式 (1)；Skoltech 博士论文评审报告式 (2.18) |
| 微球尺度 | 天然欧泊变彩来自 **150~350 nm** 单分散 SiO₂ 微球的面心立方有序排列（Sanders 1964 电镜证实；Jones 等 TEM 给出尺寸区间） | 成都理工 2025；iScience 2021 |
| 色散 / 火彩 | 白光经刻面折射后分解；钻石 `n≈2.417`、色散值 `0.044`、临界角 `24.4°`；石英色散 `0.013`、蓝宝石 `0.018`、金红石 `0.330` | 科普中国《宝石为什么那么好看》；crystals.eu 钻石物理与光学特性 |
| Cauchy 色散近似 | `n(λ) = n_D + B·(589²/λ² − 1)`（归一化到钠 D 线 589nm） | 通用色散模型；`B` 用上表实测色散值标定 |
| 双折射 / 多色性 | 两束互相垂直的偏振光速度、路径、吸收程度不同 → 出射异色；一轴晶二色性、二轴晶三色性 | 维基百科「多色性」、百度百科「多色性」 |
| CIE 1931 → sRGB | 分段高斯拟合 CMF：x̄ 三项、ȳ 两项、z̄ 两项，再乘 XYZ→linear-sRGB 矩阵 | Wyman, Sloan & Shirley, *Simple Analytic Approximations to the CIE XYZ Color Matching Functions*, JCGT 2(2), 2013 |

**火彩的一条实践结论被直接用于环境光设计**：钻石的火彩在"小而亮的点光源"下
最明显，宽大漫射光只会强调刻面轮廓与整体亮度。因此 `envLum()` 是两颗高指数
高亮的点光 + 一条细灯带 + 极弱的渐变底光，而不是均匀环境球。

## 3. 形态侧：晶系 ≠ 晶习

调研反复强调：**晶系**是内部原子对称性（对某一矿物种是固定属性），**晶习**是
外部生长形态（随冷却速率、过饱和度、可用空间、杂质、竞争晶体数量而变）。
同一种石英可以长出十几种晶习，而晶系始终是三方。本 demo 的 8 个形态因此按
「可见几何」命名，不冒充 7 大晶系的严格分类。

| 形态 | 几何 | 现实对应 | 调研事实 |
|------|------|----------|----------|
| 0 立方 `{100}` | box | 萤石、黄铁矿、石盐 | 等轴晶系典型单形 |
| 1 八面体 `{111}` | L1 球 | 金刚石、磁铁矿、萤石 | 钻石常呈八面体；`n=2.417` |
| 2 菱形十二面体 `{110}` | `max(|x|+|y|, |y|+|z|, |x|+|z|)` | 石榴石 | 12 个菱形面 |
| 3 六方柱 + 双锥头 | 六边形棱柱 ∩ 双锥 | 石英、绿柱石、磷灰石 | 六方 / 三方晶系的柱状习态 |
| 4 板状 + 燕尾双晶 | 扁盒 + 镜像连生 | 石膏、重晶石 | 双晶按确定对称规律连生（Carlsbad / 穿插 / 燕尾） |
| 5 针状放射晶簇 | 黄金角放射的 9 根细柱 | 辉锑矿、钠沸石、金红石发晶 | acicular / radiating habit |
| 6 漏斗晶（hopper） | 5 层嵌套内缩方框骨架 | **铋**、石盐、方解石、石英"窗晶" | 高过饱和度下**边缘生长快于面心**（Berg 效应 / 扩散限制），面心来不及填平 → 阶梯凹坑；慢冷得大台阶，>100℃/s 只剩细小乱晶 |
| 7 枝晶（dendritic） | 主干 + 6 层 × 4 向 + 二级分枝 | 自然铜、锰/铁氧化物 | 极高过饱和度下晶体失去初始习态、以大表面积分枝生长；**枝晶仍是单晶** |

铋的补充事实：原子结构是菱面体，但骨架生长造成**伪立方**外观；纯铋新鲜面银白，
彩虹色来自纳米级氧化膜的薄膜干涉，且**颜色本质是氧化膜的厚度地图**——先冷却 /
暴露更久的面与刚露出的内面颜色不同，倾斜样品时颜色会移动。demo 里用面哈希生成
`filmNm` 的空间变化，就是这个结论的直接编码。

## 4. 来源清单（含借用范围与许可说明）

| 来源 | URL | 借用范围 | 许可 / 性质 |
|------|-----|----------|-------------|
| 河北省地矿局《揭秘！矿物颜色千变万化的真相》 | http://www.hebdkj.cn/dkj3w/mldz/120/14801.html | 五类致色机理、F 心、Fe-Ti 电荷迁移、萤石/烟晶加热褪色事实 | 政府科普网页，事实性知识；未复制文案 |
| 科普中国《矿物》 | https://www.kepuchina.cn/article/articleinfo?business_type=100&ar_id=340285 | 自色/他色/假色分类、锖色/晕色/变彩定义、光泽与透明度 | 科普条目，事实性知识 |
| 科普中国《宝石为什么那么好看》 | https://cloud.kepuchina.cn/newSearch/imgText?id=7256783822494724096 | 钻石色散 0.044、玻璃 0.009、蓝宝石 0.018、火彩与点光源关系 | 科普条目，数值事实 |
| 上海市地质学会 / 矿物晶体知识课堂第 18 期（公众号转载） | http://mp.weixin.qq.com/s?__biz=Mzk2NDA1NTIxNA==&mid=2247484742 | 三要素（色调/明度/饱和度）、自色/他色/假色、五种成因 | 科普转载，事实性知识；未复制成段文字 |
| 《炫彩斑斓的矿物晶体，颜色成因大揭秘》（公众号） | http://mp.weixin.qq.com/s?__biz=MzI5MTcwNzU3NQ==&mid=2247485066 | Cr³⁺ 在红宝石/祖母绿中配位场差异、电荷迁移强度、能带理论 | 科普文章，事实性知识 |
| 中国地质大学《宝石晕彩效应的成因机理综述》 | https://jogg.cug.edu.cn/cn/article/pdf/preview/047828dc-5037-4d4b-9fa7-4a58d13fcdba.pdf | 光程差公式 (2)、干涉条件 (3)、多缝衍射 (4) | 学术期刊公开 PDF；**仅引用公式**（数学事实不受著作权保护） |
| 《宝石与宝石学》2025 巴林左旗晕彩石榴石 | https://jogg.cug.edu.cn/article/doi/10.15964/j.cnki.027jgg.2025.06.004 | 干涉强度式 (1)、π 相移条件、438~766nm 互层厚度 | 学术论文；仅引用公式与结论 |
| iScience 24, 102121 (2021) *Designable structural coloration by colloidal particle assembly* | https://pmc.ncbi.nlm.nih.gov/articles/PMC7892991/pdf/main.pdf | 欧泊 SiO₂ 微球有序堆积、Bragg 公式 | PMC 开放获取（作者/出版社许可见原文）；仅引用公式 |
| 成都理工《宝石学中缺陷研究现状与展望》2025 | https://kwys.cdut.edu.cn/cn/article/pdf/preview/10.19719/j.cnki.1001-6872.2025.01.09.pdf | 150~350nm 微球、`λ_Bragg = 2a√(n_eff²−sin²θ)`、Sanders/Jones 结论 | 期刊公开 PDF；仅引用公式与事实 |
| Skoltech 博士论文评审报告（Opal-like photonic crystals 章） | https://back.skoltech.ru/storage/app/media/defenses/2021/timur-ermatov/council/report_Mailis.pdf | Bragg–Snell 式 (2.18)、`n_eff` 与填充率 0.7405 | 公开学位论文材料；仅引用公式 |
| crystals.eu《铋：元素金属、漏斗几何形态及由氧化膜构建的颜色》 | https://www.crystals.eu/zh/blogs/crystals/bismuth | 铋六个生长步骤、氧化膜厚度/角度/光照对色的影响、金→绿→蓝→紫→粉的色序 | 商业科普站；仅取事实，未复制文案 |
| crystals.eu《钻石：物理 & 光学特性》 | https://www.crystals.eu/zh/blogs/crystals-d/diamond-physical-optical-characteristics | n≈2.417、临界角 24.4°、色散 0.044、各向同性单折射 | 商业科普站；仅取数值事实 |
| ScienceInsights / Geology Science 铋晶体 | https://scienceinsights.org/why-bismuth-crystals-form-their-rainbow-staircase-shape 、https://geologyscience.com/gemstones/gemstone-blog/bismuth-crystals-the-rainbow-metal-where-art-meets-science | 冷却速率 <10℃/s / >100℃/s 对形态的影响、氧化膜厚度地图 | 科普站；仅取事实 |
| JACS 2025 *Classical View on Nonclassical Crystal Growth*（PMC 镜像） | https://pmc.ncbi.nlm.nih.gov/articles/PMC11726565/ | 高过饱和度 → hopper → dendritic 的生长机制序列、Berg 效应、枝晶仍为单晶 | PMC 开放获取；仅取机理结论 |
| Trinity College Dublin Geological Museum《Diagnostic Features》 | https://trinitygeologicalmuseum.com/minerals/diagnostic-features | 7 晶系、单形/闭形开形、双晶（Carlsbad/穿插/燕尾）、晶习词汇 | 高校博物馆教学页；仅取术语事实 |
| Wikipedia *Crystal habit* / *Hopper crystal*（镜像） | https://en.chped.com/w/index.php?title=Crystal_habit 、https://wiki.smallrat.net/wikipedia_en_all_maxi/A/Hopper_crystal | 晶习清单（acicular/botryoidal/dendritic/drusy/hopper/tabular…）、hopper 边缘快于面心 | Wikipedia 文本为 **CC BY-SA**（镜像页自述）；**仅用事实与术语，未复制文本，故不触发署名/相同方式共享义务**；如需引用原句须按 CC BY-SA 处理 |
| AzMineral Education《Crystal Habit: The Vocabulary of Crystal Shapes》 | https://azmineraleducation.org/lessons/crystal-habit-vocabulary-of-shapes | 晶系 vs 晶习的区分、生长条件决定习态 | 教学页；仅取事实 |
| 维基百科「多色性」/ 百度百科「多色性」 | http://zh.wikipedia.com/wiki/多色性 、https://baike.baidu.com/item/多色性 | 二色性/三色性、一轴晶/二轴晶对应、与变彩/变色效应的区别 | 百科条目；仅取事实 |
| Wyman, Sloan & Shirley (JCGT 2013) | 公式见 shader 注释 `cieXYZ()` | CIE 1931 CMF 的分段高斯解析拟合 | 学术论文中的数学公式；**公式属事实，代码为本项目独立实现** |

> 未采用：`WebSearch` 曾返回一条与本主题无关的体育数据噪声，已剔除，未写入任何结论。

## 5. 从调研到代码的对应关系（便于复核）

### 能力反哺（2026-11-14）

按「演示所代表的能力全部保留在引擎内」的分工，可复用部分已上提到引擎仓
`siwpan/3dworld`（本演示仍是**自包含快照**，能力本体在引擎）：

| 能力 | 引擎落点 | 本仓对应 |
|------|----------|----------|
| CIE 1931 拟合 / XYZ→sRGB / Beer-Lambert / 薄膜干涉 / Bragg–Snell / Cauchy / 8 类吸收谱 | `world3d/engine/spectrum.py`（纯标准库、零 GL 的 CPU 黄金参考） | `FRAGMENT_SRC` 里的 GLSL 实现 |
| 可注入 GLSL 片段（`spectral_glsl(prefix)` 支持命名空间） | `world3d/engine/spectral_glsl.py` | `pgauss` / `cieXYZ` / `xyz2rgb` / `band` 与片段**逐字一致** |
| 公式不漂移的守卫 | `tests/test_spectrum.py`（11 项，零 GL）+ `scripts/run_all_tests.py`（47→48 模块） | 一次性逐字比对（`CONSISTENCY_OK`） |

不直接 `import` 引擎新模块的原因：本仓 venv 装的是 PyPI 发行版，引入未发版模块
会让 65 个 demo 的运行环境分叉；这与 `README.md`「terrain / voxel / particles /
隧道 / SDF 五项演示保持自包含」的既有惯例一致。引擎侧把常量进了
`world3d.engine.__all__` 公共面；工厂函数 `spectral_glsl()` 因与子模块同名，
刻意不进公共面（否则会遮蔽模块属性），需从子模块直接导入。

| shader 位置 | 对应调研结论 |
|-------------|--------------|
| `absorption()` | 五类致色机理 → 吸收带中心/宽度/强度 |
| `insidePath()` | Beer-Lambert 的 `L` 必须是实际光程，而非固定常数 |
| `interfaceMod()` | 薄膜干涉 `Δ=2nd·cosθ_t` + π 相移；Bragg–Snell 峰 |
| `shadeCrystal()` 的 `nl` | Cauchy 色散；逐波长独立 `refract` → 火彩 |
| `envLum()` | 小而亮的点光源才出火彩 |
| `filmNm` 的面哈希 | 铋的彩虹 = 氧化膜厚度地图 |
| `sdBismuthHopper()` / `sdDendrite()` | 高过饱和度下的骨架生长 / 扩散限制分枝 |
| 底部光谱尺 `background()` | 用同一套 `cieXYZ()` 显示 400~700nm，把颜色模型自证出来 |
| `u_anneal` | 色心加热褪色（萤石 / 烟晶） |

## 6. 演示参数与验收

- 八幕自动演出，每幕 6 秒、共 48 秒；每幕开头 1.2 秒晶体从 0 生长到 1。
- 光谱采样默认 14 点（`--flag low` 降到 8）；主步进 84 步、内部光程 24 步，硬上限。
- `Q/E` 可脱离演出独立切换 8 种致色机理，`1~8` 独立切换 8 种形态，便于把
  "同一个形态 / 不同机理" 或 "同一个机理 / 不同形态" 并排对照。
- 剧照：`docs/crystalward_01..05.png`（4 / 10 / 14 / 22 / 40 秒），刻意取五种互不
  相同的色彩机制与形态：F 心紫萤石·立方 / 钻石色散火彩·八面体 / 祖母绿 Cr³⁺·六方柱锥头 /
  铋氧化膜干涉·漏斗晶 / 欧泊变彩·枝晶。
- 验收（2026-11-14，仓库 `.venv` / pyglet 2.1.16 / 1280×720）：
  `python scripts/verify_demos.py --only crystalward` → **ALL_DEMOS_OK (1 demos)**
  （smoke `ok`、5 张 shot `ok`、耗时 2.3s + 41.3s）；
  `--bench 3` 首幕 mean **1142** / median 1198 / p1 **594** / 最慢 2.02ms / jank 0%；
  最重场景 `--bench 3 --flag hold,shape=6,mech=3`（漏斗晶 + 薄膜干涉）
  mean **766** / median 838 / p1 **491** / 最慢 2.60ms / jank 0%。

## 7. 待复核项

- 本 demo 是**实时近似**，不是光谱路径追踪：晶体内部只做一次折射 + 一次出射，
  未模拟多次内全反射、偏振态传播与真实相位；`n_eff≈1.45` / 氧化膜 `n≈1.92`
  为演示取值，用于把色序调到可见范围，不代表特定样品的实测折射率。
- 钻石幕的火彩做了**显式的演示增益**：真实单次折射的 n(λ) 差只有约 ±0.05，
  加上临界角 24.4° 使大半光线走全反射分支，逐波长出射方向几乎不分离；且 14 点
  光谱积分会把任何窄峰平均成白。代码里做了三件事：① 把 `(n(λ)−n_D)` 乘 7 转成
  切向偏角，给"内反射回程"一个随波长变化的方向；② 用一条**随空间双频扫动的
  窄缝**每个像素只放行 1~2 个波长（等价于白光经棱镜色散后，不同角度各有一盏
  窄缝点光）；③ 让镜面反射也随缝走，否则高折射率下接近 1 的 Fresnel 白反射
  会盖掉火彩。这是观感补偿，**不是**物理仿真结果，不可用于计算真实色散值，
  也不代表真实钻石表面会出现这种连续的彩色斑纹。
- 「Cr³⁺ 两套吸收带」「色心 555nm 单峰」是对吸收谱的**定性高斯拟合**，
  不是分光光度计实测数据；用于展示机制差异，不用于鉴定。
- 形态 2 的菱形十二面体与形态 4 的燕尾双晶是几何近似；7 大晶系未逐一严格建模。
