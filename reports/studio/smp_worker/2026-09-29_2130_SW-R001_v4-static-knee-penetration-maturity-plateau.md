# smp_worker 报告 001 — MuJoCo sim2sim 秒倒诊断：静态正例、膝碰撞穿地与成熟度平台

- 报告 ID: SW-R001
- 时间: 2026-09-29 晚（session 起始于 Studio 激活）
- 作者: smp_worker
- 基线提交: main @ 4fba1f80（工作区另有未提交诊断脚本，本次一并提交，见 §6）
- 关联实验: E-v4-STATIC-01/02/03, E-v4-KNEE-01, E-v4-RESET-01, E-v4-NAN-01

## 摘要

针对 v4 代 MuJoCo sim2sim 持续 0/4 秒倒（smpv4f_it2100, reward~0.185, `output/sim2sim_v4f_it2100.json`），本轮做了 5 组本地 MuJoCo 诊断。**已观测**：(a) 完美静态初始化下 hold-home 在所有增益集（v4 低增益 / v3 高增益 kp450 / 踝增强 / 1.5x）1.6-1.8s 倒，力矩远未饱和（ratio≤0.33）；根因是 home 姿态双膝碰撞胶囊穿地 4.4mm，初始接触全部为膝-地。(b) v3 训练数据集（v4 训练同款，`data/envs/smp_x1_env_v4.yaml` L31 → `dataset_x1_run_v3.yaml`）12/12 clips 的膝胶囊 25-68% 帧穿地 0-5.2mm——质量门 R8 只查脚底。(c) 但分组对照显示初始穿地 reset 与干净 reset 的 fell_t 无显著差异（med 0.67 vs 0.60s, n=48）——膝穿地不是秒倒的充分机制。(d) 20-seed 闭环 0 次 NaN、仅 2/20 出现 |qvel|>50 先于倒下——数值发散非主导。(e) **成熟度平台：v4 全代（it700→abs16928）MuJoCo up_frac 停在 0.04-0.07，同期 Isaac reward 0.05→0.185 翻 4 倍**——成熟度提升未传导到 MuJoCo 存活，「纯不成熟」假设被显著削弱但未证伪。

## 1. 训练状态（已观测）

TASK_20260929_110（x1-smp-v4-lowgain-dr-r5b，namode4811 账号，PRO_20260929_019）**运行中**（status 3）。it3100 checkpoint 已上传（18:33），日志 in-task it3013+，Smp_Reward_Mean ~0.19-0.20 缓慢上升，DR 全开（kp±20%/friction[0,1]/motor offset ±0.035/mass x0.9-1.1/1-step latency）。约 7 min/100 it。未重复启动。注意：`gm task list` 在该账号返回空（接口行为），须用 `gm task info --task-id` 直查。

## 2. 静态正例对照（E-v4-STATIC-01/02/03，已观测）

工具：`tools/x1_pipeline/v4_static_stance_probe.py`、`v4_static_geometry_audit*.py`、`v4_flatfoot_hold.py`；工件 `output/v4_static_stance_probe.json`、`output/v4_static_geometry_audit.json`、`output/v4_flatfoot_hold.json`。

| 增益集 | 静态 hold-home | tau_ratio_max |
|---|---|---|
| v4 (踝50/1 膝150/8) | FELL 1.73s | 0.28 |
| v3 高增益 (踝200/20 膝450/45) | FELL 1.83s | 0.22 |
| v4+踝100/2 | FELL 1.63s | 0.28 |
| v4+踝150/3 | FELL 1.60s | 0.28 |
| v4 全关节 x1.5 | FELL 1.70s | 0.33 |

初始化为完美静态（qpos=home、qvel=0、root_z 二分校准至脚底 +1mm）。**与增益无关**。此前 commit 575fb4dd 的「static home hold ~1.85s under both」与此一致，但其归因（"knee static sag 0.30 rad"）不完整。

方法学纠错（自我证伪记录）：初版几何审计把 sole 盒厚度（0.024m，两层级 8 角点属正常平底盒几何）误读为「脚翘 24mm」，并把 mesh 枚举记错；更正后（`v4_static_geometry_audit3.py`）：home 双脚平底、COM 在支撑域内（margin +0.084m）、支撑域 x 跨 0.197m。**静态失败的真因是双膝穿地**：

- `x1_sim_v4.xml` L101/L129：膝碰撞胶囊 `fromto="0 0 0 0 -0.30494 0.0336" size="0.048"`（覆盖全小腿）
- home（膝 0.632 rad）下两膝胶囊最低点 −4.4mm；初始 ncon=2，接触对全部为 `floor|*_knee_pitch_link_col`
- 时序：pitch 0.8s@6° → 1.2s@19° → 1.6s@53°，com_x 前移 0.6m——以膝为支点前倾加速

home 数组（`sim2sim_validate.py` L181-184）本身不是站姿（它是静态跪姿），用作正例对照无效；但该缺陷不直接进入正式 sim2sim（reset 来自 demo 帧）。

## 3. 数据集膝穿地（E-v4-KNEE-01，已观测）

工具 `tools/x1_pipeline/v4_knee_penetration_audit.py`，工件 `output/v4_knee_penetration_audit.json`。

- v3 数据集（12 clips）：每 clip **25.0%-67.6% 帧的膝胶囊在地平面以下**（最深 −5.2mm）；sole 全部 +0.5mm（R8 门生效）
- v2 数据集（已淘汰）：膝最深 −50.5mm、sole 最深 −86.2mm
- 根因：重定向质量门 R8（`tools/x1_pipeline/validate_retarget_v3.py` 的最终帧闭环）只对 sole 角点做抬升，**未覆盖膝/小腿碰撞胶囊**
- 含义：跑步深屈膝姿态下小腿胶囊下缘扫过地平面是数据级缺陷；真机跑步膝不会持续蹭地

## 4. 分组对照与数值审计（E-v4-RESET-01 / E-v4-NAN-01，已观测）

工具 `v4_reset_penetration_group.py`、`v4_nan_audit.py`；工件 `output/v4_reset_penetration_group.json`、`output/v4_nan_audit.json`。ckpt = smpv4f_it2100（= smpv4_eval.pt, abs 16928）。

- 48 个指定帧 reset（穿地 26 / 干净 22）：fell_t med **0.67s vs 0.60s**，无显著差异，两组 0/26 与 0/22 存活 5s
- 20 seeds（101-120）：NaN 0/20；|qvel|>50 先于倒下 2/20（seed102: 0.125s→fell 0.97s）；其余平滑快速倒下

**结论：初始膝穿地不是秒倒的充分机制；数值发散非主导。** 膝穿地仍是需修的数据缺陷（重置保真度、demo-replay 正例、Isaac/MuJoCo 接触语义潜在分叉点），但预期单独修复不解决 sim2sim。

## 5. 成熟度平台（已观测，汇总自 output/sim2sim_v4*.json 全系列）

| 代 | it | Isaac reward | MuJoCo up_frac | MuJoCo mean_speed |
|---|---|---|---|---|
| v4 | 700-1000 | ~0.05 | 0.052-0.054 | 0.12-0.13 |
| v4b | 800-3814 | ~0.06-0.09 | 0.042-0.071 | 0.18-0.31 |
| v4c | 1200-2500 | ~0.10-0.14 | 0.042 | 0.25-0.29 |
| v4e | 700-2500 | ~0.14 | 0.043-0.053 | 0.32-0.36 |
| v4f | 2100 (abs16928) | ~0.185 | 0.047 | 0.26 |

Isaac reward 翻 4 倍期间 MuJoCo 存活（up_frac）零改善。speed 有改善（0.12→0.36）。**推断**：存在与成熟度无关的硬 gap（引擎动力学差异为主嫌疑，v3 报告 §4.3 的求解器残差链条仍是最佳候选解释，但 v4 已把残差力矩从 108 → ≤6 N·m（commit 05f8b905）而 MuJoCo 存活不变——残差**力矩**放大假设也被削弱，残差本身（首步关节速度差 2.4 rad/s）在 v4 代尚未重测，是当前最大的未测项）。

## 6. 工件清单（本轮新增，全部未提交→将随本报告提交）

- `tools/x1_pipeline/v4_static_stance_probe.py` + `output/v4_static_stance_probe.json`
- `tools/x1_pipeline/v4_static_geometry_audit.py`（初版，含已声明的方法错误，保留作纠错记录）
- `tools/x1_pipeline/v4_static_geometry_audit2.py`（mesh 假设错误版本，保留作纠错记录）
- `tools/x1_pipeline/v4_static_geometry_audit3.py`（更正版）+ `output/v4_static_geometry_audit.json`
- `tools/x1_pipeline/v4_flatfoot_hold.py` + `output/v4_flatfoot_hold.json`
- `tools/x1_pipeline/v4_knee_penetration_audit.py` + `output/v4_knee_penetration_audit.json`
- `tools/x1_pipeline/v4_reset_penetration_group.py` + `output/v4_reset_penetration_group.json`
- `tools/x1_pipeline/v4_nan_audit.py` + `output/v4_nan_audit.json`

## 7. 下一步（按优先级，未执行）

1. **v4 代首步残差重测**（本地+复用 `dump_traj_v3b_forces` 类方法或已有 Isaac dump）：干净无接触 reset 帧、t=0 obs 一致前提下，量化 v4 低 kd 的 Isaac-vs-MuJoCo 首步 dof_vel 差。若仍 ~2.4 rad/s：残差速度差在低 kd 下无法用力矩解释，需换机制（如接触时序/积分器差异）；若显著缩小：成熟度/其他因素重新上位。
2. task110 自然结束后下载末期 ckpt 做代内 discriminative eval（确认平台是否持续到 500M cap）。
3. 修数据：R8 门扩展为全身碰撞几何零穿地（含膝胶囊），重生成 v3.1 数据集（本地）。注意换数据→prior 需重训，成本决策留给首步残差实验结果之后。
4. （低成本高价值）home 数组修正为真平足站姿（解膝穿地的踝角），恢复静态正例对照的判别力。

## 8. 诚实性声明

- §2 表格数据为本地 MuJoCo 3.1.6（`.venv`，Apple Silicon）实测，脚本可复跑
- §3/§4 数字来自落盘 JSON 与脚本 stdout（JSON 含逐帧/逐 seed 明细）
- 「Isaac reward」取自 git log 的代内估计值与 task110 实时日志（reward 数值为 Smp_Reward_Mean，非完整 return）
- 未宣称 sim2sim 达标；未宣称根因已终局定位——v4 代首步残差未测，机制竞争解释仍开放
- 未执行真机相关操作
