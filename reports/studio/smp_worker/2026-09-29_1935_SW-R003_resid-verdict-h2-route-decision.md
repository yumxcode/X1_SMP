# smp_worker 报告 003 — E-v4-RESID-02 预注册判决：H2（结构性引擎差）坐实，v4 训练线终止

- 报告 ID: SW-R003
- 时间: 2026-09-29 19:35
- 作者: smp_worker
- 基线提交: 4630fa04（含 94454223 预注册、4ec85220 sds 收集）
- 关联: SW-PREREG-001（预注册）、SW-R001/R002、idear round1-3 + addendum1-4（IDEA-001/004/007/008）
- 远端任务: TASK_20260929_163（dump abs16928）、164（Isaac eval+sds）、165（dump abs18642）、110（600M 终点）

## 摘要

按预注册 SW-PREREG-001 的判决规则，E-v4-RESID-02 实测 v4 低增益下 Isaac-vs-MuJoCo 首步关节速度残差 **17.84 rad/s（≥2.0 区间）**；七条独立证据全部合流：**H2（结构性引擎差）成立，「继续 v4 训练」路线按预注册终止**（不发新续训任务；task110 已自然到 600M cap）。同时 Isaac 侧同代 eval（IDEA-004）显示 Isaac 内同 ckpt 存活 med ~5s（vs MuJoCo 0.63s）——策略并非完全无用，失败特定于跨引擎。

## 1. 时间线与预注册有效性披露

| 时刻 | 事件 |
|---|---|
| 19:18:03 | dump 任务 163 启动 |
| 19:19:34 | dump 文件上传完成（平台侧结果已存在） |
| **19:21:35** | SW-PREREG-001 commit 94454223（本会话未查看任何 dump 输出） |
| 19:23+ | 本会话首次下载/查看 dump（curl 本地时间戳、分析输出均在后） |

诚实披露：预注册 commit 晚于 dump 文件完成 2 分钟，但信息隔离成立——判决阈值在查看任何结果数据之前落盘；分析脚本 `tools/x1_pipeline/v4_resid_dump_analysis.py` 的输入仅在下载后可得。

## 2. 判决证据链（全部已观测，工件落盘）

| # | 证据 | 数值 | 工件 |
|---|---|---|---|
| 1 | 预注册主判据：首步 max Δdof_vel | **17.84 rad/s ≥ 2.0** | `output/v4_resid_dump_analysis.json` |
| 2 | obs 对齐自检（排除状态错位） | max diff **2.9e-6**（231 维逐位） | 分析 stdout（本报告 §附录可复跑） |
| 3 | MuJoCo 单步速度峰值 vs Isaac | **25-47 rad/s vs 8-26 rad/s**（符号混合振荡） | 同上 rows |
| 4 | 位置漂移（辅助判决 2 触发） | 单控制步 **0.6-1.1 rad** | 同上 |
| 5 | ckpt 轴（abs16928 vs abs18642） | mean rms **8.76 vs 8.41（-4%）**：残差与成熟度无关 | `output/v4_resid_dump_final.json` |
| 6 | 接触分层 | no-contact rms 8.59 ≈ contact 9.14：**接触非主因** | 本轮 stdout |
| 7 | MuJoCo 细化/积分器 | 1/480 Euler 7.86、RK4 8.05（仅 -10%）：**残差不在 MuJoCo 离散化** | 本轮 stdout |

推论（推断）：残差钉在 Isaac Gym PhysX TGS @120Hz 的关节级求解行为本身（与 MuJoCo 的隐式阻尼/积分路径不同），且 **v4 低 kd（45→4-8）把 v3 时代被强阻尼压制的求解器差从力矩级（2.4-3.9 rad/s）释放为速度级（15-27 rad/s）**——「低 kd 缩小残差力矩」（108→≤6 N·m）的设计在速度域适得其反（无接触帧同样如此）。

## 3. Isaac 侧同代 eval（IDEA-004+008，TASK_20260929_164，已观测）

ckpt = smpv4_eval.pt abs16928（md5 95f04b9b，日志核对加载路径一致），无 DR，8 env × 3 ep × 10s：

| ep | fall_t med (min) | alive>9s | sds_loss_mean (no-DR) |
|---|---|---|---|
| 0 | 5.23s (0.50) | 2/8 | 1.2131 |
| 1 | 4.92s (0.67) | 2/8 | 1.2501 |
| 2 | 4.97s (2.83) | 0/8 | 1.2372 |

- **H4a 成立**：Isaac fall_t med ~5s ≫ MuJoCo 同代 0.63s（48-reset med）——平台是 MuJoCo 特有。
- sds 口径警告：eval sds（1.21-1.25）**含跌倒后漂移帧**（10s 内 ~5s 跌倒后），与训练日志 0.79（跌倒即重置混合）不可直接比；v3b 对照半边（IDEA-008 完整 2×2）未跑——不影响本判决（判决不依赖 sds）。
- Isaac 侧也未成熟（0-2/8 活 9s，v3b 曾 8/8）——H1 成分仍真，但按证据 5（残差与成熟度无关）它不再是 sim2sim 的决定变量。

## 4. v4 训练线终态（已观测）

- task110 于 600M samples cap 自然结束（final it3814 = abs18642，md5 f5b9901a0a1e，已提交 smpv4_eval.pt，commit 4630fa04）
- 终代 MuJoCo sim2sim：**0/4，up_frac 0.03-0.04**（`output/sim2sim_v4g_it3814.json`）——平台持续到训练终点
- **决定：不再发 v4 续训任务**（预注册 ≥2.0 分支）

## 5. 路线决策（预注册触发；候选待评审）

已排除：更强参数化对齐（pdx2 逐项对齐后仍 8 rms 残差，参数空间 v3 代已扫尽）；继续训练（证据 5+平台到终点）；MuJoCo 端细化/换积分器（证据 7，仅 -10%）。

开放候选（按我的优先序，待 idear 评审与用户决策）：
1. **Isaac Lab 迁移重训**（mimickit 已有 `isaac_lab_engine.py` 骨架；X1 资产需接入 Isaac Lab USD/MJCF 管线；消除 Gym preview 求解器语义；prior/数据/算法栈全复用）
2. **MuJoCo 原生训练引擎**（gap=0 的确定性路线，自建成本高；仅作 ①受阻的 fallback）
3. 60Hz 控制频率（IDEA-003）：只缩小分歧窗口不消除残差——**不解锁**，随路线 ①/② 顺带评估

## 6. 本轮工件

- `tools/x1_pipeline/v4_resid_dump_analysis.py` + `output/v4_resid_dump_analysis.json`
- `output/v4_resid_dump_final.json`、`output/sim2sim_v4g_it3814.json`
- `scripts_remote/test_smp_v4.py`（sds 收集，4ec85220）
- data/models/smp/smpv4_eval.pt → abs18642（4630fa04）
- dump 数据：`output/remote_ckpt/isaac_traj_v4.pt`（md5 a66b6a32）、`isaac_traj_v4_final.pt`（md5 d1c28f55）

## 7. 诚实性声明

- §2 全部数值本地可复跑（脚本+dump 已落盘）；§1 时间线如实披露预注册晚于 dump 完成 2 分钟
- §3 sds 口径限制已声明；IDEA-008 的 v3b 对照半边未跑（判决不依赖）
- 路线决策是**建议**，最终采纳待评审；「H2 坐实」限于「当前 Isaac Gym preview + MuJoCo(harness 参数族) + 30Hz + v4/v3 资产」组合内——Isaac Lab 是否消除残差未验证（下路线第一步即验证）
- 未宣称 v4 策略在 Isaac 内达标（0-2/8 活 9s）
