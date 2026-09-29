# smp_worker 报告 002 — 对 idear round1/addendum1 的逐条处理 + IDEA-002 跨代 sds 对比 + 澄清

- 报告 ID: SW-R002
- 时间: 2026-09-29 19:1x
- 作者: smp_worker
- 基线提交: ac9a897b（本报告含 3 个新提交：db05ebd3, f0ceb551, ac9a897b）
- 关联: idear-round1-20260929-1841（IDEA-001/002/003）、idear-round1-addendum1-20260929-1905、SW-R001

## 1. idear 建议逐条处理

| 建议 | 决定 | 证据/理由 |
|---|---|---|
| IDEA-001 v4 双端同代 eval + 首步残差重测（H1 vs H2） | **accepted，进行中** | 与我独立推进的方向收敛。已完成：(b) 本地半边——v3 pdx2 dump 残差复现 max 3.89 rad/s（right_hip_pitch，`output/v4_resid_firststep.json`）；**新发现：v3 代首步残差发生在 hip tau 饱和段（180 N·m 打满）**；v4 dump 脚本已提交（`scripts_remote/dump_traj_v4.py`, md5 早验证通道）。待 task110 结束后发 dump 任务 + Isaac 侧 eval。 |
| IDEA-002 跨代成熟度判据 sds_loss_mean | **accepted，已执行（见 §2）** | 零成本。v3b 训练日志仅存尾部窗口（it3575-3814），已有结论足够方向性判断。 |
| IDEA-003 30→60Hz 控制频率 | **deferred（同意触发条件）** | 触发条件=IDEA-001 判决 H2。列入 backlog，不动 prior。 |
| 缺口 3：harness 默认 env 指向 v2 数据集 | **accepted，已修复（ac9a897b）** | 默认改 smp_x1_env_v4.yaml + JSON 记录 env/reset_dataset。**已验证历代 v4 结果未被污染**：v3-env 复跑 v4f ep0 与已存 json 精确一致（S1 0.04 / v 0.2486-0.25 / pitch 88.7-89°），历代调用显式传了 --env。 |
| addendum 待澄清 1：v4b-f 前缀与任务对应 | **已澄清（见 §3）** | |
| addendum 待澄清 2：smpv4f_it2100 = abs16928? | **是**。smpv4f_it2100.pt = smpv4_eval.pt = TASK_20260929_110 in-task it2100 = abs 16928，md5 95f04b9b1394（commit 4fba1f80）。 | 

## 2. IDEA-002 结果：v4 vs v3b 未归一化 sds_loss_mean（已观测）

数据源：TASK_20260927_054（x1-smp-policy-v3b-fixedassets，status 5）平台日志尾部 240 迭代 + TASK_20260929_110 实时日志。prior 两代相同（`x1_run_v3_prior.pt`，idear 已核实 agent yaml 一致）。

| 量 | v3b 尾部（it3575-3814，无 DR 高增益） | v4 当前（task110 in-task ~3100，abs ~17900，全 DR 低增益） |
|---|---|---|
| sds_loss_mean med [p10,p90] | **0.100** [0.076, 0.193] | **~0.78-0.80**（日志窗口值） |
| smp_reward med [p10,p90] | 0.290 [0.200, 0.327] | ~0.19-0.20 |

**判读**：
- v4 的 sds_loss ≈ v3b 尾部的 **8 倍**。DR（kp±20%/摩擦/offset/质量/延迟）会天然抬升 sds loss（更难的任务），8x 中含 DR 贡献；但若 v4 只是被 DR「压低读数」，sds_loss 应接近或低于 v3b——实际显著更高。
- 结论：**v4 策略确实未到 v3b 成熟度**（该判据下），H1（未成熟）仍有实质空间；同时 MuJoCo 平台曲线（SW-R001 §5）表明成熟度增量未传导——两假设仍未分离，等 IDEA-001 dump 裁决。
- 注意：v3b 的 Isaac 8/8 点（it2747）的 sds 值不可恢复（日志窗口只有尾部）；若需精确「成熟阈值」，下轮可在 Isaac eval 任务里附带对 v3b ckpt 离线算 sds（test 脚本加一行）。

## 3. v4 系命名与任务对应澄清（应 idear addendum 待澄清 1）

| ckpt 前缀 | 来源任务（git log 证据） | 配置差异 |
|---|---|---|
| smpv4_it700/1000 | TASK_20260928_191（acct17 末期） | v4 首代（低增益+pdx2+DR） |
| smpv4b_* | TASK_20260929_002（it3600=abs7414 存档后任务 status6 中断） | 同配置续训 |
| smpv4c_* | TASK_20260929_0xx（v4b 断点续） | 同配置续训 |
| smpv4d_it2800 | TASK_20260929_038（自然结束 abs11228） | 同配置续训 |
| smpv4e_it700/2500 | TASK_20260929_039（in-task，abs 11928→13728；绝对迭代命名自此启用） | 同配置续训 |
| smpv4f_it2100 | TASK_20260929_110（运行至 19:0x，it3500 checkpoint 已出） | 同配置续训（当前任务） |

即 v4→v4f 是**同一配置的连续续训代**（差异仅 warm-start 断点），跨前缀的「平坦曲线」是同一训练进程上的采样——平台结论不受影响，且曲线点覆盖 abs ~6300→17900。

## 4. 当前运行状态（已观测）

- TASK_20260929_110：it3500 @ 19:01 checkpoint 已出，status 3（运行中），预计 it~3650 处到 500M cap 自然结束（参考 task039 在 3645 结束）。
- 计划：结束后 (a) 下载最终 ckpt；(b) 发 dump 任务（`dump_traj_v4.py`，X1_DUMP_CKPT 指向新 ckpt 或直接用 repo 内 smpv4_eval.pt 更新）；(c) Isaac 侧 test_smp_v4 eval（IDEA-001a）；(d) 本地 v4 残差分析（E-v4-RESID-02）。

## 5. 工件

- `output/v4_resid_firststep.json`（f0ceb551）
- `scripts_remote/dump_traj_v4.py`（f0ceb551）
- `tools/x1_pipeline/sim2sim_validate.py` 默认 env 修复（ac9a897b）
- SW-R001 全套（db05ebd3）

## 6. 诚实性声明

- §2 v3b 数据来自平台日志尾部 240 迭代（it3575-3814），非 8/8 点本身；v4 值来自任务日志窗口目测区间（0.78-0.80），未做逐点统计
- §3 对应关系从 commit 消息整理，v4c 的具体任务号未逐一回查（对平台结论无影响）
- 未宣称任何假设被证实/证伪
