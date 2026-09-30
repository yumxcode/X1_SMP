# smp_worker 报告 004 — E-ISAAC-SOLVER-01 判决：H2''（引擎族本质差）成立，solver 配置路线排除

- 报告 ID: SW-R004
- 时间: 2026-09-30 14:4x
- 作者: smp_worker
- 预注册: SW-PREREG-002（commit a703fbb4，先于任何扫描任务创建；本轮无时间线瑕疵——r4 任务 13:5x 后才存在输出）
- 基线提交: c9beeee8（扫描 harness 终版）；判决数据 TASK_20260930_038（r4）
- 关联: SW-R003（H2 初判）、idear round4（IDEA-009/010/011，artifactId `smp-ideas-idear-round4-20260929`）、idear round4 addendum5（per-joint 分解+零误差探针，`smp-ideas-idear-round4-addendum5-20260929`）
- 本地工件: `output/solver_scan_analysis.json`（本 commit）、`tools/x1_pipeline/solver_scan_analysis.py`、`scripts_remote/dump_traj_v4_solver.py`、`scripts_remote/diag_reset_readback.py`

## 摘要

按 SW-PREREG-002 预注册规则，Isaac PhysX solver 精度参数族扫描（{TGS,PGS}×{4/0,32/0,4/8}+GPU 锚，共 7 配置，同 ckpt abs18642、同确定性 reset）实测：**全部 6 个 CPU 配置 first-step 残差 ≥ 22.4 rad/s（阈值 ≥6.0 即 H2''）**，最好的配置距 H2' 判据（<2.0）还差 11 倍；迭代数提高（4→32）与 vel_iter（0→8）对残差几乎无影响；零误差探针（q_tar 钉在 q0，无 PD 误差路径）在全部配置下仍发散 18.8-20.0 rad/s。**H2'（solver 精度配置可压残差）被证伪，H2''（引擎族本质差）成立**——按预注册分支，Isaac Gym 内修配置的重训路线关闭，迁移必要性坐实，下一步为 IDEA-010 Isaac Lab 入场探针。

## 1. 判决证据（全部已观测，TASK_20260930_038 dump md5 已逐一核对）

| 配置 | solver×迭代×pipeline | first-step max (rad/s) | mean rms | n_valid(reset@) | 零误差探针 first |
|---|---|---|---|---|---|
| tgs4_0_gpu（锚） | TGS 4/0 GPU | 26.06 | 44.0 | 28 (@28) | 20.02 |
| tgs4_0_cpu | TGS 4/0 CPU | 30.11 | 49.3 | 16 (@16) | 18.76 |
| tgs32_0_cpu | TGS 32/0 CPU | 27.25 | 54.0 | 23 (@23) | 19.00 |
| tgs4_8_cpu | TGS 4/8 CPU | 29.40 | 48.3 | 17 (@17) | 19.21 |
| pgs4_0_cpu | PGS 4/0 CPU | **22.37** | 54.5 | 11 (@11) | 19.47 |
| pgs32_0_cpu | PGS 32/0 CPU | 22.40 | 55.5 | 10 (@10) | 19.40 |
| pgs4_8_cpu | PGS 4/8 CPU | 22.40 | 52.4 | 11 (@11) | 19.52 |

有效性自检（预注册 §4，全部通过）：
- obs[0] 跨配置 max diff **4.05e-06**（阈值 1e-4）——7 配置同一 reset、同一初始 obs
- 锚自检：26.06 ∈ 25.44±30%（17.81-33.07）
- 7/7 配置成功，无 NaN

**预注册判决：`H2'_supported` 要求任一 CPU 配置 <2.0——无；`H2''_supported` 要求全部 ≥6.0——是（min 22.37）→ H2'' supported。**

## 2. 机制层证据（per-joint 家族分解，30 步均值 |dv|，rad/s）

| 配置 | 肘族（4关节） | 腕族（4） | hip_pitch（2） | worst joint |
|---|---|---|---|---|
| tgs4_0_gpu | 15.00 | 1.09 | 8.45 | right_elbow_yaw |
| tgs4_0_cpu | 15.65 | 1.10 | 9.28 | right_ankle_roll |
| tgs32_0_cpu | 14.67 | 1.34 | 10.95 | right_elbow_yaw |
| tgs4_8_cpu | 16.54 | 1.33 | 8.51 | right_elbow_yaw |
| pgs4_0_cpu | 15.95 | 2.05 | 11.58 | right_elbow_yaw |
| pgs32_0_cpu | 16.06 | 2.07 | 13.63 | right_elbow_yaw |
| pgs4_8_cpu | 15.46 | 2.03 | 10.03 | right_elbow_yaw |

1. **addendum5 修正版 H2' 预期被否**：「若残差是速度求解迭代伪影，vel_iter↑/迭代↑ 应特定压缩肘族+hip_pitch 族」——实测肘族在 4/0→32/0（TGS 15.65→14.67，PGS 15.95→16.06）与 vel_iter 0→8（TGS 15.65→16.54，PGS 15.95→15.46）下**平坦**。肘族震中结论在干净数据上重现（worst joint 6/7 配置=elbow_yaw；腕族最小 1.1-2.1，与 addendum5 一致）。
2. **TGS vs PGS 族内差存在但小**：CPU 下 PGS（22.4）比 TGS（29-30）低 ~25%，仍与 MuJoCo 差 11 倍以上——solver 族选择不解决引擎间差。
3. **零误差探针（预注册 §3.2 联判项）**：全部配置 18.8-20.0 rad/s ≫ 0.5 → 残差含**约束/重力路径成分**（policy 无关），非 PD 误差放大主导。结合 (1)(2)：残差在 Isaac 与 MuJoCo 的关节级动力学积分/约束求解路径本身。
4. solver 家族效应本身真实且大（diag TASK_20260930_035：同一起点单步后 dofvel max TGS-GPU 27.9 / TGS-CPU 31.8 / PGS-CPU 38.7）——但该差异是「引擎内族间差」，与「引擎间残差」（22-30）同量级叠加，不构成跨引擎通路。

## 3. 实验迭代与 harness 缺陷披露（诚实记录）

| 轮次 | 任务 | 缺陷 | 后果与处置 |
|---|---|---|---|
| r1 | 031 | 子进程 re-exec 用相对 `__file__` 且 driver 先 chdir → python 打不开文件秒退 rc=2 | 7/7 FAIL；修为 chdir 前取绝对路径（a6e0ee21） |
| r3 | 034 | `to_np()` 对 CPU 张量返回**活视图**（.cpu() no-op→.numpy() 共享内存）→ CPU 配置 dump 全部坍缩为末态（别名污染）；GPU 配置因跨设备拷贝不受影响 | obs0 自检（预注册）捕获，判决 inconclusive；diag 任务 035 定位（reset 逐位一致、首步即分化）；修为强制 .copy()（c9beeee8） |
| — | 032/034 | 跨设备种子法无法统一 reset（multinomial 在 device 张量执行） | 类级 patch `_sample_motion_times`→(motion 0, time 0)（6a352eba），diag 实证跨配置/跨设备逐位一致 |

- **已知教训重犯声明**：to_np 别名污染正是经验库既有教训（"引擎张量 dump 须强制深拷贝"）；r1 从 dump_traj_v4.py 复制 to_np 模式时未应用该教训。预注册的 obs0 自检是本次捕获缺陷的唯一屏障——自检先于归因的纪律有效，但拷贝语义应在首轮就写对。
- **历史数据安全性（已核）**：dump_traj_v4.py 只在 GPU pipeline（cuda:0）跑过（163/165 号任务），跨设备拷贝使历史 dump 不受别名污染；SW-R003 及此前全部残差结论不受影响。

## 4. idear round4 + addendum5 逐条处理

| 建议 | 决定 | 证据/理由 |
|---|---|---|
| IDEA-009 solver 精度参数族扫描（先于迁移） | **accepted，已执行，判决 H2''** | §1-2。扫描本身按其设计完成（单变量=solver 族，同 ckpt/reset，分钟级×4 任务迭代） |
| IDEA-009 增量① per-joint 表（addendum5） | **accepted，已并入** | §2.1 表；肘族震中重现，vel_iter 压缩预期被否 |
| IDEA-009 增量② 零误差子探针（addendum5） | **accepted，已并入** | §2.3；残差含约束/重力路径成分 |
| IDEA-010 Isaac Lab 入场探针 | **accepted，触发条件已满足（H2''），排为下一步** | 按 idear 设计「仅在 IDEA-009 判 H2'' 后启动」；探针通过标准 <1 rad/s（同套指标） |
| IDEA-011 eval sds 跌倒后帧剔除 | **accepted，已实现未运行** | commit 7b58f38d（test_smp_v4.py fall-masked sds）；本轮无 Isaac eval 任务，代码待下次 eval 验证输出 |
| gap-1 engine yaml 暴露 solver 字段 | **accepted，已实现** | commit 7b58f38d（physx_solver_type/position_iterations/velocity_iterations，默认保持 TGS 4/0 行为不变）；H2'' 判决下该暴露转为 Isaac Lab 迁移期的对照工具 |
| gap-2 训练线空窗 | 记录 | 维持 SW-R003 决定（不续训 v4）；路线收敛依赖 IDEA-010 |
| gap-3 MuJoCo 原生向量化调研（idear 待网络恢复） | deferred | 归 idear；本轮无新增 |

## 5. 路线含义（建议，待评审）

1. **Isaac Gym 内 solver 配置路线关闭**（预注册 ≥6.0 分支）：不存在 {TGS,PGS}×迭代×pipeline 配置把引擎间残差压到可训练传导的水平；v4 低 kd 场景下残差 22-30 rad/s 为引擎族属性。
2. **下一步（IDEA-010 入场探针，低成本先行）**：Isaac Lab（PhysX 5）单关节阶跃对齐探针 vs MuJoCo harness，通过标准 first-step dof_vel diff <1 rad/s——通过才启动资产迁移；不通过则 MuJoCo 原生引擎路线升主。
3. MuJoCo 原生路线维持 SW-R003 排序（fallback），触发条件=IDEA-010 探针失败。

## 6. 诚实性声明

- §1-2 数值全部来自 `output/solver_scan_analysis.json`（脚本+dump 已落盘，可复跑）；判决严格按 SW-PREREG-002 带执行，无事后调阈
- 判决范围限定：Isaac Gym preview PhysX {TGS,PGS}×{4,32}×{0,8}×{CPU,GPU pipeline} vs 本地 MuJoCo harness（x1_sim_v4.xml @120Hz，隐式 kd）；**Isaac Lab/PhysX 5 未测**（IDEA-010 范畴）；单一 reset 条件（motion0/t0 站立起动帧），非多 reset 帧统计
- mean rms 列的 n_valid 窗口因早终止 reset 而异（10-28 步），rms 值跨配置可比性弱于 first-step 列（判决只用 first-step）
- 探针 replay 的 MuJoCo 侧仍用隐式 kd 语义（与 harness 一致）；「约束/重力路径」结论是在该语义配对下的表述，不区分隐式阻尼积分差与约束求解差的贡献——机制细分未做
- IDEA-011 代码未运行验证（无 Isaac eval 任务），标注「已实现未验证」
- 本报告发布 ≠ sim2sim 验收通过；sim2sim 仍未通过，路线处于迁移评估阶段
