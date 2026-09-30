# SW-PREREG-002 — E-ISAAC-SOLVER-01 判决阈值预注册（IDEA-009 solver 精度参数族扫描）

- 报告 ID: SW-PREREG-002
- 时间: 2026-09-30 13:45（本 commit 先于任何扫描任务启动与输出查看）
- 作者: smp_worker
- 基线: SW-R003（commit 14082dfa）已确认 H2 成立、v4 训练线终止
- 关联: idear round4 IDEA-009（artifactId `smp-ideas-idear-round4-20260929`）、
  idear round4 addendum5（`smp-ideas-idear-round4-addendum5-20260929`，per-joint 分解 + 零误差子探针增量）
- 实验载体: `scripts_remote/dump_traj_v4_solver.py`（新），复用 `dump_traj_v4.py` 协议

## 1. 实验设计（固定项，冻结）

| 项 | 值 |
|---|---|
| ckpt | `data/models/smp/smpv4_eval.pt` = abs18642（md5 f5b9901a0a1e，commit 4630fa04）——与 TASK_20260929_165 final dump 同 ckpt，直接可比 |
| env / engine yaml | `data/envs/smp_x1_env_v4.yaml` + `data/engines/isaac_gym_engine_pdx.yaml`（不变） |
| 语义 | pdx2（显式 kp 力矩 + 隐式 kd），无 DR（AgentMode.TEST），同 dump_traj_v4.py 补丁 |
| reset | 单 env，同确定性 reset（跨配置 obs[0] 一致性自检，见 §4） |
| 被扫变量（单一主变量=solver 精度族） | {solver_type} × {num_position_iterations} × {num_velocity_iterations} × pipeline |
| 配置表 | tgs4_0_gpu（锚，复现 165 号任务条件）、tgs4_0_cpu、tgs32_0_cpu、tgs4_8_cpu、pgs4_0_cpu、pgs32_0_cpu、pgs4_8_cpu |
| 附加子探针（addendum5） | 零误差探针：q_tar 钉在 q0（reset 帧 dof_pos），8 控制步，纯重力+阻尼+约束沉降路径 |
| 指标 | 每配置：MuJoCo replay 逐控制步 dof_vel diff（同 v4_resid_dump_analysis.py 算法）→ first-step max、30 步 mean rms、29 关节 per-joint mean |dv|、探针 first-step max |
| 平台 | gradmotion 单任务（goodsId ESKU000001 镜像 BJX00000001/V000124），分钟级 |

## 2. 判决规则（预注册，采用 idear round4 IDEA-009 给出的支持/否定条件）

主判据 = CPU 配置族的 first-step max |dof_vel diff|（rad/s），对照基线 = 同 ckpt GPU 锚任务实测（TASK_20260929_165: first-step max 25.44, mean rms 8.41）：

- **H2'（solver 配置可压残差）强支持**：任一 CPU 配置 first-step max **< 2.0**（相对基线降 >8x）
  → 路线留在 Isaac Gym：engine yaml 暴露 solver 字段（本轮已做），按该配置重训/重评
- **H2''（引擎族本质差）支持**：全部 CPU 配置 first-step max **≥ 6.0**
  → 迁移必要性坐实：启动 IDEA-010 Isaac Lab 入场探针 / MuJoCo 原生路线评估
- **灰区 2.0–6.0**：不做单值判决，按 §3 联判

## 3. 灰区联判规则（预注册）

1. **per-joint 模式**（addendum5 修正后的预期）：若 vel_iter↑（4/8 vs 4/0）特定地压缩肘族（elbow_pitch/elbow_yaw）与 hip_pitch 族的残差，且 pgs32/tgs32 相对 4 迭代单调改善 → 支持「速度求解迭代伪影」机制；若迭代数 ↑ 残差不动的关节族 → 该族残差在约束/积分路径。
2. **零误差探针**：探针 first-step max 仍 ≥ 2（policy 无关路径）→ 残差含约束/重力路径成分，配置路线上限受限；探针 < 0.5 而 policy dump 残差大 → PD 误差放大主导，配置扫描更可能有效。
3. GPU 锚自检：本轮 tgs4_0_gpu first-step max 应落在 165 号任务的 ±30% 内（25.44 → 17.8–33）；否则本轮任务内部一致性存疑，判决降级为 inconclusive 并先排查 reset/环境漂移。

## 4. 有效性自检（每配置必须通过才计入判决）

- 跨配置 obs[0] 一致性：各配置 dump 的 obs[0] 与 GPU 锚逐元素差 < 1e-4（否则该配置 reset 不一致，剔除）
- MuJoCo replay 与 dump 的状态对齐自检：同 E-v4-RESID-02（obs 对齐 2.9e-6 量级）
- 崩溃处理：某配置（尤其 PGS CPU）若数值崩溃/NaN，记录后继续下一配置；判决基于通过自检的配置集合；若 PGS 全崩，PGS 半边记 inconclusive，不影响 TGS 半边判决
- 决不使用累积 reward 或视频作为本实验判据

## 5. 输出契约

- 远端: `output/isaac_traj_v4_solver_{tag}.pt` × 7（SDK 自动上传）
- 本地: `tools/x1_pipeline/solver_scan_analysis.py` → `output/solver_scan_analysis.json`（per-config 表 + per-joint 表 + 探针 + 自检 + 判决）
- 报告: SW-R004（含 idear round4/addendum5 逐条 accepted/deferred/rejected 记录）

## 6. 诚实性声明

- 本预注册先于扫描脚本提交运行、先于任何扫描输出查看（时间线将以 git commit 与任务创建时间戳佐证）
- 判决只覆盖「Isaac Gym preview PhysX（TGS/PGS × 迭代数 × pipeline）vs MuJoCo harness」参数族；Isaac Lab（PhysX 5）行为不在本实验范围
- 阈值 2.0/6.0 沿用 idear round4 提出并被我采纳的数值，非本实验后调整
