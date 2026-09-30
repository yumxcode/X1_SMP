# SW-PREREG-003 — IDEA-012r Newton 入场探针判决阈值预注册（三配置矩阵 + 零误差子探针）

- 报告 ID: SW-PREREG-003
- 时间: 2026-09-30 15:4x（本 commit 先于任何 Newton 探针任务创建与输出查看）
- 作者: smp_worker
- 基线: SW-R004（d6ca851c，H2'' 判决）+ e2ed321c
- 关联: idear round5（IDEA-012 原案，`smp-ideas-idear-round5-20260930`）、round6（IDEA-012r 修订 + pdx2-mirror diff，`smp-ideas-idear-round6-20260930`）、addendum6（接口静态平价）
- 实验载体: `scripts_remote/probe_newton_entry.py`（新），复用 SW-R004 的 dump/分析协议族

## 1. 实验设计（冻结）

| 项 | 值 |
|---|---|
| 被测引擎 | mimickit 内置 Newton 引擎（`newton.solvers.SolverMuJoCo`，solver="newton", iterations=100, ls_iterations=50） |
| 资产 | `data/assets/x1/x1_v4.xml`（MJCF 直载，零转换；stiffness/damping/armature/frictionloss 为引擎增益来源） |
| reset | 同 r4：类级 patch `_sample_motion_times`→(motion 0, time 0)，角色固定于 motion0/t0 帧 |
| 对照指令 | **开环回放** r4 GPU 锚 dump（isaac_traj_v4_solver_tgs4_0_gpu.pt，已 commit d6ca851c）记录的 q_tar 序列（30 控制步）——纯动力学对照，不依赖策略 |
| 零误差子探针 | q_tar 钉 q0，8 控制步（同 r4 协议） |
| sim 频率 | 全配置统一 120Hz（新建 probe engine yaml；引擎默认 240 不用） |
| device | warp cuda:0（若镜像无 GPU warp 则任务 fail-fast 报告，不判） |

### 配置矩阵（单一主轴=kd 语义；impratio 为副轴）

| 配置 | control_mode | pd_kd_mode | impratio | 角色 |
|---|---|---|---|---|
| A_explicit | pd_explicit | explicit（引擎现行为，全显式 kd） | 10 | 负对照：v4 臂链 kd·dt/m 上界≈1.67 贴显式稳定边界（idear round6 §新核查；armature 口径上界，实际惯量更大可能更稳） |
| B_implicit | pd_explicit | implicit（pdx2-mirror：显式 kp + kd 保留为 dof_passive_damping 交求解器隐式积分） | 10 | **主判据**：与 sim2sim harness（显式 kp+隐式 kd）语义同构 |
| C_pos | pos | —（POSITION target，求解器侧 kd 语义未验证） | 10 | 顺带数据点 |
| D_impratio1 | pd_explicit | implicit | 1.0 | 副轴：harness/MuJoCo 默认 impratio 对齐 |

## 2. 判决规则（预注册，采用 idear round6 IDEA-012r 条件）

主判据 = **B_implicit** 配置的 first-step max |Δdof_vel|（Newton vs MuJoCo harness replay，rad/s）：

- **B < 1.0** → Newton 路线绿灯：训练引擎可换 MuJoCo 求解器族，H2'' 分歧源被移除；进入路线决策与 Isaac Lab 探针（IDEA-010r）并表
- **B ∈ [1.0, 5.0)** → 灰区：warp 移植对等性/特性子集排查（H-N2 路径），Newton 保留候选但不绿灯
- **B ≥ 5.0** → MuJoCo 族内部移植差坐实，Newton 路线降级，与 Isaac Lab/自建 MuJoCo 原生并评

副判据（预注册解释规则，不改变主判决）：
- A_explicit 若显著差于 B_implicit（>2x）→ 证实显式 kd 语义在 v4 增益下伪影显著（负对照检出力成立）；若 A≈B → v4 增益下显式 kd 未失稳（idear 上界分析成立），kd 语义轴对本增益组合非决定项
- D_impratio1 与 B_implicit 差异 → impratio 轴贡献量级记录
- 零误差探针（q_tar 钉 q0）：若仍 >2 → 约束/重力路径在 warp 移植内也有差；若 <0.5 而 rollout 残差大 → 差异在指令跟踪路径

## 3. 有效性自检（每配置必须通过才计入判决）

- 初始状态对齐：Newton reset 后 dof_pos/root_pos 与 r4 GPU 锚 dump 首帧 max diff < 1e-5（同一 motion0/t0 帧的独立加载，容差不设 0 因 MJCF 双端解析路径不同）
- to_np 强制 .copy()（r3 教训固化）；obs0/状态哈希落盘
- done 截断：与 r4 相同，首个 early-termination reset 处截断可比窗口
- 镜像环境报告：任务日志打印 newton/warp 版本；若 import 失败或引擎 API 不匹配 → 任务 fail-fast，本轮无判决（环境缺失不是实验结果）
- 引擎改动最小性：`pd_kd_mode`（默认 explicit）与 `impratio`（默认 10）为 yaml 配置键，默认值保持引擎现行为逐位不变

## 4. 输出契约

- 远端: `output/newton_probe_{tag}.pt` × 4（SDK 自动上传）
- 本地: `tools/x1_pipeline/newton_probe_analysis.py` → `output/newton_probe_analysis.json`（per-config 表 + per-joint 表 + 零误差探针 + 自检 + 判决 + 与 SW-R004 并表列）
- 报告: SW-R005

## 5. 诚实性声明

- 本预注册先于探针任务创建；Newton 引擎代码本地存在但从未运行（本地无 warp/newton），镜像可用性未验证——若环境不可得，按 §3 fail-fast 规则记录阻塞而非判决
- 判决仅覆盖「Newton(SolverMuJoCo/warp) vs MuJoCo CPU harness」的 first-step 残差；吞吐、SMP 训练管线兼容性、frictionloss/armature 特性对等性均为后续项（IDEA-013 清单）
- 开环回放使第 2 步起的状态轨迹与 Isaac 闭环不同——first-step 指标不受影响（同一初始状态+同一首条指令），后续步指标仅作 open-loop 发散参考
