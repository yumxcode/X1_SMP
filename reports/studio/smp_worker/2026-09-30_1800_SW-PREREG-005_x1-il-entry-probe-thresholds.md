# SW-PREREG-005 — H-X1-IL 全 X1 入场探针判决阈值预注册（Isaac Lab + MJCF 直载 + 隐式执行器）

- 报告 ID: SW-PREREG-005
- 时间: 2026-09-30（commit 时间戳为准；本 commit 先于任何全 X1 Isaac Lab 探针任务）
- 作者: smp_worker
- 基线: SW-R006（f8726b4d，H-IL_GREEN 单关节绿灯）+ 侦察工件 `output/isaaclab_mjcf_scout.json`（任务 056：MjcfFileCfg 可用、x1_v4.xml 29 关节直载成功）
- 假设: H-X1-IL（research hypotheses 在案）：单关节绿灯可传导至全 X1——接触 + 多关节耦合在 PhysX 5 隐式执行器下与 MuJoCo harness 对齐
- 实验载体: `scripts_remote/probe_x1_il_entry.py`（独立 isaaclab 脚本，不经 mimickit 引擎）

## 1. 实验设计（冻结）

| 项 | 值 |
|---|---|
| 资产 | `data/assets/x1/x1_v4.xml`（MjcfFileCfg 直载，fix_base=False） |
| 执行器 | ImplicitActuatorCfg(stiffness=None, damping=None)——增益从 MJCF joint stiffness/damping 读取；**加载后必须打印并与锚 dump kp/kd 逐项核对**（容差 1e-6 相对），不一致即配置失效（fail-fast，不判） |
| 初始状态 | 直接注入锚 dump（r4 tgs4_0_gpu，commit d6ca851c）第 0 帧：root pos/quat/vel/ang_vel + 29 dof pos/vel（write_root_pose_to_sim + write_joint_state_to_sim）；**state0 回读自检 <1e-5**（对 pos 与 vel 分别） |
| 指令 | 开环回放锚 dump q_tar 序列（30 控制步，X1_DOF_ORDER→articulation 关节序映射，映射关系打印核对） |
| 频率 | dt=1/120，4 子步/控制步，重力 (0,0,-9.81)，dt/integrator 由 SimulationCfg 给定 |
| 对照 | MuJoCo harness 回放（同 SW-R004/005 replay：x1_sim_v4.xml @120Hz，显式 kp tau + 隐式 dof_damping=kd，tlim clip） |
| 零误差子探针 | q_tar 钉 q0（注入态），8 控制步 |
| 关节序 | dump 记录统一 X1_DOF_ORDER（回读时按 articulation joint_names 反映射） |

## 2. 判决规则（预注册）

主判据 = F1_implicit_v4（唯一配置，单一变量=引擎侧 IsaacLab 隐式）first-step max |Δdof_vel|（vs MuJoCo harness，rad/s）：

- **< 1.0** → H-X1-IL 成立（绿灯）：接触/多关节耦合层对齐——解锁 SMP 训练管线 Isaac Lab 适配工作包（isaac_lab_engine 6.x 改造 → 冒烟 → 重训）
- **1.0 – 2.0** → 灰区：不判死，按 §3 联判（per-joint 分层 + 接触分层）后定
- **≥ 2.0** → H-X1-IL 证伪：单关节绿灯不可传导——接触/多关节耦合层存在族差；按预注册链（SW-R005 §5.2 → SW-R006 §5）**自建 MuJoCo 原生训练引擎升为唯一主线**，Isaac Lab 路线转入逐项定位（诊断项，非主线）

## 3. 灰区联判规则（预注册）

1. per-joint 表：残差集中于肘/腕（非承力链）→ 疑袖珍惯量链的执行器积分差，仍有修复空间；集中于髋/膝/踝 + 接触腿 → 接触求解族差，倾向证伪处理
2. 零误差探针：>2 → 裸动力学含接触即发散（无指令误差），倾向证伪处理；<0.5 而 rollout 大 → 指令跟踪路径，保留修复空间
3. 与 SW-R006 S1/S2（0.094/0.259）对比的量级跃升模式：若全尺寸残差 ≈ 单关节残差量级（<0.5）→ 传导良好；若跃升 1 个量级以上（>5）→ 耦合/接触主导

## 4. 有效性自检

- 增益核对（§1，fail-fast）；state0 自检（<1e-5，pos+vel 分别）；关节序映射打印（29 关节双序对照表）
- 读回强制拷贝（r3 教训）；AppLauncher 吞异常 → 以 dump 存在性为成功标准
- dump 记录：30 步全状态（X1_DOF_ORDER）+ 零误差探针 + 增益表 + 关节序表 + 版本

## 5. 诚实性声明

- 判决范围：IsaacLab 6.1.14 + PhysX 5 隐式执行器 vs MuJoCo 3.1.6 harness（x1_sim_v4.xml，Euler/Newton 求解器），单一起始帧开环 30 步；多 reset 帧/闭环策略行为不在本门
- 开环回放第 2 步起 IsaacLab 状态与锚 Isaac 轨迹分离（引擎不同），first-step 指标不受影响
- 灰区联判为预注册决定，不得事后放宽；≥2.0 分支的 MuJoCo 原生升主线决定为预注册（沿 SW-R005 §5.2 链）
