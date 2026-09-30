# smp_worker 报告 007 — H-X1-IL 全 X1 入场探针判决：REFUTED（F1 19.03 / F2 限位退让 19.02），MuJoCo 原生路线升唯一主线

- 报告 ID: SW-R007
- 时间: 2026-09-30 20:2x（commit 时间戳为准）
- 作者: smp_worker
- 预注册: SW-PREREG-005（15151d1a，F1）+ SW-PREREG-005a（facd57ea，F2 补充判据，先于 F2 任务）
- 判决数据: TASK_20260930_067（F1）+ TASK_20260930_069（F2）+ `output/x1_il_entry_analysis.json`、`output/x1_il_entry_F2_analysis.json`
- 关联: SW-R006（单关节 H-IL_GREEN）、SW-R004/005（IsaacGym H2'' / Newton DOWNGRADED）、idear round6 IDEA-010r
- 环境: IsaacLab 6.1.14（镜像 BJX00000335 V000386），cuda:0，MjcfFileCfg 直载 x1_v4.xml

## 摘要

按 SW-PREREG-005 预注册规则，全 X1 入场探针实测：**F1（motion0/t0 注入）first-step 19.03 rad/s ≥ 2.0 → H-X1-IL REFUTED**。判决后诊断发现初始帧多关节压限位（elbow_yaw 1.7999/1.800 等）且双端限位处理不同，按 SW-PREREG-005a 预注册补跑 F2（限位退让 0.05 rad）：**19.02 ≥ 2.0 → 限位邻域特异解释排除，一般族差坐实**。预注册链（SW-R005 §5.2 → SW-R006 §5 → 本判决）生效：三条迁移路线全部关闭，**自建 MuJoCo 原生训练引擎（gap=0 by construction）升为唯一主线**。

## 1. 判决证据（dump md5 已核对：F1 b0e6900fb8e3 / F2 18712ca9cdc7）

| 探针 | 注入状态 | first-step max | mean rms | 零误差探针 | worst joint | state0 自检 |
|---|---|---|---|---|---|---|
| F1（TASK_067） | motion0/t0 原帧 | **19.028** | 42.39 | 19.09 | left_elbow_pitch | pos 0.00 / vel 0.00 |
| F2（TASK_069） | 限位退让 0.05 rad | **19.018** | 42.44 | 19.07 | left_elbow_pitch | pos 1.43e-08 / vel 0.00 |

- 增益核对：8 组 (kp,kd,tlim,armature) 显式 ImplicitActuatorCfg，`get_dof_stiffnesses/dampings` 回读与锚 dump 逐位一致（dev 0.00）——**MjcfFileCfg 不映射 MJCF joint stiffness/damping 到 PhysX drive（r1 全零回读实证），增益必须显式注入**
- 执行器序验证：`write_joint_state_to_sim`/`set_joint_position_target` 参数序=joint_names 序；r6 诊断（全 29 向量回读恒等）+ r7 腕编码诊断双重实证；r8 根因修复（写入向量须 scatter W[perm[i]]=dp[i]，旧代码 dp[perm] 双重置换）
- 家族残差（30 步均值 |dv|）：肘族 14.7 / hip_pitch 6.2 / 腿接触族 5.3 / 腕 1.05——与 SW-R004（IsaacGym）肘族震中模式一致

## 2. 判决后机制诊断（已观测，免费本地证据）

1. **限位邻域假设被 F2 排除**（预注册判据）：退让后残差 19.03→19.02 无变化。F1 中 IL 侧 left_elbow_yaw 钉死 q=1.800/v≡0 达 6 步、right_elbow_yaw ±44.75 限位抖振是**真实差异**但非残差主因。
2. **接触排除**：t0 腾空（MuJoCo ncon=0，nefc=0）——残差发生于**无约束浮基多体动力学**。
3. **kd 语义排除**：harness 回放改显式 kd 更差（73.3 vs 19.0）——隐式阻尼语义正确。
4. **MuJoCo 离散化排除**：dt 1/120→1/960 收敛 -19.1→-17.7（elbow_pitch），非数值误差。
5. **跨引擎对称异常（未解决，如实记录）**：MuJoCo 收敛值（elbow_pitch -17.7~-19.1）与 IL dump（-0.051，且逐控制步 ±5 振荡）在物理量级上矛盾；同族现象亦见于 SW-R004（IsaacGym 25-47 尖峰）与 SW-R005（Newton -44.75 尖峰）。三个引擎族对同一 MuJoCo harness 回放均差 19-26 rad/s，引擎互差 39.8——**全尺寸 v4 增益 + t0 状态下的跨引擎一致性没有任何一对成立**（单关节摆除外，0.09-0.26）。振荡签名（±交替、量级差 4-5x）指向更深层差异（drive 模型/多体耦合积分），未在本轮定位——按预注册纪律不以此重开路线，记为 MuJoCo 原生路线时代的开源诊断项。

## 3. 路线终局（预注册链完整闭合）

| 路线 | 判决门 | 结果 | 依据 |
|---|---|---|---|
| Isaac Gym solver 配置 | SW-PREREG-002 ≥6.0 | **关闭**（22.4-30.1） | SW-R004 |
| Newton (mujoco-warp) | SW-PREREG-003 ≥5.0 | **降级**（19.12） | SW-R005 |
| Isaac Lab（PhysX 5 隐式）单关节 | SW-PREREG-004 <1.0 | GREEN（0.094/0.259） | SW-R006 |
| Isaac Lab 全 X1 | SW-PREREG-005 <1.0 / ≥2.0 | **REFUTED**（19.03；F2 排除限位混杂 19.02） | 本报告 |

**决定：自建 MuJoCo 原生训练引擎为唯一主线**——训练仿真器=sim2sim 验证仿真器，gap=0 by construction，无需任何跨引擎对齐；v4 系全部资产（x1_sim_v4.xml/prior/data）直接复用。次级含义：MuJoCo 原生路线的工程成本=向量化训练吞吐（mujoco.rollout/mjtNum 批量 or 多进程），此前 SW-R003 列为 fallback 自建成本高，现为主线必经。

## 4. harness 迭代披露（r1-r8，8 任务）

r1 增益全零（MjcfFileCfg 不映射→显式分组）→ r2 状态写入散乱（dp[perm] 双重置换；数据被 r6 全向量诊断证伪、r7 腕编码证恒等后 r8 scatter 修复）→ r4 get_joints 不存在（6.x API）→ r5 warp 单数组转换 → F2 自检比对须用裁剪后期望。全部缺陷均由预注册自检（增益/state0 fail-fast）或诊断任务捕获，无一污染已发布判决；F1/F2 判决数据在 r8 后干净。

## 5. 下一步（MuJoCo 原生主线工作包分解）

1. 吞吐侦察：mujoco.rollout（3.x 批量 API）在 M 系芯片/CPU 的 envs 规模实测 vs Isaac Gym 4096 envs 基线——决定训练时长预算（若不足，评估远端 GPU MuJoCo 或多进程 CPU）
2. 引擎骨架：mimickit engine 接口的 MuJoCo 实现（复用 mjcf_char_model/motion_lib/agent 栈，仅换 sim 层；隐式 kd 语义天然一致）
3. 冒烟：单 env 短 rollout 对齐 sim2sim_validate.py 逐步（gap=0 应给出逐位一致）
4. 训练重启 → 导出 → sim2sim 闭环验收（SW-R006 §5.5 协议）

## 6. 诚实性声明

- 两判决均严格按预注册带执行（F1/F2 同带）；F2 在查看 F1 判决后预注册（SW-PREREG-005a），角色是分层诊断非翻案
- §2.5 跨引擎对称异常未定位（含「MuJoCo harness 侧多体尺度怪异」这一未被排除的竞争解释——单关节正例不足以完全豁免 harness 多体路径）；MuJoCo 原生路线对该异常**构造性免疫**（训练=验证同引擎），这是路线选择的主要依据之一，如实说明
- F2 的退让状态不是数据集真实帧（诊断探针）；q_tar 序列未裁剪（双端对称）
- sim2sim 仍未通过；本报告是路线收敛判决，非验收
- 判决覆盖 IsaacLab 6.1.14/cuda:0 单环境；多 reset 帧/闭环行为未测（预注册 §5 声明范围）
