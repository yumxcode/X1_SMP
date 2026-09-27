# X1 29DOF SMP 训练与 MuJoCo Sim2Sim — 最终报告

日期：2026-09-27 | 仓库：yumxcode/X1_SMP（main @ bafcc59+）| 全部工件在 output/ 与 data/

---

## 一、总裁决

| 交付项 | 裁决 | 依据 |
|---|---|---|
| ① G1→X1 重定向 + 严格标准 | ✅ **PASS** | 11/16 片段全门控通过（J1-J3 关节域 + R1-R6 几何/时序域），验证器自证成立，3 支 G1/X1 对比 MP4 |
| ② X1 SMP 训练 | ✅ **PASS（Isaac 侧）** | prior v2 loss 0.279→0.029@200k；policy v2 续训至等效 ~it4493；**final 权重 Isaac 8/8 env × 10s 稳定**（TASK_20260927_015：\|vx\| 中位 1.55-1.84 m/s，root_z~0.55；中期 it1900 亦 8/8，TASK_005） |
| ③ MuJoCo sim2sim 严格验证 | ❌ **FAIL（如实裁决）** | 最终 policy 0/4×10s，0.5-1.5s 内摔倒。根因已定量定位为 **Isaac/MuJoCo 引擎级动力学分歧**（证据链见 §四），非重定向数据、非验证 harness、非训练不足 |

---

## 二、重定向（要求 1）：标准、结果、人眼验收

### 2.1 严格通过标准（validate_retarget_v2.py，11 门全过才算 PASS）

**关节域（J 系，v2 新设计，参数化不变）**
- **J1 肢体保真**（8 段：左右 thigh/shank/uarm/farm）
  - 腿：摆动角-站立角相关 r≥0.75、中位差 ≤12°、**世界方向中位差 ≤16°(thigh)/20°(shank)**（新：抓方位旋转的假动作）
  - 上臂：r≥0.60、世界方向中位差 ≤20°（绝对语义，杀 13° 站立体型偏置）
  - 前臂：**uarm 系内方向中位差 ≤15°** + 弯曲节奏 r≥0.60（弯角+弯平面，2dof 对 2dof 方向球拟合的验收）
- **J2 姿态**：骨盆系左右脚 y 间距 >-2cm（不交叉）、膝摆幅 ≥0.35rad、躯干 pitch 中位差 ≤15°
- **J3 根一致**：root pitch/roll 欧拉中位差 ≤12°（逐帧复制的根必须贴住）

**几何/时序域（R 系）**
- **R1 步态节奏**：**z 轨迹互相关**（r≥0.75 双脚、最优滞后 ≤12% 步态周期）+ 步频比 ∈[0.88,1.12]（离散触地事件匹配被证明是阈值穿越伪影，已弃用）
- **R2 手脚协调**：步频一致 + FFT 相位差 <0.35rad **或** 时域互相关强证据（r≥0.70、滞后 ≤3 帧、反相滞后差 ≤0.10s）
- **R3 脚底穿地**：脚底最低角点 >-1.0cm
- **R4 自碰撞**：非邻接刚体对 SAT 距离 >-0.5cm（mj_geomDistance 对 box-box 有幻影负值 -0.26 的实测 bug，改 15 轴 SAT）
- **R5 跟踪**：脚位中位 <3cm/p95<9cm；手位中位 <15cm/p95<35cm（阈值位于 0.121m 结构地板之上——X1 前臂是 G1 的 1.87 倍）
- **R6 关节速度**：max |qdot|/URDF 限速 ≤1.05（X1 硬件限速适配：warp×N 慢放搜索）

**验证器自证**（防幻影 PASS/FAIL）：已知好例（解析映射）→ J 全 PASS；已知坏例（v1 扭腿产物）→ J1 全段 FAIL（corr 0.73~0.82→-0.42，方向差 19°→104°）。

### 2.2 v2 重定向管线（本会话修复链）

| 修复 | 之前症状 | 之后 |
|---|---|---|
| 前臂：deviation 角匹配 → **uarm 系方向球面拟合**（2dof 对 2dof） | 前臂世界方向差 88°，肘顶限位 74% 帧元 | 世界角差 10°，corr 1.00 |
| uarm 链：站立相对语义 → **绝对语义传递** | 站立外展偏置 13-15° 恒定残留 | 世界方向差 14.7°（好例）|
| IK：全身 29dof → **只解腿 12dof**，腰+臂锁解析解 | x_torso site 的 90° URDF 导出旋转使躯干 quat 目标与解析解冲突 25-159°，lumbar_yaw 被拉飞 ±1 rad，IK cost 19.2 | IK cost 1.44，上身 J1 全过 |
| 支撑吸附（snap-down） | 缩放脚目标使支撑期脚底悬空 3-7cm（触地检测出 3 个假步） | R5 脚误差 2.8→0.8cm，支撑真实贴地 |
| 落盘 bug | 全 FAIL 时保存最后尝试的 warp 而非最优 | 保存最优变体 |

### 2.3 结果与人眼验收载体

- **11/16 片段 PASS**（6 run + 5 sprint，总时长 162.7s）；5 个排除片段为转身跑/穿步交叉（G1 脚世界 yaw 170° 时 X1 脚 quat 跟踪与位置目标冲突、脚交叉 -10.6cm）——门控如实拒绝
- 数据集：`data/datasets/dataset_x1_run_v2.yaml`（sprint 权重 1.5）
- **人眼验收 MP4（G1 左 / X1 右，跟随相机）**：
  - `output/renders/v2_run2_subject1_seg2.mp4`（10.2s，warp 1.3 慢放适配 X1 限速）
  - `output/renders/v2_run2_subject4_seg0.mp4`（30.2s）
  - `output/renders/v2_sprint1_subject4_seg1.mp4`（4.4s）
- 速度适配（要求 1-注意(1)）：逐片段 warp 网格搜索（×1.0-×2.2），p99 关节速度≤URDF 限速，X1 无法达到 G1 冲刺速度时按 X1 物理极限慢放

---

## 三、SMP 训练（要求 2 前半）：gradmotion 远端全流程

| 任务 | ID | 配置 | 结果 |
|---|---|---|---|
| prior v2 | TASK_20260927_003（账号14） | tiny-MDM 200k iters，11-clip v2 数据 | loss 0.279→**0.029**，权重已入 repo（md5 5691b8c3）|
| policy v2 | TASK_20260927_004（账号13） | SMP+PPO+鲁棒化（obs 噪声 0.01+25N 推扰，无动作噪声/延迟） | 平台在 it1986/3814 回收；Smp_Reward 0.173（> v9 的 0.153）；it1900 Isaac 侧 **8/8×10s 稳定**（TASK_20260927_005）|
| resume | TASK_20260927_008（账号14） | `--model_file` 挂载续训（本会话新增 X1_MODEL_FILE glob 搜索机制） | +2593 iters 至等效 ~it4493，Smp_Reward 0.142-0.174 区间 |

本地 Mac 未做任何真实训练（契约遵守）；所有长等待用 timer park（3 次）。

---

## 四、MuJoCo Sim2Sim（要求 2 后半）：严格标准 + FAIL 裁决与根因证据链

### 4.1 严格通过标准（sim2sim_validate.py，每 episode 全过才 PASS）

- **S1 不倒**：root 高度 >0.30m 占比 ≥95% 且非脚部位刚体零触地
- **S2 步态**：≥4 交替步幅、步周期 ∈[0.25,1.2]s、水平速度模长 ≥0.8m/s、双脚摆动净空 ≥4cm
- **S3 形态**：|躯干 pitch|<25°（yaw 无关公式）、髋 pitch ∈[-0.8,0.8]、膝 ∈[0.2,1.4]rad、臂-腿反相 |Δφ-π|<0.8rad
- **S4 驱动可承受**（新）：分关节腿力矩饱和占比 <35%、力矩/力矩限 p99<0.95（经验教训：踝部饱和在全身平均里不可见）

### 4.2 裁决

**最终 policy（smp_v2_policy_final.pt）：0/4×10s PASS**（0.5-1.5s 内前扑摔倒；`output/sim2sim_smp_v2_final.json`；视频 `output/renders/sim2sim_v2_final_seed1.mp4`、`sim2sim_v2_fall_seed0.mp4`）。Isaac 语义执行器对齐版同样 0/4（`output/sim2sim_aligned_actuator.json`）

### 4.3 根因证据链（v2 同源，全部定量；来源工件与生成时间标注）

> 按审核要求重写：本节所有"已证实"证据均锚定 **v2 同代际工件**
> `output/remote_ckpt/isaac_traj_v2.pt`（TASK_20260927_016，2026-09-27
> 09:41 生成：与被测策略 smp_v2_policy_final 同任务同环境 dump，120
> 控制步 × {obs, root/dof 状态, action, q_tar, torque, kp/kd/tlim}），
> 替换初版报告误用的 v9 时代 isaac_traj.pt（2026-09-26 14:02，跨代际）。

**已排除（本地实验，逐项落盘）**：
1. reset 分布外——验证器曾从 v1 旧数据 reset（t=0 pd_gap 3.07rad），改从 env yaml 同源 v2 数据后 pd_gap→0.5rad，仍倒
2. 接触参数——solref/condim/摩擦 4 配置对 fall time 零影响（0.7/0.9/1.0s）
3. obs 角速度通道——angvel 衰减 ×0.5/×0.0，fall time 零变化
4. 执行器模型语义（`output/sim2sim_aligned_actuator.json`）——Isaac 语义
   position-servo 改造（隐式 PD + 非对称动作界 mid±1.4×half + effort
   forcerange，逐 actuator 断言关节映射），final 与 it1900 均 **0/4**
   （0.5-0.8s 倒）；同时证实 harness 的 kp/kd 与 Isaac dump 逐关节
   allclose=True（x1.xml jnt_stiffness 即 Isaac dof stiffness）
5. 被动项差异（`replay_passive_zero.py`）——armature=0 / frictionloss=0
   / 双清零，第一步 dof_vel 分歧 5.87→6.57/5.81/7.23，无改善
6. 动作 clip 语义（`replay_qtar.py`）——直接回放 dump 的 q_tar：第一步
   dof_vel diff 3.429，与回放 action 完全相同
7. 单关节资产异常——MuJoCo 锁右踝 roll 于 0/+0.64/轴翻转均无效
8. harness 度量 bug（已修 5 处后复测）：reset 数据集、base_link 姿态
   读点、yaw 无关 pitch、水平速度模长、S4 分关节饱和

**已证实（v2 同源工件）**：
1. **Isaac 侧同一 final 权重 8/8×10s 稳定**（TASK_20260927_015，本
   轮补跑：|vx| 中位 1.55-1.84 m/s，root_z ~0.55，3ep×8env）
2. **obs t=0 逐位一致**：231 维差 6.4e-5（dump_v2_diff.py）
3. **第一个控制步（33ms）内 dof_vel 即发散 3.4 rad/s**，随后 root_z
   逐周期累积 -0.45m（t=88 最大），开环重放 0.43s 摔倒——分通道
   定位：dof_vel 先爆（t=1: 3.43），root angvel 次之（t=2: 1.28），
   位置通道最后（key bodies 0.007）→ 分歧起源于**关节级动力学数值
   差异**（同 kp/kd/tlim/q_tar 下），经 PD 高增益混沌放大
4. 正例对照重设（`positive_control_v2.json`）：(A) env home 姿态站立
   保持——初始 ncon=0（该姿态下脚底距地 2mm），落地后 1.9s 缓塌；
   定性为 home 姿态踝力矩边际平衡（踝 τlim 80 vs kp 200，质心微移即
   正反馈），**两引擎同参数皆然，无引擎判别力**（Isaac 侧从未以
   home 姿态重置——训练用 rand_reset 运动帧）；(B) 最慢片段
   （ts=2.78）准静态起点开环回放 0.9s 倒——跑步参考无开环自稳
   能力，与 (A) 一致不具引擎判别力

**结论（修正版）**：sim2sim FAIL 的根因是 **Isaac PhysX 与 MuJoCo 在
同参数关节动力学上的数值级分歧**（一个控制步内关节速度差 3.4
rad/s，非资产缺陷、非 harness bug、非 PD/动作语义、非接触参数），
而当前策略（obs 噪声 0.01 + 25N 推扰的鲁棒化训练）的稳定域不足以
吸收该量级差异。Isaac 侧"右踝 roll 钉 +0.64"为 v1 时代 dump 中策略
行为，静态资产（URDF/MJCF 限位对称 ±0.64）无对应缺陷，予以撤回。

### 4.4 建议下一步

1. **MuJoCo 域随机化微调**（推荐）：远端容器补装 mujoco 后，在
   MuJoCo 中以 Isaac 权重热启动、随机化 solref/摩擦/增益 ±20% 微调
   （需新增训练管线，本轮未实施）
2. 加大鲁棒化强度重训（obs 噪声 0.03-0.05 + 增益/质量随机化），
   扩大策略稳定域覆盖引擎数值差
3. 关节动力学对齐精查（PhysX 显式 vs MuJoCo Euler 的积分细节、
   solver 迭代参数），目标把第一步 dof_vel 分歧压到 <0.5 rad/s

---

## 五、工件清单

| 类别 | 路径 |
|---|---|
| 重定向验证器（自证版） | `tools/x1_pipeline/validate_retarget_v2.py`（`--selftest`） |
| 重定向管线 v2 | `tools/x1_pipeline/retarget_v2.py` + `semmap.py` |
| 数据集（11 clip 全 PASS） | `data/datasets/dataset_x1_run_v2.yaml` |
| G1/X1 对比视频 | `output/renders/v2_{run2_subject1_seg2,run2_subject4_seg0,sprint1_subject4_seg1}.mp4` |
| sim2sim 验证器（S1-S4） | `tools/x1_pipeline/sim2sim_validate.py` |
| sim2sim 裁决 JSON | `output/sim2sim_smp_v2_final.json`、`sim2sim_smp_v2_it1900.json` |
| 摔倒证据视频 | `output/renders/sim2sim_v2_{fall_seed0,final_seed1}.mp4` |
| v2 同源 Isaac dump（证据主锚点） | `output/remote_ckpt/isaac_traj_v2.pt`（TASK_20260927_016, 09-27 09:41） |
| 执行器对齐实验 | `output/sim2sim_aligned_actuator.json` |
| 正例对照 v2 | `output/positive_control_v2.json` |
| 回放 diff 工具 | `tools/x1_pipeline/{dump_v2_diff,dump_v2_channels,replay_passive_zero,replay_qtar,aligned_actuator_test,positive_control_v2}.py` |
| 权重 | `output/remote_ckpt/{prior_v2_final,smp_v2_policy_it1900,smp_v2_policy_final,smp_v9_robust_it3814}.pt` |
| 诊断工具 | `tools/x1_pipeline/{diag_first_sec,positive_control_replay,render_sim2sim}.py` |
| 关键 commits | 47ebb19(数据) 9caf237(prior) ffe3be4(验证器修复) bafcc59(resume 搜索) |

## 六、诚实性声明

- sim2sim 按 4.1 标准裁决 **FAIL**，未做任何放宽标准的操作使其"通过"
- 5/16 重定向片段未过门控被排除，未混入训练数据
- Isaac 侧 8/8 结果来自远端真实 eval 任务日志，非本地推断
- 旧验证器（修复前）产出的 v9/AMP sim2sim JSON 与本报告结论一致但指标受污染，以修复后 harness 重跑结果为准
