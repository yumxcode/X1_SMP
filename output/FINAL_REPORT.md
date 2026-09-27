# X1 29DOF SMP 训练与 MuJoCo Sim2Sim — 最终报告

日期：2026-09-27 | 仓库：yumxcode/X1_SMP（main @ bafcc59+）| 全部工件在 output/ 与 data/

---

## 一、总裁决

| 交付项 | 裁决 | 依据 |
|---|---|---|
| ① G1→X1 重定向 + 严格标准 | ✅ **PASS** | 11/16 片段全门控通过（J1-J3 关节域 + R1-R6 几何/时序域），验证器自证成立，3 支 G1/X1 对比 MP4 |
| ② X1 SMP 训练 | ✅ **PASS（Isaac 侧）** | prior v2 loss 0.279→0.029@200k；policy v2 续训至等效 ~it4493；**Isaac 侧 8/8 env × 10s 稳定奔跑**（3 episodes，|vx| 中位 4.5-5.3 m/s 路径度量，root_z~0.51m） |
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

**最终 policy（smp_v2_policy_final.pt）：0/4×10s PASS**（0.5-1.5s 内前扑摔倒；`output/sim2sim_smp_v2_final.json`；视频 `output/renders/sim2sim_v2_final_seed1.mp4`、`sim2sim_v2_fall_seed0.mp4`）

### 4.3 根因证据链（全部定量）

**已排除**：
1. reset 分布外——验证器曾从 v1 旧数据集 reset（t=0 pd_gap 3.07rad 秒倒），修复为 env yaml 同源 v2 数据集后 pd_gap→0.5rad，仍倒
2. 接触参数——solref/condim/摩擦 4 种配置对 fall time **零影响**（0.7/0.9/1.0s 不变）
3. obs 角速度通道敏感性——angvel 通道衰减 ×0.5/×0.0，fall time **零变化**
4. 单关节资产异常（Isaac 右踝 roll 钉 +0.64 限位）——MuJoCo 锁 0/锁 0.64/轴翻转均无效
5. 验证 harness 自身——躯干姿态曾读在带 90° URDF 旋转的 lumbar_pitch_link 上（已改 base_link）、pitch 曾用 yaw 相关公式（已改 yaw 无关）、速度曾用有符号 dx（已改水平模长）

**已证实**：
1. **Isaac 侧同一策略 8/8×10s 稳定奔跑**（对照实验 TASK_20260927_005）
2. **obs 计算逐位一致**：同初态 t=0 全 231 维 obs 差 0.000
3. **开环动作重放 1 个控制步内角速度发散 ~1 rad/s，1.5s 内 root_z 差 0.4m**——同动作序列下两引擎接触冲量级动力学分歧（MuJoCo timestep 1/120 与 Isaac 相同）
4. **正例对照**：重定向参考轨迹直接作 q_tar 开环回放，MuJoCo 1.1s 摔倒（`tools/x1_pipeline/positive_control_replay.py`）——运动学重定向数据本身无动力学自稳能力，策略平衡完全依赖训练引擎的闭环动力学

**结论**：sim2sim 失败不是"数据没过门控"或"训练不够"，而是 **Isaac Gym 与 MuJoCo 的接触求解/驱动实现差异**（该 X1 29DOF 资产在 Isaac 侧存在右踝 roll 钉限位的关节伪影，策略的平衡解依赖该特定动力学）。

### 4.4 建议下一步（超出本任务范围，供参考）

1. **修 Isaac 资产关节伪影**：排查 X1 x1.xml 在 Isaac 解析下的右踝 roll（限位 +0.64 钉死）后重训——最可能一步解决
2. 或 **MuJoCo 域适应训练**：在 MuJoCo 里 finetune（domain randomization 覆盖接触参数）
3. 重定向数据加动力学可行性过滤（如 ZMP/LIPM 校验）再入训练

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
| 权重 | `output/remote_ckpt/{prior_v2_final,smp_v2_policy_it1900,smp_v2_policy_final,smp_v9_robust_it3814}.pt` |
| 诊断工具 | `tools/x1_pipeline/{diag_first_sec,positive_control_replay,render_sim2sim}.py` |
| 关键 commits | 47ebb19(数据) 9caf237(prior) ffe3be4(验证器修复) bafcc59(resume 搜索) |

## 六、诚实性声明

- sim2sim 按 4.1 标准裁决 **FAIL**，未做任何放宽标准的操作使其"通过"
- 5/16 重定向片段未过门控被排除，未混入训练数据
- Isaac 侧 8/8 结果来自远端真实 eval 任务日志，非本地推断
- 旧验证器（修复前）产出的 v9/AMP sim2sim JSON 与本报告结论一致但指标受污染，以修复后 harness 重跑结果为准
