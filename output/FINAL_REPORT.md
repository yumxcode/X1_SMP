# X1 29DOF SMP 训练与 MuJoCo Sim2Sim — 最终报告 v3

日期：2026-09-28 | 仓库：yumxcode/X1_SMP（main @ 9bb8072+）| 工件在 output/ 与 data/

---

## 一、总裁决（本轮 v3）

| 交付项 | 裁决 | 依据 |
|---|---|---|
| ① G1→X1 重定向精度（用户三项视觉问题） | ✅ **修复并定量达标** | 根因=脚底碰撞盒镜像错置（见 §二）；v3 重定向：支撑相脚底倾角 med 2.4-6.1°（v2: 21-57°）、最终帧穿模 0mm（v2: 8.0-8.6cm×63-88%帧）、hip 抖动 ≤源×1.5 且 max 26°→3-5°（外展抖动根因=摆动腿 IK 分支翻转，已钉住） |
| ② 数据集与验证器 | ✅ 12/16 片段严格 PASS（R1-R9+J1-J3，sample_step=1）；4 排除均为源级问题（穿步×3/节奏×1）；验证器自证（站立正例过/v2 坏例败） |
| ③ SMP 训练（Isaac 侧） | ✅ 固定资产代 v3b：reward 0.27@it2747，Isaac eval 8/8×3ep×10s（\|vx\| 1.3-1.8 m/s，root_z 0.58 与正确站立 0.6016 自洽） |
| ④ MuJoCo sim2sim（pos-mode 策略） | ❌ 0/4×10s（<0.5s 倒）——**引擎驱动语义差异同代坐实**（§四证据链），与用户怀疑(2)兼容：他任务可迁移的条件未被 pos-mode 训练重现 |
| ⑤ sim2sim 解决路线 | 🔄 pd_explicit 训练中（TASK_20260928_017）：Isaac 逐子步显式 clip(kp·Δq−kd·qd,±tlim)+DOF_MODE_EFFORT，与 MuJoCo harness 逐语义相同——构造性对齐，结果待续报 |

---

## 二、根因：脚底碰撞盒镜像错置（本轮最大发现）

**X1 左右 ankle_roll body 系是镜像的**（左脚底在 local y=−0.0408，右脚在 y=+0.0408；body quat 左 180°-y vs 右 180°-x），而 `build_x1_assets.py` 给两脚放同一 `pos="0 −0.043 0"` sole 盒且覆盖不足（±0.032×0.072 vs 真实足迹 ±0.056×0.098）：
- 右脚盒在**脚背上**（世界 z +7.2cm）
- 左脚盒**下挂**（−7.4cm，薄轴未对齐竖直，角点 z 跨度 14.4cm）

**污染链**：v2 数据在修正几何下实测双脚埋地 8.0-8.6cm/63-88% 帧（旧 R3 门量坏盒故"通过"）→ 训练数据、Isaac 训练资产（x1.xml 同源 bug）、sim2sim 验证全线污染。旧报告"Isaac 能跑"是策略在这对共生 bug（坏资产训练+坏数据重置）上的过拟合——旧 v2 policy 在修正资产上 0/4 秒倒（pitch ±85-139°）予以证实（`output/sim2sim_v2_policy_on_fixed_assets.json`）。

**修复**（commit d44ec33，远端 md5 eaa2715788be 逐字节验证）：按侧放置 y=±0.0303、底面比 mesh 脚底低 1.5mm、全覆盖足迹；站立高度重算 0.6016m。

> 教训入经验库：c59edcf 的 git add 因混入不存在路径而**整体静默失败**，资产修复一度未达远端（中途审查捕获）；跨机交付必须远端回读+内容指纹闭环，现已固化为训练脚本启动日志（commit+md5）。

---

## 三、重定向 v3（要求 1）：三项视觉问题的修复与度量

| 用户所见 | 定量根因 | v3 修复 | 实测 |
|---|---|---|---|
| 脚底不平 | IK quat 权重 0.5 vs 位置 12-30×3cm——姿态跟踪形同虚设；踝顶限位 | 支撑相改「脚底法线∥重力」2dof 约束（yaw 自由）+ IK 后踝关节解析调平 | 支撑倾角 med 2.4-6.1°/p90 13.7-18.4°（v2: 37-40°/53-64°） |
| 脚地穿模 | 旧盒错置+lift 在最终平滑前 | 最终落盘帧闭环（8角点 FK 迭代抬升至 +0.5mm） | min sole z=+0.5mm 全片段 |
| 腿瞬间外展 | 摆动腿 hip_roll/yaw IK 分支逐帧翻转（±0.3-1.1rad 交替） | 解析参考 4Hz 预平滑+先验权重 3.0+连续性 0.8+Hampel | hip_roll max 26°→3-5°；hip_yaw p99≤源×1.5（源本身 6.8-10.2°，绝对 6° 门为幻影标准已改相对） |

验证器 v3 新门：R7 支撑放平（planted=角点+盒中心双条件，防脚尖戳地误判）、R8 最终帧零穿模、R9 源相对抖动。mesh 渲染对比视频（X1 侧用原始 mjcf 网格，与 URDF 渲染一致）：`output/renders/v3_{run1_subject5_seg0,sprint1_subject4_seg1}.mp4`。

---

## 四、Sim2sim（要求 2）：同代证据链与引擎差异定位

### 4.1 已排除（同代同资产，全部落盘）

1. **资产/数据**：固定资产+干净数据重训（v3b）后 Isaac 8/8 vs MuJoCo 0/4 并存——非资产问题
2. **重置态**：固定代 dump 重置帧穿地 +1.2mm、ncon=0、t=0 obs 逐位一致（2.3e-5）——非病态重置
3. **模型参数**：Isaac 审计 vs MuJoCo 逐项（TASK_20260928_013）：质量/惯量/armature **100% 一致**；frictionloss Isaac 读 0（MuJoCo 0.2-4.0 已清零对齐）；velocity 限幅全 100 默认（未采 URDF）
4. **参数化对齐**：kp/kd 缩放（含 kp×2：rms 1.63→1.31，max 恒 3.3）、原生 servo+implicitfast、margin/solimp/solref 扫参（10 配置全灭）、摩擦清零、armature、effort 限幅（去 clip+ctrlrange 放宽均不贴合）、动作延迟（×0/1/2）、双重阻尼 bug 修复——全部无效或仅边际
5. **增益随机化 ±30%**（T007）：it1800 仍 0/4（0.4s）——带宽不足以覆盖引擎差

### 4.2 已坐实

- **首控制步（33ms）dof_vel 即发散 3.3 rad/s**（干净重置、无接触、t=0 obs 一致）——分歧源于关节级驱动求解语义，经 PD 高增益放大（`dump_v3b_diff.py`，dump TASK_20260928_003）
- **airborne 阶跃**（无接触）：Isaac 关节响应快 MuJoCo **1.2-2.2×**（peak 6.4-6.9 vs 3.2-5.8 rad/s）——纯驱动语义差异，与接触无关（probe T009 + `airborne_step_mj.py`）
- Isaac dof 力传感器 ≈ 1-5 N·m vs PD 公式 40-80 N·m（传感器含约束反力，非驱动扭矩——仅提示求解器语义不同，不作定量依据）

**结论**：Isaac Gym `DOF_MODE_POS` 的 PhysX TGS 隐式位置驱动与 MuJoCo 任何参数化显式/伺服实现存在**求解器级语义差异**，首步 3.3 rad/s 的关节速度差超出当前策略稳定域。用户怀疑(2)（同 URDF 他任务可通 sim2sim）与之兼容——那些任务大概率是 Isaac **Lab**（非 Gym preview）/effort-mode/低动态；本任务的 pos-mode 隐式驱动条件正是不可迁移的那类。

### 4.3 解决路线：pd_explicit（进行中）

MimicKit `pd_explicit` 引擎模式：Isaac 端逐物理子步计算 `τ=clip(kp·(q_tar−q)−kd·q̇, ±tlim)` 并以 `DOF_MODE_EFFORT` 施加——与 MuJoCo harness **逐语义相同**（同样的公式、限幅、施加点）。动作界/obs 与 pos 模式完全一致（char_env 共享 `_build_action_bounds_pos`），v3 prior 与 v3b it2700 热启动有效。TASK_20260928_015→017（平台两次回收后续训）：pd_explicit 生效验证（reward 0.001 起步重适应，属预期——等价于在 MuJoCo 同语义动力学下重学）。**裁决待收敛后补：PASS 则构造性对齐成立；FAIL 则调查接触建模差异。**

---

## 五、工件清单（v3 增量）

| 类别 | 路径 |
|---|---|
| 资产修复+验证 | `build_x1_assets.py`（d44ec33）；`output/diag_v3/{sole_box_vs_mesh,sole_footprint,verify_sole_fix}.py` |
| 重定向 v3+验证器 v3 | `tools/x1_pipeline/{retarget_v3,validate_retarget_v3,batch_v3,build_dataset_v3,render_clip_v3}.py` |
| v3 数据集（12 clip） | `data/datasets/dataset_x1_run_v3.yaml` + `data/motions/x1_v3/` |
| 对比视频（mesh） | `output/renders/v3_{run1_subject5_seg0,sprint1_subject4_seg1}.mp4` |
| sim2sim（--env 可选） | `sim2sim_validate.py`（frictionloss 对齐固化）+ `aligned_actuator_test.py`（双重阻尼修复） |
| 引擎差异证据链 | `{dump_v3b_diff,dump_v3_channels,first_step_ode_bench,drive_scale_match2,airborne_step_mj,margin_test,cushion_sweep,reset_dz_probe,replay_by_torque,drive_identify}.py`；dump `isaac_traj_v3_fixed.pt`/`isaac_traj_v3b_fixed.pt`；审计 `isaac_model_audit`（T013） |
| 远端探针 | `scripts_remote/{probe_dof_props,audit_model_parity,dump_traj_v3,dump_traj_v3b_forces}.py` |
| 训练配置族 | `data/envs/smp_x1_env_v3.yaml`、`data/engines/isaac_gym_engine_pdx.yaml`、`tinymdm_x1_run_v3.yaml`、`run_smp_pdx*.py` |
| 权重（固定资产代） | `output/remote_ckpt/{smp_v3b_it1800,smp_v3b_it2700,smp_v3c_it1800}.pt`、prior `data/models/smp_priors/x1_run_v3_prior.pt`（md5 2731005c） |

## 六、诚实性声明

- sim2sim 裁决均按 S1-S4 全门 4ep×10s，未放宽；v3b/v3c（增益随机化）均 0/4 如实报告
- 12/16 门控、4 排除片段及理由（穿步/节奏）如上；未混入
- 一次中途审查纠正了"资产已修复"的假声明（git add 静默失败），代际作废重做了受污染结论——本报告全部"已证实"项均锚定固定资产代（x1.xml md5 eaa2715788be）工件
- pd_explicit 路线训练中，未提前宣布成功
