# smp_worker 报告 005 — IDEA-012r Newton 入场探针判决：NEWTON_DOWNGRADED（warp 移植差同量级）

- 报告 ID: SW-R005
- 时间: 2026-09-30 16:1x
- 作者: smp_worker
- 预注册: SW-PREREG-003（b0dbbfc1，先于任何 Newton 探针任务）
- 判决数据: TASK_20260930_047（r5，4/4 配置成功）+ 本地 `output/newton_probe_analysis.json`
- 关联: SW-R004（H2''）、idear round5（IDEA-012/013，`smp-ideas-idear-round5-20260930`）、round6（IDEA-012r/010r，`smp-ideas-idear-round6-20260930`）、addendum6
- 环境: newton 1.2.1 / warp 1.13.0（镜像 BJX00000335 V000386，IsaacSim6.0+IsaacLab3.0，python3.12/torch2.10）；warp **CPU 后端**（预注册 device 偏离，见 §3）

## 摘要

按 SW-PREREG-003 预注册规则，Newton 引擎（`newton.solvers.SolverMuJoCo`，与 sim2sim 目标同算法族）入场探针实测：**主判据 B_implicit（pdx2-mirror：显式 kp + kd 交求解器隐式积分）first-step 残差 19.12 rad/s ≥ 5.0 → NEWTON_DOWNGRADED**。idear round5 的 H-N1（同算法族 ⇒ 残差低 1-2 个量级）被证伪；H-N2（warp 移植对等性不足）以完整强度成立。零误差探针（q_tar 钉 q0，无指令误差）全配置 19.4-19.5 rad/s——裸动力学（重力+阻尼+约束）路径发散，与控制语义/策略无关。

## 1. 判决证据（TASK_20260930_047，dump md5 已核对）

| 配置 | 语义 | first-step vs MuJoCo harness | 零误差探针 | vs Isaac(锚) 直接差 | worst joint |
|---|---|---|---|---|---|
| A_explicit | 全显式 kd（引擎原样，负对照） | 19.76 | 19.52 | 18.08 | left_elbow_pitch |
| **B_implicit** | **pdx2-mirror（主判据）** | **19.12** | **19.50** | 39.82 | right_elbow_pitch |
| C_pos | POSITION 求解器 PD | 19.12 | 19.50 | 39.82 | right_elbow_pitch |
| D_impratio1 | B + impratio 1.0 | 19.19 | 19.45 | 39.82 | right_elbow_pitch |

参考系：Isaac r4 锚同协议 first-step 26.06 / 探针 20.02（SW-R004）。

有效性自检（全部通过）：
- state0 对齐：4 配置 root max diff **0.0**、dof **2.38e-07**（<1e-5）；四元数/线速度/角速度/dof_vel 额外复核至 1e-6~1e-7（本报告 §2 补充，r5 数据）
- 双 XML 物理核心字段逐位一致（body_mass/body_inertia/dof_damping/dof_frictionloss/dof_armature/qpos0 全 0.0）——排除资产混淆（actuator_gear 与 geom 数差异不影响探针路径：显式 joint_f + 各自引擎自加地面）
- to_np 强制拷贝（r3 教训固化），4 dump md5 互异且 B/C 后续步实际分歧（最大 5.5e-4）——非别名伪影

## 2. 副产物发现（已观测）

1. **B ≡ C 到 1e-5（step1 1.3e-5，全轨迹 max 5.5e-4）**：EFFORT(显式 kp + 隐式 passive kd) 与 POSITION(求解器 PD 同 kp/kd) 在 Newton 内数学等价——pdx2-mirror 语义设计的正确性获得引擎内佐证，也说明「显式 kp+隐式 kd」与「全隐式 PD」在 MuJoCo 算法族内不是分歧源。
2. **A/B 比 1.03**：kd 语义轴在 v4 增益下非决定项——idear round6 的 armature 上界推断（真实惯量 ≥ armature ⇒ 实际比值 ≤1.67，未必失稳）被实测支持；A 的后续轨迹与 B 大幅不同（step1 diff 21.7）但残差量级相同。
3. **D vs B**：impratio 10→1.0 使 step1 直接差 1.47 rad/s，但对 vs-harness 残差几乎无改善（19.12→19.19）——impratio 非主因。
4. **三角形**：Newton-MuJoCo 19.1 < Isaac-MuJoCo 26.1 < Newton-Isaac 39.8——Newton(warp) 比 Isaac 略近 MuJoCo CPU，但同量级；MuJoCo CPU 位形介于两者之间。
5. 肘族仍是震中（A/B/C/D worst 全为 elbow_pitch）。

## 3. harness 迭代与诚实披露（r1-r5，5 个任务）

| 轮 | 任务 | 问题 | 处置 |
|---|---|---|---|
| r1 | 040 | 节点 CUDA driver <12.4：引擎 CUDA graph capture 用条件图节点直接抛错 | 改 warp CPU 后端（cdf5d76f）——**预注册 device（cuda:0）偏离，此处正式披露**；CPU/GPU warp 数值差未测，判决覆盖 warp CPU |
| r2 | 042 | `finalize(device=cpu)` 不改变 warp **全局**设备（`wp.get_device()` 仍 cuda）→ capture 仍触发 | `warp.set_device("cpu")`（d43b077b）；C_pos 在 r2 已跑通（图形未触发） |
| r3 | 044 | `pd_explicit` 三配置全败：`tar_dof - dof_pos` 形状不可广播 | 诊断任务 046 定论：newton **1.2.1** API 漂移——`joint_q`(36)=qpos 空间 vs `joint_target_pos/joint_qd/kp/kd/joint_f`(35)=dof 空间；引擎代码按 1.0.0 语义写（README 锚点） |
| r5 | 047 | — | child 内类级 dof 空间修复（dof_pos[6:]=q[7:]，88a88ef8）；引擎文件未动（版本钉扎策略待迁移期决定） |

- state0 自检在 r2 起每配置通过；r1 的镜像环境报告（newton 1.2.1/warp 1.13.0）先于任何物理数据。
- **newton 1.2.1 ≠ README tested v1.0.0**：API 已漂移（Controls 数组语义）。IDEA-012r 判决覆盖 1.2.1+warp CPU；v1.0.0+GPU 未测。

## 4. idear round5/round6 逐条处理

| 建议 | 决定 | 证据/理由 |
|---|---|---|
| IDEA-012(r) Newton 入场探针（三配置矩阵+pdx2-mirror diff） | **accepted，已执行，判决 NEWTON_DOWNGRADED** | §1；B<1 绿灯条件未达（差 19 倍），B≥5 降级分支触发 |
| IDEA-012r A 负对照（显式 kd 失稳预期） | accepted，已执行，**预期部分证伪** | A/B=1.03——v4 增益下显式 kd 未失稳（armature 上界推断正确）；负对照的「检出语义错配」能力由 r3 形状崩溃另行证明 |
| IDEA-012r C pos 模式顺带 | accepted，已执行 | §2.1 B≡C 发现 |
| IDEA-013 X1×Newton 集成差异清单（6 项） | 部分覆盖 | ①控制语义（B 配置实证）、④impratio（D 配置实证）已测；②frictionloss 导入保真、⑤执行器限幅语义、③240Hz 未逐一显式验证（r3 崩溃时一并暴露的 API 漂移属①范畴）；清单随路线裁决归档 |
| IDEA-010r Isaac Lab 探针先钉 kd 语义 | **accepted，排为下一门** | 三分支数据面现状：Isaac Gym 配置路线关闭（SW-R004）、Newton 降级（本报告）——IDEA-010r（低kd+隐式格先行）成为最后一个低成本判别点，之后只剩自建 MuJoCo 原生路线 |
| round5 论文书目（arXiv 2512.03028 v3） | 记录 | 书目级，正文仍未核验（idear 已声明） |

## 5. 路线含义（建议，待评审）

1. **Newton 路线不获绿灯**：MuJoCo 算法经 warp 移植后与 MuJoCo CPU 的 first-step 残差（19.1）与 PhysX 族（26.1）同量级；「换回同族求解器即消除残差」的预期不成立。剩余希望：warp GPU 后端/fp 精度差异贡献未测（driver 限制），以及更深层的 mujoco_warp 版本对齐——但按预注册规则这些属降级后的排查项，不再是主线依据。
2. **下一门：IDEA-010r**（Isaac Lab 低kd+隐式格单关节探针，<1 rad/s 标准）。若也失败 → 三条迁移路线全部关闭，**自建 MuJoCo 原生训练引擎**（gap=0 确定性路线）升为唯一主线。
3. 零误差探针 19.5（全引擎族共性）提示：残差的主体在**接触/重力下的关节动力学积分细节**，非指令路径——任何「训练引擎 vs MuJoCo CPU harness」的 first-step <1 目标都要求逐项对齐该路径（积分器、fp 精度、接触参数化）。

## 6. 诚实性声明

- 判决严格按 SW-PREREG-003 带执行（B 19.12 ≥5 → DOWNGRADED），无事后调整；预注册 device 偏离（warp CPU）与 newton 版本偏离（1.2.1 vs README 1.0.0）均已披露
- §1-2 数值全部来自落盘工件（4 dump + analysis JSON），可本地复跑
- 判决覆盖「newton 1.2.1 + warp 1.13.0 CPU 后端 + SolverMuJoCo(iter=100,ls=50) vs MuJoCo 3.1.6 CPU harness(x1_sim_v4.xml@120Hz, Euler, impratio 模型默认)」；warp GPU 后端、newton 1.0.0、fp64 配置未测
- 单一 reset 条件（motion0/t0）；30 控制步开环回放使 2 步后的 rms 仅作发散参考
- Newton 侧 add_mjcf 对 armature/frictionloss 的导入保真未直接逐项验证（双 XML 字段一致 + state0 对齐 + 动力学对比间接覆盖）
- 本报告 ≠ sim2sim 验收；sim2sim 仍未通过；路线处于第三门（IDEA-010r）评估位
