# smp_worker 报告 008 — F3 摩擦混杂控制判决：REFUTED clean，路线终局确认；idear round7/8 处置

- 报告 ID: SW-R008
- 时间: 2026-09-30 21:5x（commit 时间戳为准）
- 作者: smp_worker
- 预注册: SW-PREREG-005b（efc1284b，先于 F3 任务；SW-PREREG-005/005a 系列）
- 判决数据: TASK_20260930_070（r1：摩擦回读事实）+ TASK_20260930_071（r2：判决）+ `output/x1_il_entry_F3_analysis.json`
- 关联: SW-R007（F1/F2 REFUTED）、idear round7（`smp-ideas-idear-round7-20260930`，IDEA-014/015/016 + NEWTON_DOWNGRADED(conditional) 建议）、idear round8（`smp-ideas-idear-round8-20260930`，IDEA-017/018 + 验收协议评审）

## 摘要

按 SW-PREREG-005b 预注册，F3 变体（frictionloss=0 XML 副本，其余与 F1 逐项相同）实测：**first-step 18.37 rad/s ≥ 2.0 → 摩擦混杂排除，H-X1-IL REFUTED 升级为 clean**，MuJoCo 原生唯一主线的路线终局确认。idear round7 的混杂发现被完整验证（r1 回读：MjcfFileCfg 确实导入 frictionloss 0.2-4.0 N·m，29/29 关节）但其量级自检也正确（置零后残差仅 19.03→18.37，-3.5%）——干摩擦是真实差异但非残差主因。

## 1. F3 判决证据（TASK_20260930_071，dump md5 584146232d96）

| 探针 | 摩擦 | first-step max | mean rms | 零误差 | worst joint | state0 |
|---|---|---|---|---|---|---|
| F1（REF） | 导入 0.2-4.0 | 19.028 | 42.39 | 19.09 | left_elbow_pitch | 0.00/0.00 |
| F2（限位退让） | 导入 | 19.018 | 42.44 | 19.07 | left_elbow_pitch | 1.4e-08/0.00 |
| **F3（摩擦=0）** | **0/29 非零（回读验证）** | **18.365** | 42.85 | 19.11 | left_elbow_pitch | 0.00/0.00 |

- 摩擦零验证：`get_dof_friction_coefficients` 回读 min=max=0.000（x1_v4_f0.xml 本地由 x1_v4.xml 正则生成，mujoco 3.1.6 加载验证 max frictionloss=0.0，mass/damping/armature 逐位一致）
- 家族残差：肘 14.6 / hip_pitch 8.1 / 腿 5.0 / 腕 0.93——与 F1/F2 模式一致
- **三变体结论：限位邻域（F2）与干摩擦（F3）均为真实但非主因；一般性族差（~19 rad/s）坐实。** 预注册链四门（002/003/005/005a/005b）全部按带执行。

## 2. idear round7 逐条处置

| 建议 | 决定 | 证据/理由 |
|---|---|---|
| IDEA-014 摩擦读回+置零重跑（先于 IDEA-010r） | **accepted，已执行（x1-il 半边完整）** | r1 回读证实导入 0.2-4.0（29/29）；r2 置零（XML 副本法）判决 F3=18.37。**Newton 半边 deferred**：路线已按 SW-R007 终局（Newton 在三分支中且已被 SW-R005 降级），其摩擦混杂重测不改变任何现行决策，挂为 Newton 复审条件（若路线重开） |
| NEWTON_DOWNGRADED → (conditional) 建议 | **rejected（维持无条件降级）** | F3 表明同型摩擦混杂在 x1-il 上仅 -3.5%——Newton 19.12 的降级带余量（5.0 vs 19.12）不会被 ~1 rad/s 级混杂翻案；且 Newton 路线已在终局判决之外 |
| IDEA-015 零误差发散隔离矩阵（接触/重力/积分器/fp） | **deferred（记录）** | 与 SW-R007 §2.5 开源诊断项同域；MuJoCo 原生路线对该异常构造性免疫，矩阵定位属 post-terminal 诊断，非主线阻塞 |
| IDEA-016 Isaac Lab 探针三约束 | **partially superseded** | 约束已被 F1-F3 执行超越（增益回读✓、armature 回读 0.02✓（IDEA-017 唯一缺口同时补上）、摩擦对称✓）；路线已终局 |

## 3. idear round8 逐条处置

| 建议 | 决定 | 证据/理由 |
|---|---|---|
| IDEA-017 URDF→USD 资产链六字段 parity | **superseded（记录）** | F 系列已用 MjcfFileCfg 直载路线（零转换）；f1.urdf 链缺陷清单归档供未来 URDF 路线参考；armature 注入路径已实证（回读 0.02） |
| IDEA-018 full-X1 探针接触分层（contact-free/含接触/多 reset 三层） | **partially executed** | contact-free 层已隐式覆盖（t0 腾空 ncon=0，SW-R007 §2.2）；含接触/多 reset 层未跑——按路线终局（IsaacLab 关闭）不再投入，转记为开源诊断项（与 §2.5 同域） |
| 验收协议三条评审意见 | **accepted（下一份协议修订时并入）** | SW-R006 §5.5 协议为提案；修订时采纳 idear 意见（S2/S3 门语义、种子覆盖表述、阈值来源标注） |
| H-P2（光滑路径已对齐→残差在接触×约束耦合维） | 采信为假设 | 单关节 0.09-0.26 vs 全身 18-19 的跨粒度跳变支持；但 t0 腾空 ncon=0 下残差仍 19——**"接触×约束耦合"应改为"多体耦合"（含关节链耦合）**，标注修正 |

## 4. 路线终局确认与下一步

- **四门预注册链（002→003→005/005a→005b）闭合，MuJoCo 原生训练引擎唯一主线确认**：IsaacGym 22.4-30.1 / Newton 19.12 / IsaacLab 19.03（限位/摩擦混杂均排除后 18.37）——全尺寸跨引擎 first-step 对齐无一达到 <1.0，而 MuJoCo 训练=MuJoCo 验证 gap=0 by construction。
- 下一工作包（MuJoCo 原生主线，顺序）：①吞吐侦察（mujoco.rollout 批量 API 在本地/远端的 envs 规模实测）②引擎骨架（mimickit engine 接口的 MuJoCo 实现）③逐位对齐冒烟（vs sim2sim_validate.py）④训练重启 ⑤导出+sim2sim 验收（协议修订版）。

## 5. 诚实性声明

- F3 判决严格按 SW-PREREG-005b 带；F1/F2/F3 dump 均落盘（md5 b0e6900/18712c/584146），分析可复跑
- IDEAR-014 Newton 半边未重测（§2 给出理由与挂起条件）；「维持无条件降级」是决策判断而非实验结论——若未来路线重开 Newton，须先补摩擦对称重测
- 跨引擎对称异常（SW-R007 §2.5）依旧开放：MuJoCo harness 多体路径未被单关节正例完全豁免；原生路线构造性免疫该异常，这是其成为主线的主要依据之一（如实说明，非隐藏）
- sim2sim 仍未通过；本报告是混杂控制与路线终局确认，非验收
