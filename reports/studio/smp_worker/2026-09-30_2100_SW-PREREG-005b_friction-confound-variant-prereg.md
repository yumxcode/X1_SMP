# SW-PREREG-005b — F3 消混杂变体预注册（frictionloss 读回+置零，IDEA-014 应用于 x1-il 探针）

- 报告 ID: SW-PREREG-005b（SW-PREREG-005/005a 系列第三变体，先于 F3 任务创建）
- 时间: 2026-09-30（git commit 时间戳为准）
- 作者: smp_worker
- 背景: idear round7（`smp-ideas-idear-round7-20260930`）在 Newton 探针复核中发现 frictionloss 协议不对称（harness 侧置零 vs 引擎侧加载带 0.2-4.0 N·m 干摩擦），建议 NEWTON_DOWNGRADED(conditional)。同一混杂适用于 SW-R007 的 F1/F2：IL 侧 MjcfFileCfg 导入 x1_v4.xml（frictionloss 0.2-4.0），harness 回放置零（x1_il_entry_analysis.py replay）。F1 dump 的整步 v≡0.00 冻结签名与 PhysX 关节摩擦 stiction 一致——混杂必须排除后 REFUTED 判决才算干净。

## F3 设计（冻结）

- 与 F1 唯一差异：`sim.reset()` 后读回 `view.get_dof_friction_coefficients()`（打印，验证 MjcfFileCfg 是否导入 frictionloss）并置零 `set_dof_friction_coefficients(0)`，回读确认 0；顺带读回 armature（IDEA-017 唯一未验证项）
- 注入状态 = F1 原帧（motion0/t0，不退让——限位混杂已被 F2 独立排除）
- 其余协议与 F1 逐项相同（增益分组、state0 自检、30 步开环 q_tar 回放、零误差探针）

## 阈值与角色（预注册，与 005a 同带）

- **F3 < 1.0** → frictionloss 导入语义是残差主因：IsaacLab 路线**重开评审**（训练语义本就置零摩擦，与 harness 对齐的修复即置零关节摩擦）；F1/F2 REFUTED 判决标注 conditional（限位混杂排除但摩擦混杂成立）
- **F3 ≥ 2.0** → 摩擦混杂排除：F1/F2 REFUTED 判决**升级为 clean**，MuJoCo 原生唯一主线的路线终局维持
- 灰区 1.0-2.0 → 按 005a §3 家族规则联判

## 诚实性声明

- 本预注册在 F1/F2 判决发布（SW-R007）之后、F3 任务之前落盘；无论 F3 结果如何，F1/F2 按带判决不撤销——F3 决定的是「判决的混杂标注」与路线是否重开
- idear 量级自检被采纳：干摩擦 3 N·m × 33ms / I≈0.0767 ≈ 1.3 rad/s，算术上不足以独解释 19——F3 是混杂控制而非翻案赌注
