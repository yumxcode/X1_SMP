# E-v4-RESID-02 预注册判决阈值（pre-registration）

- 登记 ID: SW-PREREG-001
- 写入时间: 2026-09-29 19:21（本地，commit 时间戳为证）
- 作者: smp_worker
- 背景: idear round3 IDEA-007（reports/studio/idear/20260929_1930_idear-round3-sw-r002-review-residual-gate-sds-decomposition.md）要求在 v4 dump 结果到达前预注册判决阈值；TASK_20260929_163（x1-smp-v4-dump-traj，ckpt=smpv4_eval.pt abs16928）已于 19:18:03 启动，**截至本文件写入，本会话尚未查看该任务的任何日志/输出/模型**。
- 适用对象: E-v4-RESID-02 = Isaac(v4 低增益+pdx2 无 DR) vs MuJoCo(x1_sim_v4) 单控制步重放的 dof_vel 差（与 E-v4-RESID-01 同方法：同状态同 q_tar 同 kp/kd，4×120Hz 单控制步后逐关节差）。主指标：首步（step 0）max |Δdof_vel| 与 rms；辅助：前 5 步轨迹、逐关节分解、tau 饱和率。

## 判决规则（预注册，dump 结果到达后不得修改）

以首步 max|Δdof_vel|（rad/s）为主判据，饱和分层报告（|tau|/tlim ≥95% 的关节单列）：

| 区间 | 判决 | 行动 |
|---|---|---|
| **< 1.0 rad/s** | H2（引擎残差主导）显著削弱：v4 低增益下残差已缩到策略稳定域内量级 | 主线转向 H1（继续训练至成熟），同时膝穿地数据修复照做 |
| **1.0 – 2.0 rad/s** | 灰区：残差仍可观但较 v3 的 3.89 缩小；单一阈值无法裁决 | 必须做 ckpt 轴第二 dump（task110 末期 ckpt）与成熟度交互分析后才裁决 |
| **≥ 2.0 rad/s** | H2 增强：残差与增益无关（速度级求解器差），成熟度提升无法跨越 | 优先启动 idear addendum4 的 Isaac Lab 迁移预研（IDEA-006 路线）与/或 60Hz 控制频率升级（IDEA-003 解除 defer） |

辅助判决（任一触发即记录，不推翻主判据）：
1. 若残差 max 出现在 tau 饱和关节且非饱和关节残差 <1.0 → 判"饱和语义差"，单独登记（Isaac 饱和 clip 与 MuJoCo clip 的相位/语义差）
2. 若首步残差 <1.0 但 5 步内累计轨迹发散（任一关节位置差 >0.3 rad）→ 残差小但快速放大，登记为"积分放大"型，与 H1/H2 均相容，需 DR/鲁棒性轴分析
3. 若 Isaac dump 自身出现 |dof_vel|>20 rad/s 或 NaN → dump 无效，重新发任务，本预注册继续适用

## ckpt 轴（对照设计，预注册）

IDEA-007 竞争解释对照要求 ≥2 个 ckpt：本 dump（abs16928, reward~0.185）+ task110 末期 ckpt（到 600M cap 后补第二个 dump）。若两 ckpt 残差同量级（差 <30%）→ 残差与成熟度无关（支持 H2 速度差解释）；若末期 ckpt 残差显著更小 → 策略输出平滑度参与残差（H1/H2 混合）。

## 诚实性

- 19:21 前本会话只做过：任务创建/run 调用（其返回只含启动成功信息，无 rollout 数据）与 task info 状态查询（未成功执行，key 提取失败）
- 预注册后如需改动，只能以"SW-PREREG-001-amended"追加并说明原因
