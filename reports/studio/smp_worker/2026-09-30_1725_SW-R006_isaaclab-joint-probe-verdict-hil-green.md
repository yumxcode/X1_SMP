# smp_worker 报告 006 — IDEA-010r Isaac Lab 单关节探针判决：H-IL_GREEN，迁移路线解锁

- 报告 ID: SW-R006
- 时间: 2026-09-30 17:2x
- 作者: smp_worker
- 预注册: SW-PREREG-004（1212f42a，先于任何 Isaac Lab 任务）
- 判决数据: TASK_20260930_054（r3，3/3 配置）+ `output/isaaclab_joint_probe_analysis.json`
- 关联: SW-R005（NEWTON_DOWNGRADED，三分支已关二）、idear round6 IDEA-010r（`smp-ideas-idear-round6-20260930`）、round3 addendum4（执行器语义）
- 环境: IsaacLab 6.1.14（镜像 BJX00000335 V000386），device cuda:0；URDF 运行时导入（UrdfFileCfg），零 USD 管线

## 摘要

按 SW-PREREG-004 预注册规则，Isaac Lab（PhysX 5）单关节入场探针实测：**S1_lowkd_implicit first-step max 0.0937 rad/s、S2_v4kd_implicit 0.2586 rad/s，均 <1.0 → H-IL_GREEN**。PhysX 5 隐式执行器（kp/kd 求解器内积分）在单关节粒度（重力+惯量+armature+PD，无接触）与 MuJoCo harness 对齐到 0.1-0.3 rad/s——比 Isaac Gym TGS 全尺寸残差（26.06）低两个量级、比 Newton/warp（19.12）低近两个量级。按预注册分支：**Isaac Lab 资产迁移工作包解锁**。S3 显式 kd 对照 5.07 rad/s 实证了 idear 的执行器语义警告——mimickit 现行 `isaac_lab_engine.py` 的 pd_explicit（IdealPD 全显式）不可原样使用，迁移必须用隐式执行器路径。

## 1. 判决证据（TASK_20260930_054，md5 已核对）

| 配置 | 执行器语义 | kp/kd | first-step max（3 偏置取 max） | 零误差变体 |
|---|---|---|---|---|
| **S1_lowkd_implicit（主判据）** | ImplicitActuator（PhysX 关节驱动，kp/kd 求解器内积分） | 40/0.5 | **0.0937** | 0.0077 |
| **S2_v4kd_implicit（主判据）** | 同上 | 40/2.0 | **0.2586** | 0.0105 |
| S3_lowkd_explicit（kd 轴对照） | 逐子步显式 kp(tar-q)-kd·qd | 40/0.5 | 5.0691 | 2.3994 |

- 模型：1-DOF 摆（URDF 运行时导入），杆 0.833 kg @ 0.245 m（I_joint≈0.0667+armature 0.01），铰轴 x，重力 -9.81，dt=1/120 Euler，4 子步/控制步，3 组 q0 ∈ {0.2,-0.35,0.6} 阶跃 + 零误差变体
- 双端参数核对：arm 质量 dump=0.833（逐位），惯量/轴/重力在脚本内同源生成
- 参考系并表：Isaac Gym TGS@120Hz 全尺寸 26.06（SW-R004 锚）；Newton/warp 19.12（SW-R005 B）；本探针为单关节无接触粒度——**跨粒度不可直接比大小**，但 H-IL 判据（<1 rad/s 门）是预注册的绝对标准
- **预注册判决：S1、S2 均 <1.0 → H-IL_GREEN**

## 2. 发现（已观测）

1. **kd 语义轴贡献 5.07 vs 0.09-0.26（54x）**：同为 kp=40/kd=0.5，隐式（求解器积分）与显式（逐子步外力矩）在 PhysX 5 内差一个量级以上——显式 kd 的「kd·dt/I 边缘失稳+积分语义差」在单关节上复现了全尺寸残差的量级。**idear round6 IDEA-010r 的语义警告被直接实证**：若用现行引擎 IdealPD 配置直接跑全尺寸探针，<1 rad/s 判据会被语义噪声污染（S3 的 5.07 正是污染的量化）。
2. kd 量级轴（0.5→2.0 隐式）：0.094→0.259（2.8x）——kd 越大隐式积分差越显现，但仍在门内。
3. 零误差变体：隐式配置 0.008-0.011（纯重力+阻尼沉降路径对齐极好）；显式 2.40——发散主体在显式 kd 路径而非重力/积分路径。
4. 单关节粒度下 PhysX 5 vs MuJoCo 的可对齐性存在——与 Isaac Gym（PhysX preview）和 Newton（warp 移植）形成路线对比：三分支中唯一过门。

## 3. harness 迭代与披露（r1-r3，3 任务）

| 轮 | 任务 | 问题 | 处置 |
|---|---|---|---|
| r1 | 051 | `SimulationCfg(use_gpu_pipeline=...)` 在 IsaacLab 6.x 已移除；**AppLauncher 吞异常使 rc=0**（"OK but dump missing"） | 删 kwarg；dump 存在性为成功标准 |
| r2 | 052 | `get_masses()` 返回非 torch 数组（.clone() 失败）；close() 硬退出使检查失效 | torch.as_tensor 包装；存在性检查移到 close() 前 |
| r3 | 054 | — | 3/3 成功，masses 逐位核对 |

- 版本偏离披露：IsaacLab 6.1.14 vs mimickit 引擎代码时代（fork 2026-06，2.x）——引擎 API 兼容性未测，属迁移工作包第一项
- 本探针为独立 isaaclab 脚本（不经 mimickit 引擎），单关节判别不依赖引擎适配

## 4. idear 建议处置（本轮增量）

| 建议 | 决定 | 证据/理由 |
|---|---|---|
| IDEA-010r 探针先钉 kd 语义、低kd+隐式格先行 | **accepted，已执行，H-IL_GREEN** | §1；低kd+隐式格（S1）正是最干净格，0.094 过门 |
| IDEA-010r 显式 kd 轴 | accepted，已执行 | §2.1 S3=5.07，语义警告实证 |
| （隐含）迁移用隐式执行器 | accepted | S1/S2 vs S3；pd_explicit IdealPD 路线在迁移中禁用或改造 |

## 5. 分支推进（按预注册与审核指令）

**GREEN 分支生效：启动 Isaac Lab 资产迁移工作包**，分解（按依赖序）：
1. URDF→USD 转换冒烟（X1 29-DOF URDF → USD，核对关节数/序/限位/惯量）——转换后逐项与 MJCF 对照
2. mimickit `isaac_lab_engine.py` 适配 IsaacLab 6.x API + 执行器改隐式（S3 教训）
3. 全 X1 入场探针（复用 SW-R004/005 协议：确定性 reset + 开环 q_tar 回放 + state0 自检 + 零误差探针）——单关节绿灯≠全尺寸过门（接触/多关节耦合未测，prereg §5 已声明）
4. 通过后：SMP 训练管线在 Isaac Lab 的适配与冒烟 → 重启策略训练 → 导出 → MuJoCo sim2sim 闭环验收（协议见 §6）
5. sim2sim 验收协议（可测阈值，先于训练重启确认）：速度跟踪（|vx_cmd−vx| 中位 <0.3 m/s，S1 门）、存活 ≥10s 比例 ≥6/8 seeds、跌倒率（10s 内 <2/8）、足端滑移（接触期足速 <0.5 m/s 中位）、力矩饱和（≥95% 限幅帧 <5%）、4 seeds 覆盖——阈值来源：v3b Isaac 内 8/8 存活的既证能力 + MuJoCo harness 物理合理性，属提案待评审确认后冻结

## 6. 诚实性声明

- 判决严格按 SW-PREREG-004 带执行；灰区/S3 联判条款未触发
- 判决范围：单关节、无接触、重力+PD 粒度的求解器族判别；**不宣称全 X1 迁移可行**（全尺寸入场探针是下一步且可能失败）；不宣称 sim2sim 通过
- IsaacLab 6.1.14 + cuda:0 单一环境；CPU 管线/其它版本未测
- 隐式配置的 kp 积分语义（PhysX 驱动内隐式 vs harness 显式 kp）存在一阶差异，实测残留 0.09-0.26 即含此语义差——已在判据内一并度量，无需分离（灰区未触发）
- §5 验收协议为提案（阈值来源已注明），未获评审确认前不作为验收标准
