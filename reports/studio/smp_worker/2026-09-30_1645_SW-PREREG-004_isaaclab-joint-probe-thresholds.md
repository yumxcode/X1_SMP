# SW-PREREG-004 — IDEA-010r Isaac Lab（PhysX 5）单关节入场探针判决阈值预注册

- 报告 ID: SW-PREREG-004
- 时间: 2026-09-30 16:4x（本 commit 先于任何 Isaac Lab 探针任务创建与输出查看）
- 作者: smp_worker
- 基线: SW-R005（b7784d8d，NEWTON_DOWNGRADED）——三分支已关二，本探针为 H-IL 最后一门
- 关联: idear round6 IDEA-010r（`smp-ideas-idear-round6-20260930`：先钉 kd 语义、低kd+隐式格先行）、round3 addendum4（Isaac Lab 执行器语义）、SW-R005 §5.2
- 实验载体: `scripts_remote/probe_isaaclab_joint.py`（新，独立 isaaclab 脚本，不用 mimickit isaac_lab_engine——其 UsdFileCfg 硬路径需资产转换，探针用 UrdfFileCfg 运行时导入）

## 1. 实验设计（冻结）

**被测系统**：单关节摆（fixed base + 1 revolute joint + 杆件），参数取 X1 臂链代表值：
- 有效惯量 I=0.05 kg·m²（杆 0.833 kg @ 0.245 m，与肘族量级一致），armature=0.01，重力 -9.81，关节轴水平（重力产生恢复矩=约束/重力路径激励——SW-R005 零误差探针指向的发散主体）
- 控制 @120Hz 子步重算，控制步 30Hz（4 子步/控制步），dt_sim=1/120
- 初始状态：3 组关节角偏置 q0 ∈ {0.2, -0.35, 0.6} rad，qdot=0，目标角 0（阶跃响应，激励 kp 路径）；零误差变体：目标=q0（纯重力+阻尼沉降，SW-R004/R005 协议）

**配置矩阵**（idear IDEA-010r 轴：{kd 语义}×{kd 量级}，先跑低kd+隐式格）：

| 配置 | 执行器 | kp | kd | kd·dt/I（上界口径 I=armature+杆件） |
|---|---|---|---|---|
| S1_lowkd_implicit（**主判据**） | ImplicitActuator（PhysX 关节驱动，kp/kd 求解器内积分） | 40 | 0.5 | 0.083（<0.5，idear「最干净格」） |
| S2_v4kd_implicit | ImplicitActuator | 40 | 2.0 | 0.33（v4 肘实际值） |
| S3_lowkd_explicit | IdealPDActuator（显式 kp+kd） | 40 | 0.5 | 0.083 |

**MuJoCo 对照（harness 语义，pdx2-mirror）**：显式 tau=kp*(tar-q) 逐步 + dof_damping=kd 隐式（S1/S2）；S3 显式 tau=kp*(tar-q)-kd*qd + damping=0。模型经 MJCF 内嵌字符串构建，惯量/重力/轴与 URDF 逐项一致（脚本内打印核对）。

**指标**：首控制步（4 子步）后 dof_vel diff（IsaacLab − MuJoCo），3 个 q0 偏置取 max；零误差变体同指标。

## 2. 判决规则（预注册，采用审核给定与 idear round6 条件）

主判据 = S1_lowkd_implicit 与 S2_v4kd_implicit 的 first-step |Δdof_vel|（rad/s）：

- **两配置均 < 1.0** → H-IL 存在（绿灯）：PhysX 5 在单关节粒度可与 MuJoCo harness 对齐 → 启动全 X1 Isaac Lab 资产迁移（下一工作包：URDF→USD + 引擎适配）
- **任一配置 ≥ 2.0** → H-IL 证伪：PhysX 5 与 Isaac Gym 同族残留 → **自建 MuJoCo 原生训练引擎升为唯一主线**（预注册分支，按 SW-R005 §5.2）
- **灰区（1.0–2.0）**：S3 对照联判——若 S3 显式 kd 明显更差（>2x）且 S1/S2 落灰区下沿（<1.5），视为 kd 语义可修、迁移有条件继续但需全 X1 探针复核；否则按证伪处理

辅助判读（不改变主判决）：
- S3 vs S1/S2 差异 → kd 语义轴贡献（Isaac Lab 侧）
- 零误差变体仍 >1 → 重力/积分路径族差（与 SW-R005 零误差 19.5 的全尺寸现象对照）
- 8 控制步序列的发散曲线作为稳定性观察（不判据）

## 3. 有效性自检

- 双端模型参数打印核对（惯量张量、质量、轴、armature、重力）；URDF 与 MJCF 的惯量值由脚本各打印一次并断言一致（<1e-9 相对差）
- IsaacLab 侧关节状态读回用强制拷贝（r3 教训）
- 环境 fail-fast：isaaclab/isaacsim 版本打印；import/启动失败记录为环境阻塞而非实验结果
- 版本偏离披露：镜像 IsaacLab 版本 vs 引擎代码时代（fork 2026-06，IsaacLab 2.x）——API 漂移属 harness 迭代项

## 4. 输出契约

- 远端: `output/isaaclab_joint_probe_{tag}.pt`（3 配置 × {阶跃, 零误差}）
- 本地: `tools/x1_pipeline/isaaclab_joint_probe_analysis.py` → `output/isaaclab_joint_probe_analysis.json`（判决 + 与 SW-R004/005 表并表）
- 报告: SW-R006

## 5. 诚实性声明

- 本预注册先于探针任务创建；Isaac Lab 环境从未在本项目运行过，首轮任务可能需要 harness 迭代（Newton 先例：5 任务）
- 判决范围：单关节粒度的求解器族差判别；绿灯≠全 X1 迁移可行（全尺寸接触/闭链/多关节耦合仍需入场探针），只解锁资产迁移工作包
- 灰区联判规则为预注册决定，不得事后放宽
