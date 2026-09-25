# G1 -> X1 Retargeting Validation

## x1_run1_subject2_seg0.pkl  **FAIL**
- frames: 321 @ 30 fps
- R1 节奏: G1 13 steps vs X1 15, cadence ratio 0.9791666666666666, strike offset med 0.06666666666666667s -> PASS
- R2 手脚相位: phase diff -0.61 rad, freq 1.21/1.21 Hz (ratio 1.00) -> FAIL
- R3 地面穿模: min sole z -2.4 mm -> PASS
- R4 自穿模: min pair dist 3.8 mm (('base_link', 'lumbar_pitch_link') @ 256) -> PASS
- R6 关节速度: max |qdot| 15.1 rad/s (ratio 1.692 of URDF limit) -> FAIL
- R5 IK跟踪: foot med/p95 0.4/2.3 cm, hand med/p95 1.4/10.5 cm -> PASS

## x1_run1_subject5_seg0.pkl  **FAIL**
- frames: 717 @ 30 fps
- R1 节奏: G1 23 steps vs X1 31, cadence ratio 0.9411764705882354, strike offset med 0.03333333333333333s -> FAIL
- R2 手脚相位: phase diff -0.13 rad, freq 1.21/1.21 Hz (ratio 1.00) -> PASS
- R3 地面穿模: min sole z 0.8 mm -> PASS
- R4 自穿模: min pair dist -6.6 mm (('base_link', 'lumbar_pitch_link') @ 8) -> FAIL
- R6 关节速度: max |qdot| 17.1 rad/s (ratio 1.685 of URDF limit) -> FAIL
- R5 IK跟踪: foot med/p95 0.3/2.2 cm, hand med/p95 0.6/11.7 cm -> PASS

## x1_run1_subject5_seg1.pkl  **FAIL**
- frames: 946 @ 30 fps
- R1 节奏: G1 20 steps vs X1 23, cadence ratio 0.979591836734694, strike offset med 0.1s -> FAIL
- R2 手脚相位: phase diff 0.21 rad, freq 0.63/0.63 Hz (ratio 1.00) -> PASS
- R3 地面穿模: min sole z -2.6 mm -> PASS
- R4 自穿模: min pair dist -335.8 mm (('left_ankle_roll_link', 'right_ankle_roll_link') @ 704) -> FAIL
- R6 关节速度: max |qdot| 13.2 rad/s (ratio 1.482 of URDF limit) -> FAIL
- R5 IK跟踪: foot med/p95 0.4/2.1 cm, hand med/p95 0.9/9.2 cm -> PASS

## x1_run1_subject5_seg2.pkl  **FAIL**
- frames: 313 @ 30 fps
- R1 节奏: G1 16 steps vs X1 16, cadence ratio 0.9523809523809524, strike offset med 0.03333333333333333s -> PASS
- R2 手脚相位: phase diff 0.08 rad, freq 1.44/1.44 Hz (ratio 1.00) -> PASS
- R3 地面穿模: min sole z -1.2 mm -> PASS
- R4 自穿模: min pair dist -17.0 mm (('left_knee_pitch_link', 'right_ankle_roll_link') @ 16) -> FAIL
- R6 关节速度: max |qdot| 13.3 rad/s (ratio 1.305 of URDF limit) -> FAIL
- R5 IK跟踪: foot med/p95 0.1/1.5 cm, hand med/p95 0.2/5.6 cm -> PASS

## x1_run2_subject1_seg0.pkl  **FAIL**
- frames: 375 @ 30 fps
- R1 节奏: G1 16 steps vs X1 14, cadence ratio 1.0833333333333333, strike offset med 0.05s -> PASS
- R2 手脚相位: phase diff -1.31 rad, freq 1.28/1.20 Hz (ratio 0.94) -> FAIL
- R3 地面穿模: min sole z 0.4 mm -> PASS
- R4 自穿模: min pair dist -15.9 mm (('left_knee_pitch_link', 'right_knee_pitch_link') @ 348) -> FAIL
- R6 关节速度: max |qdot| 21.8 rad/s (ratio 1.650 of URDF limit) -> FAIL
- R5 IK跟踪: foot med/p95 0.7/9.3 cm, hand med/p95 1.7/17.0 cm -> FAIL

## x1_run2_subject1_seg1.pkl  **FAIL**
- frames: 338 @ 30 fps
- R1 节奏: G1 14 steps vs X1 14, cadence ratio 1.0, strike offset med 0.06666666666666667s -> PASS
- R2 手脚相位: phase diff -0.03 rad, freq 1.24/1.24 Hz (ratio 1.00) -> PASS
- R3 地面穿模: min sole z -1.9 mm -> PASS
- R4 自穿模: min pair dist -7.4 mm (('left_elbow_yaw_link', 'left_hip_yaw_link') @ 76) -> FAIL
- R6 关节速度: max |qdot| 12.6 rad/s (ratio 1.295 of URDF limit) -> FAIL
- R5 IK跟踪: foot med/p95 0.8/4.3 cm, hand med/p95 6.3/24.0 cm -> FAIL

## x1_run2_subject1_seg2.pkl  **FAIL**
- frames: 270 @ 30 fps
- R1 节奏: G1 11 steps vs X1 11, cadence ratio 1.0, strike offset med 0.03333333333333333s -> PASS
- R2 手脚相位: phase diff -0.11 rad, freq 1.22/1.22 Hz (ratio 1.00) -> PASS
- R3 地面穿模: min sole z -2.1 mm -> PASS
- R4 自穿模: min pair dist 1.3 mm (('left_ankle_roll_link', 'right_ankle_roll_link') @ 28) -> PASS
- R6 关节速度: max |qdot| 13.7 rad/s (ratio 1.246 of URDF limit) -> FAIL
- R5 IK跟踪: foot med/p95 0.8/2.7 cm, hand med/p95 8.2/27.1 cm -> FAIL

## x1_sprint1_subject2_seg0.pkl  **PASS**
- frames: 292 @ 30 fps
- R1 节奏: G1 5 steps vs X1 6, cadence ratio 1.008695652173913, strike offset med 0.06666666666666667s -> PASS
- R2 手脚相位: phase diff -0.10 rad, freq 0.51/0.51 Hz (ratio 1.00) -> PASS
- R3 地面穿模: min sole z -0.4 mm -> PASS
- R4 自穿模: min pair dist 3.9 mm (('base_link', 'lumbar_pitch_link') @ 188) -> PASS
- R6 关节速度: max |qdot| 9.2 rad/s (ratio 0.678 of URDF limit) -> PASS
- R5 IK跟踪: foot med/p95 0.6/5.6 cm, hand med/p95 2.3/10.0 cm -> PASS

## x1_sprint1_subject4_seg0.pkl  **PASS**
- frames: 227 @ 30 fps
- R1 节奏: G1 8 steps vs X1 9, cadence ratio 0.896551724137931, strike offset med 0.03333333333333333s -> PASS
- R2 手脚相位: phase diff 0.03 rad, freq 1.06/1.06 Hz (ratio 1.00) -> PASS
- R3 地面穿模: min sole z 2.3 mm -> PASS
- R4 自穿模: min pair dist -0.1 mm (('left_elbow_yaw_link', 'left_hip_yaw_link') @ 12) -> PASS
- R6 关节速度: max |qdot| 10.1 rad/s (ratio 0.910 of URDF limit) -> PASS
- R5 IK跟踪: foot med/p95 0.3/1.7 cm, hand med/p95 2.8/15.1 cm -> PASS

## x1_sprint1_subject4_seg1.pkl  **FAIL**
- frames: 273 @ 30 fps
- R1 节奏: G1 5 steps vs X1 6, cadence ratio 0.9259259259259259, strike offset med 0.1s -> FAIL
- R2 手脚相位: phase diff -0.08 rad, freq 0.55/0.55 Hz (ratio 1.00) -> PASS
- R3 地面穿模: min sole z 6.3 mm -> PASS
- R4 自穿模: min pair dist 0.2 mm (('left_ankle_roll_link', 'right_knee_pitch_link') @ 176) -> PASS
- R6 关节速度: max |qdot| 8.1 rad/s (ratio 0.613 of URDF limit) -> PASS
- R5 IK跟踪: foot med/p95 0.3/3.8 cm, hand med/p95 2.5/13.6 cm -> PASS

## x1_sprint1_subject4_seg2.pkl  **FAIL**
- frames: 230 @ 30 fps
- R1 节奏: G1 5 steps vs X1 5, cadence ratio 1.0555555555555556, strike offset med 0.03333333333333333s -> PASS
- R2 手脚相位: phase diff 0.01 rad, freq 0.78/0.78 Hz (ratio 1.00) -> PASS
- R3 地面穿模: min sole z 1.5 mm -> PASS
- R4 自穿模: min pair dist -244.7 mm (('left_ankle_roll_link', 'right_ankle_roll_link') @ 56) -> FAIL
- R6 关节速度: max |qdot| 11.3 rad/s (ratio 1.266 of URDF limit) -> FAIL
- R5 IK跟踪: foot med/p95 0.7/5.3 cm, hand med/p95 6.1/22.8 cm -> FAIL

## x1_sprint1_subject4_seg3.pkl  **PASS**
- frames: 303 @ 30 fps
- R1 节奏: G1 8 steps vs X1 9, cadence ratio 0.9078947368421052, strike offset med 0.06666666666666667s -> PASS
- R2 手脚相位: phase diff 0.06 rad, freq 0.79/0.79 Hz (ratio 1.00) -> PASS
- R3 地面穿模: min sole z 6.5 mm -> PASS
- R4 自穿模: min pair dist 3.4 mm (('base_link', 'lumbar_pitch_link') @ 152) -> PASS
- R6 关节速度: max |qdot| 8.9 rad/s (ratio 0.812 of URDF limit) -> PASS
- R5 IK跟踪: foot med/p95 0.4/2.6 cm, hand med/p95 3.3/15.4 cm -> PASS

## x1_sprint1_subject4_seg6.pkl  **FAIL**
- frames: 230 @ 30 fps
- R1 节奏: G1 6 steps vs X1 7, cadence ratio 0.9166666666666667, strike offset med 0.03333333333333333s -> PASS
- R2 手脚相位: phase diff 0.10 rad, freq 0.91/0.91 Hz (ratio 1.00) -> PASS
- R3 地面穿模: min sole z 1.7 mm -> PASS
- R4 自穿模: min pair dist -261.0 mm (('left_ankle_roll_link', 'right_ankle_roll_link') @ 124) -> FAIL
- R6 关节速度: max |qdot| 14.7 rad/s (ratio 1.650 of URDF limit) -> FAIL
- R5 IK跟踪: foot med/p95 0.4/3.4 cm, hand med/p95 8.7/21.4 cm -> FAIL
