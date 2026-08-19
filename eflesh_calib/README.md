# eflesh_calib — eFlesh 电子皮肤标定数据采集系统

RM75-6FB（六维力）+ 尖头探针 → 压固定 eFlesh 皮肤 → 采集 (15 通道磁场, 位置, 六维力) 配对数据 → 训练力+位置模型。

## 架构

```
笔记本 (开发) ──rsync/ssh──▶ Orin 192.168.101.16 (/ssd/curobo_env)
                              ├─ Robotic_Arm SDK ──▶ RM75 @192.168.101.120 + 六维力
                              └─ 皮肤 USB → /dev/ttyUSB0 (921600, 64B 帧, ~170Hz)
```

全部运行时代码在 Orin 执行；本仓库在笔记本开发后同步。

## 同步到 Orin

```bash
rsync -av --exclude data --exclude __pycache__ --exclude .git \
    /home/xyz/projects/eflesh-touch/eflesh_calib/ robot@192.168.101.16:~/eflesh_calib/
```

## 软件自检（已全部通过，可随时重跑）

```bash
# 环境自检: 依赖/SDK import/HDF5 往返/事件字段/torch
python scripts/orin_selftest.py
# 合成数据端到端: 录制→build_dataset→sync_check→ridge 学习验证
python scripts/synth_e2e.py
```

## 现场流程（按顺序）

```bash
cd ~/eflesh_calib
# 0) 皮肤链路自检（USB 插 Orin 后）— 期望 ~170Hz, lost=0, badck=0
python -m eflesh_calib.skin_stream --port /dev/ttyUSB0 -t 30

# 1) 六维力自检（上电→读数→清零→手压观察）
python scripts/preflight.py --ft-only

# 2) 示教平面：键盘点动（x/X y/Y z/Z 移动，1 切步长 5/0.5/0.1mm，
#    p 打印 Fz，m 标记）轻触 4 角+中心 → data/skin_frame_*.json
python scripts/teach_plane.py

# 3) 全项预检（磁干扰飞越是门禁）— 阈值自动回写 JSON
python scripts/preflight.py --calib data/skin_frame_*.json

# 4) 空中演练（不接触皮肤，全流程 + HDF5 检查）
python scripts/collect.py --session dry1 --calib data/skin_frame_*.json \
    --grid 3 --depths 1.0 --air-gap 5
python scripts/inspect_h5.py data/batch_*_dry1.h5 --plot dry1.png

# 5) M1 试采（3×3，手放急停）→ 验收三指标（掉帧0/对齐p95<5ms/可回放）
python scripts/collect.py --session m1_trial --calib data/skin_frame_*.json \
    --grid 3 --depths 1.0
python scripts/m1_accept.py data/batch_*_m1_trial.h5

# 6) 正式批（9×9×3=243 压 ≈30min，中断后 --resume 续采）
python scripts/collect.py --session batch001 --calib data/skin_frame_*.json \
    --pitch 5 --depths 0.5,1.0,1.5

# 7) 数据集 → 训练 → 评估
python eflesh_calib/build_dataset.py --h5 "data/batch_*_batch00*.h5" --out data/ds_v1.npz
python train/train_model.py --data data/ds_v1.npz --model mlp
python train/evaluate.py --data data/ds_v1.npz --artifact data/ds_v1_mlp.pt
```

之后 ≥2 天各补一批（batch002/003）做跨天泛化；可信后升 `--pitch 2`（21×21 论文级）。

## 单次按压流程

```
APPROACH(上方5mm) → TARE(清零) → DESCEND(0.25mm步进, |Fz|>thr 停)
→ PRESS(至 z0−depth) → HOLD(0.8s 采样) → RETRACT
```

- 每压各自 touch-off 定 z0（吸收表面不平）；位置标签用实测投影
- HDF5 三流: skin(~170Hz×15) / ft(125Hz×6) / pose(100Hz)，单一单调时钟
- 切分：棋盘格空间 80/20 + 按批留出（绝不逐帧随机切）

## 安全

- Fz 中止 15N / 任意轴 20N / 工作空间盒（角点±10mm, z∈[面−5, 面+60mm]）/ 皮肤断流软告警
- Ctrl+C → 慢停，双击 → 急停；控制器急停为主
- 先 `--air-gap` 空中演练再上皮肤；磁干扰飞越不过 = 不采

## 依赖（Orin /ssd/curobo_env，已装好）

pyserial, h5py, numpy, scipy, torch, scikit-learn；`Robotic_Arm` SDK 与 `~/rm75_curobo_demo`（import 复用其 `RM75Robot`）。
