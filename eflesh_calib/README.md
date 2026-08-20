# eflesh_calib — eFlesh 电子皮肤标定数据采集与训练系统

RM75-6FB(腕部六维力)+ 探针 → 压/拖/拧固定 eFlesh 皮肤(5×MLX90393, 15 通道, ~184Hz)
→ 采集 (ΔB, 位置, 6 维力) 配对数据 → 训练 **位置 + 三维力** 模型 → 实时 web 热图。

## 当前能力(2026-08-20, batch003+004 训练)

| 输出 | 验证集 RMSE | 说明 |
|---|---|---|
| x / y 位置 | 1.22 / 1.22 mm | R²=0.982, 边缘格点外插是瓶颈 |
| 压深 z | 0.39 mm | 量程 3mm 的 13% |
| 法向力 Fz | 0.080 N | ~2% 量程 |
| 剪切力 Fx/Fy | **0.053 / 0.057 N** | 量程 8%, 与 Fz 同级 |
| Mz | 1.0 mN·m | 尖头探针下≈0(扭转待平头探针) |

## 架构

```
笔记本 (开发/git) ──rsync──▶ Orin 192.168.101.16 (/ssd/curobo_env/bin/python)
                              ├─ Robotic_Arm SDK ──▶ RM75 @192.168.101.119 + 六维力
                              └─ 皮肤 USB → /dev/ttyUSB0 (921600, 64B 帧, ~184Hz)
笔记本 ──rsync/scp──▶ 4090 192.168.101.111 (训练, env_isaaclab)
```

## 同步纪律(重要)

```bash
# 笔记本 → Orin / 4090: 永远不带 --delete, 必须 exclude data/
rsync -az --exclude '__pycache__' --exclude 'data/' \
    ~/projects/eflesh-touch/eflesh_calib/ robot@192.168.101.16:eflesh_calib/
rsync -az --exclude '__pycache__' --exclude 'data/' \
    ~/projects/eflesh-touch/eflesh_calib/ robot@192.168.101.111:eflesh_calib/

# git: 笔记本 commit, 用户 push (origin: Jnkdog-debug/eflesh-touch)
```

**每批数据采完第一件事 = 备份**(2026-08-20 rsync 事故教训,见 memory):

```bash
scp 'robot@192.168.101.16:eflesh_calib/data/*<session>*.h5' ~/eflesh_backup/
scp robot@192.168.101.16:eflesh_calib/calib/skin_frame.json ~/eflesh_backup/  # 示教文件同样关键
```

## 现场流程(Orin, `cd ~/eflesh_calib`, python 一律 `/ssd/curobo_env/bin/python`)

```bash
# 0) 皮肤链路自检 — 期望 ~184Hz, lost=0
python -m eflesh_calib.skin_stream --port /dev/ttyUSB0 -t 30
# 1) 六维力自检
python scripts/preflight.py --ft-only
# 2) 示教平面(键盘点动, 轻触 4 角+中心; 几何不合格会拒存) → calib/skin_frame.json
python scripts/teach_plane.py
# 3) 全项预检(磁干扰飞越门禁)
python scripts/preflight.py --calib calib/skin_frame.json
# 4) 空中演练(--air-gap; drag 协议会在悬停高度侧拖验证几何)
python scripts/collect.py --session dry1 --calib calib/skin_frame.json \
    --grid 3 --depths 1.0 --air-gap 5
```

### 正式采集(--resume 可断点续采)

```bash
# 法向批(位置/深度/法向力)
python scripts/collect.py --session batchNNN --calib calib/skin_frame.json \
    --pitch 2 --margin-mm 2 --depths 2,2.5,3,3.5,4,4.5,5 --resume
# 剪切批(--protocol drag: 每点每深度 ×4 方向定深侧拖 2.5mm)
python scripts/collect.py --session batchNNN --calib calib/skin_frame.json \
    --protocol drag --pitch 5 --margin-mm 2 --depths 3,4,5 --resume
# 扭转批(--protocol twist: ±12° 绕探针轴; 尖头探针传不进扭矩, 需平头/橡胶头)
python scripts/collect.py --session batchNNN --calib calib/skin_frame.json \
    --protocol twist --pitch 5 --margin-mm 2 --depths 4,5 --resume
# 试采: --limit 5 先看 [drag]/[twist] 行的 Fx/Fy/Mz 信噪比
```

## 训练闭环

```bash
# ① Orin: h5 → npz(法向批 4 列; --full-force 合并剪切批出 9 列 6 维力标签)
python eflesh_calib/build_dataset.py --h5 'data/*batch00[34]*.h5' \
    --full-force --out data/ds_b34_force.npz
# ② scp npz → 4090, 然后 4090 上:
python train/train_model.py --data data/ds_b34_force.npz --model mlp --out data/ds_b34_mlp.pt
python train/evaluate.py --data data/ds_b34_force.npz --artifact data/ds_b34_mlp.pt \
    --out data/eval_b34.png
# ③ 4090 → Orin 回传 .pt → 笔记本拉 png 读图
```

**力螺旋变换**:`--full-force` 的力标签经 `wrench_to_skin` 从 FT 传感器系变换到
**皮肤系接触点**(力: `F_皮肤=(−Fy,−Fx,+Fz)_传感器`;力矩: 先扣杆臂
`t=(−9.3,−6.0,+88.8)mm` 再映射;常数在 `config.FT_R_TO_SKIN / FT_LEVER_SENSOR_M`,
由 batch004 360 压标定)。换探针/换装夹后需重标 t。

## 实时热图(Orin)

```bash
fuser -k 8899/tcp   # 旧实例在跑先杀
python scripts/infer_live_web.py --artifact data/ds_b34_mlp.pt
# 浏览器 http://<Orin-IP>:8899 — 辉斑=接触位置, 青色箭头=剪切方向(>0.08N 显示)
```

未接触时预测被屏蔽(训练分布外防幻觉);基线在线 EMA 慢跟零点。

## 单次按压流程

```
法向: APPROACH → TARE → DESCEND(0.25mm 步进 touch-off) → PRESS → HOLD(0.8s) → RETRACT(两段式)
剪切: … PRESS → DRAG(定深侧拖 0.25mm/步) → HOLD@偏移点 → RETRACT
扭转: … PRESS → TWIST(绕探针轴 2°/步) → HOLD@最大角 → RETRACT
```

- 位置标签用实测投影(剪切批用拖动终点);每压同姿态 TARE 消重力;HOLD 静止采样
- HDF5 三流 skin/ft/pose + phases + events, 单调时钟;棋盘格空间切分防泄漏

## 安全

- Fz 中止 15N; drag/twist 切向中止 |Fx|,|Fy|>8N、|Mz|>1.5N·m
- 先 `--air-gap` 空跑再上皮肤;磁干扰飞越不过 = 不采
- 采集时手放控制器急停;Ctrl+C 慢停

## 依赖

Orin `/ssd/curobo_env`: pyserial, h5py, numpy, scipy, torch, scikit-learn + `Robotic_Arm` SDK
(`~/rm75_curobo_demo` 的 `RM75Robot`)。4090: numpy/torch/sklearn/matplotlib。
