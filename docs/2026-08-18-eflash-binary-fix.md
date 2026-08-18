# eflash_binary 固件排障记录（2026-08-18）

一天之内从"完全不正常"修到"零失败、~170Hz"。本文记录完整排查过程、证据和根因，
既是项目文档，也是以后再遇到类似问题时的参考。

## 结果对比

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 串口数据 | 2M baud 全乱码 | 921600 零误码（badck=0） |
| 传感器映射 | S1/S3 数据互换 | 与 eflesh.ino 一致 |
| 有效帧率 | ~16Hz（readData 阻塞） | ~170Hz（流水线） |
| 读失败率 | 66.7%（每 3 帧丢 2 帧） | 0% |
| 视觉表现 | 箭头一闪一闪 | 稳定常亮 |

## 环境事实（排查前提）

- 板子：经典 ESP32 DevKit，USB 串口芯片 **CP2102**（`lsusb` 显示 `10c4:ea60 Silicon Labs CP210x`）
  —— **波特率硬上限 921600**（CP2102N 才支持 2M/3M；CH340 支持 2M）
- 接线：5×MLX90393 共享 VSPI（SCK=18 MISO=19 MOSI=23），CS = {5, 2, 4, 13, 12}
- 库：Adafruit_MLX90393 **v2.0.5**（`~/Arduino/libraries/`，以下源码行号均指此版本）
- 数据量核算：64B × 170Hz ≈ 87 kbps ≪ 921600，带宽余量 10 倍

## Bug ①串口乱码（主因）

- **症状**：文本固件（115200）正常，二进制固件（2000000）全是乱码
- **根因**：CP2102 上限 921600，2M 超出芯片能力
- **修复**：`SERIAL_BAUD = 921600`。若以后换 ESP32-S2/S3（原生 USB CDC）可改回 2M

## Bug ②S1/S3 数据互换

- **证据**：`eflesh.ino` 和 README 的 CS 表都是 `{5,2,4,13,12}`，二进制固件误写成
  `{2,4,5,13,12}`（头部注释还写着"same wiring as eflash.ino"）
- **修复**：改回 `{5, 2, 4, 13, 12}`

## Bug ③帧率只有 ~16Hz

- **证据**：库源码 `readData()`（Adafruit_MLX90393.cpp:362-371）：
  ```cpp
  delay(mlx90393_tconv[_dig_filt][_osr] + 10);   // 每个传感器固定阻塞!
  ```
  5 个传感器 × ~12.6ms ≈ 63ms/帧 → ~16Hz
- **修复**：改用库的公开底层 API 流水线化：
  ```cpp
  for (5 个传感器) startSingleMeasurement();   // 同时启动，芯片内并行转换
  delayMicroseconds(MEAS_WAIT_US);             // 整体只等一次 tconv
  for (5 个传感器) readMeasurement();           // 依次读数
  ```

## Bug ④箭头闪烁（每 3 帧丢 2 帧）—— 最精彩的一个

- **症状**：3D 可视化里箭头一闪一闪；终端每 2 秒打印的 mags 交替出现全零
- **定位工具**：`eflesh-upper/logger.py`（为此专门写的无界面统计器），30 秒输出：
  ```
  f=155.8Hz lost=0 badck=0 skipB=0 最大帧间隔=7ms     ← 链路完美，无重启
  各传感器失败率全部 66.7%，连发最长恰好 2 帧，1558 次
  k 分布: k=0:1557  k=5:3116  (k=1..4 全为 0)          ← 五路同生共死，纯周期性
  ```
- **推理**：失败率精确 2/3 + 连发恰 2 帧 + k 只有 0/5 → 周期性时序问题，与接线/
  供电/链路无关。设帧周期 P=6.42ms，"每 3 帧 1 帧成功" ⟹ 实际转换时间
  tconv ∈ (2P, 3P) = (12.8, 19.3)ms
- **根因实锤**：查库的 `_init()`（Adafruit_MLX90393.cpp）：
  ```cpp
  if (!setOversampling(MLX90393_OSR_3))   // begin_SPI() 默认设成最慢的 OSR!
  ```
  固件只调了 `setFilter(FILTER_3)` 从没碰 OSR → 查表 `tconv[FILTER_3][OSR_3]`
  = **16.05ms**，正好落在推理区间内。MLX90393 的 RM 命令在测量进行中会返回
  ERR，所以 16ms 的测量对上 6.4ms 的帧周期，就是每 3 帧只有 1 帧能读到数
- **修复**：
  ```cpp
  sensors[i].setOversampling(MLX90393_OSR_0);   // tconv 16.05ms → 2.61ms
  ```
  等待时间改为直接查库头文件公开表计算，以后改滤波设置自动同步：
  ```cpp
  const uint32_t MEAS_WAIT_US =
      (uint32_t)(1000.0f * (mlx90393_tconv[MLX90393_FILTER_3][MLX90393_OSR_0] + 2.0f));
  ```

## 经验教训

1. **MLX90393 快速轮询的三个坑**：库默认 OSR_3（慢）、`readData()` 内置固定
   +10ms 阻塞、RM 命令测量中会被拒。想快必须：显式设 OSR + 用
   startSingleMeasurement/readMeasurement 流水线 + tconv 查表定时。
2. **转换时间表**（手册 Table 18，即库里 `mlx90393_tconv[8][4]`，单位 ms）节选：
   | DIGFILT | OSR_0 | OSR_1 | OSR_2 | OSR_3 |
   |---|---|---|---|---|
   | 3（本固件） | 2.61 | 4.53 | 8.37 | 16.05 |
   | 5（文本固件） | 7.22 | 13.75 | 26.80 | 52.92 |
   **规律：tconv 必须小于帧周期，否则按倍数关系出现"每 N 帧 1 帧成功"。**
3. **波特率看桥接芯片**：CP2102→921600、CP2102N/CH340→2M、原生 USB→随意。
   先 `lsusb` 再定波特率。
4. **统计模式比肉眼强**：2 秒一次的打印采样不出 2/3 这种精确比例；
   logger.py 的失败率/连发长度/k 分布三个统计量直接把问题分类成
   "单路接线 / 公共总线 / 周期时序 / 链路误码"四种。

## 变更文件

| 文件 | 内容 |
|---|---|
| `eflash_binary/eflash_binary.ino` | 4 项修复（文件头注释含完整说明）+ 初始化重试 |
| `eflesh-upper/visualize_3d.py` | 二进制模式默认 921600；状态行加 f/lost/badck/resync/skipB |
| `eflesh-upper/logger.py` | 新增：二进制帧失败模式诊断工具 |
| `README.md` | 波特率/帧率说明同步，CP2102 注意事项 |

## 遗留事项

- `eflesh-upper/test.py` 还是旧的 15 通道配置（S1~S15），配合本固件只用前 5 路
- PlatformIO 环境待配（Core 6.1.19 已装于 `~/.platformio`，VSCode 扩展已装，
  仓库尚无 `platformio.ini`）
