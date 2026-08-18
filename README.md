# eflesh-touch

eFlesh 磁场触觉传感器 —— ESP32 + 5× MLX90393 (SPI)，含 Arduino 固件与 Python 实时上位机可视化。

## 目录结构

```
eflesh-touch/
├── eflesh/                  # 固件：文本协议 (115200 baud)
│   └── eflesh.ino
├── eflash_binary/           # 固件：二进制协议 (921600 baud, ~170 Hz)
│   └── eflash_binary.ino
└── eflesh-upper/            # Python 上位机
    ├── test.py              # 15 通道传感器实时曲线 (matplotlib)
    └── visualize_3d.py      # 5 传感器 3D 磁场矢量可视化 (pyqtgraph OpenGL)
```

## 硬件

- ESP32
- 5× MLX90393 三轴磁力计，共享 SPI 总线，各自独立 CS 引脚（5 / 2 / 4 / 13 / 12）

## 固件

两份固件接线相同，区别在串口输出协议：

| 固件 | 协议 | 波特率 | 帧格式 |
|------|------|--------|--------|
| `eflesh/eflesh.ino` | 文本 | 115200 | `FRAME: [S1:x,y,z] [S2:x,y,z] ...` |
| `eflash_binary/eflash_binary.ino` | 二进制 | 921600 | 64 字节定长帧：`AA 55` 同步 + seq + 5×(float x,y,z) LE + XOR 校验 |

> 注意：波特率上限取决于板子上的 USB 串口芯片 —— CP2102 最高 921600（ESP32 DevKit 常见），
> CH340/CP2102N 可到 2M。921600 对 64B×140Hz ≈ 90 kbps 的数据量绰绰有余。

## 上位机使用

依赖：

```bash
pip install pyserial numpy matplotlib PyQt5 pyqtgraph
```

3D 磁场矢量可视化（无需硬件可用 `--sim` 演示模式）：

```bash
python eflesh-upper/visualize_3d.py                        # 文本协议
python eflesh-upper/visualize_3d.py --binary               # 二进制协议（高速）
python eflesh-upper/visualize_3d.py --sim                  # 模拟数据演示
python eflesh-upper/visualize_3d.py --port /dev/ttyUSB0 --baud 921600 --binary
```

15 通道实时曲线：

```bash
python eflesh-upper/test.py    # 修改脚本内 COM_PORT 为实际串口
```

## License

MIT
