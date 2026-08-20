/*
  eFlesh Binary SPI Firmware — High-Speed Sensor Reader (fixed 2026-08-18)
  ======================================================
  Hardware: ESP32 + 5×MLX90393 on shared SPI bus, individual CS pins.
  接线与 eflesh.ino 完全一致：CS = {5, 2, 4, 13, 12}，VSPI: SCK=18, MISO=19, MOSI=23

  本次修复的 4 个问题：
    1.【乱码】波特率 2000000 → 921600。
       ESP32 DevKit 的 USB 串口芯片是 CP2102（lsusb: 10c4:ea60），
       它最高只支持 921600，2M baud 出来全是乱码 —— 这就是之前"不正常"的主因。
       921600 对本协议绰绰有余（64B×140Hz ≈ 90 kbps）。
    2.【数据错位】CS 引脚顺序改回 {5,2,4,13,12}，与 eflesh.ino / README 一致。
       旧版误写成 {2,4,5,13,12}，S1 和 S3 的数据是互换的。
    3.【帧率】不再用 readData()（它对每个传感器阻塞 tconv+10ms，5 个只有 ~16Hz）。
       改为：5 个传感器同时启动测量 → 整体只等一次转换时间 → 依次读数。
    4.【闪烁/每3帧2帧全零】库的 _init() 会把过采样设成最慢的 MLX90393_OSR_3
       （tconv=16.05ms），旧代码只改了 Filter 没改 OSR。测量时间 16ms ≈ 2.5×帧周期，
       RM 命令每 3 帧只有 1 帧能读到数（上位机箭头"一闪一闪"）。
       现在显式设 MLX90393_OSR_0（tconv=2.61ms），全帧有效，实测 ~170Hz。

  Binary frame (64 bytes) — 与上位机 visualize_3d.py --binary 完全兼容:
    ┌────────┬────────┬────────┬──────────────────────────┬──────────┐
    │  0xAA  │  0x55  │  seq   │  5×(float x,y,z) = 60 B  │  xor ck  │
    │  sync  │  sync  │ uint8  │       little-endian      │  uint8   │
    └────────┴────────┴────────┴──────────────────────────┴──────────┘
*/

#include <SPI.h>
#include <Adafruit_MLX90393.h>

#define NUM_SENSORS 5

// 【修复2】与 eflesh.ino 相同的 CS 顺序 —— S1..S5 依次是 5, 2, 4, 13, 12
// 物理位置（2026-08-20 拔线定案，详见 docs/2026-08-20-hardware-layout.md）：
//   S1=右下角  S2=左下角  S3=右上角  S4=左上角(头顶磁体反装S朝上,保留不拆)  S5=中心
const int CS_PINS[NUM_SENSORS] = {5, 2, 4, 13, 12};

// 【修复1】CP2102 的上限是 921600。
// 如果以后换ESP32-S2/S3（原生 USB CDC），可以改回 2000000。
const uint32_t SERIAL_BAUD = 921600;

// 一轮测量的等待时间：直接查库头文件里的公开转换时间表
// mlx90393_tconv[DIGFILT][OSR] + 2ms 安全余量（手册 Table 18 的值）。
// 改 setFilter/setOversampling 时这里自动跟着变（保持下标一致即可）。
const uint32_t MEAS_WAIT_US =
    (uint32_t)(1000.0f * (mlx90393_tconv[MLX90393_FILTER_3][MLX90393_OSR_0] + 2.0f));

Adafruit_MLX90393 sensors[NUM_SENSORS];
bool sensor_ok[NUM_SENSORS] = {false};

uint8_t frame_seq = 0;

static const uint8_t SYNC0 = 0xAA;
static const uint8_t SYNC1 = 0x55;

void setup() {
  Serial.begin(SERIAL_BAUD);

  // 最多等 1.5s：普通 ESP32 立即通过；USB-CDC 板子不会因没开串口监视器而卡死
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 1500) { delay(5); }

  // 【千万别删：上电立刻把所有 CS 拉高，防止 SPI 总线冲突】
  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(CS_PINS[i], OUTPUT);
    digitalWrite(CS_PINS[i], HIGH);
  }

  SPI.begin();             // ESP32 VSPI: SCK=18, MISO=19, MOSI=23

  for (int i = 0; i < NUM_SENSORS; i++) {
    // 重试 3 次：异常复位（掉电/看门狗）会把 SPI 事务打断在半途，传感器挂在
    // 脏状态导致 begin_SPI 失败、sensor_ok 全 false、输出整帧全零。
    // 库内部有 spi_dev 判空，重复 begin_SPI 无泄漏，可安全重试。
    for (int attempt = 0; attempt < 3 && !sensor_ok[i]; attempt++) {
      if (attempt > 0) delay(50);   // 给传感器留出复位时间
      sensor_ok[i] = sensors[i].begin_SPI(CS_PINS[i], &SPI);
    }
    if (sensor_ok[i]) {
      sensors[i].setGain(MLX90393_GAIN_1X);
      // 【量程修复 2026-08-20】RES_16 时 XY 满量程 ±4.92mT(0.150µT/LSB)、Z ±7.93mT(0.242µT/LSB)。
      // 静态场就有 z≈8.1mT 顶轨，按压必溢出：S1z 实测被钉在 ±7930µT 来回翻，
      // 表观 ΔB 恒 ≈15.86mT(=2^16×0.242µT)；手指深压时 XY 也超 ±4.92mT 同样翻转。
      // RES_19: XY ±39.4mT / Z ±63.4mT，量化 1.2/1.9µT << 噪声(~10-25µT)，SNR 无损。
      sensors[i].setResolution(MLX90393_X, MLX90393_RES_19);
      sensors[i].setResolution(MLX90393_Y, MLX90393_RES_19);
      sensors[i].setResolution(MLX90393_Z, MLX90393_RES_19);
      // FILTER_3：噪声比 FILTER_5 略高，但转换时间从 7.2ms 降到 2.6ms
      sensors[i].setFilter(MLX90393_FILTER_3);
      // 【修复4，关键！】库的 _init() 默认把 OSR 设成最慢的 OSR_3（tconv=16.05ms），
      // 不显式改掉的话测量时间远超帧周期，RM 每 3 帧才成功 1 帧 → 箭头闪烁
      sensors[i].setOversampling(MLX90393_OSR_0);   // tconv=2.61ms
    }
  }

  // Small delay to let serial settle
  delay(10);
}

// -------------------------------------------------------------------
void loop() {
  float x, y, z;
  uint8_t buf[64];
  uint8_t* p = buf;

  // 【修复3】流水线测量：先给 5 个传感器都发"开始测量"(SM, 全轴)，
  // 它们在芯片内部并行转换，然后整体只等一次 tconv。
  for (int i = 0; i < NUM_SENSORS; i++) {
    if (sensor_ok[i]) sensors[i].startSingleMeasurement();
  }
  delayMicroseconds(MEAS_WAIT_US);

  *p++ = SYNC0;
  *p++ = SYNC1;
  *p++ = frame_seq++;

  for (int i = 0; i < NUM_SENSORS; i++) {
    // 读失败（掉线/未就绪）→ 该帧此传感器输出 0：
    // 在上位机上看得到异常，也不会残留上一帧的旧数据
    if (!sensor_ok[i] || !sensors[i].readMeasurement(&x, &y, &z)) {
      x = y = z = 0.0f;
    }
    memcpy(p, &x, 4);  p += 4;
    memcpy(p, &y, 4);  p += 4;
    memcpy(p, &z, 4);  p += 4;
  }

  // XOR checksum over bytes 2..62
  uint8_t ck = 0;
  for (int i = 2; i < 63; i++) ck ^= buf[i];
  buf[63] = ck;

  Serial.write(buf, 64);
}
