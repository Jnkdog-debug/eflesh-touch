/*
  单 CS 逐个测试固件 —— 传感器身份 / 极性诊断
  ============================================================
  用途：
    1. 拔线测试：哪根 CS 上没有芯片，选择它时初始化直接失败
    2. 身份测试：选中某一路后用手指压皮肤的角/中心，看哪路有响应
       → 直接建立 GPIO(数据槽) ↔ 物理芯片位置 的对应关系
    3. 极性测试：静态 z 的符号 = 该芯片头顶磁体的极性
       （芯片在磁体正下方时：N 朝上 → z 静态为正；S 朝上 → 为负）

  用法：烧录后打开串口监视器 (115200)，单行发送：
    s  —— 扫描模式：5 路各测 2 秒，输出汇总表（先跑这个）
    1..5 —— 锁定只测 S1..S5 中的一路（GPIO 5,2,4,13,12），
            每帧打印实时读数，每 1 秒打印中位数统计
            （用中位数：即使总线偶发被搅脏，坏帧也不影响统计）

  CS 映射与生产固件 eflash_binary.ino 完全一致：S1..S5 = {5,2,4,13,12}
*/

#include <SPI.h>
#include <Adafruit_MLX90393.h>

const int CS_PINS[5] = {5, 2, 4, 13, 12};

Adafruit_MLX90393 sensors[5];
bool slot_ok[5] = {false, false, false, false, false};
int cur = -1;              // 当前锁定的槽号 (0..4)，-1 = 未锁定

const int WIN = 20;        // 统计窗口：20 帧 ≈ 1s
float wx[WIN], wy[WIN], wz[WIN];
int nwin = 0;

// ------------------------------------------------------------------
void allCsHigh() {
  // 关键：所有 CS 一直被 ESP32 主动驱动为高（不悬空），
  // 只有被测那路在事务期间由库拉低 —— 别的芯片永远不抢总线
  for (int i = 0; i < 5; i++) {
    pinMode(CS_PINS[i], OUTPUT);
    digitalWrite(CS_PINS[i], HIGH);
  }
}

bool initSlot(int i) {
  bool ok = false;
  for (int attempt = 0; attempt < 3 && !ok; attempt++) {
    if (attempt > 0) delay(50);
    ok = sensors[i].begin_SPI(CS_PINS[i], &SPI);
  }
  if (ok) {
    // 与生产固件相同的量程/滤波设置，静态值可直接对比
    sensors[i].setGain(MLX90393_GAIN_1X);
    sensors[i].setResolution(MLX90393_X, MLX90393_RES_19);
    sensors[i].setResolution(MLX90393_Y, MLX90393_RES_19);
    sensors[i].setResolution(MLX90393_Z, MLX90393_RES_19);
    sensors[i].setFilter(MLX90393_FILTER_5);
  }
  slot_ok[i] = ok;
  return ok;
}

float medianOf(float* a, int n) {
  // 就地排序取中位数（n ≤ 20，插入排序足够）
  for (int i = 1; i < n; i++) {
    float v = a[i];
    int j = i - 1;
    while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; }
    a[j + 1] = v;
  }
  return (n & 1) ? a[n / 2] : 0.5f * (a[n / 2 - 1] + a[n / 2]);
}

void printStatsLine(const char* tag) {
  float mx = medianOf(wx, nwin), my = medianOf(wy, nwin), mz = medianOf(wz, nwin);
  int glitch = 0;
  for (int k = 0; k < nwin; k++)
    if (fabsf(wx[k]) > 40000 || fabsf(wy[k]) > 40000 || fabsf(wz[k]) > 40000) glitch++;
  Serial.print(tag);
  Serial.print(" 中位数 x,y,z = ");
  Serial.print(mx, 1); Serial.print(", ");
  Serial.print(my, 1); Serial.print(", ");
  Serial.print(mz, 1); Serial.print(" uT   坏帧 ");
  Serial.print(glitch); Serial.print("/"); Serial.println(nwin);
}

void selectSlot(int i) {
  cur = i;
  nwin = 0;
  Serial.println("\n==============================================");
  Serial.print(">> 锁定 S"); Serial.print(i + 1);
  Serial.print(" (GPIO"); Serial.print(CS_PINS[i]); Serial.println(")");
  Serial.print("   初始化: ");
  if (initSlot(i)) {
    Serial.println("OK   —— 现在用手指压皮肤各位置，看这路的实时响应");
  } else {
    Serial.println("FAILED —— 这根 CS 线上没有应答的芯片（线被拔/断焊/无芯片）");
  }
  Serial.println("==============================================");
}

// ------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  allCsHigh();
  SPI.begin();   // VSPI: SCK=18 MISO=19 MOSI=23

  Serial.println("\n=== MLX90393 单 CS 测试 ===");
  Serial.println("发 s  = 扫描全部 5 路（各 2 秒，出汇总表）");
  Serial.println("发 1..5 = 锁定一路实时看（手指按压测试用这个）");
  Serial.println("初始先跑一次扫描...\n");

  scanAll();
}

// 扫描模式：每路 2 秒，只打印统计，适合一眼看清 5 路状态
void scanAll() {
  Serial.println("---------- 扫描 5 路 ----------");
  Serial.println("槽    GPIO   初始化   z中位数(uT)   坏帧");
  for (int i = 0; i < 5; i++) {
    Serial.print("S"); Serial.print(i + 1);
    Serial.print("    GPIO"); Serial.print(CS_PINS[i]);
    Serial.print("   ");
    if (!initSlot(i)) {
      Serial.println("FAILED   --            --");
      continue;
    }
    nwin = 0;
    float x, y, z;
    for (int k = 0; k < WIN; k++) {
      if (sensors[i].readData(&x, &y, &z)) {
        wx[k] = x; wy[k] = y; wz[k] = z;
      } else {
        wx[k] = wy[k] = wz[k] = 1e9f;   // 读失败记为坏帧
      }
      delay(100);                       // 20 帧 × 100ms = 2s
    }
    nwin = WIN;
    float mz = medianOf(wz, nwin);
    int glitch = 0;
    for (int k = 0; k < nwin; k++)
      if (wx[k] > 9e8f || fabsf(wx[k]) > 40000 || fabsf(wz[k]) > 40000) glitch++;
    Serial.print("OK       ");
    Serial.print(mz, 1);
    Serial.print("        ");
    Serial.println(glitch);
    delay(200);
  }
  Serial.println("------------------------------");
  Serial.println("提示：发 1..5 锁定一路做手指按压测试");
}

// ------------------------------------------------------------------
void loop() {
  // 串口命令
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 's' || c == 'S') { cur = -1; scanAll(); return; }
    if (c >= '1' && c <= '5') selectSlot(c - '1');
  }

  if (cur < 0 || !slot_ok[cur]) { delay(50); return; }

  float x, y, z;
  if (sensors[cur].readData(&x, &y, &z)) {
    Serial.print("S"); Serial.print(cur + 1);
    Serial.print("  x="); Serial.print(x, 1);
    Serial.print("  y="); Serial.print(y, 1);
    Serial.print("  z="); Serial.print(z, 1);
    Serial.println(" uT");

    wx[nwin] = x; wy[nwin] = y; wz[nwin] = z;
    nwin++;
    if (nwin >= WIN) {
      printStatsLine("[统计]");
      nwin = 0;
    }
  } else {
    Serial.println("S?  读取失败");
  }
  delay(50);   // ~20Hz，串口监视器刚好看得清
}
