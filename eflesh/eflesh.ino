#include <SPI.h>
#include <Adafruit_MLX90393.h>

// 定义 5 个传感器的 CS 引脚
const int CS_PINS[5] = {5, 2, 4, 13, 12};

// 创建 5 个传感器对象数组
Adafruit_MLX90393 sensors[5];

// 记录传感器是否初始化成功的标志，防止坏板子卡死程序
bool sensor_active[5] = {false, false, false, false, false};

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); } // 等待串口连接
  
  Serial.println("MLX90393 5-Sensor Array Test Started!");

  // 【核心修复：上电立刻把所有 CS 引脚设为输出并拉高，防止总线冲突】
  for (int i = 0; i < 5; i++) {
    pinMode(CS_PINS[i], OUTPUT);
    digitalWrite(CS_PINS[i], HIGH);
  }

  // 循环初始化每一个传感器
  for (int i = 0; i < 5; i++) {
    Serial.print("Initializing Sensor ");
    Serial.print(i + 1);
    Serial.print(" on CS pin ");
    Serial.print(CS_PINS[i]);
    Serial.print("... ");

    // 传入 CS 引脚，并绑定到硬件 SPI
    if (sensors[i].begin_SPI(CS_PINS[i], &SPI)) {
      Serial.println("OK!");
      sensor_active[i] = true;
      
      // 设置传感器的增益和分辨率 (可根据磁铁强度在后续微调)
      sensors[i].setGain(MLX90393_GAIN_1X);
      sensors[i].setResolution(MLX90393_X, MLX90393_RES_16);
      sensors[i].setResolution(MLX90393_Y, MLX90393_RES_16);
      sensors[i].setResolution(MLX90393_Z, MLX90393_RES_16);
      // 设置数字滤波，降低噪声
      sensors[i].setFilter(MLX90393_FILTER_5);
    } else {
      Serial.println("FAILED! Check wiring or soldering.");
    }
  }
  
  Serial.println("===========================================");
  delay(1000);
}

void loop() {
  float x, y, z;
  
  // 打印一个数据帧的开头，方便后续如果你要写 Python 脚本解析
  Serial.print("FRAME: ");

  for (int i = 0; i < 5; i++) {
    if (sensor_active[i]) {
      // 读取数据 (Adafruit 库自带轮询 Status Byte 的等待逻辑)
      sensors[i].readData(&x, &y, &z);
      
      // 按特定格式打印： S1_X,S1_Y,S1_Z | S2_X...
      Serial.print("[S");
      Serial.print(i + 1);
      Serial.print(":");
      Serial.print(x, 1); Serial.print(",");
      Serial.print(y, 1); Serial.print(",");
      Serial.print(z, 1);
      Serial.print("] ");
    } else {
      Serial.print("[S");
      Serial.print(i + 1);
      Serial.print(":ERR] ");
    }
  }
  
  Serial.println(); // 换行，结束这一帧
  
  // 延时 50ms (大概 20Hz 刷新率，用来测试串口看数据刚好)
  delay(50);
}