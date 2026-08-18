/*
  eFlesh Binary SPI Firmware — High-Speed Sensor Reader
  ======================================================
  Hardware: ESP32 + 5×MLX90393 on shared SPI bus, individual CS pins.
  Same wiring as eflash.ino, but:
    - Binary Serial.write() instead of text Serial.print()
    - No delay() — runs as fast as sensor reads allow (~100+ Hz)
    - 64-byte fixed frame with sync + seq + checksum

  Binary frame (64 bytes):
    ┌────────┬────────┬────────┬──────────────────────────┬──────────┐
    │  0xAA  │  0x55  │  seq   │  5×(float x,y,z) = 60 B  │  xor ck  │
    │  sync  │  sync  │ uint8  │       little-endian      │  uint8   │
    └────────┴────────┴────────┴──────────────────────────┴──────────┘
*/

#include <SPI.h>
#include <Adafruit_MLX90393.h>

#define NUM_SENSORS 5

// CS pins — same as eflash.ino
const int CS_PINS[NUM_SENSORS] = {2, 4, 5, 13, 12};

Adafruit_MLX90393 sensors[NUM_SENSORS];
bool sensor_ok[NUM_SENSORS] = {false};

uint8_t frame_seq = 0;

static const uint8_t SYNC0 = 0xAA;
static const uint8_t SYNC1 = 0x55;

void setup() {
  Serial.begin(2000000);   // ESP32 USB-CDC
  while (!Serial) { delay(5); }

  // 【千万别忘了加回这段！防止上电时 SPI 总线冲突】
  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(CS_PINS[i], OUTPUT);
    digitalWrite(CS_PINS[i], HIGH);
  }

  SPI.begin();             // ESP32 VSPI: SCK=18, MISO=19, MOSI=23

  for (int i = 0; i < NUM_SENSORS; i++) {
    if (sensors[i].begin_SPI(CS_PINS[i], &SPI)) {
      sensor_ok[i] = true;
      sensors[i].setGain(MLX90393_GAIN_1X);
      sensors[i].setResolution(MLX90393_X, MLX90393_RES_16);
      sensors[i].setResolution(MLX90393_Y, MLX90393_RES_16);
      sensors[i].setResolution(MLX90393_Z, MLX90393_RES_16);
      // 降低滤波等级到 FILTER_3，牺牲一点点噪声换取更高的采样率，非常正确的决定！
      sensors[i].setFilter(MLX90393_FILTER_3);   
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

  *p++ = SYNC0;
  *p++ = SYNC1;
  *p++ = frame_seq++;

  for (int i = 0; i < NUM_SENSORS; i++) {
    if (sensor_ok[i]) {
      sensors[i].readData(&x, &y, &z);
    } else {
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
