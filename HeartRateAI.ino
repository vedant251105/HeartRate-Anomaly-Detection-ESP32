#define BLYNK_TEMPLATE_ID   "TMPL3PMzPzLMN"
#define BLYNK_TEMPLATE_NAME "HEART MONITOR 1"
#define BLYNK_AUTH_TOKEN    "e-BwPLpOIagGKqYZuWxMo4hhV1A3-gQ1"
#define BLYNK_PRINT         Serial

#include <Wire.h>
#include <Preferences.h>
#include <WiFi.h>
#include <BlynkSimpleEsp32.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "MAX30105.h"
#include "MPU6050.h"
#include "model.h"
#include "UserProfile.h"
#include "Recommendations.h"
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

// ============================================================
// CREDENTIALS
// ============================================================
const char* ssid          = "Wifi";
const char* password_wifi = "12345789";

// ============================================================
// PINS
// ============================================================
#define SDA_PIN           21
#define SCL_PIN           22
#define BUZZER_PIN        25
#define RED_LED_PIN       26
#define GREEN_LED_PIN     27
#define ACK_BUTTON_PIN    14
#define RESET_BUTTON_PIN  13

// ── LEDC buzzer (no tone() on ESP32) ─────────────────────────
#define BUZZER_CH   0
#define BUZZER_RES  8

void buzzerTone(int freq, int ms) {
  ledcSetup(BUZZER_CH, freq, BUZZER_RES);
  ledcAttachPin(BUZZER_PIN, BUZZER_CH);
  ledcWrite(BUZZER_CH, 128);
  if (ms > 0) { delay(ms); ledcWrite(BUZZER_CH, 0); }
}
void buzzerOff() { ledcWrite(BUZZER_CH, 0); }

// ============================================================
// OLED
// ============================================================
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT  64
#define OLED_ADDR    0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// ============================================================
// BLYNK VIRTUAL PINS
// ============================================================
#define V_BPM       V0
#define V_STATUS    V1
#define V_ALERT     V2
#define V_LED       V3
#define V_EMERGENCY V4
#define V_RESPONSE  V5
#define V_CHART     V6
#define V_TERMINAL  V8

WidgetTerminal terminal(V_TERMINAL);
BlynkTimer     blynkTimer;

// ============================================================
// TFLITE + SENSOR CONSTANTS
// ============================================================
#define FINGER_THRESHOLD  8000

#define BEAT_BUF_SIZE      8
#define RR_BUF_SIZE       32
#define MIN_CALIB_BEATS   40
#define CALIB_HR_MIN      45.0f
#define CALIB_HR_MAX     120.0f

const int TENSOR_ARENA_SIZE = 10 * 1024;
uint8_t   tensor_arena[TENSOR_ARENA_SIZE];

const tflite::Model*      tf_model    = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor*             tfl_input   = nullptr;
TfLiteTensor*             tfl_output  = nullptr;

MAX30105    particleSensor;
MPU6050     mpu;
UserProfile profile;
Preferences prefs;

// ============================================================
// BEAT DETECTION STATE
// ============================================================
float beat_hr_buf[BEAT_BUF_SIZE];
int   beat_buf_idx   = 0;
int   beat_buf_count = 0;

float rr_buf[RR_BUF_SIZE];
int   rr_buf_idx   = 0;
int   rr_buf_count = 0;

float calib_beats[300];
int   calib_beats_count = 0;

float         beat_dc       = 0;
float         beat_peak     = 0;
float         beat_valley   = 999999;
bool          beat_rising   = false;
unsigned long beat_lastTime = 0;
float         hr_bpm_stable = 75.0f;
int           current_act   = ACT_REST;

float personal_threshold = 0.5f;

// ============================================================
// MONITORING STATE
// ============================================================
int    current_bpm    = 0;
float  current_mse    = 0;
float  current_rmssd  = 0;
float  rmssd_smooth   = 20.0f;

#define RMSSD_ALPHA 0.3f

String current_status   = "Normal";
String current_act_name = "REST";

// ── Alert state ───────────────────────────────────────────────
bool          alertActive    = false;
unsigned long alertStartTime = 0;
const unsigned long ACK_TIMEOUT     = 15000UL;
const unsigned long ALERT_COOLDOWN  = 30000UL;
unsigned long lastAlertTime  = 0;

bool lowAlertSent  = false;
bool highAlertSent = false;

int consecutiveAnomalyCount  = 0;
#define NOTIFY_AFTER_CONSECUTIVE  5

// Simple BPM thresholds
const int CRITICAL_LOW  =  35;
const int LOW_BPM       =  50;
const int HIGH_BPM      = 120;
const int CRITICAL_HIGH = 150;

bool csvHeaderDone = false;

// ============================================================
// FORWARD DECLARATIONS
// ============================================================
void  calibrateNewUser();
float measureRestingHR(int sec);
void  computePersonalThreshold(float rest_hr);
int   detectActivity(float mag);
void  runInference(float hr_bpm);
float detectBeat(long ir, unsigned long now);
void  resetBeatDetector();
float computeInferenceMSE(float hr_mean, float hr_std,
                           float rmssd, float hr_delta, int act);
void  triggerAlert(String reason, bool critical);
void  respondToAlert(String method);
void  checkAckTimeout();
void  checkAckButton();
void  checkResetButton();
void  oledShowHR();
void  oledShowWarning(String line1, String line2);
void  oledShowCalibrating(String msg);
void  oledShowNoFinger();
void  oledShowRestarting();
void  blynkSync();
void  printCSVHeader();
void  logCSV(String status);
float robustMean(float* arr, int n);
void  toneStartup();
void  toneLow();
void  toneHigh();
void  toneCritical();
void  toneEmergency();
void  toneReset();

// ============================================================
// BEAT DETECTOR
// ============================================================
void resetBeatDetector() {
  beat_dc       = 0;
  beat_peak     = 0;
  beat_valley   = 999999;
  beat_rising   = false;
  beat_lastTime = 0;
}

float detectBeat(long irValue, unsigned long now) {
  float ir = (float)irValue;
  beat_dc  = beat_dc * 0.97f + ir * 0.03f;
  float ac = ir - beat_dc;

  if (ac > beat_peak)   beat_peak   = ac;
  if (ac < beat_valley) beat_valley = ac;

  float range = beat_peak - beat_valley;
  if (range < 50.0f) {
    beat_peak   *= 0.99f;
    beat_valley *= 0.99f;
    return 0;
  }

  float thr = beat_valley + range * 0.6f;
  beat_peak   *= 0.998f;
  beat_valley *= 0.998f;

  if ((now - beat_lastTime) < 400UL) {
    if (ac > thr) beat_rising = true;
    return 0;
  }
  if (ac > thr && !beat_rising) beat_rising = true;

  float hr_det = 0;
  if (ac < thr && beat_rising) {
    beat_rising = false;
    long delta  = (long)(now - beat_lastTime);
    if (delta > 400 && delta < 2000 && beat_lastTime > 0) {
      float hr = 60000.0f / (float)delta;
      if (abs(hr - hr_bpm_stable) < 25.0f || hr_bpm_stable < 40.0f) {
        hr_det        = hr;
        hr_bpm_stable = hr_bpm_stable * 0.85f + hr * 0.15f;
      }
    }
    beat_lastTime = now;
  }
  return hr_det;
}

// ============================================================
// INFERENCE HELPER
// ============================================================
float computeInferenceMSE(float hr_mean, float hr_std,
                           float rmssd, float hr_delta, int act) {
  float zs  = profile.getZScore(hr_mean, act);
  float ahr = profile.getActivityHRRatio(hr_mean, act);
  float f[8] = { zs, hr_std, rmssd, (float)act,
                 1.0f, 0.1f, hr_delta, ahr };
  float n[8];
  for (int i = 0; i < 8; i++)
    n[i] = (f[i] - scaler_mean[i]) / scaler_scale[i];
  for (int i = 0; i < 8; i++)
    tfl_input->data.f[i] = n[i];
  interpreter->Invoke();
  float mse = 0;
  for (int i = 0; i < 8; i++) {
    float d = n[i] - tfl_output->data.f[i];
    mse += d * d;
  }
  return mse / 8.0f;
}

// ============================================================
// CSV LOGGING
// ============================================================
void printCSVHeader() {
  Serial.println("TIMESTAMP_MS,SESSION_S,ACTIVITY,HR_BPM,"
                 "HRV_MS,MSE,THRESHOLD,STATUS");
  csvHeaderDone = true;
}

void logCSV(String status) {
  if (!csvHeaderDone) printCSVHeader();
  Serial.printf("CSV,%lu,%lu,%s,%d,%.1f,%.4f,%.4f,%s\n",
    millis(), millis() / 1000UL,
    current_act_name.c_str(),
    current_bpm,
    current_rmssd,
    current_mse, personal_threshold,
    status.c_str());
}

// ============================================================
// OLED SCREENS
// ============================================================
void oledShowHR() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(35, 0);
  display.print("HEART RATE");
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  display.setTextSize(3);
  int x = (current_bpm >= 100) ? 22 : (current_bpm >= 10 ? 34 : 46);
  display.setCursor(x, 14);
  display.print(current_bpm);
  display.setTextSize(1);
  display.setCursor(98, 22);
  display.print("BPM");

  display.drawLine(0, 43, 127, 43, SSD1306_WHITE);
  display.setCursor(0, 47);
  display.print(current_act_name);
  display.setCursor(72, 47);
  display.print(current_status);

  display.setCursor(0, 57);
  display.print("HRV:");
  display.print((int)current_rmssd);
  display.print("ms");

  if (WiFi.isConnected())
    display.fillCircle(122, 59, 3, SSD1306_WHITE);
  else
    display.drawCircle(122, 59, 3, SSD1306_WHITE);

  display.display();
}

void oledShowWarning(String line1, String line2) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.fillRect(0, 0, 128, 12, SSD1306_WHITE);
  display.setTextColor(SSD1306_BLACK);
  display.setTextSize(1);
  display.setCursor(18, 2);
  display.print("!! WARNING !!");
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(2);
  int x = (current_bpm >= 100) ? 14 : (current_bpm >= 10 ? 26 : 38);
  display.setCursor(x, 14);
  display.print(current_bpm);
  display.setTextSize(1);
  display.setCursor(82, 20);
  display.print("BPM");

  display.setCursor(0, 34);
  display.print("Type: "); display.print(line1);
  display.setCursor(0, 44);
  display.print(line2);

  display.drawLine(0, 54, 127, 54, SSD1306_WHITE);
  display.setCursor(2, 56);
  display.print("Press button to ACK");
  display.display();
}

void oledShowCalibrating(String msg) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(15, 4);  display.print("CALIBRATING...");
  display.drawLine(0, 14, 127, 14, SSD1306_WHITE);
  display.setCursor(0, 20);  display.print(msg);
  display.setCursor(0, 50);  display.print("Keep finger still");
  display.display();
}

void oledShowNoFinger() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(10, 15); display.print("Place finger on");
  display.setCursor(18, 28); display.print("MAX30102 sensor");
  display.drawLine(0, 42, 127, 42, SSD1306_WHITE);
  display.setCursor(5, 48);  display.print("GPIO13 = Full Restart");
  display.setCursor(5, 57);  display.print("GPIO14 = Acknowledge");
  display.display();
}

void oledShowRestarting() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.fillRect(0, 0, 128, 12, SSD1306_WHITE);
  display.setTextColor(SSD1306_BLACK);
  display.setTextSize(1);
  display.setCursor(22, 2);  display.print("RESTARTING...");
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(10, 20); display.print("Full system");
  display.setCursor(10, 32); display.print("restart in 2s...");
  display.setCursor(10, 48); display.print("Please wait");
  display.display();
}

// ============================================================
// ALERT LOGIC
// ============================================================
void triggerAlert(String reason, bool critical) {
  alertActive    = true;
  alertStartTime = millis();
  lastAlertTime  = millis();

  String msg = reason + " | BPM=" + String(current_bpm);
  Blynk.logEvent("heart_alert", msg);
  Blynk.virtualWrite(V_ALERT, "ALERT: " + msg + " (ACK 15s)");
  Blynk.virtualWrite(V_LED, 0);
  terminal.println("ALERT: " + msg);
  terminal.flush();

  oledShowWarning(reason, "BPM=" + String(current_bpm));
  digitalWrite(GREEN_LED_PIN, LOW);
  digitalWrite(RED_LED_PIN,   HIGH);

  if (critical) toneCritical();
  else          toneHigh();

  Serial.println("ALERT: " + msg);
}

void respondToAlert(String method) {
  alertActive             = false;
  lowAlertSent            = false;
  highAlertSent           = false;
  consecutiveAnomalyCount = 0;
  buzzerOff();
  digitalWrite(RED_LED_PIN,   LOW);
  digitalWrite(GREEN_LED_PIN, HIGH);

  Blynk.logEvent("driver_ok", "Acknowledged via " + method);
  Blynk.virtualWrite(V_ALERT, "OK — Acknowledged (" + method + ")");
  Blynk.virtualWrite(V_LED, 255);
  terminal.println("Acknowledged via " + method);
  terminal.flush();
  Serial.println("Acknowledged via: " + method);
  oledShowHR();
}

void checkAckTimeout() {
  if (!alertActive) return;
  if (millis() - alertStartTime >= ACK_TIMEOUT) {
    alertActive = false;
    buzzerOff();

    String msg = "No ACK! BPM=" + String(current_bpm) +
                 " Status=" + current_status;
    Blynk.logEvent("emergency", msg);
    Blynk.virtualWrite(V_ALERT, "NO ACK — " + msg);
    Blynk.virtualWrite(V_EMERGENCY, msg);
    terminal.println("NO ACK TIMEOUT: " + msg);
    terminal.flush();

    Serial.println("NO ACK TIMEOUT: " + msg);

    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.fillRect(0, 0, 128, 12, SSD1306_WHITE);
    display.setTextColor(SSD1306_BLACK);
    display.setTextSize(1);
    display.setCursor(8, 2);  display.print("NO RESPONSE!");
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 18); display.print("Alert was not");
    display.setCursor(0, 28); display.print("acknowledged in 15s");
    display.setCursor(0, 42); display.print("BPM: " + String(current_bpm));
    display.setCursor(0, 54); display.print(current_status);
    display.display();

    toneEmergency();
    delay(3000);
    oledShowHR();
  }
}

BLYNK_WRITE(V_RESPONSE) {
  if (param.asInt() == 1 && alertActive)
    respondToAlert("Blynk App");
}

// ============================================================
// BUTTON HANDLERS
// ============================================================
void checkAckButton() {
  static bool last = HIGH;
  bool cur = digitalRead(ACK_BUTTON_PIN);
  if (last == HIGH && cur == LOW) {
    if (alertActive) respondToAlert("Button");
    delay(200);
  }
  last = cur;
}

void checkResetButton() {
  static bool          last      = HIGH;
  static unsigned long pressTime = 0;
  bool cur = digitalRead(RESET_BUTTON_PIN);
  if (last == HIGH && cur == LOW)  pressTime = millis();
  if (last == LOW  && cur == HIGH) {
    if (millis() - pressTime > 50) {
      Serial.println("Reset button pressed — restarting ESP32...");
      oledShowRestarting();
      toneReset();
      delay(2000);
      ESP.restart();
    }
  }
  last = cur;
}

// ============================================================
// BLYNK SYNC
// ============================================================
void blynkSync() {
  if (current_bpm < 1) return;
  Blynk.virtualWrite(V_BPM,    current_bpm);
  Blynk.virtualWrite(V_STATUS, current_status);
  Blynk.virtualWrite(V_CHART,  current_bpm);
}

// ============================================================
// ACTIVITY DETECTION
// ============================================================
int detectActivity(float mag) {
  float dev = abs(mag - 1.0f);
  if (dev < 0.20f) return ACT_REST;
  if (dev < 0.60f) return ACT_WALK;
  if (dev < 1.20f) return ACT_RUN;
  return ACT_INTENSE;
}

// ============================================================
// OUTLIER REJECTION
// ============================================================
float robustMean(float* arr, int n) {
  if (n <= 0) return 75.0f;
  float sum = 0;
  for (int i = 0; i < n; i++) sum += arr[i];
  float mean = sum / n;
  float var  = 0;
  for (int i = 0; i < n; i++) var += sq(arr[i] - mean);
  float sd = sqrt(var / n);
  float sum2 = 0; int cnt = 0;
  for (int i = 0; i < n; i++) {
    if (abs(arr[i] - mean) <= 2.0f * sd) { sum2 += arr[i]; cnt++; }
  }
  return (cnt > 0) ? (sum2 / cnt) : mean;
}

// ============================================================
// MAIN INFERENCE
// ============================================================
void runInference(float hr_bpm) {

  int   n = min(beat_buf_count, BEAT_BUF_SIZE);
  float hr_mean = 0;
  for (int i = 0; i < n; i++) hr_mean += beat_hr_buf[i];
  hr_mean /= n;

  float hr_std = 0;
  for (int i = 0; i < n; i++) hr_std += sq(beat_hr_buf[i] - hr_mean);
  hr_std = sqrt(hr_std / n);

  int   n_rr     = min(rr_buf_count, RR_BUF_SIZE);
  float rmssd_raw = 0;
  int   pairs    = 0;
  for (int i = 1; i < n_rr; i++) {
    float d    = rr_buf[i] - rr_buf[i-1];
    rmssd_raw += d * d;  pairs++;
  }
  rmssd_raw  = (pairs > 0) ? sqrt(rmssd_raw / pairs) : 20.0f;
  rmssd_smooth = RMSSD_ALPHA * rmssd_raw +
                 (1.0f - RMSSD_ALPHA) * rmssd_smooth;
  float rmssd = constrain(rmssd_smooth, 8.0f, 80.0f);

  int16_t ax_r, ay_r, az_r, gx_r, gy_r, gz_r;
  mpu.getMotion6(&ax_r, &ay_r, &az_r, &gx_r, &gy_r, &gz_r);
  float ax  = ax_r / 16384.0f;
  float ay  = ay_r / 16384.0f;
  float az  = az_r / 16384.0f;
  float mag = sqrt(ax*ax + ay*ay + az*az);
  current_act = detectActivity(mag);

  const char* act_names[] = {"REST", "WALK", "RUN", "INTENSE"};
  current_act_name = act_names[current_act];

  float hr_zscore = profile.getZScore(hr_mean, current_act);
  float ahr       = profile.getActivityHRRatio(hr_mean, current_act);

  static float prev_hr = -1;
  float hr_delta = (prev_hr > 0) ? (hr_bpm - prev_hr) : 0.0f;
  prev_hr = hr_bpm;

  float features[8] = {
    hr_zscore, hr_std, rmssd, (float)current_act,
    mag, 0.1f, hr_delta, ahr
  };
  float norm[8];
  for (int i = 0; i < 8; i++)
    norm[i] = (features[i] - scaler_mean[i]) / scaler_scale[i];
  for (int i = 0; i < 8; i++)
    tfl_input->data.f[i] = norm[i];
  interpreter->Invoke();

  float mse = 0;
  for (int i = 0; i < 8; i++) {
    float d  = norm[i] - tfl_output->data.f[i];
    mse     += d * d;
  }
  mse /= 8.0f;

  current_bpm   = (int)round(hr_mean);
  current_mse   = mse;
  current_rmssd = rmssd;

  String statusStr = "Normal";
  bool   isAnomaly = (mse > personal_threshold);
  AnomalyType atype = ANOMALY_GENERAL;

  if (isAnomaly) {
    atype = classifyAnomaly(hr_zscore, rmssd, hr_delta, ahr);
    switch (atype) {
      case ANOMALY_TACHY:   statusStr = "TACHY";   break;
      case ANOMALY_BRADY:   statusStr = "BRADY";   break;
      case ANOMALY_HRV_LOW: statusStr = "HRV_LOW"; break;
      case ANOMALY_SPIKE:   statusStr = "SPIKE";   break;
      default:              statusStr = "ANOMALY";  break;
    }
  }
  current_status = statusStr;

  Serial.printf("[%s] HR:%d BPM | HRV:%d ms | "
                "MSE:%.4f | Thresh:%.4f | %s\n",
    current_act_name.c_str(),
    current_bpm, (int)current_rmssd,
    mse, personal_threshold, statusStr.c_str());

  logCSV(statusStr);

  Blynk.virtualWrite(V_BPM,    current_bpm);
  Blynk.virtualWrite(V_STATUS, statusStr);
  Blynk.virtualWrite(V_CHART,  current_bpm);

  if (current_bpm >= LOW_BPM && current_bpm <= HIGH_BPM) {
    lowAlertSent = false;  highAlertSent = false;
  } else if (current_bpm < LOW_BPM && current_bpm >= CRITICAL_LOW
             && !lowAlertSent && !alertActive) {
    lowAlertSent = true;
    Blynk.logEvent("bpm_warning", "Low HR: " + String(current_bpm));
    toneLow();
  } else if (current_bpm > HIGH_BPM && current_bpm <= CRITICAL_HIGH
             && !highAlertSent && !alertActive) {
    highAlertSent = true;
    Blynk.logEvent("bpm_warning", "High HR: " + String(current_bpm));
  }

  if (isAnomaly) {
    consecutiveAnomalyCount++;
  } else {
    consecutiveAnomalyCount = 0;
  }

  bool cooldownOK = (millis() - lastAlertTime > ALERT_COOLDOWN);
  if (isAnomaly && !alertActive && cooldownOK) {
    bool critical = (current_bpm < CRITICAL_LOW ||
                     current_bpm > CRITICAL_HIGH);
    triggerAlert(statusStr, critical);
    printRecommendation(atype);
  }

  if (consecutiveAnomalyCount == NOTIFY_AFTER_CONSECUTIVE) {
    Blynk.logEvent("heart_alert",
      "5 consecutive anomalies: " + statusStr +
      " BPM=" + String(current_bpm));
    terminal.println("PERSISTENT ANOMALY: " + statusStr +
                     " x5 | BPM=" + String(current_bpm));
    terminal.flush();
    Serial.printf("PERSISTENT ANOMALY: %s x5 | BPM=%d\n",
      statusStr.c_str(), current_bpm);
  }

  if (!alertActive) {
    oledShowHR();
    digitalWrite(GREEN_LED_PIN, HIGH);
    digitalWrite(RED_LED_PIN,   LOW);
  }
}

// ============================================================
// CALIBRATION
// ============================================================
float measureRestingHR(int seconds) {
  Serial.printf("Measuring resting HR for %d sec...\n", seconds);
  Serial.printf("Need at least %d valid beats.\n", MIN_CALIB_BEATS);
  oledShowCalibrating("Measuring HR...");
  resetBeatDetector();
  hr_bpm_stable     = 75.0f;
  calib_beats_count = 0;

  float readings[300];
  int   count      = 0;
  bool  fingerWasOff = false;
  unsigned long start = millis();

  while (millis() - start < (unsigned long)seconds * 1000) {
    long ir = particleSensor.getIR();

    if (ir < FINGER_THRESHOLD) {
      static unsigned long lm = 0;
      unsigned long now = millis();
      if (now - lm > 2000) {
        Serial.printf("Place finger... IR=%ld (need >%d)\n",
          ir, FINGER_THRESHOLD);
        String msg = "Beats: " + String(count) +
                     "/" + String(MIN_CALIB_BEATS) +
                     "\nIR too low: " + String((int)ir);
        oledShowCalibrating(msg);
        lm = now;
      }
      resetBeatDetector();
      fingerWasOff = true;
      Blynk.run();
      continue;
    }

    if (fingerWasOff) {
      fingerWasOff = false;
      resetBeatDetector();
      unsigned long settle = millis();
      while (millis() - settle < 1500) {
        particleSensor.getIR();
        Blynk.run();
      }
      continue;
    }

    float hr = detectBeat(ir, millis());
    if (hr > 0) {
      if (hr >= CALIB_HR_MIN && hr <= CALIB_HR_MAX) {
        if (count < 300) readings[count++] = hr;
        if (calib_beats_count < 300)
          calib_beats[calib_beats_count++] = hr;
        Serial.printf("  Beat %d: %d BPM  (IR=%ld)\n",
          count, (int)round(hr), ir);
        String msg = "Beats: " + String(count) +
                     "/" + String(MIN_CALIB_BEATS);
        oledShowCalibrating(msg);
      } else {
        Serial.printf("  Beat skipped (out of range): %d BPM\n",
          (int)round(hr));
      }
    }
    Blynk.run();
  }

  if (count < MIN_CALIB_BEATS) {
    Serial.printf(
      "WARNING: Only %d beats (need %d). Using default 72 BPM.\n",
      count, MIN_CALIB_BEATS);
    oledShowCalibrating("Too few beats!\nUsing default 72");
    delay(2000);
    return 72.0f;
  }

  float result = robustMean(readings, count);
  if (result < CALIB_HR_MIN || result > CALIB_HR_MAX) {
    Serial.printf(
      "WARNING: Computed HR %d out of range. Using 72 BPM.\n",
      (int)round(result));
    return 72.0f;
  }

  Serial.printf("Resting HR: %d BPM (%d beats)\n",
    (int)round(result), count);
  return result;
}

void computePersonalThreshold(float rest_hr) {
  Serial.println("Computing personal threshold...");
  oledShowCalibrating("Computing thresh...");

  float rr0 = 60000.0f / rest_hr;
  for (int i = 0; i < RR_BUF_SIZE;   i++) rr_buf[i]     = rr0;
  for (int i = 0; i < BEAT_BUF_SIZE; i++) beat_hr_buf[i] = rest_hr;
  rr_buf_idx   = rr_buf_count   = RR_BUF_SIZE;
  beat_buf_idx = beat_buf_count = BEAT_BUF_SIZE;
  rmssd_smooth = 20.0f;

  float calib_mse[300];
  int   cc = 0;

  for (int b = 0; b < calib_beats_count; b++) {
    float hr = calib_beats[b];
    beat_hr_buf[beat_buf_idx % BEAT_BUF_SIZE] = hr;
    beat_buf_idx++; beat_buf_count++;
    rr_buf[rr_buf_idx % RR_BUF_SIZE] = 60000.0f / hr;
    rr_buf_idx++; rr_buf_count++;

    if (beat_buf_count < 8) continue;

    int   n = min(beat_buf_count, BEAT_BUF_SIZE);
    float hm = 0;
    for (int i = 0; i < n; i++) hm += beat_hr_buf[i]; hm /= n;
    float hs = 0;
    for (int i = 0; i < n; i++) hs += sq(beat_hr_buf[i] - hm);
    hs = sqrt(hs / n);

    int n_rr = min(rr_buf_count, RR_BUF_SIZE);
    float rm_raw = 0; int p = 0;
    for (int i = 1; i < n_rr; i++) {
      float d = rr_buf[i] - rr_buf[i-1]; rm_raw += d*d; p++;
    }
    rm_raw    = (p > 0) ? sqrt(rm_raw / p) : 20.0f;
    rmssd_smooth = RMSSD_ALPHA * rm_raw +
                   (1.0f - RMSSD_ALPHA) * rmssd_smooth;
    float rm  = constrain(rmssd_smooth, 8.0f, 80.0f);
    float hd  = (b > 0) ? (hr - calib_beats[b-1]) : 0.0f;
    float mse = computeInferenceMSE(hm, hs, rm, hd, ACT_REST);
    if (cc < 300) calib_mse[cc++] = mse;
  }

  if (cc < 5) {
    personal_threshold = 0.5f;
    Serial.println("Too few calib points — default 0.5");
    return;
  }

  float mm = 0;
  for (int i = 0; i < cc; i++) mm += calib_mse[i]; mm /= cc;
  float ms = 0;
  for (int i = 0; i < cc; i++) ms += sq(calib_mse[i] - mm);
  ms = sqrt(ms / cc);

  personal_threshold = max(mm + 4.0f * ms, 0.3f);
  Serial.printf("Threshold: mean=%.4f std=%.4f => %.4f\n",
    mm, ms, personal_threshold);

  prefs.begin("threshold", false);
  prefs.putFloat("thresh", personal_threshold);
  prefs.end();
}

void calibrateNewUser() {
  Serial.println("============================");
  Serial.println("NEW USER CALIBRATION (90sec)");
  Serial.println("============================");
  oledShowCalibrating("NEW USER\n90s calibration");
  Blynk.virtualWrite(V_STATUS, "Calibrating...");
  terminal.println("Calibration started — keep still 90s");
  terminal.flush();

  float rest_hr = measureRestingHR(90);

  profile.resting_hr = rest_hr;
  profile.stats[ACT_REST]    = {rest_hr,               8.0f,  true};
  float hrr = 220.0f - 30.0f - rest_hr;
  profile.stats[ACT_WALK]    = {rest_hr + hrr * 0.35f, 10.0f, true};
  profile.stats[ACT_RUN]     = {rest_hr + hrr * 0.65f, 12.0f, true};
  profile.stats[ACT_INTENSE] = {rest_hr + hrr * 0.85f, 15.0f, true};
  profile.user_id    = (int)round(rest_hr);
  profile.calibrated = true;
  profile.save(prefs);
  hr_bpm_stable = rest_hr;

  computePersonalThreshold(rest_hr);

  Serial.printf("Done! Rest HR:%d  Thresh:%.4f\n",
    (int)round(rest_hr), personal_threshold);
  Blynk.virtualWrite(V_STATUS, "Ready");
  terminal.printf("Done! Rest HR:%d | Thresh:%.4f\n",
    (int)round(rest_hr), personal_threshold);
  terminal.flush();
}

// ============================================================
// BUZZER TONES
// ============================================================
void toneStartup() {
  buzzerTone(1000, 200); delay(50);
  buzzerTone(1500, 200); delay(50);
  buzzerOff();
}
void toneLow() {
  buzzerTone(800, 120); delay(40);
  buzzerTone(800, 120); delay(40);
  buzzerOff();
}
void toneHigh() {
  buzzerTone(1600, 150); delay(50);
  buzzerTone(1600, 150); delay(50);
  buzzerOff();
}
void toneCritical() {
  for (int i = 0; i < 4; i++) {
    buzzerTone(2400, 150); delay(80);
  }
  buzzerOff();
}
void toneEmergency() {
  for (int i = 0; i < 8; i++) {
    buzzerTone(3000, 100); delay(50);
    buzzerTone(2000, 100); delay(50);
  }
  buzzerOff();
}
void toneReset() {
  buzzerTone(1200, 80); delay(40);
  buzzerTone(1200, 80); delay(40);
  buzzerOff();
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  delay(500);

  pinMode(RED_LED_PIN,      OUTPUT);
  pinMode(GREEN_LED_PIN,    OUTPUT);
  pinMode(ACK_BUTTON_PIN,   INPUT_PULLUP);
  pinMode(RESET_BUTTON_PIN, INPUT_PULLUP);
  digitalWrite(GREEN_LED_PIN, HIGH);
  digitalWrite(RED_LED_PIN,   LOW);

  ledcSetup(BUZZER_CH, 2000, BUZZER_RES);
  ledcAttachPin(BUZZER_PIN, BUZZER_CH);
  ledcWrite(BUZZER_CH, 0);

  // OLED
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("OLED init failed!"); while (1);
  }
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(20, 18); display.print("HeartRate AI");
  display.setCursor(15, 32); display.print("Initialising...");
  display.display();
  delay(1000);

  // WiFi + Blynk
  Serial.println("Connecting WiFi + Blynk...");
  display.clearDisplay();
  display.setCursor(10, 22);
  display.print("Connecting WiFi...");
  display.display();
  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, password_wifi);
  Serial.println("Blynk connected");
  Blynk.virtualWrite(V_STATUS, "Starting...");
  Blynk.virtualWrite(V_LED, 255);
  terminal.println("HeartRate AI starting...");
  terminal.flush();

  // MAX30102
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found!"); while (1);
  }
  particleSensor.setup(0x1F, 1, 2, 100, 411, 4096);
  particleSensor.setPulseAmplitudeRed(0x1F);
  particleSensor.setPulseAmplitudeIR(0x1F);
  particleSensor.setPulseAmplitudeGreen(0);
  Serial.println("MAX30102 OK");

  // MPU6050
  Wire.beginTransmission(0x68); byte e68 = Wire.endTransmission();
  Wire.beginTransmission(0x69); byte e69 = Wire.endTransmission();
  Wire.beginTransmission(0x68);
  Wire.write(0x6B); Wire.write(0x00);
  Wire.endTransmission(true);
  delay(100);
  if      (e68 == 0) mpu = MPU6050(0x68);
  else if (e69 == 0) mpu = MPU6050(0x69);
  else { Serial.println("MPU6050 not found!"); while (1); }
  mpu.initialize(); delay(100);
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  if (ax == 0 && ay == 0 && az == 0) {
    Serial.println("MPU6050 zero!"); while (1);
  }
  Serial.printf("MPU6050 OK ax=%d ay=%d az=%d\n", ax, ay, az);

  // TFLite
  tf_model = tflite::GetModel(heart_model_data);
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddFullyConnected(); resolver.AddRelu();
  resolver.AddQuantize(); resolver.AddDequantize(); resolver.AddReshape();
  static tflite::MicroErrorReporter err_rep;
  static tflite::MicroInterpreter interp(
    tf_model, resolver, tensor_arena, TENSOR_ARENA_SIZE, &err_rep);
  interpreter = &interp;
  interpreter->AllocateTensors();
  tfl_input  = interpreter->input(0);
  tfl_output = interpreter->output(0);
  Serial.printf("TFLite OK — arena: %d bytes\n",
    interpreter->arena_used_bytes());

  for (int i = 0; i < BEAT_BUF_SIZE; i++) beat_hr_buf[i] = 75.0f;
  for (int i = 0; i < RR_BUF_SIZE;   i++) rr_buf[i]      = 750.0f;
  rmssd_smooth = 20.0f;

  // ── Remove these 2 lines after first successful calibration ──
  Preferences t1; t1.begin("userprofile", false); t1.clear(); t1.end();
  Preferences t2; t2.begin("threshold",   false); t2.clear(); t2.end();
  Serial.println("Profile cleared — will recalibrate");
  // ─────────────────────────────────────────────────────────────

  bool loaded = profile.load(prefs);
  if (!loaded) {
    calibrateNewUser();
  } else {
    Serial.printf("Profile loaded — ID:%d  Rest HR:%d\n",
      profile.user_id, (int)round(profile.resting_hr));
    prefs.begin("threshold", true);
    personal_threshold = prefs.getFloat("thresh", 0.5f);
    prefs.end();
    Serial.printf("Threshold: %.4f\n", personal_threshold);

    float quick_hr = measureRestingHR(15);
    if (!profile.isSameUser(quick_hr)) {
      Serial.println("Different user — recalibrating...");
      calibrateNewUser();
    } else {
      Serial.println("Same user confirmed.");
    }
  }

  hr_bpm_stable = profile.resting_hr;
  float rr0     = 60000.0f / profile.resting_hr;
  for (int i = 0; i < RR_BUF_SIZE;   i++) rr_buf[i]      = rr0;
  for (int i = 0; i < BEAT_BUF_SIZE; i++) beat_hr_buf[i]  = profile.resting_hr;
  rr_buf_idx   = rr_buf_count   = RR_BUF_SIZE;
  beat_buf_idx = beat_buf_count = BEAT_BUF_SIZE;
  rmssd_smooth = 20.0f;

  blynkTimer.setInterval(2000L, blynkSync);
  blynkTimer.setInterval(500L,  checkAckTimeout);

  printCSVHeader();
  toneStartup();

  Blynk.virtualWrite(V_STATUS, "Monitoring");
  Blynk.virtualWrite(V_LED, 255);
  terminal.printf("Monitoring | Rest HR:%d | Thresh:%.4f\n",
    (int)round(profile.resting_hr), personal_threshold);
  terminal.flush();

  Serial.println("=== Monitoring started ===");
  Serial.printf("Personal threshold: %.4f\n", personal_threshold);
  Serial.printf("Finger threshold:   %d\n",  FINGER_THRESHOLD);
  oledShowNoFinger();
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  Blynk.run();
  blynkTimer.run();
  checkAckButton();
  checkResetButton();

  unsigned long now     = millis();
  long          irValue = particleSensor.getIR();

  if (irValue < FINGER_THRESHOLD) {
    static unsigned long lastMsg = 0;
    if (now - lastMsg > 3000) {
      Serial.printf("No finger — IR=%ld (need >%d)\n",
        irValue, FINGER_THRESHOLD);
      lastMsg = now;
      if (!alertActive) oledShowNoFinger();
    }
    resetBeatDetector();
    beat_buf_count          = 0;
    rr_buf_count            = 0;
    rmssd_smooth            = 20.0f;
    consecutiveAnomalyCount = 0;
    return;
  }

  float hr_bpm = detectBeat(irValue, now);

  if (hr_bpm > 0) {
    beat_hr_buf[beat_buf_idx % BEAT_BUF_SIZE] = hr_bpm;
    beat_buf_idx++; beat_buf_count++;

    rr_buf[rr_buf_idx % RR_BUF_SIZE] = 60000.0f / hr_bpm;
    rr_buf_idx++; rr_buf_count++;

    Serial.printf("Beat: %d BPM  (IR=%ld)\n",
      (int)round(hr_bpm), irValue);
    if (beat_buf_count < 8) return;
    runInference(hr_bpm);
  }
}
