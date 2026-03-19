#include <Wire.h>
#include <Preferences.h>
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

#define SDA_PIN          21
#define SCL_PIN          22
#define FINGER_THRESHOLD 30000
#define BEAT_BUF_SIZE    8
#define RR_BUF_SIZE      32

const int TENSOR_ARENA_SIZE = 10 * 1024;
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

const tflite::Model*      tf_model    = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor*             input       = nullptr;
TfLiteTensor*             output      = nullptr;

MAX30105    particleSensor;
MPU6050     mpu;
UserProfile profile;
Preferences prefs;

// Beat rolling buffer
float beat_hr_buf[BEAT_BUF_SIZE];
int   beat_buf_idx   = 0;
int   beat_buf_count = 0;

// RR interval rolling buffer
float rr_buf[RR_BUF_SIZE];
int   rr_buf_idx   = 0;
int   rr_buf_count = 0;

// Calibration beats storage
float calib_beats[200];
int   calib_beats_count = 0;

// Beat detector state
float beat_dc        = 0;
float beat_peak      = 0;
float beat_valley    = 999999;
bool  beat_rising    = false;
unsigned long beat_lastTime = 0;
float hr_bpm_stable  = 75.0;
int   current_act    = ACT_REST;

// Personal threshold — computed from calibration
float personal_threshold = 0.5f;

// Forward declarations
void  calibrateNewUser();
float measureRestingHR(int seconds);
int   detectActivity(float accel_mag);
void  runInference(float hr_bpm);
float detectBeat(long irValue, unsigned long now);
void  resetBeatDetector();
float computeInferenceMSE(float hr_mean, float hr_std,
                           float rmssd, float hr_delta,
                           int act_code);

// ─────────────────────────────────────────────────────────────
void resetBeatDetector() {
  beat_dc       = 0;
  beat_peak     = 0;
  beat_valley   = 999999;
  beat_rising   = false;
  beat_lastTime = 0;
}

// ─────────────────────────────────────────────────────────────
float detectBeat(long irValue, unsigned long now) {
  float ir = (float)irValue;

  beat_dc = beat_dc * 0.97f + ir * 0.03f;
  float ac = ir - beat_dc;

  if (ac > beat_peak)   beat_peak   = ac;
  if (ac < beat_valley) beat_valley = ac;

  float range = beat_peak - beat_valley;

  if (range < 50.0f) {
    beat_peak   *= 0.99f;
    beat_valley *= 0.99f;
    return 0;
  }

  float threshold = beat_valley + range * 0.6f;
  beat_peak   *= 0.998f;
  beat_valley *= 0.998f;

  if ((now - beat_lastTime) < 600UL) {
    if (ac > threshold) beat_rising = true;
    return 0;
  }

  if (ac > threshold && !beat_rising) beat_rising = true;

  float hr_detected = 0;
  if (ac < threshold && beat_rising) {
    beat_rising = false;
    long delta  = (long)(now - beat_lastTime);

    if (delta > 600 && delta < 2000 && beat_lastTime > 0) {
      float hr = 60000.0f / (float)delta;
      float diff = abs(hr - hr_bpm_stable);
      if (diff < 20.0f || hr_bpm_stable < 40.0f) {
        hr_detected   = hr;
        hr_bpm_stable = hr_bpm_stable * 0.8f + hr * 0.2f;
      }
    }
    beat_lastTime = now;
  }
  return hr_detected;
}

// ─────────────────────────────────────────────────────────────
float computeInferenceMSE(float hr_mean, float hr_std,
                           float rmssd, float hr_delta,
                           int act_code) {
  float hr_zscore         = profile.getZScore(hr_mean, act_code);
  float activity_hr_ratio = profile.getActivityHRRatio(hr_mean,
                                                        act_code);

  float features[8] = {
    hr_zscore,
    hr_std,
    rmssd,
    (float)act_code,
    1.0f,   // a_mean — rest default
    0.1f,   // a_std  — rest default
    hr_delta,
    activity_hr_ratio
  };

  float normalized[8];
  for (int i = 0; i < 8; i++)
    normalized[i] = (features[i] - scaler_mean[i]) / scaler_scale[i];

  for (int i = 0; i < 8; i++)
    input->data.f[i] = normalized[i];
  interpreter->Invoke();

  float mse = 0;
  for (int i = 0; i < 8; i++) {
    float diff = normalized[i] - output->data.f[i];
    mse += diff * diff;
  }
  return mse / 8.0f;
}

// ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  delay(1000);

  // ── TEMPORARY: clear saved profile ───────────────────────
  // Remove these 5 lines after first successful calibration
  Preferences tempPrefs;
  tempPrefs.begin("userprofile", false);
  tempPrefs.clear();
  tempPrefs.end();
  Preferences tempThresh;
  tempThresh.begin("threshold", false);
  tempThresh.clear();
  tempThresh.end();
  Serial.println("Profile cleared");
  // ── END TEMPORARY ─────────────────────────────────────────

  Serial.println("Heart Rate AI - ESP32");
  Serial.println("Initialising sensors...");

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found!");
    while (1);
  }
  particleSensor.setup(0x1F, 1, 2, 100, 411, 4096);
  particleSensor.setPulseAmplitudeRed(0x1F);
  particleSensor.setPulseAmplitudeIR(0x1F);
  particleSensor.setPulseAmplitudeGreen(0);
  Serial.println("MAX30102 OK");

  Wire.beginTransmission(0x68);
  byte error68 = Wire.endTransmission();
  Wire.beginTransmission(0x69);
  byte error69 = Wire.endTransmission();
  Serial.printf("MPU6050 at 0x68: %s\n",
                error68 == 0 ? "FOUND" : "not found");
  Serial.printf("MPU6050 at 0x69: %s\n",
                error69 == 0 ? "FOUND" : "not found");

  Wire.beginTransmission(0x68);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);
  delay(100);

  if (error68 == 0)      mpu = MPU6050(0x68);
  else if (error69 == 0) mpu = MPU6050(0x69);
  else { Serial.println("MPU6050 not found!"); while (1); }

  mpu.initialize();
  delay(100);

  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  if (ax == 0 && ay == 0 && az == 0) {
    Serial.println("MPU6050 zero data!");
    while (1);
  }
  Serial.printf("MPU6050 OK — ax=%d ay=%d az=%d\n", ax, ay, az);

  tf_model = tflite::GetModel(heart_model_data);
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddReshape();

  static tflite::MicroErrorReporter micro_error_reporter;
  static tflite::MicroInterpreter static_interpreter(
      tf_model, resolver, tensor_arena, TENSOR_ARENA_SIZE,
      &micro_error_reporter);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("TFLite model loaded OK");
  Serial.printf("Tensor arena used: %d bytes\n",
                interpreter->arena_used_bytes());

  // Init buffers
  for (int i = 0; i < BEAT_BUF_SIZE; i++)
    beat_hr_buf[i] = 75.0f;
  for (int i = 0; i < RR_BUF_SIZE; i++)
    rr_buf[i] = 750.0f;

  // Load user profile
  bool loaded = profile.load(prefs);
  if (!loaded) {
    Serial.println("No profile — starting calibration...");
    calibrateNewUser();
  } else {
    Serial.printf("Profile loaded — User ID: %d\n", profile.user_id);
    Serial.printf("Resting HR: %.1f BPM\n", profile.resting_hr);

    // Load personal threshold
    prefs.begin("threshold", true);
    personal_threshold = prefs.getFloat("thresh", 0.5f);
    prefs.end();
    Serial.printf("Personal threshold: %.4f\n", personal_threshold);

    float quick_hr = measureRestingHR(15);
    if (!profile.isSameUser(quick_hr)) {
      Serial.println("Different user — recalibrating...");
      calibrateNewUser();
    } else {
      Serial.println("Same user confirmed.");
    }
  }

  // Seed buffers from calibrated profile
  hr_bpm_stable = profile.resting_hr;
  float rest_rr = 60000.0f / profile.resting_hr;
  for (int i = 0; i < RR_BUF_SIZE; i++)
    rr_buf[i] = rest_rr;
  rr_buf_idx   = RR_BUF_SIZE;
  rr_buf_count = RR_BUF_SIZE;
  for (int i = 0; i < BEAT_BUF_SIZE; i++)
    beat_hr_buf[i] = profile.resting_hr;
  beat_buf_idx   = BEAT_BUF_SIZE;
  beat_buf_count = BEAT_BUF_SIZE;

  Serial.println("\nMonitoring every heartbeat...");
  Serial.printf("Personal threshold: %.4f\n", personal_threshold);
  Serial.println("Place finger on MAX30102.");
}

// ─────────────────────────────────────────────────────────────
void loop() {
  unsigned long now = millis();
  long irValue      = particleSensor.getIR();

  if (irValue < FINGER_THRESHOLD) {
    static unsigned long lastMsg = 0;
    if (now - lastMsg > 3000) {
      Serial.printf("No finger — IR=%ld\n", irValue);
      lastMsg = now;
    }
    resetBeatDetector();
    beat_buf_count = 0;
    rr_buf_count   = 0;
    return;
  }

  float hr_bpm = detectBeat(irValue, now);

  if (hr_bpm > 0) {
    beat_hr_buf[beat_buf_idx % BEAT_BUF_SIZE] = hr_bpm;
    beat_buf_idx++;
    beat_buf_count++;

    float rr_ms = 60000.0f / hr_bpm;
    rr_buf[rr_buf_idx % RR_BUF_SIZE] = rr_ms;
    rr_buf_idx++;
    rr_buf_count++;

    Serial.printf("Beat: %.1f BPM\n", hr_bpm);

    if (beat_buf_count < 8) return;

    runInference(hr_bpm);
  }
}

// ─────────────────────────────────────────────────────────────
int detectActivity(float accel_mag) {
  float dev = abs(accel_mag - 1.0f);
  if (dev < 0.20f) return ACT_REST;
  if (dev < 0.60f) return ACT_WALK;
  if (dev < 1.20f) return ACT_RUN;
  return ACT_INTENSE;
}

// ─────────────────────────────────────────────────────────────
void runInference(float hr_bpm) {
  int n = min(beat_buf_count, BEAT_BUF_SIZE);
  float hr_mean = 0;
  for (int i = 0; i < n; i++)
    hr_mean += beat_hr_buf[i];
  hr_mean /= n;

  float hr_std = 0;
  for (int i = 0; i < n; i++)
    hr_std += sq(beat_hr_buf[i] - hr_mean);
  hr_std = sqrt(hr_std / n);

  int n_rr = min(rr_buf_count, RR_BUF_SIZE);
  float rmssd = 0; int pairs = 0;
  for (int i = 1; i < n_rr; i++) {
    float diff = rr_buf[i] - rr_buf[i-1];
    rmssd += diff * diff; pairs++;
  }
  rmssd = (pairs > 0) ? sqrt(rmssd / pairs) : 30.0f;
  rmssd = min(rmssd, 150.0f);
  rmssd = max(rmssd, 5.0f);

  int16_t ax_r, ay_r, az_r, gx_r, gy_r, gz_r;
  mpu.getMotion6(&ax_r, &ay_r, &az_r, &gx_r, &gy_r, &gz_r);
  float ax  = ax_r / 16384.0f;
  float ay  = ay_r / 16384.0f;
  float az  = az_r / 16384.0f;
  float mag = sqrt(ax*ax + ay*ay + az*az);
  current_act = detectActivity(mag);

  float a_mean = mag;
  float a_std  = 0.1f;

  static float prev_hr = -1;
  float hr_delta = (prev_hr > 0) ? (hr_bpm - prev_hr) : 0.0f;
  prev_hr = hr_bpm;

  float hr_zscore         = profile.getZScore(hr_mean, current_act);
  float activity_hr_ratio = profile.getActivityHRRatio(hr_mean,
                                                        current_act);

  float features[8] = {
    hr_zscore, hr_std, rmssd, (float)current_act,
    a_mean, a_std, hr_delta, activity_hr_ratio
  };

  float normalized[8];
  for (int i = 0; i < 8; i++)
    normalized[i] = (features[i] - scaler_mean[i]) / scaler_scale[i];

  for (int i = 0; i < 8; i++)
    input->data.f[i] = normalized[i];
  interpreter->Invoke();

  float mse = 0;
  for (int i = 0; i < 8; i++) {
    float diff = normalized[i] - output->data.f[i];
    mse += diff * diff;
  }
  mse /= 8.0f;

  const char* act_names[] = {"REST","WALK","RUN","INTENSE"};
  Serial.printf("[%s] HR: %.1f BPM | HRV: %.1f ms | "
                "MSE: %.4f | Thresh: %.4f\n",
                act_names[current_act], hr_mean,
                rmssd, mse, personal_threshold);

  if (mse > personal_threshold) {
    AnomalyType atype = classifyAnomaly(hr_zscore, rmssd,
                                        hr_delta,
                                        activity_hr_ratio);
    printRecommendation(atype);
  }
}

// ─────────────────────────────────────────────────────────────
float measureRestingHR(int seconds) {
  Serial.printf("Measuring resting HR for %d sec...\n", seconds);
  Serial.println("Keep finger firmly on sensor...");

  resetBeatDetector();
  hr_bpm_stable    = 75.0f;
  calib_beats_count = 0;

  float readings[200];
  int   count = 0;
  unsigned long start = millis();

  while (millis() - start < (unsigned long)seconds * 1000) {
    long irValue = particleSensor.getIR();

    if (irValue < FINGER_THRESHOLD) {
      static unsigned long lastMsg = 0;
      unsigned long now = millis();
      if (now - lastMsg > 2000) {
        Serial.println("Place finger on sensor...");
        lastMsg = now;
      }
      resetBeatDetector();
      continue;
    }

    float hr = detectBeat(irValue, millis());
    if (hr > 0 && count < 200) {
      readings[count++] = hr;
      // Store for threshold computation
      if (calib_beats_count < 200)
        calib_beats[calib_beats_count++] = hr;
      Serial.printf("  Beat %d: %.1f BPM\n", count, hr);
    }
  }

  if (count < 3) {
    Serial.println("Too few beats — using default 72 BPM");
    return 72.0f;
  }

  // Sort
  for (int i = 0; i < count - 1; i++)
    for (int j = i + 1; j < count; j++)
      if (readings[j] < readings[i]) {
        float t     = readings[i];
        readings[i] = readings[j];
        readings[j] = t;
      }

  // Middle 50%
  int   s = count / 4, e = count * 3 / 4;
  float sum = 0; int used = 0;
  for (int i = s; i < e; i++) { sum += readings[i]; used++; }

  float result = sum / used;
  Serial.printf("Resting HR: %.1f BPM (from %d beats)\n",
                result, used);
  return result;
}

// ─────────────────────────────────────────────────────────────
void computePersonalThreshold(float rest_hr) {
  Serial.println("Computing personal threshold from calibration...");

  // Seed buffers with resting HR
  float rest_rr = 60000.0f / rest_hr;
  for (int i = 0; i < RR_BUF_SIZE; i++)
    rr_buf[i] = rest_rr;
  rr_buf_idx   = RR_BUF_SIZE;
  rr_buf_count = RR_BUF_SIZE;
  for (int i = 0; i < BEAT_BUF_SIZE; i++)
    beat_hr_buf[i] = rest_hr;
  beat_buf_idx   = BEAT_BUF_SIZE;
  beat_buf_count = BEAT_BUF_SIZE;

  float calib_mse[200];
  int   calib_count = 0;

  for (int b = 0; b < calib_beats_count; b++) {
    float hr = calib_beats[b];

    // Update buffers
    beat_hr_buf[beat_buf_idx % BEAT_BUF_SIZE] = hr;
    beat_buf_idx++;
    beat_buf_count++;
    float rr = 60000.0f / hr;
    rr_buf[rr_buf_idx % RR_BUF_SIZE] = rr;
    rr_buf_idx++;
    rr_buf_count++;

    if (beat_buf_count < 8) continue;

    // Compute features
    int n = min(beat_buf_count, BEAT_BUF_SIZE);
    float hr_mean = 0;
    for (int i = 0; i < n; i++) hr_mean += beat_hr_buf[i];
    hr_mean /= n;

    float hr_std = 0;
    for (int i = 0; i < n; i++)
      hr_std += sq(beat_hr_buf[i] - hr_mean);
    hr_std = sqrt(hr_std / n);

    int n_rr = min(rr_buf_count, RR_BUF_SIZE);
    float rmssd = 0; int pairs = 0;
    for (int i = 1; i < n_rr; i++) {
      float diff = rr_buf[i] - rr_buf[i-1];
      rmssd += diff * diff; pairs++;
    }
    rmssd = (pairs > 0) ? sqrt(rmssd / pairs) : 30.0f;
    rmssd = min(rmssd, 150.0f);
    rmssd = max(rmssd, 5.0f);

    float hr_delta = (b > 0) ? (hr - calib_beats[b-1]) : 0.0f;

    float mse = computeInferenceMSE(hr_mean, hr_std,
                                     rmssd, hr_delta,
                                     ACT_REST);
    if (calib_count < 200)
      calib_mse[calib_count++] = mse;
  }

  if (calib_count < 5) {
    Serial.println("Too few calibration points — using default 0.5");
    personal_threshold = 0.5f;
    return;
  }

  // Mean and std of calibration MSE
  float mse_mean = 0;
  for (int i = 0; i < calib_count; i++)
    mse_mean += calib_mse[i];
  mse_mean /= calib_count;

  float mse_std = 0;
  for (int i = 0; i < calib_count; i++)
    mse_std += sq(calib_mse[i] - mse_mean);
  mse_std = sqrt(mse_std / calib_count);

  // Threshold = mean + 3 * std
  // Covers 99.7% of this person's normal variation
  personal_threshold = mse_mean + 3.0f * mse_std;
  personal_threshold = max(personal_threshold, 0.2f);  // minimum floor

  Serial.printf("Personal threshold:\n");
  Serial.printf("  Calib MSE mean : %.4f\n", mse_mean);
  Serial.printf("  Calib MSE std  : %.4f\n", mse_std);
  Serial.printf("  Threshold set  : %.4f\n", personal_threshold);

  // Save to NVS flash
  prefs.begin("threshold", false);
  prefs.putFloat("thresh", personal_threshold);
  prefs.end();
}

// ─────────────────────────────────────────────────────────────
void calibrateNewUser() {
  Serial.println("========================================");
  Serial.println("NEW USER CALIBRATION");
  Serial.println("Sit still with finger on sensor");
  Serial.println("for 90 seconds...");
  Serial.println("========================================");

  float rest_hr = measureRestingHR(90);

  profile.resting_hr            = rest_hr;
  profile.stats[ACT_REST].mean  = rest_hr;
  profile.stats[ACT_REST].std   = 8.0f;
  profile.stats[ACT_REST].valid = true;

  float hrr = 220.0f - 30.0f - rest_hr;
  profile.stats[ACT_WALK].mean     = rest_hr + hrr * 0.35f;
  profile.stats[ACT_WALK].std      = 10.0f;
  profile.stats[ACT_WALK].valid    = true;
  profile.stats[ACT_RUN].mean      = rest_hr + hrr * 0.65f;
  profile.stats[ACT_RUN].std       = 12.0f;
  profile.stats[ACT_RUN].valid     = true;
  profile.stats[ACT_INTENSE].mean  = rest_hr + hrr * 0.85f;
  profile.stats[ACT_INTENSE].std   = 15.0f;
  profile.stats[ACT_INTENSE].valid = true;

  profile.user_id    = (int)rest_hr;
  profile.calibrated = true;
  profile.save(prefs);

  hr_bpm_stable = rest_hr;

  // Compute personal threshold from calibration beats
  computePersonalThreshold(rest_hr);

  Serial.println("Calibration complete!");
  Serial.printf("Rest HR    : %.1f BPM\n", rest_hr);
  Serial.printf("Walk HR    : %.1f BPM\n",
                profile.stats[ACT_WALK].mean);
  Serial.printf("Run HR     : %.1f BPM\n",
                profile.stats[ACT_RUN].mean);
  Serial.printf("Intense HR : %.1f BPM\n",
                profile.stats[ACT_INTENSE].mean);
  Serial.printf("Threshold  : %.4f\n", personal_threshold);
}