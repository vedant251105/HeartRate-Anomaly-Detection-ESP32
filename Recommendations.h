#ifndef RECOMMENDATIONS_H
#define RECOMMENDATIONS_H

// Anomaly types based on which features triggered
enum AnomalyType {
  ANOMALY_TACHY,       // HR too high for activity
  ANOMALY_BRADY,       // HR too low
  ANOMALY_HRV_LOW,     // HRV dropped — stress/fatigue
  ANOMALY_SPIKE,       // sudden HR jump
  ANOMALY_GENERAL
};

struct Recommendation {
  const char* warning;
  const char* action;
  const char* recovery;
};

// Recommendations shown on Serial / display
// This is what makes your project unique vs just detection
const Recommendation RECOMMENDATIONS[] = {
  // ANOMALY_TACHY
  {
    "WARNING: Heart rate too high for current activity!",
    "Action: Slow down or stop current activity immediately.",
    "Recovery: Sit down, breathe deeply (4 sec in, 6 sec out), hydrate."
  },
  // ANOMALY_BRADY
  {
    "WARNING: Heart rate unusually low!",
    "Action: Stop exercise, avoid sudden position changes.",
    "Recovery: Sit or lie down, breathe normally, monitor for dizziness."
  },
  // ANOMALY_HRV_LOW
  {
    "WARNING: Heart rate variability low — possible fatigue or stress!",
    "Action: Reduce exercise intensity, consider rest day.",
    "Recovery: Rest, hydrate, avoid caffeine, try slow breathing."
  },
  // ANOMALY_SPIKE
  {
    "WARNING: Sudden heart rate spike detected!",
    "Action: Stop activity, sit down immediately.",
    "Recovery: Breathe slowly, wait 5 minutes, check if HR normalises."
  },
  // ANOMALY_GENERAL
  {
    "WARNING: Abnormal heart rate pattern detected!",
    "Action: Reduce activity level.",
    "Recovery: Rest and monitor. Consult doctor if warning persists."
  }
};

// Determine anomaly type from raw feature values
AnomalyType classifyAnomaly(float hr_zscore, float rmssd,
                             float hr_delta, float activity_hr_ratio) {
  if (hr_zscore > 2.5 || activity_hr_ratio > 2.0)  return ANOMALY_TACHY;
  if (hr_zscore < -2.5)                              return ANOMALY_BRADY;
  if (rmssd < 10.0)                                  return ANOMALY_HRV_LOW;
  if (abs(hr_delta) > 20.0)                          return ANOMALY_SPIKE;
  return ANOMALY_GENERAL;
}

void printRecommendation(AnomalyType type) {
  Serial.println("========================================");
  Serial.println(RECOMMENDATIONS[type].warning);
  Serial.println(RECOMMENDATIONS[type].action);
  Serial.println(RECOMMENDATIONS[type].recovery);
  Serial.println("========================================");
}

#endif