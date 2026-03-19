#ifndef USER_PROFILE_H
#define USER_PROFILE_H

#include <Preferences.h>

// Activity codes — must match Python training
#define ACT_REST    0
#define ACT_WALK    1
#define ACT_RUN     2
#define ACT_INTENSE 3

// Expected HR per activity (population average)
// Used for activity_hr_ratio feature
const float EXPECTED_HR[4] = {70.0, 95.0, 140.0, 155.0};

struct ActivityStats {
  float mean;
  float std;
  bool  valid;
};

class UserProfile {
public:
  int         user_id;
  float       resting_hr;
  ActivityStats stats[4];   // one per activity class
  bool        calibrated;

  UserProfile() {
    user_id    = -1;
    resting_hr = 75.0;
    calibrated = false;
    for (int i = 0; i < 4; i++) {
      stats[i] = {75.0, 10.0, false};
    }
  }

  // Save profile to ESP32 NVS flash
  void save(Preferences& prefs) {
    prefs.begin("userprofile", false);
    prefs.putInt("user_id",    user_id);
    prefs.putFloat("rest_hr",  resting_hr);
    prefs.putBool("calib",     calibrated);
    for (int i = 0; i < 4; i++) {
      prefs.putFloat(("m" + String(i)).c_str(), stats[i].mean);
      prefs.putFloat(("s" + String(i)).c_str(), stats[i].std);
      prefs.putBool(("v" + String(i)).c_str(),  stats[i].valid);
    }
    prefs.end();
    Serial.println("Profile saved to flash");
  }

  // Load profile from NVS flash
  bool load(Preferences& prefs) {
    prefs.begin("userprofile", true);
    user_id    = prefs.getInt("user_id",   -1);
    resting_hr = prefs.getFloat("rest_hr", -1.0);
    calibrated = prefs.getBool("calib",    false);
    for (int i = 0; i < 4; i++) {
      stats[i].mean  = prefs.getFloat(("m" + String(i)).c_str(), 75.0);
      stats[i].std   = prefs.getFloat(("s" + String(i)).c_str(), 10.0);
      stats[i].valid = prefs.getBool(("v"  + String(i)).c_str(), false);
    }
    prefs.end();
    return (user_id >= 0 && calibrated && resting_hr > 0);
  }

  // Check if measured resting HR matches stored profile
  // Returns true if same user, false if new user
  bool isSameUser(float measured_resting_hr) {
    if (!calibrated) return false;
    return abs(measured_resting_hr - resting_hr) < 15.0;
  }

  // Get personal z-score — same formula as Python training
  float getZScore(float hr, int act_code) {
    if (!stats[act_code].valid) {
      // Fallback to rest stats if activity not calibrated
      act_code = ACT_REST;
    }
    float denom = max(stats[act_code].std, 1.0f);
    return (hr - stats[act_code].mean) / denom;
  }

  float getActivityHRRatio(float hr, int act_code) {
    float expected = EXPECTED_HR[act_code];
    float denom    = max(stats[act_code].std, 1.0f) + 1.0f;
    return (hr - expected) / denom;
  }
};

#endif