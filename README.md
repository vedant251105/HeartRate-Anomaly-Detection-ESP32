# Heart Rate Anomaly Detection During Physical Activity
### ESP32 + TinyML | Embedded AI | PCCOE Pune | 2026

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![ESP32](https://img.shields.io/badge/Platform-ESP32-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview
A TinyML-based heart rate anomaly detection system deployed on 
ESP32. The system automatically adapts to any user — someone with 
resting HR of 70 BPM and someone with 120 BPM both get accurate 
anomaly detection without retraining the model.

## Key Results
| Metric | Value |
|--------|-------|
| ROC-AUC Score | 0.998 |
| Anomaly Separation Ratio | 44.7x |
| Model Size (INT8 TFLite) | 4.7 KB |
| Total Parameters | 636 |
| Tensor Arena Used | 1,292 bytes |
| Training Subjects | 24 (PAMAP2 + PPG-DaLiA) |
| Training Windows | 3,336 |

## Hardware
| Component | Purpose |
|-----------|---------|
| ESP32 Dev Module | Main processor + TFLite inference |
| MAX30102 | PPG sensor for heart rate detection |
| MPU6050 | IMU for activity classification |

## How It Works
1. **90-second calibration** — device learns your personal 
   resting HR baseline
2. **Personal threshold** — computed from your own calibration 
   data using mean + 3 x std of MSE
3. **Per-beat inference** — every heartbeat checked against 
   your personal threshold
4. **Activity-aware** — 4 activity classes: rest, walk, run, 
   intense
5. **Recommendations** — specific health advice given for 
   each anomaly type

## System Architecture
```
Sensor Data (MAX30102 + MPU6050)
        ↓
Feature Extraction (8 features on ESP32)
        ↓
TFLite Autoencoder Inference (4.7KB model)
        ↓
Reconstruction MSE vs Personal Threshold
        ↓
Anomaly Classification + Health Recommendation
```

## The 8 Features
| # | Feature | Description |
|---|---------|-------------|
| 1 | HR Z-Score | (HR - personal_mean) / personal_std |
| 2 | HR Std | Stability of HR in rolling window |
| 3 | RMSSD | Heart rate variability metric |
| 4 | Activity Code | 0=rest 1=walk 2=run 3=intense |
| 5 | Accel Mean | Mean acceleration magnitude |
| 6 | Accel Std | Movement variability |
| 7 | HR Delta | Rate of change per beat |
| 8 | Activity HR Ratio | HR vs expected for activity |

## Model Architecture
```
Input(8) → Dense(16) → Dense(8) → Dense(4) [bottleneck]
         → Dense(8)  → Dense(16) → Output(8)
Loss: MSE (reconstruction error = anomaly score)
```

## Why Person-Agnostic
The model trains on z-scores not raw HR values. Person A 
with HR=70 at rest and Person B with HR=120 at rest both 
produce z-score=0. The model sees them as identical. Only 
the personal threshold differs — computed from each user's 
own 90-second calibration data.

## Datasets
- **PAMAP2** — 9 subjects, 18 activities, IMU + HR 
  (UCI ML Repository ID: 231)
- **PPG-DaLiA** — 15 subjects, wrist PPG + accelerometer 
  (UCI ML Repository ID: 495)

## Repository Structure
```
├── python/
│   ├── step1_load_data.py          # Load PAMAP2 + PPG-DaLiA
│   ├── step2_feature_extraction.py # Extract 8 features
│   ├── step3_train_model.py        # Train autoencoder
│   ├── step4_convert_tflite.py     # Convert to TFLite INT8
│   └── eda_analysis.py             # Exploratory data analysis
│
├── arduino/
│   ├── HeartRateAI.ino             # Main ESP32 firmware
│   ├── model.h                     # TFLite model as C array
│   ├── UserProfile.h               # Adaptive user profiling
│   └── Recommendations.h           # Health recommendations
│
└── results/
    └── (training plots + EDA visualizations)
```

## Setup

### Python — Training Pipeline
```bash
pip install tensorflow==2.13.0 numpy pandas scipy scikit-learn matplotlib seaborn
python step1_load_data.py
python step2_feature_extraction.py
python step3_train_model.py
python step4_convert_tflite.py
```

### Arduino — ESP32 Deployment
Install these libraries in Arduino IDE:
- TensorFlowLite_ESP32
- SparkFun MAX3010x
- MPU6050 by Electronic Cats
- Preferences (built-in)

Copy `model.h`, `UserProfile.h`, `Recommendations.h` into 
the same folder as `HeartRateAI.ino` and upload.

## Wiring
```
MAX30102  →  ESP32
VIN       →  3.3V
GND       →  GND
SDA       →  GPIO 21
SCL       →  GPIO 22

MPU6050   →  ESP32
VCC       →  3.3V
GND       →  GND
SDA       →  GPIO 21  (shared I2C bus)
SCL       →  GPIO 22  (shared I2C bus)
```

## Authors
**Vedant Gadge** — B.Tech Electronics & Telecommunication, 
PCCOE Pune

**[Aagam Katariya]** — B.Tech Electronics & Telecommunication, 
PCCOE Pune

*Project for Computional Tools for AIML subject | Academic Year 2025-26
