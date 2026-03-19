# Heart Rate Anomaly Detection During Physical Activity
### ESP32 + TinyML | Embedded AI Project | 2026

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)]()
[![ESP32](https://img.shields.io/badge/Platform-ESP32-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

## Project Overview
A TinyML-based heart rate anomaly detection system that runs 
entirely on an ESP32 microcontroller. The system automatically 
adapts to any user — someone with resting HR of 70 BPM and 
someone with 120 BPM both get accurate anomaly detection without 
retraining the model.

## Key Results
| Metric | Value |
|--------|-------|
| ROC-AUC | 0.998 |
| Anomaly Separation Ratio | 44.7x |
| Model Size (INT8 TFLite) | 4.7 KB |
| Parameters | 636 |
| Tensor Arena Used | 1,292 bytes |
| Training Subjects | 24 (PAMAP2 + PPG-DaLiA) |
| Training Windows | 3,336 |

## Hardware Required
- ESP32 Dev Module
- MAX30102 Pulse Oximeter Sensor
- MPU6050 6-axis IMU

## How It Works
1. 90-second calibration captures your personal resting HR
2. Model computes your personal anomaly threshold from 
   calibration data
3. Every heartbeat is checked against your threshold
4. Anomaly type classified and health recommendation given

## Repository Structure
- python/ — data loading, feature extraction, training, TFLite 
  conversion
- arduino/ — ESP32 firmware with TFLite inference
- results/ — training plots and EDA visualizations

## Datasets
- PAMAP2 (UCI ML Repository, ID 231)
- PPG-DaLiA (UCI ML Repository, ID 495)

## Setup
### Python (Training)
pip install tensorflow==2.13.0 numpy pandas scipy scikit-learn

### Arduino (Deployment)
Install libraries: TensorFlowLite_ESP32, MAX30105, MPU6050, 
Preferences

## Authors
Vedant Gadge | B.Tech Electronics & Telecommunication | PCCOE Pune
[Aagam katariya] | B.Tech Electronics & Telecommunication | PCCOE Pune

Project submitted for: Computional Tools for AIML 
Academic Year: 2025-26
