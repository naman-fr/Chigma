# Architecture Overview

Chigma is designed as a modular, industrial-grade platform combining AI, Computer Vision, Generative AI, and autonomous drone capabilities.

## High-Level Architecture

The platform consists of four primary modules running within a unified MLOps pipeline:

1. **Defect Detection (Module 2)**
   - Core engine: FD-YOLO11
   - Custom implementations of `SC-C3k2` (Self-Calibrated Convolutions), `FSPPF` (Feature-fusion SPPF), and `DySample` (Dynamic Upsampling).
   - Designed to improve accuracy on industrial defect datasets like NEU-DET and GC10-DET.

2. **Synthetic Data Generation (Module 2b - SynGen-Vision)**
   - Leverages Stable Diffusion XL and ControlNet.
   - Orchestrated via Python scripts to prompt VLMs for diverse descriptions, generate textures, render them onto 3D surfaces, and auto-label bounding boxes in YOLO format.

3. **Vision-Language Copilot (Module 3)**
   - Powered by Qwen2.5-VL-7B.
   - Provides a natural language interface for visual querying, automated inspection reporting, and severity rating.

4. **Drone Autonomy (Module 1)**
   - **Perception**: Runs YOLO11n for real-time tracking (BoT-SORT).
   - **SLAM**: Uses ORB-SLAM3 for mapping and pose estimation in GPS-denied environments.
   - **Navigation**: Uses Stable-Baselines3 (PPO) for obstacle avoidance.
   - **Control**: Communicates via MAVLink to autopilots (like PX4/ArduPilot).

## MLOps and Infrastructure

- **API Layer**: FastAPI application serving HTTP endpoints for inference, drone control, and VLM querying.
- **Data Versioning**: DVC manages datasets and model weights in cloud storage.
- **Experiment Tracking**: MLflow logs parameters, metrics, and models.
- **Monitoring**: Prometheus scrapes metrics from the FastAPI app, visualized via Grafana.
- **Deployment**: Containerized via Docker and orchestrated with Kubernetes (manifests included).
