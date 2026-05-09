# 🏭 Chigma — GenAI-Enhanced Industrial Vision & Drone Autonomy

> Production-grade platform combining AI-powered drone autonomy, industrial defect detection, GenAI synthetic data, vision-language copilot, and full MLOps infrastructure.

[![CI](https://github.com/chigma/chigma/actions/workflows/ci.yml/badge.svg)](https://github.com/chigma/chigma/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CHIGMA PLATFORM                              │
├────────────────┬────────────────┬──────────────┬────────────────────┤
│  Module 1      │  Module 2      │  Module 3    │  Module 4          │
│  Drone         │  Defect        │  VLM         │  MLOps             │
│  Autonomy      │  Detection     │  Copilot     │  Pipeline          │
│                │                │              │                    │
│  • YOLO11 +    │  • FD-YOLO11   │  • Qwen-VL   │  • MLflow          │
│    BoT-SORT    │  • SC-C3k2     │  • NL Queries│  • DVC             │
│  • ORB-SLAM3   │  • FSPPF       │  • Auto      │  • Docker/K8s      │
│  • PPO/SAC RL  │  • DySample    │    Reports   │  • Prometheus      │
│  • MAVLink     │  • SynGen-     │              │  • Grafana         │
│  • AirSim      │    Vision      │              │  • GitHub Actions  │
└────────────────┴────────────────┴──────────────┴────────────────────┘
```

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/chigma/chigma.git
cd chigma

# Install
pip install -e ".[dev]"

# Run API server
python -m src.api.main

# Run with Docker
docker-compose up -d
```

## 📋 Modules

### Module 1: Drone Autonomy
- **YOLO11n** real-time perception with BoT-SORT tracking
- **Visual SLAM** (ORB-SLAM3) for GPS-denied navigation
- **PPO Reinforcement Learning** agent for obstacle avoidance
- **MAVLink** flight controller (Antigravity A1 / PX4)
- **VLM flight commands**: "Fly to the red building at 30 meters"

### Module 2: Industrial Defect Detection (FD-YOLO11)
- **SC-C3k2**: Self-Calibrated Convolution for multi-scale features
- **FSPPF**: Feature-fusion Spatial Pyramid Pooling Fast
- **DySample**: Dynamic content-aware upsampling
- **+4.6% mAP** improvement over baseline YOLO11s
- **SynGen-Vision**: Synthetic defect data via Stable Diffusion XL

### Module 3: VLM Copilot
- Natural language queries: *"Show me all cracked products"*
- Automated inspection report generation (HTML/PDF)
- Quality assessment with severity rating
- Root cause analysis prompts

### Module 4: MLOps Pipeline
- **DVC** data/model versioning
- **MLflow** experiment tracking + model registry
- **Prometheus + Grafana** monitoring
- **Docker + Kubernetes** deployment
- **GitHub Actions** CI/CD

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | System health check |
| `/api/v1/detection/predict` | POST | Run defect detection on image |
| `/api/v1/detection/predict/batch` | POST | Batch inference |
| `/api/v1/vlm/query` | POST | Natural language image query |
| `/api/v1/vlm/report` | POST | Generate inspection report |
| `/api/v1/drone/status` | GET | Drone telemetry |
| `/api/v1/drone/command/natural` | POST | NL flight command |
| `/metrics` | GET | Prometheus metrics |

## 🧪 Testing

```bash
pytest tests/ -v --cov=src
ruff check src/ tests/
mypy src/
```

## 📚 Research Papers

| Paper | Contribution |
|-------|-------------|
| FD-YOLO11 (IEEE Access, 2025) | SC-C3k2 + FSPPF + DySample |
| SynGen-Vision (2025) | Synthetic defect data pipeline |
| Survey of LVLMs (2025) | VLM architectures & benchmarks |

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
