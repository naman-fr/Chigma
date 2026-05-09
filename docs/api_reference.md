# API Reference

The Chigma API provides RESTful endpoints to interact with the platform's modules.

## Base URL
`/api/v1`

## Authentication
Add `X-API-Key` to headers for protected routes.

---

## 1. Detection Endpoints

### `POST /detection/predict`
Run FD-YOLO11 defect detection on an uploaded image.
- **Params**: `conf` (Confidence threshold, default 0.25), `iou` (IoU threshold, default 0.45).
- **Body**: `multipart/form-data` with `image` file.
- **Response**: JSON with bounding boxes, classes, and confidence scores.

### `POST /detection/predict/batch`
Run inference on multiple images simultaneously.

### `GET /detection/model/info`
Get currently loaded model configurations and classes.

---

## 2. Vision-Language Copilot (VLM) Endpoints

### `POST /vlm/query`
Ask a natural language question about an image.
- **Params**: `query` (e.g., "What defects are visible?").
- **Body**: `multipart/form-data` with `image` file.

### `POST /vlm/report`
Generate an automated, structured inspection report (JSON format).

---

## 3. Drone Endpoints

### `GET /drone/status`
Retrieve real-time drone telemetry (battery, altitude, mode, etc.).

### `POST /drone/command/natural`
Send a natural language flight command (e.g., "fly to the red building").
Parsed via VLM and converted into actionable waypoints.

### `POST /drone/command/emergency-stop`
Immediately halt and hover the drone.

---

## 4. System Endpoints

### `GET /health`
System health status and GPU availability.

### `GET /metrics` (Internal)
Prometheus metrics scraping endpoint.
