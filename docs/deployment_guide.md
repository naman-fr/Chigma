# Deployment Guide

Chigma is designed for deployment in production environments using Docker and Kubernetes.

## Prerequisites

1. **Docker Engine** and **Docker Compose** installed.
2. **NVIDIA Container Toolkit** installed for GPU support.
3. (Optional) **Kubernetes Cluster** (minikube, EKS, GKE, AKS) with NGINX Ingress Controller.

---

## Local Deployment (Docker Compose)

The easiest way to spin up the entire stack locally is using `docker-compose`.

1. **Build and start services**:
   ```bash
   docker-compose up -d --build
   ```

2. **Verify services**:
   - API Server: `http://localhost:8000/docs`
   - MLflow UI: `http://localhost:5000`
   - Prometheus: `http://localhost:9090`
   - Grafana: `http://localhost:3000` (Default login: `admin` / `chigma2024`)

3. **Stop services**:
   ```bash
   docker-compose down
   ```

---

## Production Deployment (Kubernetes)

We provide standard Kubernetes manifests in `mlops/k8s/`.

1. **Build and push the Docker image** to your registry (e.g., GHCR):
   ```bash
   docker build -t ghcr.io/your-org/chigma:latest .
   docker push ghcr.io/your-org/chigma:latest
   ```

2. **Apply Kubernetes Manifests**:
   ```bash
   kubectl create namespace chigma
   kubectl apply -f mlops/k8s/deployment.yaml
   kubectl apply -f mlops/k8s/service.yaml
   kubectl apply -f mlops/k8s/ingress.yaml
   ```

3. **Verify Deployment**:
   ```bash
   kubectl get pods -n chigma
   kubectl get ingress -n chigma
   ```
   
> **Note**: Ensure your Kubernetes cluster has GPU nodes enabled and the NVIDIA device plugin installed to allow the `nvidia.com/gpu` resource requests to be fulfilled.
