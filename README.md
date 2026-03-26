# Week 7: Stress Testing, Observability & Scaling the IRIS Pipeline

## Overview
This assignment validates the deployed IRIS API under high concurrency using wrk, monitors Pod behavior through GCP Cloud Monitoring and Cloud Logging, observes Kubernetes autoscaling, and identifies bottlenecks when scaling is constrained.

## Repository Structure
```
.
├── .github/workflows/main.yml    # CI/CD pipeline with stress testing
├── k8s/
│   ├── deployment.yaml           # IRIS API deployment with resource limits
│   ├── service.yaml              # LoadBalancer service
│   ├── hpa.yaml                  # HPA - max 3 replicas (Task 3)
│   └── hpa_constrained.yaml      # HPA - max 1 replica (Task 5)
├── stress_test.lua               # wrk POST request script
├── main.py                       # FastAPI IRIS app
├── Dockerfile                    # Container definition
└── requirements.txt              # Python dependencies
```

## Tasks

### Task 1: CI/CD with Stress Testing
- Extended GitHub Actions workflow with a dedicated `stress-test` job
- Runs automatically after successful deployment to GKE
- Installs wrk on the runner and executes load tests against the live API

### Task 2: High-Concurrency Traffic with wrk
```bash
wrk -t4 -c1000 -d30s --timeout 10s -s stress_test.lua http://<EXTERNAL_IP>/predict/
```
- 4 threads, 1000 concurrent connections, 30 second duration
- Metrics recorded: requests/sec, average latency, error count

### Task 3: Horizontal Pod Autoscaler (max 3 replicas)
```bash
kubectl apply -f k8s/hpa.yaml
kubectl get hpa
kubectl get pods
```
- minReplicas: 1, maxReplicas: 3
- CPU utilization target: 50%
- HPA scales pods up automatically under load

### Task 4: GCP Cloud Monitoring & Cloud Logging
- Observed CPU and memory usage across Pod replicas in real time
- Filtered logs by Pod name in Logs Explorer
- Identified load distribution across pods

### Task 5: Bottleneck Analysis (max 1 replica vs max 3 replicas)
```bash
# Apply constrained HPA
kubectl apply -f k8s/hpa_constrained.yaml

# Run with 2000 connections
wrk -t4 -c2000 -d30s --timeout 10s -s stress_test.lua http://<EXTERNAL_IP>/predict/
```

| Scenario | Replicas | Connections | Expected Result |
|----------|----------|-------------|-----------------|
| Task 3   | max 3    | 1000        | Lower latency, higher throughput |
| Task 5   | max 1    | 2000        | Higher latency, more errors |

## Key Findings
- With maxReplicas: 3, the HPA distributes load across pods keeping latency low
- With maxReplicas: 1, CPU saturates quickly causing latency spikes and errors
- First bottleneck observed: CPU exhaustion leading to request queuing
