# MLOps Weekly Assignment - 21f3001527

## Week 06 - Continuous Deployment with Docker + GKE

### Overview
Containerized the IRIS inference API with Docker and deployed it to Google Kubernetes Engine (GKE) on GCP, automating the entire build, push, and deploy cycle through GitHub Actions.

### Tasks Completed

#### Task 1 - Pod vs Container
- **Docker Container**: A running instance of a Docker image. Single isolated process with its own filesystem and network.
- **Kubernetes Pod**: Smallest deployable unit in K8s. Wraps one or more containers sharing the same network and storage. K8s manages Pods, not raw containers.

#### Task 2 - Dockerfile
- Built Docker image using `python:3.11-slim`
- Copies `model.joblib` and `main.py` into container
- Serves predictions via `uvicorn` on port 8080

#### Task 3 - GCP Service Account
- Created `github-actions-sa` service account
- Granted `artifactregistry.writer` and `container.developer` roles
- Configured credentials as GitHub Actions secrets

#### Task 4 - Build & Push via GitHub Actions
- GitHub Actions workflow triggers on push to `week_06`
- Builds Docker image and pushes to Google Artifact Registry
- Image: `us-central1-docker.pkg.dev/meta-territory-488805-q1/iris-repo/iris-api`

#### Task 5 - Deploy to GKE
- Created GKE cluster `ml-cluster` in `us-east1-b`
- Deployed using `k8s/deployment.yaml` and `k8s/service.yaml`
- Live API endpoint: `http://34.73.134.197`

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | Health check |
| `/predict/` | POST | Predict IRIS species |
| `/docs` | GET | Swagger UI |

### Test the API
```bash
# Health check
curl http://34.73.134.197/health

# Predict
curl -X POST http://34.73.134.197/predict/ \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

### File Structure
```
├── Dockerfile                        # Container definition
├── main.py                           # FastAPI inference API
├── model.joblib                      # Trained IRIS model
├── requirements.txt                  # Python dependencies
├── k8s/
│   ├── deployment.yaml               # Kubernetes Deployment
│   └── service.yaml                  # Kubernetes LoadBalancer
└── .github/
    └── workflows/
        └── main.yml                    # GitHub Actions CD pipeline
```

### Tech Stack
- **FastAPI** - Inference API framework
- **Docker** - Containerization
- **Google Artifact Registry** - Container image storage
- **Google Kubernetes Engine** - Container orchestration
- **GitHub Actions** - CI/CD automation

---


