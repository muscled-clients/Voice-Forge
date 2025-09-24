# VoiceForge STT Kubernetes Deployment

This directory contains comprehensive Kubernetes manifests and deployment tools for the VoiceForge Speech-to-Text service.

## Overview

The deployment consists of the following components:

- **API Servers**: FastAPI application with GPU support for real-time transcription
- **Background Workers**: Celery workers for batch processing and async tasks
- **PostgreSQL**: Primary database with TimescaleDB extension
- **Redis**: Caching and message broker
- **Monitoring**: Prometheus metrics, Flower for Celery monitoring
- **Ingress**: NGINX-based load balancing with SSL termination

## Architecture

```
Internet → Ingress (NGINX) → API Pods (3x) → Workers (4x)
                                ↓
                           PostgreSQL ← Redis
                                ↓
                           Persistent Storage
```

## Prerequisites

### Required Tools

- `kubectl` >= 1.24
- `kustomize` >= 4.5.0 (optional, for advanced deployments)
- `helm` >= 3.8.0 (optional, for monitoring stack)

### Cluster Requirements

- Kubernetes cluster >= 1.24
- GPU nodes (recommended: NVIDIA T4, V100, or A10)
- Minimum resources:
  - 32 vCPU
  - 128 GB RAM
  - 200 GB fast SSD storage
  - 100 GB shared storage (NFS/EFS)

### Storage Classes

The deployment expects two storage classes:

```yaml
# Fast SSD for databases
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: ebs.csi.aws.com  # Adjust for your cloud provider
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
volumeBindingMode: WaitForFirstConsumer

# Shared storage for model files
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: shared-storage
provisioner: efs.csi.aws.com  # Adjust for your cloud provider
volumeBindingMode: WaitForFirstConsumer
```

## Quick Start

1. **Clone and navigate to the k8s directory**:
   ```bash
   cd k8s/
   ```

2. **Update secrets** (IMPORTANT):
   ```bash
   # Edit secrets.yaml with actual production values
   vi secrets.yaml
   
   # Generate new passwords and keys
   openssl rand -base64 32  # For JWT_SECRET_KEY
   openssl rand -base64 32  # For DATABASE_PASSWORD
   ```

3. **Deploy the application**:
   ```bash
   chmod +x deploy.sh
   ./deploy.sh deploy production voiceforge-stt
   ```

4. **Verify deployment**:
   ```bash
   ./deploy.sh verify production voiceforge-stt
   ```

## Configuration

### Secrets Management

**⚠️ SECURITY WARNING**: The included `secrets.yaml` contains placeholder values. Update all secrets before production deployment.

Required secrets:
- Database passwords
- JWT signing keys
- Redis authentication
- External API keys (OpenAI, HuggingFace)
- TLS certificates

For production, consider using:
- [Sealed Secrets](https://sealed-secrets.netlify.app/)
- [External Secrets Operator](https://external-secrets.io/)
- [Kubernetes Secrets Store CSI Driver](https://secrets-store-csi-driver.sigs.k8s.io/)

### Environment Configuration

The deployment supports multiple environments through ConfigMaps:

- **Development**: Lower resource limits, debug enabled
- **Staging**: Production-like settings with test data
- **Production**: Full resource allocation, security hardened

### Resource Scaling

#### Horizontal Pod Autoscaler (HPA)

The API and Worker deployments include HPA configurations:

```yaml
# API scaling: 3-10 pods based on CPU/memory/queue length
minReplicas: 3
maxReplicas: 10

# Worker scaling: 2-8 pods based on task queue depth
minReplicas: 2
maxReplicas: 8
```

#### Vertical Pod Autoscaler (VPA)

For VPA support, install the [VPA controller](https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler) and add VPA resources.

## Deployment Options

### Standard Deployment

```bash
# Deploy to production namespace
./deploy.sh deploy production voiceforge-stt

# Deploy to staging
./deploy.sh deploy staging voiceforge-staging
```

### Using Kustomize

For advanced customization:

```bash
# Preview changes
kubectl kustomize . | head -50

# Deploy with kustomize
kubectl apply -k .
```

### Helm Chart (Alternative)

A Helm chart is also available:

```bash
# Add the VoiceForge Helm repository
helm repo add voiceforge https://charts.voiceforge.ai

# Install with custom values
helm install voiceforge-stt voiceforge/voiceforge-stt \
  --namespace voiceforge-stt \
  --create-namespace \
  --values production-values.yaml
```

## Monitoring and Observability

### Metrics Collection

The deployment includes comprehensive monitoring:

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization dashboards
- **Flower**: Celery task monitoring
- **Jaeger**: Distributed tracing

### Health Checks

Multiple health check endpoints:

- `/health`: Basic application health
- `/health/ready`: Readiness probe
- `/health/live`: Liveness probe
- `/metrics`: Prometheus metrics

### Alerting Rules

Pre-configured alerts for:

- High latency (>500ms p95)
- High error rate (>5%)
- Resource exhaustion
- Service unavailability
- Queue backlog

## Security

### Network Policies

The deployment includes network policies to:

- Restrict pod-to-pod communication
- Allow only necessary ingress traffic
- Isolate database access

### Pod Security Standards

All pods run with restricted security contexts:

- Non-root user
- Read-only root filesystem
- Dropped capabilities
- No privilege escalation

### TLS/SSL

- Automatic certificate management with cert-manager
- TLS termination at ingress
- Internal encryption for sensitive communications

## Backup and Recovery

### Database Backup

Automated backup strategy:

```bash
# Create backup CronJob
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: voiceforge-stt
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15-alpine
            command:
            - sh
            - -c
            - |
              pg_dump -h voiceforge-postgres -U postgres voiceforge | \
              gzip > /backup/voiceforge-$(date +%Y%m%d-%H%M%S).sql.gz
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
EOF
```

### Disaster Recovery

1. **Database**: Point-in-time recovery with WAL-E or pgBackRest
2. **Models**: Shared storage with cross-region replication
3. **Configuration**: GitOps with ArgoCD or Flux

## Troubleshooting

### Common Issues

#### Pods Stuck in Pending State

```bash
# Check resource availability
kubectl describe nodes
kubectl top nodes

# Check scheduling constraints
kubectl describe pod <pod-name> -n voiceforge-stt
```

#### GPU Allocation Issues

```bash
# Verify GPU plugin
kubectl get nodes -o yaml | grep nvidia.com/gpu

# Check GPU resource requests
kubectl describe pod <pod-name> -n voiceforge-stt
```

#### Database Connection Issues

```bash
# Check PostgreSQL logs
kubectl logs -n voiceforge-stt voiceforge-postgres-0

# Test database connectivity
kubectl exec -n voiceforge-stt -it voiceforge-api-0 -- \
  pg_isready -h voiceforge-postgres -p 5432
```

### Debugging Commands

```bash
# Get all resources in namespace
kubectl get all -n voiceforge-stt

# Check pod logs
kubectl logs -n voiceforge-stt -l app.kubernetes.io/component=api --tail=100

# Execute shell in pod
kubectl exec -n voiceforge-stt -it voiceforge-api-0 -- /bin/bash

# Port forward for local testing
kubectl port-forward -n voiceforge-stt svc/voiceforge-api 8000:8000
```

### Performance Tuning

#### Database Optimization

```sql
-- Check connection stats
SELECT * FROM pg_stat_activity;

-- Analyze query performance
SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC;

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats WHERE schemaname = 'public';
```

#### Redis Optimization

```bash
# Check Redis performance
kubectl exec -n voiceforge-stt -it voiceforge-redis-0 -- redis-cli info stats

# Monitor slow queries
kubectl exec -n voiceforge-stt -it voiceforge-redis-0 -- redis-cli slowlog get 10
```

## Upgrades

### Rolling Updates

The deployment supports zero-downtime rolling updates:

```bash
# Update image version
kubectl set image statefulset/voiceforge-api -n voiceforge-stt \
  api=voiceforge/stt-api:v1.1.0

# Check rollout status
kubectl rollout status statefulset/voiceforge-api -n voiceforge-stt
```

### Database Migrations

```bash
# Run migrations
kubectl exec -n voiceforge-stt -it voiceforge-api-0 -- \
  alembic upgrade head
```

## Cost Optimization

### Resource Right-Sizing

- Use VPA recommendations
- Monitor actual resource usage
- Adjust requests/limits accordingly

### Spot Instances

Configure node affinity for cost-effective compute:

```yaml
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 50
      preference:
        matchExpressions:
        - key: karpenter.sh/capacity-type
          operator: In
          values:
          - spot
```

## Support

### Documentation

- [API Documentation](../docs/api.md)
- [Architecture Overview](../docs/architecture.md)
- [Development Guide](../docs/development.md)

### Monitoring Dashboards

- Grafana: `https://monitoring.voiceforge.internal/grafana`
- Flower: `https://monitoring.voiceforge.internal/flower`
- Prometheus: `https://monitoring.voiceforge.internal/prometheus`

### Getting Help

For issues and questions:

1. Check the troubleshooting section above
2. Review pod logs and events
3. Search existing GitHub issues
4. Create a new issue with detailed information

---

**Note**: This deployment guide assumes familiarity with Kubernetes concepts. For production deployments, thorough testing in a staging environment is recommended.