# 🚀 VoiceForge Deployment Guide & Cost Analysis

## 📋 Table of Contents
1. [Deployment Options](#deployment-options)
2. [AWS Deployment](#aws-deployment)
3. [Google Cloud Platform](#gcp-deployment)
4. [Azure Deployment](#azure-deployment)
5. [Cost Analysis](#cost-analysis)
6. [Production Checklist](#production-checklist)

---

## 🎯 Deployment Options

### Option 1: AWS (Recommended)
**Best for**: Scalability, comprehensive services, market leader
- **EC2** for compute
- **EKS** for Kubernetes
- **RDS** for PostgreSQL
- **ElastiCache** for Redis
- **S3** for storage
- **CloudFront** for CDN

### Option 2: Google Cloud Platform
**Best for**: AI/ML workloads, competitive pricing
- **Compute Engine** for VMs
- **GKE** for Kubernetes
- **Cloud SQL** for PostgreSQL
- **Memorystore** for Redis
- **Cloud Storage** for files

### Option 3: Microsoft Azure
**Best for**: Enterprise integration, hybrid cloud
- **Virtual Machines**
- **AKS** for Kubernetes
- **Azure Database** for PostgreSQL
- **Azure Cache** for Redis
- **Blob Storage**

---

## 🌐 AWS Deployment

### Step 1: Infrastructure Setup

```bash
# Install AWS CLI
pip install awscli
aws configure

# Create VPC and subnets
aws ec2 create-vpc --cidr-block 10.0.0.0/16

# Launch EC2 instances (GPU-enabled for AI models)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g4dn.xlarge \
  --count 2 \
  --key-name voiceforge-key \
  --security-groups voiceforge-sg
```

### Step 2: EKS Cluster Setup

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create EKS cluster
eksctl create cluster \
  --name voiceforge-cluster \
  --region us-west-2 \
  --nodegroup-name gpu-nodes \
  --node-type g4dn.xlarge \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 10 \
  --managed
```

### Step 3: Database Setup

```bash
# Create RDS PostgreSQL instance
aws rds create-db-instance \
  --db-instance-identifier voiceforge-db \
  --db-instance-class db.m5.large \
  --engine postgres \
  --engine-version 14.7 \
  --master-username voiceforge \
  --master-user-password $DB_PASSWORD \
  --allocated-storage 100 \
  --backup-retention-period 7 \
  --multi-az
```

### Step 4: Deploy Application

```bash
# Build and push Docker image to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ECR_URI

docker build -t voiceforge .
docker tag voiceforge:latest $ECR_URI/voiceforge:latest
docker push $ECR_URI/voiceforge:latest

# Deploy to Kubernetes
kubectl apply -f k8s/
```

### Step 5: Setup Load Balancer & SSL

```bash
# Install AWS Load Balancer Controller
kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller/crds"

# Create SSL certificate
aws acm request-certificate \
  --domain-name api.voiceforge.ai \
  --validation-method DNS
```

---

## ☁️ GCP Deployment

### Step 1: Project Setup

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash

# Initialize project
gcloud init
gcloud config set project voiceforge-prod

# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable container.googleapis.com
gcloud services enable sqladmin.googleapis.com
```

### Step 2: GKE Cluster

```bash
# Create GKE cluster with GPU nodes
gcloud container clusters create voiceforge-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 10
```

### Step 3: Cloud SQL Setup

```bash
# Create PostgreSQL instance
gcloud sql instances create voiceforge-db \
  --database-version=POSTGRES_14 \
  --tier=db-n1-standard-2 \
  --region=us-central1 \
  --network=default \
  --backup \
  --backup-start-time=03:00
```

---

## 💰 Cost Analysis

### AWS Pricing (Monthly)

| Service | Configuration | Cost/Month |
|---------|--------------|------------|
| **EC2 (GPU)** | 2x g4dn.xlarge (NVIDIA T4) | $752 |
| **EKS** | Control plane | $73 |
| **RDS PostgreSQL** | db.m5.large, 100GB, Multi-AZ | $280 |
| **ElastiCache Redis** | cache.t3.medium | $50 |
| **S3 Storage** | 1TB | $23 |
| **CloudFront CDN** | 10TB transfer | $850 |
| **Load Balancer** | Application LB | $25 |
| **Data Transfer** | 5TB egress | $450 |
| **Backup & Snapshots** | - | $50 |
| **Total** | | **$2,553/month** |

**Annual Cost**: $30,636

### GCP Pricing (Monthly)

| Service | Configuration | Cost/Month |
|---------|--------------|------------|
| **Compute Engine** | 2x n1-standard-4 + T4 GPU | $680 |
| **GKE** | Management fee | $0 |
| **Cloud SQL** | PostgreSQL, 100GB | $250 |
| **Memorystore Redis** | 1GB | $40 |
| **Cloud Storage** | 1TB | $20 |
| **Cloud CDN** | 10TB transfer | $800 |
| **Load Balancer** | HTTP(S) LB | $25 |
| **Network Egress** | 5TB | $420 |
| **Total** | | **$2,235/month** |

**Annual Cost**: $26,820

### Azure Pricing (Monthly)

| Service | Configuration | Cost/Month |
|---------|--------------|------------|
| **Virtual Machines** | 2x NC4as T4 v3 | $876 |
| **AKS** | Management | $0 |
| **Azure Database** | PostgreSQL | $290 |
| **Azure Cache** | Redis, 1GB | $50 |
| **Blob Storage** | 1TB | $20 |
| **Azure CDN** | 10TB | $810 |
| **Load Balancer** | Standard | $30 |
| **Bandwidth** | 5TB | $430 |
| **Total** | | **$2,506/month** |

**Annual Cost**: $30,072

---

## 📊 Cost Optimization Strategies

### 1. Reserved Instances (Save 40-60%)
```
AWS: 3-year reserved = $1,021/month (save $1,532)
GCP: 3-year committed = $894/month (save $1,341)
Azure: 3-year reserved = $1,002/month (save $1,504)
```

### 2. Spot/Preemptible Instances (Save 70-90%)
- Use for batch processing workloads
- Not recommended for API servers
- Great for model training

### 3. Auto-scaling Configuration
```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: voiceforge-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: voiceforge-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 4. Storage Optimization
- Use lifecycle policies for old transcriptions
- Compress audio files after processing
- Move to cold storage after 30 days

---

## 🔧 Production Deployment Steps

### 1. Environment Setup

```bash
# Create production namespace
kubectl create namespace production

# Create secrets
kubectl create secret generic voiceforge-secrets \
  --from-literal=db-password=$DB_PASSWORD \
  --from-literal=jwt-secret=$JWT_SECRET \
  --from-literal=api-key=$API_KEY \
  -n production
```

### 2. Database Migration

```bash
# Run Alembic migrations
kubectl run migration --rm -it \
  --image=voiceforge:latest \
  --restart=Never \
  -n production \
  -- alembic upgrade head
```

### 3. Deploy Services

```bash
# Deploy all services
kubectl apply -f k8s/production/ -n production

# Verify deployment
kubectl get pods -n production
kubectl get services -n production
```

### 4. SSL/TLS Setup

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create certificate
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: voiceforge-tls
  namespace: production
spec:
  secretName: voiceforge-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - api.voiceforge.ai
  - www.voiceforge.ai
EOF
```

### 5. Monitoring Setup

```bash
# Install Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring

# Install Grafana dashboards
kubectl apply -f monitoring/dashboards/
```

---

## ✅ Production Checklist

### Security
- [ ] SSL/TLS certificates configured
- [ ] Secrets management (AWS Secrets Manager / GCP Secret Manager)
- [ ] Network policies configured
- [ ] WAF rules enabled
- [ ] DDoS protection active
- [ ] Regular security scans

### Performance
- [ ] Auto-scaling configured
- [ ] CDN enabled
- [ ] Database indexes optimized
- [ ] Redis caching active
- [ ] Load testing completed (target: 1000 req/s)

### Monitoring
- [ ] Prometheus metrics collection
- [ ] Grafana dashboards created
- [ ] Log aggregation (ELK/CloudWatch)
- [ ] Alerts configured
- [ ] Uptime monitoring (99.9% SLA)

### Backup & Recovery
- [ ] Database automated backups
- [ ] Point-in-time recovery tested
- [ ] Disaster recovery plan
- [ ] Multi-region failover

### Compliance
- [ ] GDPR compliance
- [ ] Data encryption at rest
- [ ] Data encryption in transit
- [ ] Audit logging enabled
- [ ] Terms of Service & Privacy Policy

---

## 🚀 Quick Deployment Commands

### One-Command Deploy (AWS)
```bash
# Deploy everything
./deploy.sh --environment production --provider aws --region us-west-2
```

### Health Check
```bash
# Verify deployment
curl -X GET https://api.voiceforge.ai/health

# Expected response:
{
  "status": "healthy",
  "version": "2.0.0",
  "checks": {
    "api": "healthy",
    "model": "healthy",
    "database": "healthy",
    "redis": "healthy"
  }
}
```

---

## 📞 Support

For deployment assistance:
- **Email**: devops@voiceforge.ai
- **Slack**: #voiceforge-deployment
- **Documentation**: https://docs.voiceforge.ai

---

## 💡 Recommendations

### For Startups (< 100K requests/month)
- **Provider**: GCP (lowest cost)
- **Configuration**: Single GPU instance, managed services
- **Cost**: ~$1,200/month

### For Scale-ups (100K - 1M requests/month)  
- **Provider**: AWS (best tooling)
- **Configuration**: Auto-scaling cluster, reserved instances
- **Cost**: ~$2,500/month

### For Enterprise (> 1M requests/month)
- **Provider**: Multi-cloud (AWS + GCP)
- **Configuration**: Global distribution, dedicated support
- **Cost**: ~$10,000+/month

---

*Last Updated: August 2024*