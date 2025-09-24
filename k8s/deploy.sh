#!/bin/bash

# VoiceForge STT Kubernetes Deployment Script
# Usage: ./deploy.sh [environment] [namespace]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ENVIRONMENT="production"
DEFAULT_NAMESPACE="voiceforge-stt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
ENVIRONMENT="${1:-$DEFAULT_ENVIRONMENT}"
NAMESPACE="${2:-$DEFAULT_NAMESPACE}"

log_info "Starting VoiceForge STT deployment"
log_info "Environment: $ENVIRONMENT"
log_info "Namespace: $NAMESPACE"

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT. Must be development, staging, or production."
    exit 1
fi

# Check kubectl availability
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Check cluster connectivity
if ! kubectl cluster-info &> /dev/null; then
    log_error "Unable to connect to Kubernetes cluster"
    exit 1
fi

log_info "Kubernetes cluster connectivity verified"

# Check for required tools
REQUIRED_TOOLS=("kustomize" "helm")
for tool in "${REQUIRED_TOOLS[@]}"; do
    if ! command -v "$tool" &> /dev/null; then
        log_warning "$tool is not installed. Some features may not work correctly."
    fi
done

# Validation function
validate_secrets() {
    log_info "Validating secrets configuration..."
    
    # Check if secrets file exists and contains placeholder values
    if [[ -f "$SCRIPT_DIR/secrets.yaml" ]]; then
        if grep -q "cG9zdGdyZXNfcGFzc3dvcmQ=" "$SCRIPT_DIR/secrets.yaml"; then
            log_warning "Default placeholder passwords detected in secrets.yaml"
            log_warning "Please update secrets.yaml with actual production values"
            
            if [[ "$ENVIRONMENT" == "production" ]]; then
                read -p "Continue with placeholder values? (y/N): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    log_error "Deployment cancelled. Please update secrets before proceeding."
                    exit 1
                fi
            fi
        fi
    else
        log_error "secrets.yaml not found"
        exit 1
    fi
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check if namespace exists
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Namespace $NAMESPACE already exists"
    else
        log_info "Namespace $NAMESPACE will be created"
    fi
    
    # Check storage classes
    log_info "Checking storage classes..."
    if ! kubectl get storageclass fast-ssd &> /dev/null; then
        log_warning "Storage class 'fast-ssd' not found. You may need to create it or update the PVC specifications."
    fi
    
    if ! kubectl get storageclass shared-storage &> /dev/null; then
        log_warning "Storage class 'shared-storage' not found. You may need to create it or update the PVC specifications."
    fi
    
    # Check for GPU nodes if using GPU workloads
    GPU_NODES=$(kubectl get nodes -l nvidia.com/gpu-present=true --no-headers 2>/dev/null | wc -l)
    if [[ $GPU_NODES -eq 0 ]]; then
        log_warning "No GPU nodes found. GPU-dependent workloads may fail to schedule."
    else
        log_info "Found $GPU_NODES GPU node(s)"
    fi
}

# Deploy function
deploy_component() {
    local component=$1
    local file=$2
    
    log_info "Deploying $component..."
    
    if kubectl apply -f "$file"; then
        log_success "$component deployed successfully"
    else
        log_error "Failed to deploy $component"
        return 1
    fi
}

# Wait for deployment to be ready
wait_for_deployment() {
    local deployment=$1
    local namespace=$2
    local timeout=${3:-300}
    
    log_info "Waiting for $deployment to be ready..."
    
    if kubectl wait --for=condition=available --timeout=${timeout}s deployment/"$deployment" -n "$namespace"; then
        log_success "$deployment is ready"
    else
        log_error "$deployment failed to become ready within ${timeout}s"
        return 1
    fi
}

# Wait for StatefulSet to be ready
wait_for_statefulset() {
    local statefulset=$1
    local namespace=$2
    local replicas=$3
    local timeout=${4:-300}
    
    log_info "Waiting for StatefulSet $statefulset to be ready..."
    
    if kubectl wait --for=jsonpath='{.status.readyReplicas}'="$replicas" --timeout=${timeout}s statefulset/"$statefulset" -n "$namespace"; then
        log_success "StatefulSet $statefulset is ready"
    else
        log_error "StatefulSet $statefulset failed to become ready within ${timeout}s"
        return 1
    fi
}

# Main deployment function
deploy() {
    log_info "Starting deployment process..."
    
    # Validate secrets
    validate_secrets
    
    # Run pre-deployment checks
    pre_deployment_checks
    
    # Deploy components in order
    COMPONENTS=(
        "Namespace:namespace.yaml"
        "Secrets:secrets.yaml"
        "ConfigMaps:configmap.yaml"
        "PostgreSQL:postgres.yaml"
        "Redis:redis.yaml"
        "API:api.yaml"
        "Workers:worker.yaml"
        "Monitoring:monitoring.yaml"
        "Ingress:ingress.yaml"
    )
    
    for component_spec in "${COMPONENTS[@]}"; do
        IFS=':' read -r component file <<< "$component_spec"
        
        if [[ -f "$SCRIPT_DIR/$file" ]]; then
            deploy_component "$component" "$SCRIPT_DIR/$file"
        else
            log_error "File not found: $file"
            exit 1
        fi
        
        # Add delays between certain components
        case "$component" in
            "Namespace"|"Secrets"|"ConfigMaps")
                sleep 5
                ;;
            "PostgreSQL"|"Redis")
                sleep 10
                ;;
        esac
    done
    
    # Wait for critical components to be ready
    log_info "Waiting for critical components to be ready..."
    
    # Wait for PostgreSQL
    wait_for_statefulset "voiceforge-postgres" "$NAMESPACE" 1 300
    
    # Wait for Redis
    wait_for_statefulset "voiceforge-redis" "$NAMESPACE" 1 300
    
    # Wait for API
    wait_for_statefulset "voiceforge-api" "$NAMESPACE" 3 600
    
    # Wait for Workers
    wait_for_deployment "voiceforge-worker" "$NAMESPACE" 300
    wait_for_deployment "voiceforge-worker-batch" "$NAMESPACE" 300
    
    log_success "All components deployed successfully!"
}

# Post-deployment verification
verify_deployment() {
    log_info "Running post-deployment verification..."
    
    # Check pod status
    log_info "Checking pod status..."
    kubectl get pods -n "$NAMESPACE"
    
    # Check services
    log_info "Checking services..."
    kubectl get services -n "$NAMESPACE"
    
    # Check ingress
    log_info "Checking ingress..."
    kubectl get ingress -n "$NAMESPACE"
    
    # Test API health endpoint
    log_info "Testing API health endpoint..."
    API_POD=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=api -o jsonpath='{.items[0].metadata.name}')
    
    if [[ -n "$API_POD" ]]; then
        if kubectl exec -n "$NAMESPACE" "$API_POD" -c api -- curl -f http://localhost:8000/health; then
            log_success "API health check passed"
        else
            log_warning "API health check failed"
        fi
    else
        log_warning "No API pod found for health check"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up deployment..."
    
    read -p "Are you sure you want to delete the $NAMESPACE deployment? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete namespace "$NAMESPACE" --ignore-not-found
        log_success "Deployment cleaned up"
    else
        log_info "Cleanup cancelled"
    fi
}

# Show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [ENVIRONMENT] [NAMESPACE]"
    echo ""
    echo "Commands:"
    echo "  deploy     Deploy the application (default)"
    echo "  verify     Verify existing deployment"
    echo "  cleanup    Remove the deployment"
    echo "  help       Show this help message"
    echo ""
    echo "Environments: development, staging, production (default: production)"
    echo "Namespace: Kubernetes namespace (default: voiceforge-stt)"
    echo ""
    echo "Examples:"
    echo "  $0 deploy production voiceforge-stt"
    echo "  $0 verify staging voiceforge-staging"
    echo "  $0 cleanup development voiceforge-dev"
}

# Main script logic
case "${1:-deploy}" in
    deploy)
        deploy
        verify_deployment
        ;;
    verify)
        verify_deployment
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_usage
        exit 0
        ;;
    *)
        log_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac

log_success "Script completed successfully!"