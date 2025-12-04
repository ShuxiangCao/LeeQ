#!/bin/bash
# Build and run EPII daemon in Docker with example configuration
# Usage: ./build_and_run.sh [mode] [options]
# Modes: daemon (default), jupyter, both

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="leeq-epii"
IMAGE_TAG="latest"
CONTAINER_NAME_DAEMON="epii-daemon"
CONTAINER_NAME_JUPYTER="epii-jupyter"
CONFIG_PATH="$(dirname "$0")/../../configs/docker"
DATA_PATH="$(dirname "$0")/../../data"
MODE="${1:-daemon}"

# Print colored message
print_msg() {
    echo -e "${GREEN}[LeeQ Docker]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_msg "Docker found: $(docker --version)"
}

# Create necessary directories
create_directories() {
    print_msg "Creating necessary directories..."
    mkdir -p "$CONFIG_PATH"
    mkdir -p "$DATA_PATH"
    mkdir -p "$DATA_PATH/logs"
    mkdir -p "$DATA_PATH/chronicle_data"
    mkdir -p "$DATA_PATH/notebooks"
    print_msg "Directories created/verified"
}

# Build Docker image
build_image() {
    print_msg "Building Docker image ${IMAGE_NAME}:${IMAGE_TAG}..."
    
    # Check if Dockerfile exists
    if [ ! -f "Dockerfile" ]; then
        print_error "Dockerfile not found in current directory"
        exit 1
    fi
    
    # Build with progress
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" . || {
        print_error "Docker build failed"
        exit 1
    }
    
    print_msg "Docker image built successfully"
    
    # Show image info
    echo -e "\n${GREEN}Image Information:${NC}"
    docker images "${IMAGE_NAME}:${IMAGE_TAG}"
}

# Stop and remove existing container
cleanup_container() {
    local container_name=$1
    if docker ps -a | grep -q "$container_name"; then
        print_warning "Stopping and removing existing container: $container_name"
        docker stop "$container_name" 2>/dev/null || true
        docker rm "$container_name" 2>/dev/null || true
    fi
}

# Run daemon mode
run_daemon() {
    print_msg "Starting EPII daemon mode..."
    
    cleanup_container "$CONTAINER_NAME_DAEMON"
    
    # Check if config file exists
    if [ ! -f "$CONFIG_PATH/example_daemon.json" ]; then
        print_warning "Config file not found at $CONFIG_PATH/example_daemon.json"
        print_msg "Using default configuration"
        CONFIG_MOUNT=""
    else
        CONFIG_MOUNT="-v $(realpath $CONFIG_PATH):/home/jovyan/config:ro"
        print_msg "Using config: $CONFIG_PATH/example_daemon.json"
    fi
    
    # Run daemon container
    docker run -d \
        --name "$CONTAINER_NAME_DAEMON" \
        -e CONTAINER_MODE=daemon \
        -e EPII_CONFIG=/home/jovyan/config/example_daemon.json \
        -e EPII_LOG_LEVEL=INFO \
        -p 50051:50051 \
        -p 8051:8051 \
        $CONFIG_MOUNT \
        -v "$(realpath $DATA_PATH)":/home/jovyan/work:rw \
        "${IMAGE_NAME}:${IMAGE_TAG}"
    
    print_msg "EPII daemon started"
    echo -e "\n${GREEN}Daemon Access:${NC}"
    echo "  - gRPC Service: localhost:50051"
    echo "  - Chronicle Viewer: http://localhost:8051"
    echo "  - Container Name: $CONTAINER_NAME_DAEMON"
    echo ""
    echo "View logs: docker logs -f $CONTAINER_NAME_DAEMON"
    echo "Stop daemon: docker stop $CONTAINER_NAME_DAEMON"
}

# Run Jupyter mode
run_jupyter() {
    print_msg "Starting Jupyter notebook mode..."
    
    cleanup_container "$CONTAINER_NAME_JUPYTER"
    
    # Run Jupyter container
    docker run -d \
        --name "$CONTAINER_NAME_JUPYTER" \
        -e CONTAINER_MODE=jupyter \
        -p 8888:8888 \
        -v "$(realpath $DATA_PATH/notebooks)":/home/jovyan/work:rw \
        "${IMAGE_NAME}:${IMAGE_TAG}"
    
    print_msg "Jupyter notebook started"
    
    # Wait for Jupyter to start and get token
    sleep 3
    
    echo -e "\n${GREEN}Jupyter Access:${NC}"
    echo "  - URL: http://localhost:8888"
    echo "  - Container Name: $CONTAINER_NAME_JUPYTER"
    echo ""
    echo "View logs: docker logs -f $CONTAINER_NAME_JUPYTER"
    echo "Stop Jupyter: docker stop $CONTAINER_NAME_JUPYTER"
}

# Show container status
show_status() {
    echo -e "\n${GREEN}Container Status:${NC}"
    docker ps --filter "name=epii-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# Test daemon connection
test_daemon() {
    print_msg "Testing daemon connection..."
    
    # Wait for daemon to be ready
    sleep 5
    
    # Check if container is running
    if ! docker ps | grep -q "$CONTAINER_NAME_DAEMON"; then
        print_error "Daemon container is not running"
        return 1
    fi
    
    # Check daemon logs for startup message
    if docker logs "$CONTAINER_NAME_DAEMON" 2>&1 | grep -q "EPII daemon ready"; then
        print_msg "✓ Daemon started successfully"
    else
        print_warning "Daemon may still be starting up. Check logs for details."
    fi
    
    # Test gRPC port
    if nc -zv localhost 50051 2>/dev/null; then
        print_msg "✓ gRPC port 50051 is accessible"
    else
        print_warning "gRPC port 50051 is not accessible yet"
    fi
}

# Main execution
main() {
    echo -e "${GREEN}=====================================${NC}"
    echo -e "${GREEN}   LeeQ EPII Docker Builder & Runner${NC}"
    echo -e "${GREEN}=====================================${NC}\n"
    
    check_docker
    create_directories
    
    case "$MODE" in
        daemon)
            build_image
            run_daemon
            test_daemon
            show_status
            ;;
        jupyter)
            build_image
            run_jupyter
            show_status
            ;;
        both)
            build_image
            run_daemon
            run_jupyter
            test_daemon
            show_status
            ;;
        build-only)
            build_image
            ;;
        status)
            show_status
            ;;
        stop)
            cleanup_container "$CONTAINER_NAME_DAEMON"
            cleanup_container "$CONTAINER_NAME_JUPYTER"
            print_msg "All containers stopped"
            ;;
        logs-daemon)
            docker logs -f "$CONTAINER_NAME_DAEMON"
            ;;
        logs-jupyter)
            docker logs -f "$CONTAINER_NAME_JUPYTER"
            ;;
        *)
            print_error "Unknown mode: $MODE"
            echo "Usage: $0 [daemon|jupyter|both|build-only|status|stop|logs-daemon|logs-jupyter]"
            echo ""
            echo "Modes:"
            echo "  daemon      - Build and run EPII daemon (default)"
            echo "  jupyter     - Build and run Jupyter notebook"
            echo "  both        - Run both daemon and Jupyter"
            echo "  build-only  - Only build the Docker image"
            echo "  status      - Show container status"
            echo "  stop        - Stop all containers"
            echo "  logs-daemon - Show daemon logs"
            echo "  logs-jupyter - Show Jupyter logs"
            exit 1
            ;;
    esac
    
    echo -e "\n${GREEN}Done!${NC}"
}

# Run main function
main "$@"