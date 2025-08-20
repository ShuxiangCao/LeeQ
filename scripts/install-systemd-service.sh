#!/bin/bash
#
# LeeQ EPII Systemd Service Installation Script
#
# This script installs the LeeQ EPII daemon as a systemd service
# and sets up the necessary directories and permissions.
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEEQ_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="leeq-epii@.service"
SERVICE_FILE="$SCRIPT_DIR/systemd/$SERVICE_NAME"
SYSTEMD_DIR="/etc/systemd/system"
CONFIG_DIR="/etc/leeq-epii"
LOG_DIR="/var/log/leeq-epii"
RUN_DIR="/var/run/leeq-epii"
INSTALL_DIR="/opt/leeq"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Check if systemd is available
check_systemd() {
    if ! command -v systemctl &> /dev/null; then
        log_error "systemctl not found. This system doesn't appear to use systemd."
        exit 1
    fi
}

# Create system user and group
create_user() {
    if ! getent group leeq >/dev/null 2>&1; then
        log_info "Creating leeq group..."
        groupadd --system leeq
    else
        log_info "Group leeq already exists"
    fi

    if ! getent passwd leeq >/dev/null 2>&1; then
        log_info "Creating leeq user..."
        useradd --system --gid leeq --shell /bin/false \
            --home-dir /opt/leeq --no-create-home \
            --comment "LeeQ EPII daemon user" leeq
    else
        log_info "User leeq already exists"
    fi
}

# Create necessary directories
create_directories() {
    log_info "Creating directories..."
    
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$RUN_DIR"
    mkdir -p "$INSTALL_DIR"
    
    # Set ownership and permissions
    chown leeq:leeq "$LOG_DIR"
    chown leeq:leeq "$RUN_DIR"
    chown -R leeq:leeq "$CONFIG_DIR"
    
    chmod 755 "$CONFIG_DIR"
    chmod 750 "$LOG_DIR"
    chmod 750 "$RUN_DIR"
    
    log_success "Directories created and configured"
}

# Install LeeQ code
install_leeq() {
    log_info "Installing LeeQ to $INSTALL_DIR..."
    
    # Copy LeeQ source code
    if [[ -d "$INSTALL_DIR/leeq" ]]; then
        log_warning "LeeQ already installed, updating..."
        rm -rf "$INSTALL_DIR/leeq"
    fi
    
    cp -r "$LEEQ_ROOT/leeq" "$INSTALL_DIR/"
    
    # Copy virtual environment if it exists
    if [[ -d "$LEEQ_ROOT/venv" ]]; then
        if [[ -d "$INSTALL_DIR/venv" ]]; then
            log_warning "Virtual environment already exists, updating..."
            rm -rf "$INSTALL_DIR/venv"
        fi
        cp -r "$LEEQ_ROOT/venv" "$INSTALL_DIR/"
    else
        log_warning "Virtual environment not found at $LEEQ_ROOT/venv"
        log_warning "You may need to create a virtual environment manually"
    fi
    
    # Set ownership
    chown -R leeq:leeq "$INSTALL_DIR"
    
    log_success "LeeQ installed to $INSTALL_DIR"
}

# Install systemd service
install_service() {
    log_info "Installing systemd service..."
    
    if [[ ! -f "$SERVICE_FILE" ]]; then
        log_error "Service file not found: $SERVICE_FILE"
        exit 1
    fi
    
    # Copy service file
    cp "$SERVICE_FILE" "$SYSTEMD_DIR/"
    chmod 644 "$SYSTEMD_DIR/$SERVICE_NAME"
    
    # Reload systemd
    systemctl daemon-reload
    
    log_success "Systemd service installed"
}

# Install sample configuration
install_sample_config() {
    log_info "Installing sample configuration..."
    
    # Create simulation_2q.json config
    cat > "$CONFIG_DIR/simulation_2q.json" << EOF
{
    "setup_type": "simulation",
    "setup_name": "simulation_2q",
    "description": "2-qubit simulation setup for EPII testing",
    "max_workers": 10,
    "timeout": 300,
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    },
    "simulation": {
        "backend": "numpy",
        "qubits": 2
    }
}
EOF

    # Create hardware example (commented out)
    cat > "$CONFIG_DIR/hardware_lab1.json.example" << EOF
{
    "setup_type": "hardware",
    "setup_name": "hardware_lab1",
    "description": "Hardware setup example for lab environment",
    "max_workers": 5,
    "timeout": 600,
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    },
    "hardware": {
        "type": "qubic_lbnl",
        "config_file": "/etc/leeq-epii/hardware/lab1_config.json"
    }
}
EOF

    chown -R leeq:leeq "$CONFIG_DIR"
    chmod 640 "$CONFIG_DIR"/*.json*
    
    log_success "Sample configurations installed"
}

# Show usage information
show_usage() {
    cat << EOF

${GREEN}Installation completed successfully!${NC}

To use the LeeQ EPII service:

${BLUE}1. Enable and start a service instance:${NC}
   sudo systemctl enable leeq-epii@simulation_2q.service
   sudo systemctl start leeq-epii@simulation_2q.service

${BLUE}2. Check service status:${NC}
   sudo systemctl status leeq-epii@simulation_2q.service

${BLUE}3. View logs:${NC}
   sudo journalctl -u leeq-epii@simulation_2q.service -f

${BLUE}4. Stop the service:${NC}
   sudo systemctl stop leeq-epii@simulation_2q.service

${BLUE}Configuration files:${NC}
   - System config: $CONFIG_DIR/
   - Logs: $LOG_DIR/
   - Service file: $SYSTEMD_DIR/$SERVICE_NAME

${BLUE}Available configurations:${NC}
   - simulation_2q: Basic 2-qubit simulation setup
   - hardware_lab1.example: Template for hardware setup

${YELLOW}Note:${NC} Create additional .json config files in $CONFIG_DIR/ 
to define new service instances (e.g., for different setups).

EOF
}

# Main installation function
main() {
    log_info "Starting LeeQ EPII systemd service installation..."
    
    check_root
    check_systemd
    create_user
    create_directories
    install_leeq
    install_service
    install_sample_config
    
    log_success "Installation completed!"
    show_usage
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--help]"
        echo ""
        echo "Install LeeQ EPII as a systemd service"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac