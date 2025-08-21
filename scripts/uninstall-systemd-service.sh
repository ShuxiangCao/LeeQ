#!/bin/bash
#
# LeeQ EPII Systemd Service Uninstallation Script
#
# This script removes the LeeQ EPII daemon systemd service
# and optionally cleans up directories and user accounts.
#

set -e

# Configuration
SERVICE_NAME="leeq-epii@*.service"
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

# Stop and disable all running services
stop_services() {
    log_info "Stopping and disabling LeeQ EPII services..."
    
    # Find all enabled leeq-epii services
    local services
    services=$(systemctl list-unit-files "leeq-epii@*.service" --state=enabled --no-legend | awk '{print $1}' || true)
    
    if [[ -n "$services" ]]; then
        for service in $services; do
            log_info "Stopping and disabling $service..."
            systemctl stop "$service" || true
            systemctl disable "$service" || true
        done
    else
        log_info "No enabled leeq-epii services found"
    fi
    
    # Also check for any running instances
    local running_services
    running_services=$(systemctl list-units "leeq-epii@*.service" --state=active --no-legend | awk '{print $1}' || true)
    
    if [[ -n "$running_services" ]]; then
        for service in $running_services; do
            log_warning "Stopping running service $service..."
            systemctl stop "$service" || true
        done
    fi
}

# Remove systemd service files
remove_service_files() {
    log_info "Removing systemd service files..."
    
    if [[ -f "$SYSTEMD_DIR/leeq-epii@.service" ]]; then
        rm -f "$SYSTEMD_DIR/leeq-epii@.service"
        log_success "Removed systemd service file"
    else
        log_info "No systemd service file found"
    fi
    
    # Reload systemd
    systemctl daemon-reload
    systemctl reset-failed || true
}

# Remove directories
remove_directories() {
    local remove_data="$1"
    
    log_info "Removing runtime directories..."
    
    # Always remove runtime directory
    if [[ -d "$RUN_DIR" ]]; then
        rm -rf "$RUN_DIR"
        log_success "Removed runtime directory: $RUN_DIR"
    fi
    
    if [[ "$remove_data" == "yes" ]]; then
        log_warning "Removing data directories (logs and config)..."
        
        if [[ -d "$LOG_DIR" ]]; then
            rm -rf "$LOG_DIR"
            log_success "Removed log directory: $LOG_DIR"
        fi
        
        if [[ -d "$CONFIG_DIR" ]]; then
            rm -rf "$CONFIG_DIR"
            log_success "Removed config directory: $CONFIG_DIR"
        fi
        
        if [[ -d "$INSTALL_DIR" ]]; then
            log_warning "Removing installation directory: $INSTALL_DIR"
            rm -rf "$INSTALL_DIR"
            log_success "Removed installation directory"
        fi
    else
        log_info "Keeping data directories (use --remove-data to remove them)"
        if [[ -d "$LOG_DIR" ]]; then
            log_info "Logs preserved in: $LOG_DIR"
        fi
        if [[ -d "$CONFIG_DIR" ]]; then
            log_info "Config preserved in: $CONFIG_DIR"
        fi
    fi
}

# Remove user and group
remove_user() {
    local remove_user="$1"
    
    if [[ "$remove_user" == "yes" ]]; then
        log_info "Removing leeq user and group..."
        
        if getent passwd leeq >/dev/null 2>&1; then
            userdel leeq
            log_success "Removed leeq user"
        fi
        
        if getent group leeq >/dev/null 2>&1; then
            groupdel leeq
            log_success "Removed leeq group"
        fi
    else
        log_info "Keeping leeq user and group (use --remove-user to remove them)"
    fi
}

# Show help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Remove LeeQ EPII systemd service installation.

Options:
    --remove-data    Remove all data directories (logs, config, installation)
    --remove-user    Remove the leeq system user and group
    --all            Remove everything (equivalent to --remove-data --remove-user)
    --help, -h       Show this help message

Examples:
    $0                          # Remove service only, keep data and user
    $0 --remove-data            # Remove service and data, keep user
    $0 --all                    # Remove everything

EOF
}

# Main uninstallation function
main() {
    local remove_data="no"
    local remove_user="no"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --remove-data)
                remove_data="yes"
                shift
                ;;
            --remove-user)
                remove_user="yes"
                shift
                ;;
            --all)
                remove_data="yes"
                remove_user="yes"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    log_info "Starting LeeQ EPII systemd service uninstallation..."
    
    check_root
    stop_services
    remove_service_files
    remove_directories "$remove_data"
    remove_user "$remove_user"
    
    log_success "Uninstallation completed!"
    
    if [[ "$remove_data" == "no" ]] || [[ "$remove_user" == "no" ]]; then
        log_info "Some components were preserved. Use --all to remove everything."
    fi
}

main "$@"