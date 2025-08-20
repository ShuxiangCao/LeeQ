#!/bin/bash
#
# LeeQ EPII Service Management Script
#
# This script provides convenient commands for managing LeeQ EPII daemon instances.
#

set -e

# Configuration
SERVICE_NAME="leeq-epii"
CONFIG_DIR="/etc/leeq-epii"
LOG_DIR="/var/log/leeq-epii"

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

# Check if systemctl is available
check_systemd() {
    if ! command -v systemctl &> /dev/null; then
        log_error "systemctl not found. This system doesn't appear to use systemd."
        exit 1
    fi
}

# List available configurations
list_configs() {
    log_info "Available configurations:"
    
    if [[ -d "$CONFIG_DIR" ]]; then
        for config in "$CONFIG_DIR"/*.json; do
            if [[ -f "$config" ]]; then
                local basename
                basename=$(basename "$config" .json)
                echo "  - $basename"
                
                # Show if service is enabled/active
                local service_name="${SERVICE_NAME}@${basename}.service"
                local enabled
                local active
                enabled=$(systemctl is-enabled "$service_name" 2>/dev/null || echo "disabled")
                active=$(systemctl is-active "$service_name" 2>/dev/null || echo "inactive")
                
                echo "    Status: $enabled, $active"
                
                # Show description from config if available
                if command -v jq &> /dev/null; then
                    local description
                    description=$(jq -r '.description // "No description"' "$config" 2>/dev/null)
                    echo "    Description: $description"
                fi
            fi
        done
    else
        log_warning "Configuration directory not found: $CONFIG_DIR"
        log_info "Run the installation script first: sudo ./install-systemd-service.sh"
    fi
}

# Start a service instance
start_service() {
    local instance="$1"
    
    if [[ -z "$instance" ]]; then
        log_error "Instance name required"
        echo "Usage: $0 start <instance_name>"
        echo "Example: $0 start simulation_2q"
        exit 1
    fi
    
    local service_name="${SERVICE_NAME}@${instance}.service"
    local config_file="$CONFIG_DIR/${instance}.json"
    
    # Check if config exists
    if [[ ! -f "$config_file" ]]; then
        log_error "Configuration file not found: $config_file"
        exit 1
    fi
    
    log_info "Starting $service_name..."
    
    # Enable and start the service
    systemctl enable "$service_name"
    systemctl start "$service_name"
    
    # Check if it started successfully
    sleep 2
    if systemctl is-active "$service_name" > /dev/null; then
        log_success "Service $service_name started successfully"
        
        # Show status
        echo ""
        systemctl status "$service_name" --no-pager -l
    else
        log_error "Service $service_name failed to start"
        echo ""
        log_info "Check logs with: journalctl -u $service_name"
        exit 1
    fi
}

# Stop a service instance
stop_service() {
    local instance="$1"
    
    if [[ -z "$instance" ]]; then
        log_error "Instance name required"
        echo "Usage: $0 stop <instance_name>"
        echo "Example: $0 stop simulation_2q"
        exit 1
    fi
    
    local service_name="${SERVICE_NAME}@${instance}.service"
    
    log_info "Stopping $service_name..."
    
    systemctl stop "$service_name"
    
    # Optionally disable if requested
    if [[ "${2:-}" == "--disable" ]]; then
        systemctl disable "$service_name"
        log_info "Service disabled (will not start on boot)"
    fi
    
    log_success "Service $service_name stopped"
}

# Restart a service instance
restart_service() {
    local instance="$1"
    
    if [[ -z "$instance" ]]; then
        log_error "Instance name required"
        echo "Usage: $0 restart <instance_name>"
        echo "Example: $0 restart simulation_2q"
        exit 1
    fi
    
    local service_name="${SERVICE_NAME}@${instance}.service"
    
    log_info "Restarting $service_name..."
    
    systemctl restart "$service_name"
    
    # Check if it restarted successfully
    sleep 2
    if systemctl is-active "$service_name" > /dev/null; then
        log_success "Service $service_name restarted successfully"
    else
        log_error "Service $service_name failed to restart"
        echo ""
        log_info "Check logs with: journalctl -u $service_name"
        exit 1
    fi
}

# Show status of service instances
show_status() {
    local instance="$1"
    
    if [[ -n "$instance" ]]; then
        # Show specific instance
        local service_name="${SERVICE_NAME}@${instance}.service"
        systemctl status "$service_name" --no-pager -l
    else
        # Show all instances
        log_info "Status of all LeeQ EPII services:"
        systemctl status "${SERVICE_NAME}@*.service" --no-pager -l 2>/dev/null || {
            log_warning "No LeeQ EPII services found"
            log_info "Available configurations:"
            list_configs
        }
    fi
}

# Show logs for a service instance
show_logs() {
    local instance="$1"
    local follow="${2:-}"
    
    if [[ -z "$instance" ]]; then
        log_error "Instance name required"
        echo "Usage: $0 logs <instance_name> [--follow]"
        echo "Example: $0 logs simulation_2q --follow"
        exit 1
    fi
    
    local service_name="${SERVICE_NAME}@${instance}.service"
    
    if [[ "$follow" == "--follow" || "$follow" == "-f" ]]; then
        log_info "Following logs for $service_name (Ctrl+C to exit)..."
        journalctl -u "$service_name" -f
    else
        log_info "Recent logs for $service_name:"
        journalctl -u "$service_name" --no-pager -l
    fi
}

# Validate a configuration file
validate_config() {
    local instance="$1"
    
    if [[ -z "$instance" ]]; then
        log_error "Instance name required"
        echo "Usage: $0 validate <instance_name>"
        echo "Example: $0 validate simulation_2q"
        exit 1
    fi
    
    local config_file="$CONFIG_DIR/${instance}.json"
    
    if [[ ! -f "$config_file" ]]; then
        log_error "Configuration file not found: $config_file"
        exit 1
    fi
    
    log_info "Validating configuration: $config_file"
    
    # Use the daemon's validation feature
    if command -v python3 &> /dev/null && [[ -d "/opt/leeq" ]]; then
        cd /opt/leeq
        python3 -m leeq.epii.daemon --config "$config_file" --validate
        log_success "Configuration is valid"
    else
        log_warning "Cannot validate - LeeQ installation not found in /opt/leeq"
        
        # Basic JSON validation
        if command -v jq &> /dev/null; then
            if jq empty "$config_file" 2>/dev/null; then
                log_success "Configuration is valid JSON"
            else
                log_error "Configuration is invalid JSON"
                exit 1
            fi
        else
            log_warning "Cannot validate - jq not available"
        fi
    fi
}

# Show help
show_help() {
    cat << EOF
LeeQ EPII Service Management Script

Usage: $0 <command> [arguments]

Commands:
    list                         List available configurations and their status
    start <instance>             Start and enable a service instance
    stop <instance> [--disable]  Stop a service instance (optionally disable)
    restart <instance>           Restart a service instance
    status [instance]            Show status of service instance(s)
    logs <instance> [--follow]   Show logs for a service instance
    validate <instance>          Validate a configuration file
    help                         Show this help message

Examples:
    $0 list                      # List all available configurations
    $0 start simulation_2q       # Start the simulation_2q instance
    $0 stop simulation_2q        # Stop the simulation_2q instance
    $0 restart simulation_2q     # Restart the simulation_2q instance
    $0 status                    # Show status of all instances
    $0 status simulation_2q      # Show status of specific instance
    $0 logs simulation_2q        # Show recent logs
    $0 logs simulation_2q -f     # Follow logs in real-time
    $0 validate simulation_2q    # Validate configuration

Configuration files are stored in: $CONFIG_DIR/
Service logs are available via: journalctl -u ${SERVICE_NAME}@<instance>.service

EOF
}

# Main function
main() {
    # Check for systemd availability
    check_systemd
    
    # Parse command
    case "${1:-}" in
        list|ls)
            list_configs
            ;;
        start)
            start_service "$2"
            ;;
        stop)
            stop_service "$2" "$3"
            ;;
        restart)
            restart_service "$2"
            ;;
        status)
            show_status "$2"
            ;;
        logs)
            show_logs "$2" "$3"
            ;;
        validate)
            validate_config "$2"
            ;;
        help|--help|-h)
            show_help
            ;;
        "")
            log_error "No command specified"
            echo ""
            show_help
            exit 1
            ;;
        *)
            log_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"