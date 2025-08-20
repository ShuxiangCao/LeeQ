#!/bin/bash
#
# Test script for LeeQ EPII systemd integration
#
# This script tests the systemd service functionality including
# installation, service management, and logging verification.
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_INSTANCE="simulation_2q"
SERVICE_NAME="leeq-epii@${TEST_INSTANCE}.service"
LEEQ_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

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

log_test_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_test_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
    FAILED_TESTS+=("$1")
}

# Test execution wrapper
run_test() {
    local test_name="$1"
    local test_function="$2"
    
    log_info "Running test: $test_name"
    
    if $test_function; then
        log_test_pass "$test_name"
    else
        log_test_fail "$test_name"
    fi
    
    echo ""
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This test script must be run as root (use sudo)"
        log_info "Some tests require systemctl operations which need root privileges"
        exit 1
    fi
}

# Check prerequisites
test_prerequisites() {
    local all_good=true
    
    # Check systemctl
    if ! command -v systemctl &> /dev/null; then
        log_error "systemctl not found"
        all_good=false
    fi
    
    # Check if service file exists
    if [[ ! -f "/etc/systemd/system/leeq-epii@.service" ]]; then
        log_error "Systemd service file not installed. Run install-systemd-service.sh first."
        all_good=false
    fi
    
    # Check if config exists
    if [[ ! -f "/etc/leeq-epii/${TEST_INSTANCE}.json" ]]; then
        log_error "Test configuration not found: /etc/leeq-epii/${TEST_INSTANCE}.json"
        all_good=false
    fi
    
    # Check if LeeQ is installed
    if [[ ! -d "/opt/leeq" ]]; then
        log_error "LeeQ not installed in /opt/leeq. Run install-systemd-service.sh first."
        all_good=false
    fi
    
    return $all_good
}

# Test service installation
test_service_installation() {
    # Check if service file is properly installed
    if [[ -f "/etc/systemd/system/leeq-epii@.service" ]]; then
        log_info "✓ Service file is installed"
    else
        log_error "✗ Service file not found"
        return 1
    fi
    
    # Check if systemd recognizes the service
    if systemctl list-unit-files "leeq-epii@*.service" &> /dev/null; then
        log_info "✓ Systemd recognizes the service template"
    else
        log_error "✗ Systemd does not recognize the service template"
        return 1
    fi
    
    return 0
}

# Test service enable/disable
test_service_enable_disable() {
    # Enable the service
    if systemctl enable "$SERVICE_NAME" &> /dev/null; then
        log_info "✓ Service enabled successfully"
    else
        log_error "✗ Failed to enable service"
        return 1
    fi
    
    # Check if it's enabled
    if systemctl is-enabled "$SERVICE_NAME" &> /dev/null; then
        log_info "✓ Service is enabled"
    else
        log_error "✗ Service is not enabled"
        return 1
    fi
    
    # Disable the service
    if systemctl disable "$SERVICE_NAME" &> /dev/null; then
        log_info "✓ Service disabled successfully"
    else
        log_error "✗ Failed to disable service"
        return 1
    fi
    
    return 0
}

# Test service start/stop
test_service_start_stop() {
    # Start the service
    log_info "Starting service..."
    if systemctl start "$SERVICE_NAME"; then
        log_info "✓ Service started"
    else
        log_error "✗ Failed to start service"
        systemctl status "$SERVICE_NAME" --no-pager -l || true
        return 1
    fi
    
    # Wait a moment for startup
    sleep 3
    
    # Check if it's active
    if systemctl is-active "$SERVICE_NAME" &> /dev/null; then
        log_info "✓ Service is active"
    else
        log_error "✗ Service is not active"
        systemctl status "$SERVICE_NAME" --no-pager -l || true
        return 1
    fi
    
    # Check if it's listening on port (basic connectivity test)
    local port
    port=$(jq -r '.port // 50051' "/etc/leeq-epii/${TEST_INSTANCE}.json" 2>/dev/null || echo "50051")
    
    if timeout 5 bash -c "echo > /dev/tcp/localhost/$port" 2>/dev/null; then
        log_info "✓ Service is listening on port $port"
    else
        log_warning "⚠ Service might not be listening on port $port (this could be normal during startup)"
    fi
    
    # Stop the service
    log_info "Stopping service..."
    if systemctl stop "$SERVICE_NAME"; then
        log_info "✓ Service stopped"
    else
        log_error "✗ Failed to stop service"
        return 1
    fi
    
    # Check if it's inactive
    if ! systemctl is-active "$SERVICE_NAME" &> /dev/null; then
        log_info "✓ Service is inactive"
    else
        log_error "✗ Service is still active"
        return 1
    fi
    
    return 0
}

# Test service restart
test_service_restart() {
    # Start the service first
    systemctl start "$SERVICE_NAME"
    sleep 2
    
    # Get initial PID if possible
    local initial_pid
    initial_pid=$(systemctl show "$SERVICE_NAME" --property=MainPID --value 2>/dev/null || echo "")
    
    # Restart the service
    log_info "Restarting service..."
    if systemctl restart "$SERVICE_NAME"; then
        log_info "✓ Service restarted"
    else
        log_error "✗ Failed to restart service"
        return 1
    fi
    
    # Wait for restart
    sleep 3
    
    # Check if it's active after restart
    if systemctl is-active "$SERVICE_NAME" &> /dev/null; then
        log_info "✓ Service is active after restart"
    else
        log_error "✗ Service is not active after restart"
        systemctl status "$SERVICE_NAME" --no-pager -l || true
        return 1
    fi
    
    # Check if PID changed (indicating actual restart)
    local new_pid
    new_pid=$(systemctl show "$SERVICE_NAME" --property=MainPID --value 2>/dev/null || echo "")
    
    if [[ -n "$initial_pid" && -n "$new_pid" && "$initial_pid" != "$new_pid" ]]; then
        log_info "✓ Service PID changed ($initial_pid -> $new_pid)"
    elif [[ -n "$new_pid" && "$new_pid" != "0" ]]; then
        log_info "✓ Service has valid PID after restart: $new_pid"
    else
        log_warning "⚠ Could not verify PID change"
    fi
    
    # Stop for cleanup
    systemctl stop "$SERVICE_NAME"
    
    return 0
}

# Test logging configuration
test_logging() {
    # Start the service
    systemctl start "$SERVICE_NAME"
    sleep 3
    
    # Check if logs are being generated
    local log_entries
    log_entries=$(journalctl -u "$SERVICE_NAME" --since "1 minute ago" --no-pager | wc -l)
    
    if [[ $log_entries -gt 0 ]]; then
        log_info "✓ Service is generating log entries ($log_entries entries)"
    else
        log_error "✗ No log entries found"
        systemctl stop "$SERVICE_NAME"
        return 1
    fi
    
    # Check log format (should contain structured messages)
    local sample_log
    sample_log=$(journalctl -u "$SERVICE_NAME" --since "1 minute ago" --no-pager -n 1 2>/dev/null || echo "")
    
    if [[ -n "$sample_log" ]]; then
        log_info "✓ Sample log entry captured"
        echo "    Sample: $sample_log"
    else
        log_warning "⚠ Could not capture sample log entry"
    fi
    
    # Test different log levels by looking for specific patterns
    if journalctl -u "$SERVICE_NAME" --since "1 minute ago" --no-pager | grep -q "INFO\|ERROR\|WARNING"; then
        log_info "✓ Structured log levels detected"
    else
        log_warning "⚠ Could not detect structured log levels"
    fi
    
    # Stop for cleanup
    systemctl stop "$SERVICE_NAME"
    
    return 0
}

# Test graceful shutdown
test_graceful_shutdown() {
    # Start the service
    systemctl start "$SERVICE_NAME"
    sleep 3
    
    # Send SIGTERM and check if it shuts down gracefully
    local pid
    pid=$(systemctl show "$SERVICE_NAME" --property=MainPID --value 2>/dev/null || echo "0")
    
    if [[ "$pid" != "0" && -n "$pid" ]]; then
        log_info "Sending SIGTERM to PID $pid"
        
        # Send SIGTERM
        kill -TERM "$pid" 2>/dev/null || true
        
        # Wait for graceful shutdown (max 30 seconds as configured in service)
        local count=0
        while [[ $count -lt 30 ]] && systemctl is-active "$SERVICE_NAME" &> /dev/null; do
            sleep 1
            ((count++))
        done
        
        if ! systemctl is-active "$SERVICE_NAME" &> /dev/null; then
            log_info "✓ Service shut down gracefully in ${count}s"
        else
            log_error "✗ Service did not shut down gracefully"
            systemctl stop "$SERVICE_NAME" || true
            return 1
        fi
    else
        log_error "✗ Could not get service PID"
        systemctl stop "$SERVICE_NAME" || true
        return 1
    fi
    
    return 0
}

# Test configuration validation
test_config_validation() {
    # Test with valid config
    if python3 -c "
import sys
sys.path.insert(0, '/opt/leeq')
from leeq.epii.daemon import load_config, validate_config
config = load_config('/etc/leeq-epii/${TEST_INSTANCE}.json')
result = validate_config(config)
exit(0 if result else 1)
" &> /dev/null; then
        log_info "✓ Configuration validation passed"
    else
        log_error "✗ Configuration validation failed"
        return 1
    fi
    
    return 0
}

# Test service management script
test_service_management_script() {
    local script_path="$SCRIPT_DIR/leeq-epii-service.sh"
    
    if [[ ! -f "$script_path" ]]; then
        log_error "✗ Service management script not found: $script_path"
        return 1
    fi
    
    # Test list command
    if "$script_path" list &> /dev/null; then
        log_info "✓ Service management script 'list' command works"
    else
        log_error "✗ Service management script 'list' command failed"
        return 1
    fi
    
    # Test validate command
    if "$script_path" validate "$TEST_INSTANCE" &> /dev/null; then
        log_info "✓ Service management script 'validate' command works"
    else
        log_error "✗ Service management script 'validate' command failed"
        return 1
    fi
    
    return 0
}

# Run all tests
run_all_tests() {
    log_info "Starting LeeQ EPII systemd integration tests..."
    echo ""
    
    run_test "Prerequisites check" test_prerequisites
    run_test "Service installation" test_service_installation
    run_test "Service enable/disable" test_service_enable_disable
    run_test "Service start/stop" test_service_start_stop
    run_test "Service restart" test_service_restart
    run_test "Logging configuration" test_logging
    run_test "Graceful shutdown" test_graceful_shutdown
    run_test "Configuration validation" test_config_validation
    run_test "Service management script" test_service_management_script
}

# Show test results
show_results() {
    echo ""
    echo "=============================="
    echo "Test Results Summary"
    echo "=============================="
    echo ""
    
    log_success "Tests passed: $TESTS_PASSED"
    
    if [[ $TESTS_FAILED -gt 0 ]]; then
        log_error "Tests failed: $TESTS_FAILED"
        echo ""
        log_error "Failed tests:"
        for test in "${FAILED_TESTS[@]}"; do
            echo "  - $test"
        done
        echo ""
        log_info "Check the logs above for details on failures"
        exit 1
    else
        echo ""
        log_success "All tests passed! LeeQ EPII systemd integration is working correctly."
        echo ""
        log_info "You can now use the service management commands:"
        echo "  sudo $SCRIPT_DIR/leeq-epii-service.sh start $TEST_INSTANCE"
        echo "  sudo $SCRIPT_DIR/leeq-epii-service.sh status $TEST_INSTANCE"
        echo "  sudo $SCRIPT_DIR/leeq-epii-service.sh logs $TEST_INSTANCE"
        echo "  sudo $SCRIPT_DIR/leeq-epii-service.sh stop $TEST_INSTANCE"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up test environment..."
    
    # Stop service if running
    systemctl stop "$SERVICE_NAME" 2>/dev/null || true
    
    # Disable service
    systemctl disable "$SERVICE_NAME" 2>/dev/null || true
    
    log_info "Cleanup completed"
}

# Set up signal handlers for cleanup
trap cleanup EXIT INT TERM

# Main execution
main() {
    case "${1:-}" in
        --help|-h)
            echo "Usage: $0 [--help]"
            echo ""
            echo "Test LeeQ EPII systemd integration"
            echo ""
            echo "This script tests:"
            echo "  - Service installation and recognition"
            echo "  - Service enable/disable operations"
            echo "  - Service start/stop/restart operations"
            echo "  - Logging configuration and output"
            echo "  - Graceful shutdown handling"
            echo "  - Configuration validation"
            echo "  - Service management scripts"
            echo ""
            echo "Prerequisites:"
            echo "  - Must be run as root (sudo)"
            echo "  - LeeQ EPII systemd service must be installed"
            echo "  - Test configuration (simulation_2q) must exist"
            exit 0
            ;;
        *)
            check_root
            run_all_tests
            show_results
            ;;
    esac
}

main "$@"