#!/bin/bash
# LeeQ EPII Service Test Script

set -e

# Configuration
CONFIG_NAME=${1:-simulation_2q}
SERVICE_NAME="leeq-epii@$CONFIG_NAME"
TIMEOUT=30
GRPC_PORT=50051

echo "Testing LeeQ EPII service with configuration: $CONFIG_NAME"
echo "========================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check service status
echo "1. Checking service status..."
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "   ✓ Service is running"
    systemctl status "$SERVICE_NAME" --no-pager -l
else
    echo "   ✗ Service is not running"
    echo "   Starting service..."
    sudo systemctl start "$SERVICE_NAME"
    
    # Wait for service to start
    echo "   Waiting for service to start..."
    for i in $(seq 1 $TIMEOUT); do
        if systemctl is-active --quiet "$SERVICE_NAME"; then
            echo "   ✓ Service started successfully"
            break
        fi
        sleep 1
        if [ $i -eq $TIMEOUT ]; then
            echo "   ✗ Service failed to start within $TIMEOUT seconds"
            echo "   Checking logs:"
            sudo journalctl -u "$SERVICE_NAME" --no-pager -l
            exit 1
        fi
    done
fi

echo

# Check port is listening
echo "2. Checking network connectivity..."
if netstat -tlnp | grep -q ":$GRPC_PORT "; then
    echo "   ✓ Port $GRPC_PORT is listening"
else
    echo "   ✗ Port $GRPC_PORT is not listening"
    echo "   Check service logs:"
    sudo journalctl -u "$SERVICE_NAME" --no-pager -l
    exit 1
fi

echo

# Test gRPC endpoint with grpcurl if available
echo "3. Testing gRPC endpoint..."
if command_exists grpcurl; then
    echo "   Testing with grpcurl..."
    
    # Test service listing
    if grpcurl -plaintext -max-time 10 localhost:$GRPC_PORT list >/dev/null 2>&1; then
        echo "   ✓ gRPC service responds to list request"
    else
        echo "   ✗ gRPC service does not respond to list request"
        exit 1
    fi
    
    # Test ping
    if ping_result=$(grpcurl -plaintext -max-time 10 localhost:$GRPC_PORT ExperimentPlatformService/Ping 2>/dev/null); then
        echo "   ✓ Ping successful"
        echo "   Response: $ping_result"
    else
        echo "   ✗ Ping failed"
        exit 1
    fi
else
    echo "   ⚠ grpcurl not available, skipping direct gRPC tests"
    echo "   Install with: sudo apt-get install grpcurl"
fi

echo

# Test with Python client if available
echo "4. Testing with Python client..."
python_test_script=$(cat <<'EOF'
import sys
import grpc
import os

# Add the LeeQ directory to Python path
sys.path.insert(0, '/opt/leeq')

try:
    from leeq.epii.proto import epii_pb2, epii_pb2_grpc
    
    # Connect to service
    channel = grpc.insecure_channel('localhost:50051')
    stub = epii_pb2_grpc.ExperimentPlatformServiceStub(channel)
    
    # Test ping
    response = stub.Ping(epii_pb2.PingRequest(), timeout=10)
    print(f"   ✓ Python client ping successful: {response.message}")
    
    # Test capabilities
    response = stub.GetCapabilities(epii_pb2.Empty(), timeout=10)
    experiments = list(response.experiments)
    print(f"   ✓ Available experiments: {', '.join(experiments)}")
    
    # Test parameter listing
    response = stub.ListParameters(epii_pb2.Empty(), timeout=10)
    param_count = len(response.parameters)
    print(f"   ✓ Found {param_count} parameters")
    
    channel.close()
    print("   ✓ All Python client tests passed")

except ImportError as e:
    print(f"   ✗ Import error: {e}")
    print("   Make sure LeeQ is properly installed")
    sys.exit(1)
except grpc.RpcError as e:
    print(f"   ✗ gRPC error: {e.code()} - {e.details()}")
    sys.exit(1)
except Exception as e:
    print(f"   ✗ Unexpected error: {e}")
    sys.exit(1)
EOF
)

if command_exists python3; then
    if echo "$python_test_script" | python3; then
        echo "   All tests completed successfully"
    else
        echo "   Python client test failed"
        exit 1
    fi
else
    echo "   ✗ Python3 not available"
    exit 1
fi

echo

# Check recent logs for errors
echo "5. Checking recent logs for errors..."
if sudo journalctl -u "$SERVICE_NAME" --since "5 minutes ago" | grep -i error >/dev/null; then
    echo "   ⚠ Found errors in recent logs:"
    sudo journalctl -u "$SERVICE_NAME" --since "5 minutes ago" | grep -i error
else
    echo "   ✓ No errors found in recent logs"
fi

echo

# Performance check
echo "6. Checking service performance..."
echo "   Service memory usage:"
ps aux | grep "$SERVICE_NAME" | grep -v grep | awk '{print "   RSS: " $6 " KB, VSZ: " $5 " KB"}'

echo "   Service uptime:"
systemctl show "$SERVICE_NAME" --property=ActiveEnterTimestamp | sed 's/ActiveEnterTimestamp=/   Started: /'

echo

# Summary
echo "========================================================="
echo "EPII Service Test Summary"
echo "========================================================="
echo "Configuration: $CONFIG_NAME"
echo "Service Status: $(systemctl is-active $SERVICE_NAME)"
echo "Port: $GRPC_PORT"
echo "✓ All tests passed - service is ready for use"
echo

echo "Next steps:"
echo "- Run example clients: cd examples/epii && python simple_client.py"
echo "- View logs: sudo journalctl -u $SERVICE_NAME -f"
echo "- Monitor service: systemctl status $SERVICE_NAME"
echo "- Check documentation: docs/epii/README.md"
echo
echo "To test with different configuration:"
echo "  $0 <config-name>"
echo "  Example: $0 hardware_lab1"