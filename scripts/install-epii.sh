#!/bin/bash
# LeeQ EPII Installation Script

set -e

# Default values
INSTALL_DIR="/opt/leeq"
USER="leeq"
CONFIG_NAME="production"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --user)
            USER="$2"
            shift 2
            ;;
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --install-dir DIR    Installation directory (default: /opt/leeq)"
            echo "  --user USER          Service user (default: leeq)"
            echo "  --config NAME        Configuration name (default: production)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Installing LeeQ EPII service..."
echo "Installation directory: $INSTALL_DIR"
echo "Service user: $USER"
echo "Configuration: $CONFIG_NAME"
echo

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

# Create user if it doesn't exist
if ! id "$USER" &>/dev/null; then
    echo "Creating user: $USER"
    useradd -r -s /bin/bash -d "$INSTALL_DIR" "$USER"
fi

# Create directories
echo "Creating directories..."
mkdir -p "$INSTALL_DIR"
mkdir -p /var/log/leeq
mkdir -p /var/lib/leeq
mkdir -p /etc/leeq/configs/epii

# Set ownership
chown -R "$USER:$USER" "$INSTALL_DIR" /var/log/leeq /var/lib/leeq
chown -R "$USER:$USER" /etc/leeq

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y python3 python3-venv python3-pip git

# Clone repository (if not already present)
if [[ ! -d "$INSTALL_DIR/.git" ]]; then
    echo "Cloning LeeQ repository..."
    sudo -u "$USER" git clone https://github.com/ShuxiangCao/LeeQ.git "$INSTALL_DIR"
fi

# Setup Python environment
echo "Setting up Python environment..."
cd "$INSTALL_DIR"
sudo -u "$USER" python3 -m venv venv
sudo -u "$USER" ./venv/bin/pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
sudo -u "$USER" ./venv/bin/pip install -r requirements.txt
sudo -u "$USER" ./venv/bin/pip install -e .

# Copy configuration files
echo "Setting up configuration..."
cp configs/epii/*.json /etc/leeq/configs/epii/

# Install systemd service
echo "Installing systemd service..."
cp scripts/systemd/leeq-epii@.service /etc/systemd/system/
systemctl daemon-reload

# Validate configuration
echo "Validating configuration..."
if [[ -f "/etc/leeq/configs/epii/$CONFIG_NAME.json" ]]; then
    sudo -u "$USER" "$INSTALL_DIR/venv/bin/python" -m leeq.epii.daemon \
        --config "/etc/leeq/configs/epii/$CONFIG_NAME.json" --validate
    echo "Configuration validation passed"
else
    echo "Warning: Configuration file $CONFIG_NAME.json not found"
    echo "Available configurations:"
    ls -1 /etc/leeq/configs/epii/*.json | xargs -n1 basename
fi

# Setup log rotation
echo "Setting up log rotation..."
cat > /etc/logrotate.d/leeq-epii << EOF
/var/log/leeq/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    sharedscripts
    postrotate
        systemctl reload-or-restart leeq-epii@* > /dev/null 2>&1 || true
    endscript
}
EOF

# Enable and start service (if config exists)
if [[ -f "/etc/leeq/configs/epii/$CONFIG_NAME.json" ]]; then
    echo "Enabling and starting service..."
    systemctl enable "leeq-epii@$CONFIG_NAME"
    systemctl start "leeq-epii@$CONFIG_NAME"
    
    # Wait a moment and check status
    sleep 3
    if systemctl is-active --quiet "leeq-epii@$CONFIG_NAME"; then
        echo "✓ Service started successfully"
        echo "✓ Service status: $(systemctl is-active leeq-epii@$CONFIG_NAME)"
    else
        echo "⚠ Service failed to start"
        echo "Check logs: journalctl -u leeq-epii@$CONFIG_NAME"
    fi
else
    echo "Service not started (no configuration file)"
fi

echo
echo "Installation complete!"
echo
echo "Next steps:"
echo "1. Edit configuration: /etc/leeq/configs/epii/$CONFIG_NAME.json"
echo "2. Start service: systemctl start leeq-epii@$CONFIG_NAME"
echo "3. Check status: systemctl status leeq-epii@$CONFIG_NAME"
echo "4. View logs: journalctl -u leeq-epii@$CONFIG_NAME -f"
echo
echo "Documentation: $INSTALL_DIR/docs/epii/"
echo "Example clients: $INSTALL_DIR/examples/epii/"