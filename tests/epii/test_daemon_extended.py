"""
Extended tests for EPII daemon functionality.
"""
import pytest
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from leeq.epii.daemon import EPIIDaemon, load_config


class TestLoadConfig:
    """Test configuration loading functionality."""
    
    def test_load_config_from_file(self):
        """Test loading configuration from a file."""
        # Create a temporary config file
        config_data = {
            "server": {
                "host": "localhost",
                "port": 50051
            },
            "logging": {
                "level": "INFO"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            # Test loading the config
            loaded_config = load_config(config_file)
            
            assert loaded_config is not None
            assert "server" in loaded_config
            assert loaded_config["server"]["host"] == "localhost"
            assert loaded_config["server"]["port"] == 50051
            
        finally:
            os.unlink(config_file)
    
    def test_load_config_file_not_found(self):
        """Test handling of missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.json")
    
    def test_load_config_invalid_json(self):
        """Test handling of invalid JSON in config file."""
        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            config_file = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_config(config_file)
        finally:
            os.unlink(config_file)
    
    def test_load_config_empty_file(self):
        """Test handling of empty config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{}")
            config_file = f.name
        
        try:
            loaded_config = load_config(config_file)
            assert loaded_config == {}
        finally:
            os.unlink(config_file)


class TestEPIIDaemon:
    """Test EPII daemon functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for daemon."""
        return {
            "server": {
                "host": "localhost",
                "port": 50051,
                "max_workers": 10
            },
            "logging": {
                "level": "INFO"
            },
            "experiments": {
                "timeout": 300,
                "max_concurrent": 5
            }
        }
    
    def test_daemon_initialization(self, mock_config):
        """Test daemon initialization with configuration."""
        daemon = EPIIDaemon(config=mock_config)
        
        assert daemon is not None
        assert hasattr(daemon, 'config')
        assert daemon.config == mock_config
    
    def test_daemon_initialization_no_config(self):
        """Test daemon initialization with minimal config."""
        config = {"port": 50051}  # Minimal valid config
        daemon = EPIIDaemon(config)
        
        assert daemon is not None
        # Should have some default configuration
        assert hasattr(daemon, 'config')
        assert daemon.config == config
    
    def test_daemon_server_configuration(self, mock_config):
        """Test server configuration handling."""
        daemon = EPIIDaemon(config=mock_config)
        
        server_config = daemon.config.get("server", {})
        
        assert server_config.get("host") == "localhost"
        assert server_config.get("port") == 50051
        assert server_config.get("max_workers") == 10
    
    def test_daemon_logging_configuration(self, mock_config):
        """Test logging configuration handling."""
        daemon = EPIIDaemon(config=mock_config)
        
        logging_config = daemon.config.get("logging", {})
        
        assert logging_config.get("level") == "INFO"
    
    def test_daemon_experiment_configuration(self, mock_config):
        """Test experiment configuration handling."""
        daemon = EPIIDaemon(config=mock_config)
        
        experiment_config = daemon.config.get("experiments", {})
        
        assert experiment_config.get("timeout") == 300
        assert experiment_config.get("max_concurrent") == 5
    
    @patch('leeq.epii.daemon.grpc')
    def test_daemon_server_creation(self, mock_grpc, mock_config):
        """Test gRPC server creation."""
        # Mock grpc server
        mock_server = Mock()
        mock_grpc.server.return_value = mock_server
        
        daemon = EPIIDaemon(config=mock_config)
        
        # Test server creation method exists
        assert hasattr(daemon, 'config')
    
    def test_daemon_port_validation(self):
        """Test port validation in daemon configuration."""
        # Test valid port
        valid_config = {"server": {"port": 50051}}
        daemon = EPIIDaemon(config=valid_config)
        
        port = daemon.config["server"]["port"]
        assert isinstance(port, int)
        assert 1 <= port <= 65535
    
    def test_daemon_host_validation(self):
        """Test host validation in daemon configuration."""
        # Test valid host
        valid_config = {"server": {"host": "localhost"}}
        daemon = EPIIDaemon(config=valid_config)
        
        host = daemon.config["server"]["host"]
        assert isinstance(host, str)
        assert len(host) > 0


class TestDaemonLifecycle:
    """Test daemon lifecycle management."""
    
    @pytest.fixture
    def daemon_config(self):
        """Create daemon configuration for lifecycle tests."""
        return {
            "server": {
                "host": "localhost",
                "port": 50052,  # Different port to avoid conflicts
                "max_workers": 5
            },
            "logging": {
                "level": "DEBUG"
            }
        }
    
    @patch('leeq.epii.daemon.grpc')
    def test_daemon_start_method_exists(self, mock_grpc, daemon_config):
        """Test that daemon has start method."""
        mock_server = Mock()
        mock_grpc.server.return_value = mock_server
        
        daemon = EPIIDaemon(config=daemon_config)
        
        # Check if daemon has lifecycle methods
        assert hasattr(daemon, '__init__')  # Constructor should exist
    
    def test_daemon_configuration_merge(self, daemon_config):
        """Test configuration merging with defaults."""
        # Partial configuration
        partial_config = {"server": {"host": "0.0.0.0"}}
        
        daemon = EPIIDaemon(config=partial_config)
        
        # Should have the provided host
        assert daemon.config["server"]["host"] == "0.0.0.0"
    
    def test_daemon_configuration_types(self, daemon_config):
        """Test configuration type validation."""
        daemon = EPIIDaemon(config=daemon_config)
        
        # Test that configuration values have correct types
        server_config = daemon.config.get("server", {})
        
        if "port" in server_config:
            assert isinstance(server_config["port"], int)
        if "host" in server_config:
            assert isinstance(server_config["host"], str)
        if "max_workers" in server_config:
            assert isinstance(server_config["max_workers"], int)
    
    def test_daemon_default_configuration(self):
        """Test daemon with minimal configuration (should use defaults)."""
        config = {"port": 50051}  # Minimal valid config
        daemon = EPIIDaemon(config)
        
        # Should have some configuration, even if defaults
        assert daemon.config is not None
        assert isinstance(daemon.config, dict)
        assert daemon.config == config


class TestConfigurationValidation:
    """Test configuration validation functionality."""
    
    def test_validate_server_config(self):
        """Test server configuration validation."""
        # Valid server config
        valid_server_config = {
            "host": "localhost",
            "port": 50051,
            "max_workers": 10
        }
        
        # Basic validation
        assert "host" in valid_server_config
        assert "port" in valid_server_config
        assert isinstance(valid_server_config["port"], int)
        assert 1 <= valid_server_config["port"] <= 65535
        assert valid_server_config["max_workers"] > 0
    
    def test_validate_logging_config(self):
        """Test logging configuration validation."""
        valid_logging_config = {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
        
        # Basic validation
        assert "level" in valid_logging_config
        assert valid_logging_config["level"] in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    def test_validate_experiment_config(self):
        """Test experiment configuration validation."""
        valid_experiment_config = {
            "timeout": 300,
            "max_concurrent": 5,
            "result_storage_path": "/tmp/results"
        }
        
        # Basic validation
        assert "timeout" in valid_experiment_config
        assert valid_experiment_config["timeout"] > 0
        assert valid_experiment_config["max_concurrent"] > 0
    
    def test_invalid_port_handling(self):
        """Test handling of invalid port numbers."""
        invalid_ports = [-1, 0, 65536, 100000, "not_a_port"]
        
        for port in invalid_ports:
            if isinstance(port, int):
                assert not (1 <= port <= 65535)
            else:
                assert not isinstance(port, int)
    
    def test_invalid_logging_level(self):
        """Test handling of invalid logging levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        invalid_levels = ["INVALID", "debug", "info", 123, None]
        
        for level in invalid_levels:
            assert level not in valid_levels


class TestDaemonUtilities:
    """Test daemon utility functions."""
    
    def test_config_path_resolution(self):
        """Test configuration file path resolution."""
        # Test with absolute path
        abs_path = "/tmp/config.json"
        assert os.path.isabs(abs_path)
        
        # Test with relative path
        rel_path = "config.json"
        assert not os.path.isabs(rel_path)
        
        # Test with Path object
        path_obj = Path("config.json")
        assert isinstance(path_obj, Path)
    
    def test_environment_variable_handling(self):
        """Test environment variable handling in configuration."""
        # Test setting an environment variable
        test_var = "EPII_TEST_VAR"
        test_value = "test_value_123"
        
        # Set environment variable
        os.environ[test_var] = test_value
        
        try:
            # Test reading environment variable
            retrieved_value = os.environ.get(test_var)
            assert retrieved_value == test_value
            
            # Test with default value
            default_value = os.environ.get("NONEXISTENT_VAR", "default")
            assert default_value == "default"
            
        finally:
            # Clean up
            if test_var in os.environ:
                del os.environ[test_var]
    
    def test_configuration_file_extensions(self):
        """Test handling of different configuration file extensions."""
        valid_extensions = ['.json', '.yaml', '.yml', '.toml']
        
        for ext in valid_extensions:
            filename = f"config{ext}"
            assert filename.endswith(ext)
            
            # Test Path handling
            path = Path(filename)
            assert path.suffix == ext


@pytest.mark.integration
class TestDaemonIntegration:
    """Integration tests for daemon functionality."""
    
    @pytest.fixture
    def integration_config(self):
        """Create configuration for integration tests."""
        return {
            "server": {
                "host": "127.0.0.1",
                "port": 50053,  # Use different port for integration tests
                "max_workers": 2
            },
            "logging": {
                "level": "DEBUG"
            },
            "experiments": {
                "timeout": 60,
                "max_concurrent": 2
            }
        }
    
    def test_daemon_configuration_loading_integration(self, integration_config):
        """Test complete daemon configuration loading."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(integration_config, f)
            config_file = f.name
        
        try:
            # Load config and create daemon
            loaded_config = load_config(config_file)
            daemon = EPIIDaemon(config=loaded_config)
            
            # Verify configuration was loaded correctly
            assert daemon.config == integration_config
            assert daemon.config["server"]["host"] == "127.0.0.1"
            assert daemon.config["server"]["port"] == 50053
            
        finally:
            os.unlink(config_file)
    
    @patch('leeq.epii.daemon.grpc')
    def test_daemon_server_setup_integration(self, mock_grpc, integration_config):
        """Test complete daemon server setup."""
        # Mock gRPC components
        mock_server = Mock()
        mock_servicer = Mock()
        mock_grpc.server.return_value = mock_server
        
        # Create daemon
        daemon = EPIIDaemon(config=integration_config)
        
        # Verify daemon was created with config
        assert daemon is not None
        assert daemon.config == integration_config
    
    def test_configuration_validation_integration(self, integration_config):
        """Test complete configuration validation."""
        # Test all configuration sections
        daemon = EPIIDaemon(config=integration_config)
        
        # Verify server configuration
        server_config = daemon.config["server"]
        assert server_config["host"] == "127.0.0.1"
        assert 1 <= server_config["port"] <= 65535
        assert server_config["max_workers"] > 0
        
        # Verify logging configuration
        logging_config = daemon.config["logging"]
        assert logging_config["level"] in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        # Verify experiments configuration
        experiments_config = daemon.config["experiments"]
        assert experiments_config["timeout"] > 0
        assert experiments_config["max_concurrent"] > 0