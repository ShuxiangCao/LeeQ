"""Unit tests for EPII daemon process management"""

import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
import os
import signal
import tempfile
import json
import time
from pathlib import Path
import psutil
import shutil

from leeq.epii.daemon import (
    PIDFileManager, HealthChecker, EPIIDaemon,
    load_config, validate_config, setup_logging, main
)


class TestPIDFileManager:
    """Test suite for PID file management"""
    
    def test_init(self, temp_pid_file):
        """Test PID file manager initialization"""
        manager = PIDFileManager(Path(temp_pid_file))
        assert manager.pid_file == Path(temp_pid_file)
        assert manager.pid == os.getpid()
    
    def test_create_new_pid_file(self, temp_pid_file):
        """Test creating a new PID file"""
        manager = PIDFileManager(Path(temp_pid_file))
        
        with patch('atexit.register'):
            result = manager.create()
        
        assert result is True
        assert Path(temp_pid_file).exists()
        
        with open(temp_pid_file, 'r') as f:
            assert int(f.read().strip()) == os.getpid()
    
    def test_create_existing_running_process(self, temp_pid_file):
        """Test handling existing PID file with running process"""
        # Create PID file with current process ID
        with open(temp_pid_file, 'w') as f:
            f.write(str(os.getpid()))
        
        manager = PIDFileManager(Path(temp_pid_file))
        result = manager.create()
        
        assert result is False  # Should refuse to create since process is running
    
    def test_create_stale_pid_file(self, temp_pid_file):
        """Test handling stale PID file with non-existent process"""
        # Create PID file with non-existent process ID
        fake_pid = 999999
        with open(temp_pid_file, 'w') as f:
            f.write(str(fake_pid))
        
        manager = PIDFileManager(Path(temp_pid_file))
        
        with patch('atexit.register'):
            result = manager.create()
        
        assert result is True
        # Should have created new PID file with current PID
        with open(temp_pid_file, 'r') as f:
            assert int(f.read().strip()) == os.getpid()
    
    def test_create_invalid_pid_file(self, temp_pid_file):
        """Test handling invalid PID file content"""
        # Create PID file with invalid content
        with open(temp_pid_file, 'w') as f:
            f.write("invalid_pid")
        
        manager = PIDFileManager(Path(temp_pid_file))
        
        with patch('atexit.register'):
            result = manager.create()
        
        assert result is True
        # Should have created new PID file with current PID
        with open(temp_pid_file, 'r') as f:
            assert int(f.read().strip()) == os.getpid()
    
    def test_cleanup_own_pid(self, temp_pid_file):
        """Test cleaning up own PID file"""
        manager = PIDFileManager(Path(temp_pid_file))
        
        # Create PID file
        with open(temp_pid_file, 'w') as f:
            f.write(str(manager.pid))
        
        manager.cleanup()
        assert not Path(temp_pid_file).exists()
    
    def test_cleanup_different_pid(self, temp_pid_file):
        """Test not cleaning up PID file with different PID"""
        manager = PIDFileManager(Path(temp_pid_file))
        
        # Create PID file with different PID
        with open(temp_pid_file, 'w') as f:
            f.write(str(manager.pid + 1))
        
        manager.cleanup()
        assert Path(temp_pid_file).exists()  # Should not be removed
    
    def test_cleanup_invalid_pid_file(self, temp_pid_file):
        """Test cleanup with invalid PID file"""
        manager = PIDFileManager(Path(temp_pid_file))
        
        # Create invalid PID file
        with open(temp_pid_file, 'w') as f:
            f.write("invalid")
        
        # Should not raise exception
        manager.cleanup()


class TestHealthChecker:
    """Test suite for health checker"""
    
    def test_init_without_service(self):
        """Test health checker initialization without service"""
        checker = HealthChecker()
        assert checker.service is None
        assert isinstance(checker.start_time, float)
    
    def test_init_with_service(self):
        """Test health checker initialization with service"""
        mock_service = Mock()
        checker = HealthChecker(mock_service)
        assert checker.service is mock_service
    
    @patch('leeq.epii.daemon.validate_config')
    def test_check_startup_health_all_pass(self, mock_validate, simulation_2q_config):
        """Test startup health checks when all pass"""
        mock_validate.return_value = True
        checker = HealthChecker()
        
        with patch.object(checker, '_check_port_available', return_value=True), \
             patch.object(checker, '_check_disk_space', return_value=True), \
             patch.object(checker, '_check_memory', return_value=True):
            
            result = checker.check_startup_health(simulation_2q_config)
        
        assert result is True
        mock_validate.assert_called_once_with(simulation_2q_config)
    
    @patch('leeq.epii.daemon.validate_config')
    def test_check_startup_health_config_fail(self, mock_validate, simulation_2q_config):
        """Test startup health checks when config validation fails"""
        mock_validate.return_value = False
        checker = HealthChecker()
        
        with patch.object(checker, '_check_port_available', return_value=True), \
             patch.object(checker, '_check_disk_space', return_value=True), \
             patch.object(checker, '_check_memory', return_value=True):
            
            result = checker.check_startup_health(simulation_2q_config)
        
        assert result is False
    
    def test_check_port_available_free_port(self):
        """Test port availability check with free port"""
        checker = HealthChecker()
        # Use a high port number that's unlikely to be in use
        result = checker._check_port_available(59999)
        assert result is True
    
    def test_check_port_available_used_port(self):
        """Test port availability check with used port"""
        import socket
        checker = HealthChecker()
        
        # Bind to a port first
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))  # Let OS choose a port
            port = s.getsockname()[1]
            
            # Now check if the same port is available (should be False)
            result = checker._check_port_available(port)
            assert result is False
    
    def test_check_disk_space_sufficient(self):
        """Test disk space check with sufficient space"""
        checker = HealthChecker()
        # Check with very low threshold
        result = checker._check_disk_space(min_free_mb=1)
        assert result is True
    
    def test_check_disk_space_insufficient(self):
        """Test disk space check with insufficient space"""
        checker = HealthChecker()
        # Check with unreasonably high threshold
        result = checker._check_disk_space(min_free_mb=999999999)
        assert result is False
    
    def test_check_memory_sufficient(self):
        """Test memory check with sufficient memory"""
        checker = HealthChecker()
        # Check with very low threshold
        result = checker._check_memory(min_free_mb=1)
        assert result is True
    
    @pytest.mark.skip(reason="psutil is available in test environment")
    def test_check_memory_no_psutil(self):
        """Test memory check when psutil is not available"""
        # This test would require removing psutil from the environment
        # which is complex in the current test setup
        pass
    
    def test_runtime_health_healthy(self):
        """Test runtime health check when healthy"""
        mock_service = Mock()
        checker = HealthChecker(mock_service)
        
        with patch.object(checker, '_check_disk_space', return_value=True), \
             patch.object(checker, '_check_memory', return_value=True):
            
            health = checker.check_runtime_health()
        
        assert health['status'] == 'healthy'
        assert 'uptime_seconds' in health
        assert health['checks']['service_available'] == 'pass'
        assert health['checks']['disk_space'] == 'pass'
        assert health['checks']['memory'] == 'pass'
    
    def test_runtime_health_degraded(self):
        """Test runtime health check when degraded"""
        checker = HealthChecker(None)  # No service
        
        with patch.object(checker, '_check_disk_space', return_value=False), \
             patch.object(checker, '_check_memory', return_value=True):
            
            health = checker.check_runtime_health()
        
        assert health['status'] == 'degraded'
        assert health['checks']['service_available'] == 'fail'
        assert health['checks']['disk_space'] == 'fail'


class TestEPIIDaemon:
    """Test suite for EPII daemon"""
    
    def test_init(self, simulation_2q_config, temp_pid_file):
        """Test daemon initialization"""
        daemon = EPIIDaemon(simulation_2q_config, port=50052, pid_file=temp_pid_file)
        
        assert daemon.config == simulation_2q_config
        assert daemon.port == 50052
        assert daemon.server is None
        assert daemon.service is None
        assert daemon._stop_requested is False
        assert daemon.pid_manager.pid_file == Path(temp_pid_file)
    
    def test_init_default_pid_file(self, simulation_2q_config):
        """Test daemon initialization with default PID file"""
        daemon = EPIIDaemon(simulation_2q_config, port=50052)
        
        expected_pid_file = Path("leeq-epii-50052.pid")
        assert daemon.pid_manager.pid_file == expected_pid_file
    
    def test_load_setup_simulation(self, simulation_2q_config):
        """Test setup loading in simulation mode"""
        daemon = EPIIDaemon(simulation_2q_config)
        setup = daemon._load_setup()
        
        # Should return None for simulation mode
        assert setup is None
    
    def test_load_setup_hardware_not_implemented(self):
        """Test setup loading for hardware mode (not yet implemented)"""
        hardware_config = {
            "setup_type": "hardware",
            "setup_name": "test_hardware"
        }
        
        daemon = EPIIDaemon(hardware_config)
        setup = daemon._load_setup()
        
        # Should return None for unimplemented hardware mode
        assert setup is None
    
    def test_signal_handler(self, simulation_2q_config):
        """Test signal handler sets stop flag"""
        daemon = EPIIDaemon(simulation_2q_config)
        
        daemon._signal_handler(signal.SIGTERM, None)
        assert daemon._stop_requested is True
    
    def test_health_check_handler(self, simulation_2q_config, capsys):
        """Test health check signal handler"""
        daemon = EPIIDaemon(simulation_2q_config)
        
        with patch.object(daemon, 'get_health_status', return_value={'status': 'healthy'}):
            daemon._health_check_handler(signal.SIGUSR1, None)
        
        captured = capsys.readouterr()
        assert 'Health Status' in captured.out
    
    def test_get_health_status(self, simulation_2q_config):
        """Test getting health status"""
        daemon = EPIIDaemon(simulation_2q_config)
        
        with patch.object(daemon.health_checker, 'check_runtime_health', return_value={'status': 'healthy'}):
            status = daemon.get_health_status()
        
        assert status['status'] == 'healthy'
    
    @pytest.mark.skip(reason="Performance test: daemon.start() contains infinite loop waiting for termination")
    @patch('grpc.server')
    @patch('leeq.epii.service.ExperimentPlatformService')
    def test_start_success(self, mock_service, mock_grpc_server, simulation_2q_config, temp_pid_file):
        """Test successful daemon start"""
        mock_server = Mock()
        mock_grpc_server.return_value = mock_server
        
        # Configure the simulation config to skip health checks for easier testing
        test_config = simulation_2q_config.copy()
        test_config['skip_health_checks'] = True
        
        daemon = EPIIDaemon(test_config, pid_file=temp_pid_file)
        
        with patch.object(daemon.pid_manager, 'create', return_value=True), \
             patch('signal.signal'), \
             patch('atexit.register'):
            
            daemon.start()
        
        assert daemon.server == mock_server
        mock_server.add_insecure_port.assert_called_once_with(f'[::]:{daemon.port}')
        mock_server.start.assert_called_once()
    
    @pytest.mark.skip(reason="Performance test: daemon.start() may hang due to infinite loop")
    def test_start_pid_file_failure(self, simulation_2q_config, temp_pid_file):
        """Test daemon start failure due to PID file"""
        # Skip health checks for easier testing
        test_config = simulation_2q_config.copy()
        test_config['skip_health_checks'] = True
        
        daemon = EPIIDaemon(test_config, pid_file=temp_pid_file)
        
        with patch.object(daemon.pid_manager, 'create', return_value=False), \
             patch('sys.exit') as mock_exit:
            
            daemon.start()
            mock_exit.assert_called_once_with(1)
    
    @pytest.mark.skip(reason="Performance test: daemon.start() may hang due to infinite loop")
    def test_start_validation_failure(self, simulation_2q_config, temp_pid_file):
        """Test daemon start failure due to validation"""
        daemon = EPIIDaemon(simulation_2q_config, pid_file=temp_pid_file)
        
        with patch.object(daemon.health_checker, 'check_startup_health', return_value=False), \
             patch('sys.exit') as mock_exit:
            
            daemon.start()
            mock_exit.assert_called_once_with(1)
    
    def test_validate_startup(self, simulation_2q_config):
        """Test startup validation"""
        daemon = EPIIDaemon(simulation_2q_config)
        
        with patch.object(daemon.health_checker, 'check_startup_health', return_value=True):
            result = daemon._validate_startup()
        
        assert result is True
    
    def test_stop_without_server(self, simulation_2q_config):
        """Test stopping daemon without server"""
        daemon = EPIIDaemon(simulation_2q_config)
        daemon.stop()  # Should not raise exception
    
    def test_stop_with_server(self, simulation_2q_config):
        """Test stopping daemon with server"""
        daemon = EPIIDaemon(simulation_2q_config)
        
        mock_server = Mock()
        daemon.server = mock_server
        
        daemon.stop(grace=1.0)
        
        mock_server.stop.assert_called_once_with(grace=1.0)


class TestDaemonFunctions:
    """Test suite for daemon utility functions"""
    
    def test_load_config_valid_file(self, minimal_config_file):
        """Test loading valid configuration file"""
        config = load_config(minimal_config_file)
        
        assert config['setup_type'] == 'simulation'
        assert config['setup_name'] == 'test_setup'
        assert config['parameters']['num_qubits'] == 2
    
    def test_load_config_nonexistent_file(self):
        """Test loading non-existent configuration file"""
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/config.json')
    
    def test_load_config_invalid_json(self, tmp_path):
        """Test loading invalid JSON configuration"""
        config_path = tmp_path / "invalid.json"
        with open(config_path, 'w') as f:
            f.write("invalid json {")
        
        with pytest.raises(json.JSONDecodeError):
            load_config(str(config_path))
    
    def test_validate_config_valid(self, simulation_2q_config):
        """Test validating valid configuration"""
        result = validate_config(simulation_2q_config)
        assert result is True
    
    def test_validate_config_missing_required_field(self):
        """Test validating configuration with missing required field"""
        config = {
            "setup_name": "test",
            "parameters": {}
        }
        
        result = validate_config(config)
        assert result is False
    
    def test_validate_config_invalid_setup_type(self):
        """Test validating configuration with invalid setup type"""
        config = {
            "setup_type": "invalid_type",
            "setup_name": "test",
            "parameters": {}
        }
        
        result = validate_config(config)
        assert result is False
    
    @patch('logging.basicConfig')
    def test_setup_logging_basic(self, mock_basic_config):
        """Test basic logging setup"""
        setup_logging(log_level="DEBUG")
        mock_basic_config.assert_called_once()
    
    @patch('logging.basicConfig')
    @patch('leeq.epii.daemon.logging.handlers')
    def test_setup_logging_systemd(self, mock_handlers, mock_basic_config):
        """Test systemd logging setup"""
        mock_systemd_handler = Mock()
        mock_handlers.SysLogHandler.return_value = mock_systemd_handler
        
        setup_logging(use_systemd=True)
        mock_basic_config.assert_called_once()
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('leeq.epii.daemon.load_config')
    @patch('leeq.epii.daemon.EPIIDaemon')
    def test_main_function(self, mock_daemon_class, mock_load_config, mock_parse_args, simulation_2q_config):
        """Test main function execution"""
        # Mock command line arguments
        mock_args = Mock()
        mock_args.config = 'test_config.json'
        mock_args.port = 50051
        mock_args.pid_file = None
        mock_args.log_level = 'INFO'
        mock_args.systemd = False
        mock_args.validate = False
        mock_parse_args.return_value = mock_args
        
        # Mock config loading
        mock_load_config.return_value = simulation_2q_config
        
        # Mock daemon
        mock_daemon = Mock()
        mock_daemon_class.return_value = mock_daemon
        
        # Run main function
        main()
        
        # Verify calls
        mock_load_config.assert_called_once_with('test_config.json')
        mock_daemon_class.assert_called_once_with(simulation_2q_config, port=50051, pid_file=None)
        mock_daemon.start.assert_called_once()
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('leeq.epii.daemon.load_config')
    @patch('leeq.epii.daemon.validate_config')
    def test_main_function_validate_only(self, mock_validate_config, mock_load_config, mock_parse_args, simulation_2q_config):
        """Test main function with validation only"""
        # Mock command line arguments for validation
        mock_args = Mock()
        mock_args.config = 'test_config.json'
        mock_args.validate = True
        mock_parse_args.return_value = mock_args
        
        # Mock config loading and validation
        mock_load_config.return_value = simulation_2q_config
        mock_validate_config.return_value = True
        
        # Run main function
        with patch('sys.exit') as mock_exit:
            main()
        
        # Should exit with code 0 for valid config
        mock_exit.assert_called_once_with(0)
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('leeq.epii.daemon.load_config')
    @patch('leeq.epii.daemon.validate_config')
    def test_main_function_validate_invalid(self, mock_validate_config, mock_load_config, mock_parse_args, simulation_2q_config):
        """Test main function with invalid configuration"""
        # Mock command line arguments for validation
        mock_args = Mock()
        mock_args.config = 'test_config.json'
        mock_args.validate = True
        mock_parse_args.return_value = mock_args
        
        # Mock config loading and validation failure
        mock_load_config.return_value = simulation_2q_config
        mock_validate_config.return_value = False
        
        # Run main function
        with patch('sys.exit') as mock_exit:
            main()
        
        # Should exit with code 1 for invalid config
        mock_exit.assert_called_once_with(1)