#!/usr/bin/env python
"""
Chronicle Viewer Troubleshooting Examples

This script demonstrates how to handle common issues when using
the Chronicle session viewer, including port conflicts, error handling,
and different launch patterns.
"""

import sys
import time
import socket
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from leeq.chronicle import Chronicle


def check_port_available(port):
    """Check if a port is available for use."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('', port))
        sock.close()
        return True
    except OSError:
        return False


def example_1_handle_port_conflicts():
    """Example 1: Handling port conflicts gracefully."""
    print("\n" + "=" * 60)
    print("Example 1: Handling Port Conflicts")
    print("=" * 60)
    
    chronicle = Chronicle()
    
    # Try default port first
    default_port = 8051
    
    if check_port_available(default_port):
        print(f"Port {default_port} is available")
        try:
            print(f"Launching viewer on port {default_port}...")
            # Note: In real usage, this would block. Using threading for demo
            import threading
            t = threading.Thread(
                target=chronicle.launch_viewer,
                kwargs={'port': default_port, 'debug': False},
                daemon=True
            )
            t.start()
            time.sleep(2)
            print(f"✓ Viewer launched at http://localhost:{default_port}")
        except Exception as e:
            print(f"✗ Failed to launch: {e}")
    else:
        print(f"Port {default_port} is in use")
        
        # Try alternative ports
        alternative_ports = [8052, 8053, 8054, 8055]
        for port in alternative_ports:
            if check_port_available(port):
                print(f"Trying alternative port {port}...")
                try:
                    import threading
                    t = threading.Thread(
                        target=chronicle.launch_viewer,
                        kwargs={'port': port, 'debug': False},
                        daemon=True
                    )
                    t.start()
                    time.sleep(2)
                    print(f"✓ Viewer launched at http://localhost:{port}")
                    break
                except Exception as e:
                    print(f"✗ Port {port} also failed: {e}")
        else:
            print("✗ No available ports found. Please free up a port.")


def example_2_check_chronicle_status():
    """Example 2: Checking Chronicle status before launching."""
    print("\n" + "=" * 60)
    print("Example 2: Checking Chronicle Status")
    print("=" * 60)
    
    chronicle = Chronicle()
    
    # Check if Chronicle is properly initialized
    print("Checking Chronicle status...")
    
    # Check if chronicle has necessary attributes
    checks = {
        "Has launch_viewer method": hasattr(chronicle, 'launch_viewer'),
        "Is singleton instance": Chronicle() is chronicle,
        "Has record book": hasattr(chronicle, '_record_book'),
    }
    
    for check, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check}: {result}")
    
    # Check if there's an active session
    try:
        # Try to get current session (this is hypothetical - actual method may vary)
        if hasattr(chronicle, 'get_current_session_entries'):
            entries = chronicle.get_current_session_entries()
            print(f"  ✓ Active session found with {len(entries)} entries")
        else:
            print("  ℹ Session check method not available")
    except Exception as e:
        print(f"  ✗ Error checking session: {e}")
    
    # Launch viewer only if everything looks good
    if all(checks.values()):
        print("\n✓ Chronicle is ready. Viewer can be launched.")
        print("  Use: chronicle.launch_viewer()")
    else:
        print("\n✗ Chronicle may not be properly initialized.")


def example_3_different_launch_patterns():
    """Example 3: Different ways to launch the viewer."""
    print("\n" + "=" * 60)
    print("Example 3: Different Launch Patterns")
    print("=" * 60)
    
    chronicle = Chronicle()
    
    print("1. Basic launch (blocking):")
    print("   chronicle.launch_viewer()")
    print("   # This will block the script until viewer is closed")
    
    print("\n2. Launch with custom port:")
    print("   chronicle.launch_viewer(port=8055)")
    
    print("\n3. Launch in production mode:")
    print("   chronicle.launch_viewer(debug=False)")
    
    print("\n4. Launch allowing external connections:")
    print("   chronicle.launch_viewer(host='0.0.0.0')")
    
    print("\n5. Launch in background thread (non-blocking):")
    print("""
   import threading
   viewer_thread = threading.Thread(
       target=chronicle.launch_viewer,
       kwargs={'port': 8051, 'debug': False},
       daemon=True
   )
   viewer_thread.start()
   """)
    
    print("\n6. Launch with error handling:")
    print("""
   try:
       chronicle.launch_viewer(port=8051)
   except OSError as e:
       if 'Address already in use' in str(e):
           print('Port in use, trying alternative...')
           chronicle.launch_viewer(port=8052)
   except Exception as e:
       print(f'Unexpected error: {e}')
   """)


def example_4_debug_no_experiments():
    """Example 4: Debugging when no experiments appear."""
    print("\n" + "=" * 60)
    print("Example 4: Debugging No Experiments Issue")
    print("=" * 60)
    
    chronicle = Chronicle()
    
    print("Common causes when viewer shows no experiments:\n")
    
    print("1. Check Chronicle logging is enabled:")
    import os
    chronicle_logging = os.environ.get('CHRONICLE_LOGGING', 'Not set')
    print(f"   CHRONICLE_LOGGING = {chronicle_logging}")
    if chronicle_logging != 'True':
        print("   ✗ Set environment variable: export CHRONICLE_LOGGING=True")
    else:
        print("   ✓ Chronicle logging is enabled")
    
    print("\n2. Check if Chronicle has a log path:")
    try:
        from leeq.chronicle.utils import get_log_path
        log_path = get_log_path()
        print(f"   Log path: {log_path}")
        if Path(log_path).exists():
            print("   ✓ Log directory exists")
        else:
            print("   ✗ Log directory does not exist")
    except Exception as e:
        print(f"   ✗ Could not check log path: {e}")
    
    print("\n3. Check if experiments are being recorded:")
    print("   Run a test experiment and check if it appears")
    print("   Example:")
    print("""
   from leeq.experiments.builtin import QubitSpectroscopy
   exp = QubitSpectroscopy(...)  # Should appear in viewer
   """)
    
    print("\n4. Check browser console for JavaScript errors:")
    print("   Open Developer Tools (F12) → Console tab")
    print("   Look for red error messages")
    
    print("\n5. Try manual refresh:")
    print("   Click the 'Refresh' button in the viewer")
    print("   Or wait 5 seconds for auto-refresh")


def example_5_performance_optimization():
    """Example 5: Optimizing viewer performance."""
    print("\n" + "=" * 60)
    print("Example 5: Performance Optimization")
    print("=" * 60)
    
    print("Tips for better viewer performance:\n")
    
    print("1. Use production mode for better performance:")
    print("   chronicle.launch_viewer(debug=False)")
    
    print("\n2. If viewer is slow with many experiments:")
    print("   - Collapse unused tree branches")
    print("   - Consider starting a new session")
    print("   - Increase polling interval (requires code modification)")
    
    print("\n3. For remote access over network:")
    print("   chronicle.launch_viewer(host='0.0.0.0', debug=False)")
    print("   # Note: Ensure firewall allows the port")
    
    print("\n4. Monitor memory usage:")
    print("   - Check browser memory in Task Manager")
    print("   - Restart viewer if memory grows too large")
    
    print("\n5. Use appropriate hardware:")
    print("   - Modern browser (Chrome, Firefox, Edge)")
    print("   - Sufficient RAM for large datasets")
    print("   - Good network connection if remote")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Chronicle Viewer Troubleshooting Guide")
    print("=" * 60)
    
    examples = [
        ("Port Conflict Handling", example_1_handle_port_conflicts),
        ("Chronicle Status Check", example_2_check_chronicle_status),
        ("Launch Patterns", example_3_different_launch_patterns),
        ("Debug No Experiments", example_4_debug_no_experiments),
        ("Performance Tips", example_5_performance_optimization)
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    try:
        choice = input("\nRun all examples? (y/n): ").lower()
        
        if choice == 'y':
            for name, func in examples:
                func()
                input("\nPress Enter to continue...")
        else:
            print("\nRun individual examples by calling their functions:")
            for name, func in examples:
                print(f"  {func.__name__}()")
    
    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == "__main__":
    main()