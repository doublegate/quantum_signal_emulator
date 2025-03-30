"""
Integration adapters for connecting with external emulators and tools.

This module provides adapters and connectors for integrating the Quantum Signal
Emulator with external emulators, hardware devices, and analysis tools.
"""

import os
import logging
import socket
import json
import time
import threading
import subprocess
import tempfile
import shutil
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, IO
from enum import Enum, auto
import queue
import requests
from datetime import datetime

logger = logging.getLogger("QuantumSignalEmulator.IntegrationAdapters")

class IntegrationType(Enum):
    """Types of external integrations."""
    EMULATOR = auto()
    HARDWARE = auto()
    LOGIC_ANALYZER = auto()
    SIGNAL_GENERATOR = auto()
    WAVEFORM_ANALYZER = auto()
    DATABASE = auto()
    CUSTOM = auto()

class ConnectionType(Enum):
    """Types of connection methods."""
    LOCAL_PROCESS = auto()
    TCP_SOCKET = auto()
    HTTP_API = auto()
    WEBSOCKET = auto()
    SERIAL = auto()
    FILE = auto()
    CUSTOM = auto()

class IntegrationAdapter:
    """
    Base class for all integration adapters.
    
    This class provides a common interface for connecting the Quantum Signal
    Emulator with external tools and emulators.
    """
    
    def __init__(self, 
                name: str,
                integration_type: IntegrationType,
                connection_type: ConnectionType):
        """
        Initialize the integration adapter.
        
        Args:
            name: Name of the integration
            integration_type: Type of integration
            connection_type: Connection method
        """
        self.name = name
        self.integration_type = integration_type
        self.connection_type = connection_type
        self.connected = False
        self.config = {}
        
        # Initialize empty event handlers
        self.on_connect = None
        self.on_disconnect = None
        self.on_error = None
        self.on_data = None
        
        logger.debug(f"Initialized {self.name} integration adapter")
    
    def connect(self) -> bool:
        """
        Connect to the external system.
        
        Returns:
            True if successful, False otherwise
        """
        if self.connected:
            logger.warning(f"{self.name} is already connected")
            return True
            
        # Implemented by subclasses
        logger.warning(f"Connect not implemented for {self.name}")
        return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the external system.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.debug(f"{self.name} is already disconnected")
            return True
            
        # Implemented by subclasses
        logger.warning(f"Disconnect not implemented for {self.name}")
        return False
    
    def send_data(self, data: Any) -> bool:
        """
        Send data to the external system.
        
        Args:
            data: Data to send
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error(f"Cannot send data: {self.name} is not connected")
            return False
            
        # Implemented by subclasses
        logger.warning(f"Send data not implemented for {self.name}")
        return False
    
    def receive_data(self) -> Optional[Any]:
        """
        Receive data from the external system.
        
        Returns:
            Received data or None if no data available
        """
        if not self.connected:
            logger.error(f"Cannot receive data: {self.name} is not connected")
            return None
            
        # Implemented by subclasses
        logger.warning(f"Receive data not implemented for {self.name}")
        return None
    
    def set_event_handler(self, event: str, handler: Callable) -> None:
        """
        Set an event handler.
        
        Args:
            event: Event name ('connect', 'disconnect', 'error', or 'data')
            handler: Handler function
        """
        if event == 'connect':
            self.on_connect = handler
        elif event == 'disconnect':
            self.on_disconnect = handler
        elif event == 'error':
            self.on_error = handler
        elif event == 'data':
            self.on_data = handler
        else:
            logger.warning(f"Unknown event: {event}")
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the adapter.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        self.config.update(config)
        logger.debug(f"Updated configuration for {self.name}")
        return True
    
    def _trigger_event(self, event: str, data: Any = None) -> None:
        """
        Trigger an event handler if set.
        
        Args:
            event: Event name
            data: Event data
        """
        handler = None
        
        if event == 'connect':
            handler = self.on_connect
        elif event == 'disconnect':
            handler = self.on_disconnect
        elif event == 'error':
            handler = self.on_error
        elif event == 'data':
            handler = self.on_data
            
        if handler:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Error in {event} handler: {e}")


class EmulatorAdapter(IntegrationAdapter):
    """
    Adapter for connecting with external emulators.
    
    This class provides functionality for connecting to and exchanging data with
    external emulators like FCEUX, Mesen, BizHawk, etc.
    """
    
    def __init__(self, 
                name: str,
                emulator_type: str,
                connection_type: ConnectionType,
                executable_path: Optional[str] = None):
        """
        Initialize the emulator adapter.
        
        Args:
            name: Name of the integration
            emulator_type: Type of emulator ('fceux', 'mesen', 'bizhawk', etc.)
            connection_type: Connection method
            executable_path: Path to emulator executable (for LOCAL_PROCESS)
        """
        super().__init__(name, IntegrationType.EMULATOR, connection_type)
        
        self.emulator_type = emulator_type
        self.executable_path = executable_path
        self.process = None
        self.connection = None
        self.data_buffer = queue.Queue()
        self.receiver_thread = None
        
        # Set default configuration
        self.config = {
            "host": "localhost",
            "port": self._get_default_port(emulator_type),
            "timeout": 5.0,
            "auto_reconnect": True,
            "reconnect_delay": 2.0,
            "log_traffic": False,
            "rom_path": None
        }
        
        logger.info(f"Initialized {self.emulator_type} adapter named {self.name}")
    
    def _get_default_port(self, emulator_type: str) -> int:
        """
        Get default port for the emulator type.
        
        Args:
            emulator_type: Type of emulator
            
        Returns:
            Default port number
        """
        # Default ports for known emulators
        ports = {
            "fceux": 4399,
            "mesen": 13579,
            "bizhawk": 8080,
            "mame": 6800,
            "stella": 6502,
            "genesis_plus_gx": 6868
        }
        
        return ports.get(emulator_type.lower(), 8000)
    
    def connect(self) -> bool:
        """
        Connect to the emulator.
        
        Returns:
            True if successful, False otherwise
        """
        if self.connected:
            return True
            
        try:
            if self.connection_type == ConnectionType.LOCAL_PROCESS:
                return self._connect_local_process()
            elif self.connection_type == ConnectionType.TCP_SOCKET:
                return self._connect_tcp_socket()
            elif self.connection_type == ConnectionType.HTTP_API:
                return self._connect_http_api()
            else:
                logger.error(f"Unsupported connection type: {self.connection_type}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to emulator: {e}")
            self._trigger_event('error', f"Connection error: {e}")
            return False
    
    def _connect_local_process(self) -> bool:
        """
        Start and connect to local emulator process.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.executable_path:
            logger.error("No executable path specified for local process")
            return False
            
        if not os.path.exists(self.executable_path):
            logger.error(f"Emulator executable not found: {self.executable_path}")
            return False
            
        try:
            # Prepare command line arguments
            args = [self.executable_path]
            
            # Add ROM path if specified
            if self.config.get("rom_path"):
                args.append(self.config["rom_path"])
                
            # Add any custom arguments
            args.extend(self.config.get("command_args", []))
            
            # Start process
            self.process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=False  # Binary mode
            )
            
            # Wait a bit for process to start
            time.sleep(1.0)
            
            # Check if process is running
            if self.process.poll() is not None:
                logger.error(f"Emulator process exited with code {self.process.returncode}")
                stderr_output = self.process.stderr.read()
                logger.error(f"Error output: {stderr_output}")
                return False
                
            # Start receiver thread
            self._start_receiver_thread()
            
            self.connected = True
            logger.info(f"Connected to {self.name} emulator process")
            self._trigger_event('connect', None)
            return True
            
        except Exception as e:
            logger.error(f"Error starting emulator process: {e}")
            self._trigger_event('error', f"Process error: {e}")
            return False
    
    def _connect_tcp_socket(self) -> bool:
        """
        Connect to emulator via TCP socket.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create socket
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.settimeout(self.config["timeout"])
            
            # Connect to emulator
            self.connection.connect((self.config["host"], self.config["port"]))
            
            # Start receiver thread
            self._start_receiver_thread()
            
            self.connected = True
            logger.info(f"Connected to {self.name} emulator at {self.config['host']}:{self.config['port']}")
            self._trigger_event('connect', None)
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to emulator socket: {e}")
            self._trigger_event('error', f"Socket error: {e}")
            return False
    
    def _connect_http_api(self) -> bool:
        """
        Connect to emulator via HTTP API.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Test connection with a simple request
            url = f"http://{self.config['host']}:{self.config['port']}/api/status"
            response = requests.get(url, timeout=self.config["timeout"])
            
            if response.status_code != 200:
                logger.error(f"Error connecting to emulator API: {response.status_code}")
                return False
                
            self.connected = True
            logger.info(f"Connected to {self.name} emulator API at {self.config['host']}:{self.config['port']}")
            self._trigger_event('connect', None)
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to emulator API: {e}")
            self._trigger_event('error', f"API error: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the emulator.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            return True
            
        try:
            # Stop receiver thread
            self._stop_receiver_thread()
            
            if self.connection_type == ConnectionType.LOCAL_PROCESS:
                return self._disconnect_local_process()
            elif self.connection_type == ConnectionType.TCP_SOCKET:
                return self._disconnect_tcp_socket()
            elif self.connection_type == ConnectionType.HTTP_API:
                return self._disconnect_http_api()
            else:
                logger.error(f"Unsupported connection type: {self.connection_type}")
                return False
        except Exception as e:
            logger.error(f"Error disconnecting from emulator: {e}")
            self._trigger_event('error', f"Disconnection error: {e}")
            return False
            
    def _disconnect_local_process(self) -> bool:
        """
        Stop and disconnect from local emulator process.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.process:
            return True
            
        try:
            # Try to terminate gracefully
            self.process.terminate()
            
            # Wait for process to exit
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't exit
                self.process.kill()
                self.process.wait()
                
            self.process = None
            self.connected = False
            logger.info(f"Disconnected from {self.name} emulator process")
            self._trigger_event('disconnect', None)
            return True
            
        except Exception as e:
            logger.error(f"Error stopping emulator process: {e}")
            self._trigger_event('error', f"Process error: {e}")
            return False
    
    def _disconnect_tcp_socket(self) -> bool:
        """
        Disconnect from emulator TCP socket.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connection:
            return True
            
        try:
            self.connection.close()
            self.connection = None
            self.connected = False
            logger.info(f"Disconnected from {self.name} emulator socket")
            self._trigger_event('disconnect', None)
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from emulator socket: {e}")
            self._trigger_event('error', f"Socket error: {e}")
            return False
    
    def _disconnect_http_api(self) -> bool:
        """
        Disconnect from emulator HTTP API.
        
        Returns:
            True if successful, False otherwise
        """
        # HTTP is stateless, so just mark as disconnected
        self.connected = False
        logger.info(f"Disconnected from {self.name} emulator API")
        self._trigger_event('disconnect', None)
        return True
    
    def send_data(self, data: Any) -> bool:
        """
        Send data to the emulator.
        
        Args:
            data: Data to send
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error(f"Cannot send data: {self.name} is not connected")
            return False
            
        try:
            if self.connection_type == ConnectionType.LOCAL_PROCESS:
                return self._send_data_local_process(data)
            elif self.connection_type == ConnectionType.TCP_SOCKET:
                return self._send_data_tcp_socket(data)
            elif self.connection_type == ConnectionType.HTTP_API:
                return self._send_data_http_api(data)
            else:
                logger.error(f"Unsupported connection type: {self.connection_type}")
                return False
        except Exception as e:
            logger.error(f"Error sending data to emulator: {e}")
            self._trigger_event('error', f"Send error: {e}")
            return False
    
    def _send_data_local_process(self, data: Any) -> bool:
        """
        Send data to local emulator process.
        
        Args:
            data: Data to send
            
        Returns:
            True if successful, False otherwise
        """
        if not self.process or not self.process.stdin:
            return False
            
        try:
            # Convert data to string if needed
            if isinstance(data, dict) or isinstance(data, list):
                data = json.dumps(data)
                
            # Convert to bytes if needed
            if isinstance(data, str):
                data = data.encode('utf-8')
                
            # Log traffic if enabled
            if self.config.get("log_traffic"):
                logger.debug(f"Sending to {self.name}: {data}")
                
            # Send data
            self.process.stdin.write(data)
            self.process.stdin.flush()
            return True
            
        except Exception as e:
            logger.error(f"Error sending data to emulator process: {e}")
            self._trigger_event('error', f"Send error: {e}")
            return False
    
    def _send_data_tcp_socket(self, data: Any) -> bool:
        """
        Send data to emulator via TCP socket.
        
        Args:
            data: Data to send
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connection:
            return False
            
        try:
            # Convert data to string if needed
            if isinstance(data, dict) or isinstance(data, list):
                data = json.dumps(data)
                
            # Convert to bytes if needed
            if isinstance(data, str):
                data = data.encode('utf-8')
                
            # Log traffic if enabled
            if self.config.get("log_traffic"):
                logger.debug(f"Sending to {self.name}: {data}")
                
            # Send data
            self.connection.sendall(data)
            return True
            
        except Exception as e:
            logger.error(f"Error sending data to emulator socket: {e}")
            self._trigger_event('error', f"Send error: {e}")
            
            # Handle disconnection
            self.connected = False
            self._trigger_event('disconnect', None)
            
            # Auto-reconnect if enabled
            if self.config.get("auto_reconnect"):
                time.sleep(self.config.get("reconnect_delay", 2.0))
                self.connect()
                
            return False
    
    def _send_data_http_api(self, data: Any) -> bool:
        """
        Send data to emulator via HTTP API.
        
        Args:
            data: Data to send
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine URL and method based on data
            if isinstance(data, dict) and "endpoint" in data:
                endpoint = data.pop("endpoint")
                method = data.pop("method", "POST").upper()
            else:
                endpoint = "command"
                method = "POST"
                
            url = f"http://{self.config['host']}:{self.config['port']}/api/{endpoint}"
            
            # Log traffic if enabled
            if self.config.get("log_traffic"):
                logger.debug(f"Sending to {self.name} ({method} {url}): {data}")
                
            # Send data
            if method == "GET":
                response = requests.get(url, params=data, timeout=self.config["timeout"])
            else:
                response = requests.post(url, json=data, timeout=self.config["timeout"])
                
            # Check response
            if response.status_code != 200:
                logger.error(f"Error sending data to emulator API: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
            # Process response
            if self.on_data:
                self._trigger_event('data', response.json() if response.text else None)
                
            return True
            
        except Exception as e:
            logger.error(f"Error sending data to emulator API: {e}")
            self._trigger_event('error', f"API error: {e}")
            
            # Handle disconnection
            self.connected = False
            self._trigger_event('disconnect', None)
            
            # Auto-reconnect if enabled
            if self.config.get("auto_reconnect"):
                time.sleep(self.config.get("reconnect_delay", 2.0))
                self.connect()
                
            return False
    
    def receive_data(self) -> Optional[Any]:
        """
        Receive data from the emulator.
        
        Returns:
            Received data or None if no data available
        """
        if not self.connected:
            logger.error(f"Cannot receive data: {self.name} is not connected")
            return None
            
        try:
            # Get data from buffer (non-blocking)
            try:
                data = self.data_buffer.get_nowait()
                self.data_buffer.task_done()
                return data
            except queue.Empty:
                return None
                
        except Exception as e:
            logger.error(f"Error receiving data from emulator: {e}")
            self._trigger_event('error', f"Receive error: {e}")
            return None
    
    def _start_receiver_thread(self) -> None:
        """Start the receiver thread."""
        if self.receiver_thread and self.receiver_thread.is_alive():
            return
            
        self.receiver_thread = threading.Thread(
            target=self._receiver_loop,
            daemon=True
        )
        self.receiver_thread.start()
    
    def _stop_receiver_thread(self) -> None:
        """Stop the receiver thread."""
        if not self.receiver_thread or not self.receiver_thread.is_alive():
            return
            
        # Let the thread exit naturally on next iteration
        # It's running as daemon, so it will be terminated when the program exits
        pass
    
    def _receiver_loop(self) -> None:
        """Receiver thread main loop."""
        while self.connected:
            try:
                if self.connection_type == ConnectionType.LOCAL_PROCESS:
                    self._receive_data_local_process()
                elif self.connection_type == ConnectionType.TCP_SOCKET:
                    self._receive_data_tcp_socket()
                else:
                    # Other connection types don't need a receiver thread
                    break
            except Exception as e:
                logger.error(f"Error in receiver thread: {e}")
                
                # Pause to avoid tight loop on error
                time.sleep(0.1)
    
    def _receive_data_local_process(self) -> None:
        """Receive data from local emulator process."""
        if not self.process or not self.process.stdout:
            return
            
        try:
            # Check if process is still running
            if self.process.poll() is not None:
                logger.warning(f"Emulator process exited with code {self.process.returncode}")
                self.connected = False
                self._trigger_event('disconnect', None)
                return
                
            # Read data (non-blocking)
            data = self.process.stdout.readline()
            
            if data:
                # Log traffic if enabled
                if self.config.get("log_traffic"):
                    logger.debug(f"Received from {self.name}: {data}")
                    
                # Add to buffer
                self.data_buffer.put(data)
                
                # Trigger event
                self._trigger_event('data', data)
                
        except Exception as e:
            logger.error(f"Error receiving data from emulator process: {e}")
            self._trigger_event('error', f"Receive error: {e}")
    
    def _receive_data_tcp_socket(self) -> None:
        """Receive data from emulator via TCP socket."""
        if not self.connection:
            return
            
        try:
            # Set socket to non-blocking mode
            self.connection.setblocking(False)
            
            try:
                # Try to receive data
                data = self.connection.recv(4096)
                
                if not data:
                    # Connection closed
                    logger.warning(f"Connection to {self.name} closed by remote host")
                    self.connected = False
                    self._trigger_event('disconnect', None)
                    return
                    
                # Log traffic if enabled
                if self.config.get("log_traffic"):
                    logger.debug(f"Received from {self.name}: {data}")
                    
                # Add to buffer
                self.data_buffer.put(data)
                
                # Trigger event
                self._trigger_event('data', data)
                
            except BlockingIOError:
                # No data available
                pass
                
            except Exception as e:
                logger.error(f"Error receiving data from emulator socket: {e}")
                self._trigger_event('error', f"Receive error: {e}")
                
                # Handle disconnection
                self.connected = False
                self._trigger_event('disconnect', None)
                
        except Exception as e:
            logger.error(f"Error in socket receiver: {e}")
            self._trigger_event('error', f"Socket error: {e}")
            
        # Small sleep to avoid tight loop
        time.sleep(0.01)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current emulator state.
        
        Returns:
            Dictionary with emulator state
        """
        if not self.connected:
            return {"error": "Not connected"}
            
        try:
            if self.emulator_type == "fceux":
                return self._get_state_fceux()
            elif self.emulator_type == "mesen":
                return self._get_state_mesen()
            elif self.emulator_type == "bizhawk":
                return self._get_state_bizhawk()
            else:
                logger.warning(f"Get state not implemented for {self.emulator_type}")
                return {"error": f"Not implemented for {self.emulator_type}"}
                
        except Exception as e:
            logger.error(f"Error getting emulator state: {e}")
            return {"error": str(e)}
    
    def _get_state_fceux(self) -> Dict[str, Any]:
        """
        Get state from FCEUX emulator.
        
        Returns:
            Dictionary with FCEUX state
        """
        if self.connection_type == ConnectionType.HTTP_API:
            try:
                url = f"http://{self.config['host']}:{self.config['port']}/api/state"
                response = requests.get(url, timeout=self.config["timeout"])
                
                if response.status_code != 200:
                    return {"error": f"HTTP error: {response.status_code}"}
                    
                return response.json()
                
            except Exception as e:
                return {"error": str(e)}
                
        elif self.connection_type == ConnectionType.TCP_SOCKET:
            # Send state request command
            self.send_data(b"STATE\n")
            
            # Wait for response
            time.sleep(0.1)
            
            # Get all available data
            data = b""
            while True:
                chunk = self.receive_data()
                if not chunk:
                    break
                data += chunk
                
            # Parse response
            try:
                return json.loads(data)
            except:
                return {"raw_data": data}
                
        else:
            return {"error": f"Not implemented for {self.connection_type}"}
    
    def _get_state_mesen(self) -> Dict[str, Any]:
        """
        Get state from Mesen emulator.
        
        Returns:
            Dictionary with Mesen state
        """
        if self.connection_type == ConnectionType.HTTP_API:
            try:
                url = f"http://{self.config['host']}:{self.config['port']}/api/state"
                response = requests.get(url, timeout=self.config["timeout"])
                
                if response.status_code != 200:
                    return {"error": f"HTTP error: {response.status_code}"}
                    
                return response.json()
                
            except Exception as e:
                return {"error": str(e)}
                
        else:
            return {"error": f"Not implemented for {self.connection_type}"}
    
    def _get_state_bizhawk(self) -> Dict[str, Any]:
        """
        Get state from BizHawk emulator.
        
        Returns:
            Dictionary with BizHawk state
        """
        if self.connection_type == ConnectionType.HTTP_API:
            try:
                url = f"http://{self.config['host']}:{self.config['port']}/api/Emulation/State"
                response = requests.get(url, timeout=self.config["timeout"])
                
                if response.status_code != 200:
                    return {"error": f"HTTP error: {response.status_code}"}
                    
                return response.json()
                
            except Exception as e:
                return {"error": str(e)}
                
        else:
            return {"error": f"Not implemented for {self.connection_type}"}
    
    def execute_command(self, command: str, args: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a command on the emulator.
        
        Args:
            command: Command to execute
            args: Command arguments
            
        Returns:
            Command result or None on error
        """
        if not self.connected:
            logger.error(f"Cannot execute command: {self.name} is not connected")
            return None
            
        try:
            # Prepare command data
            data = {
                "command": command
            }
            
            if args:
                data.update(args)
                
            # Send command
            if self.connection_type == ConnectionType.HTTP_API:
                url = f"http://{self.config['host']}:{self.config['port']}/api/command"
                response = requests.post(url, json=data, timeout=self.config["timeout"])
                
                if response.status_code != 200:
                    logger.error(f"Error executing command: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    return None
                    
                return response.json() if response.text else None
                
            else:
                # For other connection types, just send the command
                success = self.send_data(json.dumps(data).encode('utf-8') + b'\n')
                
                if not success:
                    return None
                    
                # Wait for response
                time.sleep(0.1)
                
                # Get response
                return self.receive_data()
                
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return None


class HardwareAdapter(IntegrationAdapter):
    """
    Adapter for connecting with hardware devices.
    
    This class provides functionality for connecting to and controlling
    hardware devices like logic analyzers and signal generators.
    """
    
    def __init__(self, 
                name: str,
                device_type: str,
                connection_type: ConnectionType,
                device_path: Optional[str] = None):
        """
        Initialize the hardware adapter.
        
        Args:
            name: Name of the integration
            device_type: Type of device
            connection_type: Connection method
            device_path: Path to device (for SERIAL connection)
        """
        super().__init__(name, IntegrationType.HARDWARE, connection_type)
        
        self.device_type = device_type
        self.device_path = device_path
        self.connection = None
        self.data_buffer = queue.Queue()
        self.receiver_thread = None
        
        # Set default configuration
        self.config = {
            "host": "localhost",
            "port": 0,
            "baud_rate": 115200,
            "timeout": 5.0,
            "auto_reconnect": True,
            "reconnect_delay": 2.0,
            "log_traffic": False
        }
        
        logger.info(f"Initialized {self.device_type} adapter named {self.name}")
    
    def connect(self) -> bool:
        """
        Connect to the hardware device.
        
        Returns:
            True if successful, False otherwise
        """
        if self.connected:
            return True
            
        try:
            if self.connection_type == ConnectionType.SERIAL:
                return self._connect_serial()
            elif self.connection_type == ConnectionType.TCP_SOCKET:
                return self._connect_tcp_socket()
            elif self.connection_type == ConnectionType.HTTP_API:
                return self._connect_http_api()
            else:
                logger.error(f"Unsupported connection type: {self.connection_type}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to hardware device: {e}")
            self._trigger_event('error', f"Connection error: {e}")
            return False
    
    def _connect_serial(self) -> bool:
        """
        Connect to hardware device via serial port.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure pyserial is available
            import serial
            
            if not self.device_path:
                logger.error("No device path specified for serial connection")
                return False
                
            # Create serial connection
            self.connection = serial.Serial(
                port=self.device_path,
                baudrate=self.config["baud_rate"],
                timeout=self.config["timeout"]
            )
            
            # Check if connection is open
            if not self.connection.is_open:
                self.connection.open()
                
            # Start receiver thread
            self._start_receiver_thread()
            
            self.connected = True
            logger.info(f"Connected to {self.name} device at {self.device_path}")
            self._trigger_event('connect', None)
            return True
            
        except ImportError:
            logger.error("pyserial is not installed. Please install it with: pip install pyserial")
            return False
        except Exception as e:
            logger.error(f"Error connecting to serial device: {e}")
            self._trigger_event('error', f"Serial error: {e}")
            return False
    
    def _connect_tcp_socket(self) -> bool:
        """
        Connect to hardware device via TCP socket.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create socket
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.settimeout(self.config["timeout"])
            
            # Connect to device
            self.connection.connect((self.config["host"], self.config["port"]))
            
            # Start receiver thread
            self._start_receiver_thread()
            
            self.connected = True
            logger.info(f"Connected to {self.name} device at {self.config['host']}:{self.config['port']}")
            self._trigger_event('connect', None)
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to device socket: {e}")
            self._trigger_event('error', f"Socket error: {e}")
            return False
    
    def _connect_http_api(self) -> bool:
        """
        Connect to hardware device via HTTP API.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Test connection with a simple request
            url = f"http://{self.config['host']}:{self.config['port']}/api/status"
            response = requests.get(url, timeout=self.config["timeout"])
            
            if response.status_code != 200:
                logger.error(f"Error connecting to device API: {response.status_code}")
                return False
                
            self.connected = True
            logger.info(f"Connected to {self.name} device API at {self.config['host']}:{self.config['port']}")
            self._trigger_event('connect', None)
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to device API: {e}")
            self._trigger_event('error', f"API error: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the hardware device.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            return True
            
        try:
            # Stop receiver thread
            self._stop_receiver_thread()
            
            if self.connection_type == ConnectionType.SERIAL:
                return self._disconnect_serial()
            elif self.connection_type == ConnectionType.TCP_SOCKET:
                return self._disconnect_tcp_socket()
            elif self.connection_type == ConnectionType.HTTP_API:
                return self._disconnect_http_api()
            else:
                logger.error(f"Unsupported connection type: {self.connection_type}")
                return False
        except Exception as e:
            logger.error(f"Error disconnecting from hardware device: {e}")
            self._trigger_event('error', f"Disconnection error: {e}")
            return False
    
    def _disconnect_serial(self) -> bool:
        """
        Disconnect from serial device.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connection:
            return True
            
        try:
            self.connection.close()
            self.connection = None
            self.connected = False
            logger.info(f"Disconnected from {self.name} device")
            self._trigger_event('disconnect', None)
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from serial device: {e}")
            self._trigger_event('error', f"Serial error: {e}")
            return False
    
    def _disconnect_tcp_socket(self) -> bool:
        """
        Disconnect from device TCP socket.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connection:
            return True
            
        try:
            self.connection.close()
            self.connection = None
            self.connected = False
            logger.info(f"Disconnected from {self.name} device socket")
            self._trigger_event('disconnect', None)
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from device socket: {e}")
            self._trigger_event('error', f"Socket error: {e}")
            return False
    
    def _disconnect_http_api(self) -> bool:
        """
        Disconnect from device HTTP API.
        
        Returns:
            True if successful, False otherwise
        """
        # HTTP is stateless, so just mark as disconnected
        self.connected = False
        logger.info(f"Disconnected from {self.name} device API")
        self._trigger_event('disconnect', None)
        return True
    
    def send_data(self, data: Any) -> bool:
        """
        Send data to the hardware device.
        
        Args:
            data: Data to send
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error(f"Cannot send data: {self.name} is not connected")
            return False
            
        try:
            if self.connection_type == ConnectionType.SERIAL:
                return self._send_data_serial(data)
            elif self.connection_type == ConnectionType.TCP_SOCKET:
                return self._send_data_tcp_socket(data)
            elif self.connection_type == ConnectionType.HTTP_API:
                return self._send_data_http_api(data)
            else:
                logger.error(f"Unsupported connection type: {self.connection_type}")
                return False
        except Exception as e:
            logger.error(f"Error sending data to hardware device: {e}")
            self._trigger_event('error', f"Send error: {e}")
            return False
    
    def _send_data_serial(self, data: Any) -> bool:
        """
        Send data to hardware device via serial port.
        
        Args:
            data: Data to send
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connection:
            return False
            
        try:
            # Convert data to bytes if needed
            if isinstance(data, str):
                data = data.encode('utf-8')
            elif isinstance(data, dict) or isinstance(data, list):
                data = json.dumps(data).encode('utf-8')
                
            # Log traffic if enabled
            if self.config.get("log_traffic"):
                logger.debug(f"Sending to {self.name}: {data}")
                
            # Send data
            self.connection.write(data)
            return True
            
        except Exception as e:
            logger.error(f"Error sending data to serial device: {e}")
            self._trigger_event('error', f"Serial error: {e}")
            
            # Handle disconnection
            self.connected = False
            self._trigger_event('disconnect', None)
            
            # Auto-reconnect if enabled
            if self.config.get("auto_reconnect"):
                time.sleep(self.config.get("reconnect_delay", 2.0))
                self.connect()
                
            return False
    
    def _send_data_tcp_socket(self, data: Any) -> bool:
        """
        Send data to hardware device via TCP socket.
        
        Args:
            data: Data to send
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connection:
            return False
            
        try:
            # Convert data to bytes if needed
            if isinstance(data, str):
                data = data.encode('utf-8')
            elif isinstance(data, dict) or isinstance(data, list):
                data = json.dumps(data).encode('utf-8')
                
            # Log traffic if enabled
            if self.config.get("log_traffic"):
                logger.debug(f"Sending to {self.name}: {data}")
                
            # Send data
            self.connection.sendall(data)
            return True
            
        except Exception as e:
            logger.error(f"Error sending data to device socket: {e}")
            self._trigger_event('error', f"Socket error: {e}")
            
            # Handle disconnection
            self.connected = False
            self._trigger_event('disconnect', None)
            
            # Auto-reconnect if enabled
            if self.config.get("auto_reconnect"):
                time.sleep(self.config.get("reconnect_delay", 2.0))
                self.connect()
                
            return False
    
    def _send_data_http_api(self, data: Any) -> bool:
        """
        Send data to hardware device via HTTP API.
        
        Args:
            data: Data to send
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine URL and method based on data
            if isinstance(data, dict) and "endpoint" in data:
                endpoint = data.pop("endpoint")
                method = data.pop("method", "POST").upper()
            else:
                endpoint = "command"
                method = "POST"
                
            url = f"http://{self.config['host']}:{self.config['port']}/api/{endpoint}"
            
            # Log traffic if enabled
            if self.config.get("log_traffic"):
                logger.debug(f"Sending to {self.name} ({method} {url}): {data}")
                
            # Send data
            if method == "GET":
                response = requests.get(url, params=data, timeout=self.config["timeout"])
            else:
                response = requests.post(url, json=data, timeout=self.config["timeout"])
                
            # Check response
            if response.status_code != 200:
                logger.error(f"Error sending data to device API: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
            # Process response
            if self.on_data:
                self._trigger_event('data', response.json() if response.text else None)
                
            return True
            
        except Exception as e:
            logger.error(f"Error sending data to device API: {e}")
            self._trigger_event('error', f"API error: {e}")
            
            # Handle disconnection
            self.connected = False
            self._trigger_event('disconnect', None)
            
            # Auto-reconnect if enabled
            if self.config.get("auto_reconnect"):
                time.sleep(self.config.get("reconnect_delay", 2.0))
                self.connect()
                
            return False
    
    def receive_data(self) -> Optional[Any]:
        """
        Receive data from the hardware device.
        
        Returns:
            Received data or None if no data available
        """
        if not self.connected:
            logger.error(f"Cannot receive data: {self.name} is not connected")
            return None
            
        try:
            # Get data from buffer (non-blocking)
            try:
                data = self.data_buffer.get_nowait()
                self.data_buffer.task_done()
                return data
            except queue.Empty:
                return None
                
        except Exception as e:
            logger.error(f"Error receiving data from hardware device: {e}")
            self._trigger_event('error', f"Receive error: {e}")
            return None
    
    def _start_receiver_thread(self) -> None:
        """Start the receiver thread."""
        if self.receiver_thread and self.receiver_thread.is_alive():
            return
            
        self.receiver_thread = threading.Thread(
            target=self._receiver_loop,
            daemon=True
        )
        self.receiver_thread.start()
    
    def _stop_receiver_thread(self) -> None:
        """Stop the receiver thread."""
        if not self.receiver_thread or not self.receiver_thread.is_alive():
            return
            
        # Let the thread exit naturally on next iteration
        # It's running as daemon, so it will be terminated when the program exits
        pass
    
    def _receiver_loop(self) -> None:
        """Receiver thread main loop."""
        while self.connected:
            try:
                if self.connection_type == ConnectionType.SERIAL:
                    self._receive_data_serial()
                elif self.connection_type == ConnectionType.TCP_SOCKET:
                    self._receive_data_tcp_socket()
                else:
                    # Other connection types don't need a receiver thread
                    break
            except Exception as e:
                logger.error(f"Error in receiver thread: {e}")
                
                # Pause to avoid tight loop on error
                time.sleep(0.1)
    
    def _receive_data_serial(self) -> None:
        """Receive data from hardware device via serial port."""
        if not self.connection:
            return
            
        try:
            # Check if data is available
            if self.connection.in_waiting > 0:
                # Read data
                data = self.connection.read(self.connection.in_waiting)
                
                if data:
                    # Log traffic if enabled
                    if self.config.get("log_traffic"):
                        logger.debug(f"Received from {self.name}: {data}")
                        
                    # Add to buffer
                    self.data_buffer.put(data)
                    
                    # Trigger event
                    self._trigger_event('data', data)
                    
        except Exception as e:
            logger.error(f"Error receiving data from serial device: {e}")
            self._trigger_event('error', f"Serial error: {e}")
            
            # Handle disconnection
            self.connected = False
            self._trigger_event('disconnect', None)
            
        # Small sleep to avoid tight loop
        time.sleep(0.01)
    
    def _receive_data_tcp_socket(self) -> None:
        """Receive data from hardware device via TCP socket."""
        if not self.connection:
            return
            
        try:
            # Set socket to non-blocking mode
            self.connection.setblocking(False)
            
            try:
                # Try to receive data
                data = self.connection.recv(4096)
                
                if not data:
                    # Connection closed
                    logger.warning(f"Connection to {self.name} closed by remote host")
                    self.connected = False
                    self._trigger_event('disconnect', None)
                    return
                    
                # Log traffic if enabled
                if self.config.get("log_traffic"):
                    logger.debug(f"Received from {self.name}: {data}")
                    
                # Add to buffer
                self.data_buffer.put(data)
                
                # Trigger event
                self._trigger_event('data', data)
                
            except BlockingIOError:
                # No data available
                pass
                
            except Exception as e:
                logger.error(f"Error receiving data from device socket: {e}")
                self._trigger_event('error', f"Receive error: {e}")
                
                # Handle disconnection
                self.connected = False
                self._trigger_event('disconnect', None)
                
        except Exception as e:
            logger.error(f"Error in socket receiver: {e}")
            self._trigger_event('error', f"Socket error: {e}")
            
        # Small sleep to avoid tight loop
        time.sleep(0.01)
    
    def capture_data(self, duration: float) -> Any:
        """
        Capture data from hardware device for a specified duration.
        
        Args:
            duration: Capture duration in seconds
            
        Returns:
            Captured data or None on error
        """
        if not self.connected:
            logger.error(f"Cannot capture data: {self.name} is not connected")
            return None
            
        try:
            # Clear buffer
            while not self.data_buffer.empty():
                self.data_buffer.get_nowait()
                self.data_buffer.task_done()
                
            # Start capture
            if self.device_type == "logic_analyzer":
                # Send capture command
                self.send_data({"command": "capture", "duration": duration})
            else:
                # For other devices, just collect data for the specified duration
                pass
                
            # Collect data for the specified duration
            start_time = time.time()
            all_data = []
            
            while time.time() - start_time < duration:
                data = self.receive_data()
                if data:
                    all_data.append(data)
                time.sleep(0.01)
                
            # Process captured data
            if not all_data:
                return None
                
            if isinstance(all_data[0], bytes):
                # Combine byte data
                return b''.join(all_data)
            else:
                # Return as list
                return all_data
                
        except Exception as e:
            logger.error(f"Error capturing data: {e}")
            self._trigger_event('error', f"Capture error: {e}")
            return None
    
    def configure_device(self, settings: Dict[str, Any]) -> bool:
        """
        Configure the hardware device.
        
        Args:
            settings: Device settings
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error(f"Cannot configure device: {self.name} is not connected")
            return False
            
        try:
            # Send configuration command
            return self.send_data({"command": "configure", "settings": settings})
                
        except Exception as e:
            logger.error(f"Error configuring device: {e}")
            self._trigger_event('error', f"Configuration error: {e}")
            return False


class LogicAnalyzerAdapter(HardwareAdapter):
    """
    Specialized adapter for logic analyzers.
    
    This class extends the hardware adapter with additional functionality
    specific to logic analyzers.
    """
    
    def __init__(self, 
                name: str,
                device_type: str,
                connection_type: ConnectionType,
                device_path: Optional[str] = None):
        """
        Initialize the logic analyzer adapter.
        
        Args:
            name: Name of the integration
            device_type: Type of logic analyzer
            connection_type: Connection method
            device_path: Path to device (for SERIAL connection)
        """
        super().__init__(name, device_type, connection_type, device_path)
        
        # Override integration type
        self.integration_type = IntegrationType.LOGIC_ANALYZER
        
        # Additional configuration for logic analyzers
        self.config.update({
            "sample_rate": 1000000,  # 1 MHz
            "channel_count": 8,
            "buffer_size": 1048576,  # 1 MB
            "trigger_channel": 0,
            "trigger_edge": "rising",
            "export_format": "csv"
        })
        
        logger.info(f"Initialized {self.device_type} logic analyzer adapter named {self.name}")
    
    def start_capture(self, duration: Optional[float] = None) -> bool:
        """
        Start a logic analyzer capture.
        
        Args:
            duration: Capture duration in seconds (None for continuous)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error(f"Cannot start capture: {self.name} is not connected")
            return False
            
        try:
            # Prepare capture command
            command = {
                "command": "start_capture",
                "sample_rate": self.config["sample_rate"],
                "channel_count": self.config["channel_count"],
                "buffer_size": self.config["buffer_size"],
                "trigger_channel": self.config["trigger_channel"],
                "trigger_edge": self.config["trigger_edge"]
            }
            
            if duration is not None:
                command["duration"] = duration
                
            # Send command
            return self.send_data(command)
                
        except Exception as e:
            logger.error(f"Error starting capture: {e}")
            self._trigger_event('error', f"Capture error: {e}")
            return False
    
    def stop_capture(self) -> bool:
        """
        Stop the current logic analyzer capture.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error(f"Cannot stop capture: {self.name} is not connected")
            return False
            
        try:
            # Send stop command
            return self.send_data({"command": "stop_capture"})
                
        except Exception as e:
            logger.error(f"Error stopping capture: {e}")
            self._trigger_event('error', f"Capture error: {e}")
            return False
    
    def get_capture_data(self) -> Optional[Dict[str, Any]]:
        """
        Get the captured data from the logic analyzer.
        
        Returns:
            Dictionary with captured data or None on error
        """
        if not self.connected:
            logger.error(f"Cannot get capture data: {self.name} is not connected")
            return None
            
        try:
            # Send data request command
            self.send_data({"command": "get_capture_data"})
            
            # Wait for response
            time.sleep(0.5)
            
            # Get all available data
            data = b""
            while True:
                chunk = self.receive_data()
                if not chunk:
                    break
                if isinstance(chunk, bytes):
                    data += chunk
                    
            # Parse response
            try:
                if data.startswith(b'{'):
                    # JSON response
                    return json.loads(data)
                else:
                    # Binary response
                    return {"raw_data": data}
            except:
                return {"raw_data": data}
                
        except Exception as e:
            logger.error(f"Error getting capture data: {e}")
            self._trigger_event('error', f"Data error: {e}")
            return None
    
    def export_capture(self, filename: str, format: Optional[str] = None) -> bool:
        """
        Export the captured data to a file.
        
        Args:
            filename: Output filename
            format: Export format (None to use config default)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error(f"Cannot export capture: {self.name} is not connected")
            return False
            
        try:
            # Use specified format or default from config
            export_format = format or self.config["export_format"]
            
            # Send export command
            return self.send_data({
                "command": "export_capture",
                "filename": filename,
                "format": export_format
            })
                
        except Exception as e:
            logger.error(f"Error exporting capture: {e}")
            self._trigger_event('error', f"Export error: {e}")
            return False
    
    def set_trigger(self, channel: int, edge: str) -> bool:
        """
        Set the trigger channel and edge.
        
        Args:
            channel: Trigger channel
            edge: Trigger edge ('rising', 'falling', or 'both')
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error(f"Cannot set trigger: {self.name} is not connected")
            return False
            
        try:
            # Update config
            self.config["trigger_channel"] = channel
            self.config["trigger_edge"] = edge
            
            # Send trigger command
            return self.send_data({
                "command": "set_trigger",
                "channel": channel,
                "edge": edge
            })
                
        except Exception as e:
            logger.error(f"Error setting trigger: {e}")
            self._trigger_event('error', f"Trigger error: {e}")
            return False


class DatabaseAdapter(IntegrationAdapter):
    """
    Adapter for connecting with databases for result storage.
    
    This class provides functionality for storing and retrieving analysis
    results and other data in various database systems.
    """
    
    def __init__(self, 
                name: str,
                db_type: str,
                connection_string: Optional[str] = None):
        """
        Initialize the database adapter.
        
        Args:
            name: Name of the integration
            db_type: Type of database ('sqlite', 'postgresql', 'mysql', etc.)
            connection_string: Database connection string
        """
        super().__init__(name, IntegrationType.DATABASE, ConnectionType.CUSTOM)
        
        self.db_type = db_type
        self.connection_string = connection_string
        self.connection = None
        self.cursor = None
        
        # Set default configuration
        self.config = {
            "table_prefix": "quantum_signal_",
            "batch_size": 1000,
            "timeout": 30.0,
            "auto_create_tables": True
        }
        
        logger.info(f"Initialized {self.db_type} database adapter named {self.name}")
    
    def connect(self) -> bool:
        """
        Connect to the database.
        
        Returns:
            True if successful, False otherwise
        """
        if self.connected:
            return True
            
        try:
            if self.db_type == "sqlite":
                return self._connect_sqlite()
            elif self.db_type == "postgresql":
                return self._connect_postgresql()
            elif self.db_type == "mysql":
                return self._connect_mysql()
            else:
                logger.error(f"Unsupported database type: {self.db_type}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            self._trigger_event('error', f"Connection error: {e}")
            return False
    
    def _connect_sqlite(self) -> bool:
        """
        Connect to SQLite database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure sqlite3 is available
            import sqlite3
            
            # Use connection string or default to in-memory database
            conn_str = self.connection_string or ":memory:"
            
            # Create connection
            self.connection = sqlite3.connect(conn_str)
            self.cursor = self.connection.cursor()
            
            # Enable foreign keys
            self.cursor.execute("PRAGMA foreign_keys = ON")
            
            # Create tables if needed
            if self.config["auto_create_tables"]:
                self._create_sqlite_tables()
                
            self.connected = True
            logger.info(f"Connected to SQLite database: {conn_str}")
            self._trigger_event('connect', None)
            return True
            
        except ImportError:
            logger.error("sqlite3 is not available")
            return False
        except Exception as e:
            logger.error(f"Error connecting to SQLite database: {e}")
            self._trigger_event('error', f"SQLite error: {e}")
            return False
    
    def _connect_postgresql(self) -> bool:
        """
        Connect to PostgreSQL database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure psycopg2 is available
            import psycopg2
            
            if not self.connection_string:
                logger.error("No connection string specified for PostgreSQL")
                return False
                
            # Create connection
            self.connection = psycopg2.connect(self.connection_string)
            self.cursor = self.connection.cursor()
            
            # Create tables if needed
            if self.config["auto_create_tables"]:
                self._create_postgresql_tables()
                
            self.connected = True
            logger.info(f"Connected to PostgreSQL database")
            self._trigger_event('connect', None)
            return True
            
        except ImportError:
            logger.error("psycopg2 is not installed. Please install it with: pip install psycopg2-binary")
            return False
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL database: {e}")
            self._trigger_event('error', f"PostgreSQL error: {e}")
            return False
    
    def _connect_mysql(self) -> bool:
        """
        Connect to MySQL database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure mysql-connector is available
            import mysql.connector
            
            if not self.connection_string:
                logger.error("No connection string specified for MySQL")
                return False
                
            # Parse connection string
            # Format: mysql://user:password@host:port/database
            conn_parts = self.connection_string.replace("mysql://", "").split("/")
            
            if len(conn_parts) != 2:
                logger.error(f"Invalid MySQL connection string: {self.connection_string}")
                return False
                
            auth_host, database = conn_parts
            auth_host_parts = auth_host.split("@")
            
            if len(auth_host_parts) != 2:
                logger.error(f"Invalid MySQL connection string: {self.connection_string}")
                return False
                
            user_pass, host_port = auth_host_parts
            user_pass_parts = user_pass.split(":")
            
            if len(user_pass_parts) != 2:
                logger.error(f"Invalid MySQL connection string: {self.connection_string}")
                return False
                
            user, password = user_pass_parts
            
            if ":" in host_port:
                host, port = host_port.split(":")
                port = int(port)
            else:
                host = host_port
                port = 3306
                
            # Create connection
            self.connection = mysql.connector.connect(
                user=user,
                password=password,
                host=host,
                port=port,
                database=database
            )
            
            self.cursor = self.connection.cursor()
            
            # Create tables if needed
            if self.config["auto_create_tables"]:
                self._create_mysql_tables()
                
            self.connected = True
            logger.info(f"Connected to MySQL database at {host}:{port}/{database}")
            self._trigger_event('connect', None)
            return True
            
        except ImportError:
            logger.error("mysql-connector is not installed. Please install it with: pip install mysql-connector-python")
            return False
        except Exception as e:
            logger.error(f"Error connecting to MySQL database: {e}")
            self._trigger_event('error', f"MySQL error: {e}")
            return False
    
    def _create_sqlite_tables(self) -> None:
        """Create tables in SQLite database."""
        if not self.cursor:
            return
            
        prefix = self.config["table_prefix"]
        
        # Create runs table
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            system TEXT,
            rom_path TEXT,
            analysis_mode TEXT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            metadata TEXT
        )
        """)
        
        # Create states table
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            cycle INTEGER,
            scanline INTEGER,
            dot INTEGER,
            timestamp TIMESTAMP,
            state_data TEXT,
            FOREIGN KEY (run_id) REFERENCES {prefix}runs (id)
        )
        """)
        
        # Create quantum_analysis table
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}quantum_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            cycle_range TEXT,
            analysis_mode TEXT,
            quantum_entropy REAL,
            coherence_measure REAL,
            frequency_data TEXT,
            analysis_summary TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES {prefix}runs (id)
        )
        """)
        
        # Create timing_analysis table
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}timing_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            cycle_range TEXT,
            timing_patterns TEXT,
            timing_anomalies TEXT,
            statistics TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES {prefix}runs (id)
        )
        """)
        
        # Create register_analysis table
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}register_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            register_name TEXT,
            cycle_range TEXT,
            change_frequency REAL,
            value_entropy REAL,
            common_values TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES {prefix}runs (id)
        )
        """)
        
        self.connection.commit()
    
    def _create_postgresql_tables(self) -> None:
        """Create tables in PostgreSQL database."""
        if not self.cursor:
            return
            
        prefix = self.config["table_prefix"]
        
        # Create runs table
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}runs (
            id SERIAL PRIMARY KEY,
            name TEXT,
            system TEXT,
            rom_path TEXT,
            analysis_mode TEXT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            metadata JSONB
        )
        """)
        
        # Create states table
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}states (
            id SERIAL PRIMARY KEY,
            run_id INTEGER REFERENCES {prefix}runs (id),
            cycle INTEGER,
            scanline INTEGER,
            dot INTEGER,
            timestamp TIMESTAMP,
            state_data JSONB
        )
        """)
        
        # Create quantum_analysis table
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}quantum_analysis (
            id SERIAL PRIMARY KEY,
            run_id INTEGER REFERENCES {prefix}runs (id),
            cycle_range TEXT,
            analysis_mode TEXT,
            quantum_entropy REAL,
            coherence_measure REAL,
            frequency_data JSONB,
            analysis_summary TEXT,
            timestamp TIMESTAMP
        )
        """)
        
        # Create timing_analysis table
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}timing_analysis (
            id SERIAL PRIMARY KEY,
            run_id INTEGER REFERENCES {prefix}runs (id),
            cycle_range TEXT,
            timing_patterns JSONB,
            timing_anomalies JSONB,
            statistics JSONB,
            timestamp TIMESTAMP
        )
        """)
        
        # Create register_analysis table
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}register_analysis (
            id SERIAL PRIMARY KEY,
            run_id INTEGER REFERENCES {prefix}runs (id),
            register_name TEXT,
            cycle_range TEXT,
            change_frequency REAL,
            value_entropy REAL,
            common_values JSONB,
            timestamp TIMESTAMP
        )
        """)
        
        self.connection.commit()
    
    def _create_mysql_tables(self) -> None:
        """Create tables in MySQL database."""
        if not self.cursor:
            return
            
        prefix = self.config["table_prefix"]
        
        # Create runs table
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}runs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            system VARCHAR(50),
            rom_path VARCHAR(255),
            analysis_mode VARCHAR(50),
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            metadata JSON
        )
        """)
        
        # Create states table
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}states (
            id INT AUTO_INCREMENT PRIMARY KEY,
            run_id INT,
            cycle INT,
            scanline INT,
            dot INT,
            timestamp TIMESTAMP,
            state_data JSON,
            FOREIGN KEY (run_id) REFERENCES {prefix}runs (id)
        )
        """)
        
        # Create quantum_analysis table
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}quantum_analysis (
            id INT AUTO_INCREMENT PRIMARY KEY,
            run_id INT,
            cycle_range VARCHAR(100),
            analysis_mode VARCHAR(50),
            quantum_entropy FLOAT,
            coherence_measure FLOAT,
            frequency_data JSON,
            analysis_summary TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES {prefix}runs (id)
        )
        """)
        
        # Create timing_analysis table
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}timing_analysis (
            id INT AUTO_INCREMENT PRIMARY KEY,
            run_id INT,
            cycle_range VARCHAR(100),
            timing_patterns JSON,
            timing_anomalies JSON,
            statistics JSON,
            timestamp TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES {prefix}runs (id)
        )
        """)
        
        # Create register_analysis table
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}register_analysis (
            id INT AUTO_INCREMENT PRIMARY KEY,
            run_id INT,
            register_name VARCHAR(100),
            cycle_range VARCHAR(100),
            change_frequency FLOAT,
            value_entropy FLOAT,
            common_values JSON,
            timestamp TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES {prefix}runs (id)
        )
        """)
        
        self.connection.commit()
    
    def disconnect(self) -> bool:
        """
        Disconnect from the database.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            return True
            
        try:
            if self.connection:
                self.connection.close()
                
            self.connection = None
            self.cursor = None
            self.connected = False
            
            logger.info(f"Disconnected from {self.name} database")
            self._trigger_event('disconnect', None)
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from database: {e}")
            self._trigger_event('error', f"Disconnection error: {e}")
            return False
    
    def create_run(self, name: str, system: str, 
                 rom_path: Optional[str] = None,
                 analysis_mode: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Create a new run record.
        
        Args:
            name: Run name
            system: System type
            rom_path: Path to ROM file
            analysis_mode: Analysis mode
            metadata: Additional metadata
            
        Returns:
            Run ID if successful, None otherwise
        """
        if not self.connected:
            logger.error(f"Cannot create run: {self.name} is not connected")
            return None
            
        try:
            prefix = self.config["table_prefix"]
            
            # Convert metadata to JSON
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Get current time
            now = datetime.now()
            
            if self.db_type == "sqlite":
                self.cursor.execute(
                    f"INSERT INTO {prefix}runs (name, system, rom_path, analysis_mode, start_time, metadata) "
                    f"VALUES (?, ?, ?, ?, ?, ?)",
                    (name, system, rom_path, analysis_mode, now, metadata_json)
                )
                
                self.connection.commit()
                return self.cursor.lastrowid
                
            elif self.db_type == "postgresql":
                self.cursor.execute(
                    f"INSERT INTO {prefix}runs (name, system, rom_path, analysis_mode, start_time, metadata) "
                    f"VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                    (name, system, rom_path, analysis_mode, now, metadata_json)
                )
                
                run_id = self.cursor.fetchone()[0]
                self.connection.commit()
                return run_id
                
            elif self.db_type == "mysql":
                self.cursor.execute(
                    f"INSERT INTO {prefix}runs (name, system, rom_path, analysis_mode, start_time, metadata) "
                    f"VALUES (%s, %s, %s, %s, %s, %s)",
                    (name, system, rom_path, analysis_mode, now, metadata_json)
                )
                
                run_id = self.cursor.lastrowid
                self.connection.commit()
                return run_id
                
            else:
                logger.error(f"Unsupported database type: {self.db_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating run: {e}")
            self._trigger_event('error', f"Database error: {e}")
            return None
    
    def finish_run(self, run_id: int) -> bool:
        """
        Mark a run as finished.
        
        Args:
            run_id: Run ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error(f"Cannot finish run: {self.name} is not connected")
            return False
            
        try:
            prefix = self.config["table_prefix"]
            
            # Get current time
            now = datetime.now()
            
            if self.db_type == "sqlite":
                self.cursor.execute(
                    f"UPDATE {prefix}runs SET end_time = ? WHERE id = ?",
                    (now, run_id)
                )
                
            elif self.db_type in ["postgresql", "mysql"]:
                self.cursor.execute(
                    f"UPDATE {prefix}runs SET end_time = %s WHERE id = %s",
                    (now, run_id)
                )
                
            else:
                logger.error(f"Unsupported database type: {self.db_type}")
                return False
                
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error finishing run: {e}")
            self._trigger_event('error', f"Database error: {e}")
            return False
    
    def store_states(self, run_id: int, states: List[Dict[str, Any]]) -> bool:
        """
        Store state snapshots.
        
        Args:
            run_id: Run ID
            states: List of state dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error(f"Cannot store states: {self.name} is not connected")
            return False
            
        try:
            prefix = self.config["table_prefix"]
            
            # Get current time
            now = datetime.now()
            
            # Store states in batches
            batch_size = self.config["batch_size"]
            
            for i in range(0, len(states), batch_size):
                batch = states[i:i+batch_size]
                
                if self.db_type == "sqlite":
                    for state in batch:
                        self.cursor.execute(
                            f"INSERT INTO {prefix}states (run_id, cycle, scanline, dot, timestamp, state_data) "
                            f"VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                run_id,
                                state.get("cycle"),
                                state.get("scanline"),
                                state.get("dot"),
                                now,
                                json.dumps(state)
                            )
                        )
                        
                elif self.db_type in ["postgresql", "mysql"]:
                    # Prepare values for batch insert
                    values = []
                    for state in batch:
                        values.append((
                            run_id,
                            state.get("cycle"),
                            state.get("scanline"),
                            state.get("dot"),
                            now,
                            json.dumps(state)
                        ))
                        
                    # Perform batch insert
                    if self.db_type == "postgresql":
                        args_str = ",".join(["%s"] * len(values))
                        self.cursor.execute(
                            f"INSERT INTO {prefix}states (run_id, cycle, scanline, dot, timestamp, state_data) "
                            f"VALUES {args_str}",
                            values
                        )
                    else:  # MySQL
                        args_str = ",".join(["(%s, %s, %s, %s, %s, %s)"] * len(values))
                        flat_values = [item for sublist in values for item in sublist]
                        self.cursor.execute(
                            f"INSERT INTO {prefix}states (run_id, cycle, scanline, dot, timestamp, state_data) "
                            f"VALUES {args_str}",
                            flat_values
                        )
                        
                else:
                    logger.error(f"Unsupported database type: {self.db_type}")
                    return False
                    
                self.connection.commit()
                
            return True
            
        except Exception as e:
            logger.error(f"Error storing states: {e}")
            self._trigger_event('error', f"Database error: {e}")
            return False
    
    def store_quantum_analysis(self, run_id: int, 
                             cycle_range: Optional[str] = None,
                             analysis_mode: Optional[str] = None,
                             results: Dict[str, Any] = None) -> bool:
        """
        Store quantum analysis results.
        
        Args:
            run_id: Run ID
            cycle_range: Cycle range (e.g., "0-1000")
            analysis_mode: Analysis mode
            results: Analysis results
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error(f"Cannot store quantum analysis: {self.name} is not connected")
            return False
            
        try:
            prefix = self.config["table_prefix"]
            
            # Get current time
            now = datetime.now()
            
            # Extract relevant fields
            quantum_entropy = results.get("quantum_entropy", 0.0)
            coherence_measure = results.get("coherence_measure", 0.0)
            frequency_data = json.dumps(results.get("frequency_data", {}))
            analysis_summary = results.get("analysis_summary", "")
            
            if self.db_type == "sqlite":
                self.cursor.execute(
                    f"INSERT INTO {prefix}quantum_analysis "
                    f"(run_id, cycle_range, analysis_mode, quantum_entropy, coherence_measure, "
                    f"frequency_data, analysis_summary, timestamp) "
                    f"VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        cycle_range,
                        analysis_mode,
                        quantum_entropy,
                        coherence_measure,
                        frequency_data,
                        analysis_summary,
                        now
                    )
                )
                
            elif self.db_type in ["postgresql", "mysql"]:
                self.cursor.execute(
                    f"INSERT INTO {prefix}quantum_analysis "
                    f"(run_id, cycle_range, analysis_mode, quantum_entropy, coherence_measure, "
                    f"frequency_data, analysis_summary, timestamp) "
                    f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        run_id,
                        cycle_range,
                        analysis_mode,
                        quantum_entropy,
                        coherence_measure,
                        frequency_data,
                        analysis_summary,
                        now
                    )
                )
                
            else:
                logger.error(f"Unsupported database type: {self.db_type}")
                return False
                
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error storing quantum analysis: {e}")
            self._trigger_event('error', f"Database error: {e}")
            return False
    
    def store_timing_analysis(self, run_id: int, 
                            cycle_range: Optional[str] = None,
                            results: Dict[str, Any] = None) -> bool:
        """
        Store timing analysis results.
        
        Args:
            run_id: Run ID
            cycle_range: Cycle range (e.g., "0-1000")
            results: Analysis results
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error(f"Cannot store timing analysis: {self.name} is not connected")
            return False
            
        try:
            prefix = self.config["table_prefix"]
            
            # Get current time
            now = datetime.now()
            
            # Extract relevant fields
            timing_patterns = json.dumps(results.get("cycle_patterns", []))
            timing_anomalies = json.dumps(results.get("timing_anomalies", []))
            statistics = json.dumps(results.get("statistics", {}))
            
            if self.db_type == "sqlite":
                self.cursor.execute(
                    f"INSERT INTO {prefix}timing_analysis "
                    f"(run_id, cycle_range, timing_patterns, timing_anomalies, statistics, timestamp) "
                    f"VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        cycle_range,
                        timing_patterns,
                        timing_anomalies,
                        statistics,
                        now
                    )
                )
                
            elif self.db_type in ["postgresql", "mysql"]:
                self.cursor.execute(
                    f"INSERT INTO {prefix}timing_analysis "
                    f"(run_id, cycle_range, timing_patterns, timing_anomalies, statistics, timestamp) "
                    f"VALUES (%s, %s, %s, %s, %s, %s)",
                    (
                        run_id,
                        cycle_range,
                        timing_patterns,
                        timing_anomalies,
                        statistics,
                        now
                    )
                )
                
            else:
                logger.error(f"Unsupported database type: {self.db_type}")
                return False
                
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error storing timing analysis: {e}")
            self._trigger_event('error', f"Database error: {e}")
            return False
    
    def store_register_analysis(self, run_id: int, 
                              register_name: str,
                              cycle_range: Optional[str] = None,
                              results: Dict[str, Any] = None) -> bool:
        """
        Store register analysis results.
        
        Args:
            run_id: Run ID
            register_name: Register name
            cycle_range: Cycle range (e.g., "0-1000")
            results: Analysis results
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error(f"Cannot store register analysis: {self.name} is not connected")
            return False
            
        try:
            prefix = self.config["table_prefix"]
            
            # Get current time
            now = datetime.now()
            
            # Extract relevant fields
            change_frequency = results.get("change_frequency", 0.0)
            value_entropy = results.get("value_entropy", 0.0)
            common_values = json.dumps(results.get("most_common_values", []))
            
            if self.db_type == "sqlite":
                self.cursor.execute(
                    f"INSERT INTO {prefix}register_analysis "
                    f"(run_id, register_name, cycle_range, change_frequency, value_entropy, "
                    f"common_values, timestamp) "
                    f"VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        register_name,
                        cycle_range,
                        change_frequency,
                        value_entropy,
                        common_values,
                        now
                    )
                )
                
            elif self.db_type in ["postgresql", "mysql"]:
                self.cursor.execute(
                    f"INSERT INTO {prefix}register_analysis "
                    f"(run_id, register_name, cycle_range, change_frequency, value_entropy, "
                    f"common_values, timestamp) "
                    f"VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        run_id,
                        register_name,
                        cycle_range,
                        change_frequency,
                        value_entropy,
                        common_values,
                        now
                    )
                )
                
            else:
                logger.error(f"Unsupported database type: {self.db_type}")
                return False
                
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error storing register analysis: {e}")
            self._trigger_event('error', f"Database error: {e}")
            return False
    
    def query_runs(self, system: Optional[str] = None, 
                 limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query runs from the database.
        
        Args:
            system: Filter by system type
            limit: Maximum number of results
            
        Returns:
            List of run dictionaries
        """
        if not self.connected:
            logger.error(f"Cannot query runs: {self.name} is not connected")
            return []
            
        try:
            prefix = self.config["table_prefix"]
            
            if system:
                if self.db_type == "sqlite":
                    self.cursor.execute(
                        f"SELECT * FROM {prefix}runs WHERE system = ? ORDER BY start_time DESC LIMIT ?",
                        (system, limit)
                    )
                else:  # PostgreSQL or MySQL
                    self.cursor.execute(
                        f"SELECT * FROM {prefix}runs WHERE system = %s ORDER BY start_time DESC LIMIT %s",
                        (system, limit)
                    )
            else:
                if self.db_type == "sqlite":
                    self.cursor.execute(
                        f"SELECT * FROM {prefix}runs ORDER BY start_time DESC LIMIT ?",
                        (limit,)
                    )
                else:  # PostgreSQL or MySQL
                    self.cursor.execute(
                        f"SELECT * FROM {prefix}runs ORDER BY start_time DESC LIMIT %s",
                        (limit,)
                    )
                    
            # Get column names
            if self.db_type == "sqlite":
                columns = [column[0] for column in self.cursor.description]
            else:  # PostgreSQL or MySQL
                columns = [column[0] for column in self.cursor.description]
                
            # Convert to list of dictionaries
            runs = []
            for row in self.cursor.fetchall():
                run = dict(zip(columns, row))
                
                # Convert metadata
                if "metadata" in run and run["metadata"]:
                    if isinstance(run["metadata"], str):
                        run["metadata"] = json.loads(run["metadata"])
                        
                runs.append(run)
                
            return runs
            
        except Exception as e:
            logger.error(f"Error querying runs: {e}")
            self._trigger_event('error', f"Database error: {e}")
            return []
    
    def export_data(self, run_id: int, filename: str, format: str = 'json') -> bool:
        """
        Export data for a run to a file.
        
        Args:
            run_id: Run ID
            filename: Output filename
            format: Export format ('json', 'csv', or 'sqlite')
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error(f"Cannot export data: {self.name} is not connected")
            return False
            
        try:
            prefix = self.config["table_prefix"]
            
            # Query run details
            if self.db_type == "sqlite":
                self.cursor.execute(
                    f"SELECT * FROM {prefix}runs WHERE id = ?",
                    (run_id,)
                )
            else:  # PostgreSQL or MySQL
                self.cursor.execute(
                    f"SELECT * FROM {prefix}runs WHERE id = %s",
                    (run_id,)
                )
                
            run = self.cursor.fetchone()
            
            if not run:
                logger.error(f"Run not found: {run_id}")
                return False
                
            # Get column names
            if self.db_type == "sqlite":
                run_columns = [column[0] for column in self.cursor.description]
            else:  # PostgreSQL or MySQL
                run_columns = [column[0] for column in self.cursor.description]
                
            run_dict = dict(zip(run_columns, run))
            
            # Query states
            if self.db_type == "sqlite":
                self.cursor.execute(
                    f"SELECT * FROM {prefix}states WHERE run_id = ? ORDER BY cycle",
                    (run_id,)
                )
            else:  # PostgreSQL or MySQL
                self.cursor.execute(
                    f"SELECT * FROM {prefix}states WHERE run_id = %s ORDER BY cycle",
                    (run_id,)
                )
                
            # Get column names
            if self.db_type == "sqlite":
                state_columns = [column[0] for column in self.cursor.description]
            else:  # PostgreSQL or MySQL
                state_columns = [column[0] for column in self.cursor.description]
                
            states = []
            for row in self.cursor.fetchall():
                state = dict(zip(state_columns, row))
                
                # Convert state data
                if "state_data" in state and state["state_data"]:
                    if isinstance(state["state_data"], str):
                        state["state_data"] = json.loads(state["state_data"])
                        
                states.append(state)
                
            # Query quantum analysis
            if self.db_type == "sqlite":
                self.cursor.execute(
                    f"SELECT * FROM {prefix}quantum_analysis WHERE run_id = ? ORDER BY id",
                    (run_id,)
                )
            else:  # PostgreSQL or MySQL
                self.cursor.execute(
                    f"SELECT * FROM {prefix}quantum_analysis WHERE run_id = %s ORDER BY id",
                    (run_id,)
                )
                
            # Get column names
            if self.db_type == "sqlite":
                quantum_columns = [column[0] for column in self.cursor.description]
            else:  # PostgreSQL or MySQL
                quantum_columns = [column[0] for column in self.cursor.description]
                
            quantum_analysis = []
            for row in self.cursor.fetchall():
                analysis = dict(zip(quantum_columns, row))
                
                # Convert frequency data
                if "frequency_data" in analysis and analysis["frequency_data"]:
                    if isinstance(analysis["frequency_data"], str):
                        analysis["frequency_data"] = json.loads(analysis["frequency_data"])
                        
                quantum_analysis.append(analysis)
                
            # Query timing analysis
            if self.db_type == "sqlite":
                self.cursor.execute(
                    f"SELECT * FROM {prefix}timing_analysis WHERE run_id = ? ORDER BY id",
                    (run_id,)
                )
            else:  # PostgreSQL or MySQL
                self.cursor.execute(
                    f"SELECT * FROM {prefix}timing_analysis WHERE run_id = %s ORDER BY id",
                    (run_id,)
                )
                
            # Get column names
            if self.db_type == "sqlite":
                timing_columns = [column[0] for column in self.cursor.description]
            else:  # PostgreSQL or MySQL
                timing_columns = [column[0] for column in self.cursor.description]
                
            timing_analysis = []
            for row in self.cursor.fetchall():
                analysis = dict(zip(timing_columns, row))
                
                # Convert JSON fields
                for field in ["timing_patterns", "timing_anomalies", "statistics"]:
                    if field in analysis and analysis[field]:
                        if isinstance(analysis[field], str):
                            analysis[field] = json.loads(analysis[field])
                            
                timing_analysis.append(analysis)
                
            # Query register analysis
            if self.db_type == "sqlite":
                self.cursor.execute(
                    f"SELECT * FROM {prefix}register_analysis WHERE run_id = ? ORDER BY id",
                    (run_id,)
                )
            else:  # PostgreSQL or MySQL
                self.cursor.execute(
                    f"SELECT * FROM {prefix}register_analysis WHERE run_id = %s ORDER BY id",
                    (run_id,)
                )
                
            # Get column names
            if self.db_type == "sqlite":
                register_columns = [column[0] for column in self.cursor.description]
            else:  # PostgreSQL or MySQL
                register_columns = [column[0] for column in self.cursor.description]
                
            register_analysis = []
            for row in self.cursor.fetchall():
                analysis = dict(zip(register_columns, row))
                
                # Convert common values
                if "common_values" in analysis and analysis["common_values"]:
                    if isinstance(analysis["common_values"], str):
                        analysis["common_values"] = json.loads(analysis["common_values"])
                        
                register_analysis.append(analysis)
                
            # Prepare export data
            export_data = {
                "run": run_dict,
                "states": states,
                "quantum_analysis": quantum_analysis,
                "timing_analysis": timing_analysis,
                "register_analysis": register_analysis
            }
            
            # Export based on format
            if format == 'json':
                # Convert datetime objects to strings
                def json_serial(obj):
                    if isinstance(obj, (datetime,)):
                        return obj.isoformat()
                    raise TypeError(f"Type {type(obj)} not serializable")
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, default=json_serial, indent=2)
                    
            elif format == 'csv':
                # Create directory if needed
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                # Export each table to a separate CSV file
                base_name, ext = os.path.splitext(filename)
                
                # Export run
                with open(f"{base_name}_run.csv", 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=run_columns)
                    writer.writeheader()
                    writer.writerow({k: v for k, v in run_dict.items() if k in run_columns})
                    
                # Export states
                if states:
                    with open(f"{base_name}_states.csv", 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=[c for c in state_columns if c != 'state_data'])
                        writer.writeheader()
                        for state in states:
                            writer.writerow({k: v for k, v in state.items() if k != 'state_data'})
                            
                # Export quantum analysis
                if quantum_analysis:
                    with open(f"{base_name}_quantum_analysis.csv", 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=[c for c in quantum_columns if c != 'frequency_data'])
                        writer.writeheader()
                        for analysis in quantum_analysis:
                            writer.writerow({k: v for k, v in analysis.items() if k != 'frequency_data'})
                            
                # Export timing analysis
                if timing_analysis:
                    with open(f"{base_name}_timing_analysis.csv", 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=[c for c in timing_columns if c not in ['timing_patterns', 'timing_anomalies', 'statistics']])
                        writer.writeheader()
                        for analysis in timing_analysis:
                            writer.writerow({k: v for k, v in analysis.items() if k not in ['timing_patterns', 'timing_anomalies', 'statistics']})
                            
                # Export register analysis
                if register_analysis:
                    with open(f"{base_name}_register_analysis.csv", 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=[c for c in register_columns if c != 'common_values'])
                        writer.writeheader()
                        for analysis in register_analysis:
                            writer.writerow({k: v for k, v in analysis.items() if k != 'common_values'})
                            
            elif format == 'sqlite':
                # Create a new SQLite database
                import sqlite3
                
                # Create temporary file if filename is just a directory
                if os.path.isdir(filename):
                    filename = os.path.join(filename, f"run_{run_id}.db")
                    
                # Create directory if needed
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                # Create connection
                conn = sqlite3.connect(filename)
                cursor = conn.cursor()
                
                # Create tables
                cursor.execute(f"""
                CREATE TABLE runs (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    system TEXT,
                    rom_path TEXT,
                    analysis_mode TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    metadata TEXT
                )
                """)
                
                cursor.execute(f"""
                CREATE TABLE states (
                    id INTEGER PRIMARY KEY,
                    run_id INTEGER,
                    cycle INTEGER,
                    scanline INTEGER,
                    dot INTEGER,
                    timestamp TIMESTAMP,
                    state_data TEXT
                )
                """)
                
                cursor.execute(f"""
                CREATE TABLE quantum_analysis (
                    id INTEGER PRIMARY KEY,
                    run_id INTEGER,
                    cycle_range TEXT,
                    analysis_mode TEXT,
                    quantum_entropy REAL,
                    coherence_measure REAL,
                    frequency_data TEXT,
                    analysis_summary TEXT,
                    timestamp TIMESTAMP
                )
                """)
                
                cursor.execute(f"""
                CREATE TABLE timing_analysis (
                    id INTEGER PRIMARY KEY,
                    run_id INTEGER,
                    cycle_range TEXT,
                    timing_patterns TEXT,
                    timing_anomalies TEXT,
                    statistics TEXT,
                    timestamp TIMESTAMP
                )
                """)
                
                cursor.execute(f"""
                CREATE TABLE register_analysis (
                    id INTEGER PRIMARY KEY,
                    run_id INTEGER,
                    register_name TEXT,
                    cycle_range TEXT,
                    change_frequency REAL,
                    value_entropy REAL,
                    common_values TEXT,
                    timestamp TIMESTAMP
                )
                """)
                
                # Insert run data
                run_values = []
                for col in run_columns:
                    value = run_dict.get(col)
                    if col == 'metadata' and value and not isinstance(value, str):
                        value = json.dumps(value)
                    run_values.append(value)
                    
                cursor.execute(
                    f"INSERT INTO runs VALUES ({', '.join(['?'] * len(run_columns))})",
                    run_values
                )
                
                # Insert states
                for state in states:
                    state_values = []
                    for col in state_columns:
                        value = state.get(col)
                        if col == 'state_data' and value and not isinstance(value, str):
                            value = json.dumps(value)
                        state_values.append(value)
                        
                    cursor.execute(
                        f"INSERT INTO states VALUES ({', '.join(['?'] * len(state_columns))})",
                        state_values
                    )
                    
                # Insert quantum analysis
                for analysis in quantum_analysis:
                    quantum_values = []
                    for col in quantum_columns:
                        value = analysis.get(col)
                        if col == 'frequency_data' and value and not isinstance(value, str):
                            value = json.dumps(value)
                        quantum_values.append(value)
                        
                    cursor.execute(
                        f"INSERT INTO quantum_analysis VALUES ({', '.join(['?'] * len(quantum_columns))})",
                        quantum_values
                    )
                    
                # Insert timing analysis
                for analysis in timing_analysis:
                    timing_values = []
                    for col in timing_columns:
                        value = analysis.get(col)
                        if col in ['timing_patterns', 'timing_anomalies', 'statistics'] and value and not isinstance(value, str):
                            value = json.dumps(value)
                        timing_values.append(value)
                        
                    cursor.execute(
                        f"INSERT INTO timing_analysis VALUES ({', '.join(['?'] * len(timing_columns))})",
                        timing_values
                    )
                    
                # Insert register analysis
                for analysis in register_analysis:
                    register_values = []
                    for col in register_columns:
                        value = analysis.get(col)
                        if col == 'common_values' and value and not isinstance(value, str):
                            value = json.dumps(value)
                        register_values.append(value)
                        
                    cursor.execute(
                        f"INSERT INTO register_analysis VALUES ({', '.join(['?'] * len(register_columns))})",
                        register_values
                    )
                    
                # Commit and close
                conn.commit()
                conn.close()
                
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
            logger.info(f"Exported data for run {run_id} to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            self._trigger_event('error', f"Export error: {e}")
            return False


class IntegrationManager:
    """
    Manager for multiple integration adapters.
    
    This class provides a unified interface for managing and coordinating
    multiple integration adapters.
    """
    
    def __init__(self):
        """Initialize the integration manager."""
        self.adapters = {}
        self.active_adapters = set()
        logger.info("Initialized integration manager")
    
    def register_adapter(self, adapter: IntegrationAdapter) -> None:
        """
        Register an integration adapter.
        
        Args:
            adapter: Integration adapter to register
        """
        self.adapters[adapter.name] = adapter
        logger.info(f"Registered {adapter.name} adapter")
    
    def unregister_adapter(self, name: str) -> bool:
        """
        Unregister an integration adapter.
        
        Args:
            name: Name of adapter to unregister
            
        Returns:
            True if adapter was unregistered, False if not found
        """
        if name in self.adapters:
            # Disconnect if active
            if name in self.active_adapters:
                self.adapters[name].disconnect()
                self.active_adapters.remove(name)
                
            # Remove adapter
            del self.adapters[name]
            logger.info(f"Unregistered {name} adapter")
            return True
        else:
            logger.warning(f"Adapter not found: {name}")
            return False
    
    def connect(self, name: str) -> bool:
        """
        Connect to an integration adapter.
        
        Args:
            name: Name of adapter to connect
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.adapters:
            logger.error(f"Adapter not found: {name}")
            return False
            
        # Connect adapter
        adapter = self.adapters[name]
        success = adapter.connect()
        
        if success:
            self.active_adapters.add(name)
            
        return success
    
    def connect_all(self) -> Dict[str, bool]:
        """
        Connect to all registered adapters.
        
        Returns:
            Dictionary of adapter names and connection results
        """
        results = {}
        
        for name, adapter in self.adapters.items():
            results[name] = self.connect(name)
            
        return results
    
    def disconnect(self, name: str) -> bool:
        """
        Disconnect from an integration adapter.
        
        Args:
            name: Name of adapter to disconnect
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.adapters:
            logger.error(f"Adapter not found: {name}")
            return False
            
        # Disconnect adapter
        adapter = self.adapters[name]
        success = adapter.disconnect()
        
        if success and name in self.active_adapters:
            self.active_adapters.remove(name)
            
        return success
    
    def disconnect_all(self) -> Dict[str, bool]:
        """
        Disconnect from all active adapters.
        
        Returns:
            Dictionary of adapter names and disconnection results
        """
        results = {}
        
        for name in list(self.active_adapters):
            results[name] = self.disconnect(name)
            
        return results
    
    def send_data(self, name: str, data: Any) -> bool:
        """
        Send data to an integration adapter.
        
        Args:
            name: Name of adapter to send data to
            data: Data to send
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.adapters:
            logger.error(f"Adapter not found: {name}")
            return False
            
        if name not in self.active_adapters:
            logger.error(f"Adapter not connected: {name}")
            return False
            
        # Send data
        return self.adapters[name].send_data(data)
    
    def broadcast_data(self, data: Any, adapter_type: Optional[IntegrationType] = None) -> Dict[str, bool]:
        """
        Broadcast data to multiple adapters.
        
        Args:
            data: Data to send
            adapter_type: Only send to adapters of this type (None for all)
            
        Returns:
            Dictionary of adapter names and send results
        """
        results = {}
        
        for name in self.active_adapters:
            adapter = self.adapters[name]
            
            # Filter by type if specified
            if adapter_type is not None and adapter.integration_type != adapter_type:
                continue
                
            # Send data
            results[name] = adapter.send_data(data)
            
        return results
    
    def receive_data(self, name: str) -> Optional[Any]:
        """
        Receive data from an integration adapter.
        
        Args:
            name: Name of adapter to receive data from
            
        Returns:
            Received data or None if no data available
        """
        if name not in self.adapters:
            logger.error(f"Adapter not found: {name}")
            return None
            
        if name not in self.active_adapters:
            logger.error(f"Adapter not connected: {name}")
            return None
            
        # Receive data
        return self.adapters[name].receive_data()
    
    def get_adapter(self, name: str) -> Optional[IntegrationAdapter]:
        """
        Get an integration adapter by name.
        
        Args:
            name: Adapter name
            
        Returns:
            Integration adapter or None if not found
        """
        return self.adapters.get(name)
    
    def get_adapters_by_type(self, adapter_type: IntegrationType) -> List[IntegrationAdapter]:
        """
        Get all adapters of a specific type.
        
        Args:
            adapter_type: Adapter type
            
        Returns:
            List of matching adapters
        """
        return [adapter for adapter in self.adapters.values() 
               if adapter.integration_type == adapter_type]
    
    def get_active_adapters(self) -> List[IntegrationAdapter]:
        """
        Get all active adapters.
        
        Returns:
            List of active adapters
        """
        return [self.adapters[name] for name in self.active_adapters]
    
    def create_emulator_adapter(self, name: str, 
                              emulator_type: str,
                              connection_type: ConnectionType,
                              executable_path: Optional[str] = None) -> EmulatorAdapter:
        """
        Create and register an emulator adapter.
        
        Args:
            name: Adapter name
            emulator_type: Type of emulator
            connection_type: Connection method
            executable_path: Path to emulator executable
            
        Returns:
            Created emulator adapter
        """
        adapter = EmulatorAdapter(name, emulator_type, connection_type, executable_path)
        self.register_adapter(adapter)
        return adapter
    
    def create_hardware_adapter(self, name: str, 
                              device_type: str,
                              connection_type: ConnectionType,
                              device_path: Optional[str] = None) -> HardwareAdapter:
        """
        Create and register a hardware adapter.
        
        Args:
            name: Adapter name
            device_type: Type of device
            connection_type: Connection method
            device_path: Path to device
            
        Returns:
            Created hardware adapter
        """
        adapter = HardwareAdapter(name, device_type, connection_type, device_path)
        self.register_adapter(adapter)
        return adapter
    
    def create_logic_analyzer_adapter(self, name: str, 
                                   device_type: str,
                                   connection_type: ConnectionType,
                                   device_path: Optional[str] = None) -> LogicAnalyzerAdapter:
        """
        Create and register a logic analyzer adapter.
        
        Args:
            name: Adapter name
            device_type: Type of logic analyzer
            connection_type: Connection method
            device_path: Path to device
            
        Returns:
            Created logic analyzer adapter
        """
        adapter = LogicAnalyzerAdapter(name, device_type, connection_type, device_path)
        self.register_adapter(adapter)
        return adapter
    
    def create_database_adapter(self, name: str, 
                              db_type: str,
                              connection_string: Optional[str] = None) -> DatabaseAdapter:
        """
        Create and register a database adapter.
        
        Args:
            name: Adapter name
            db_type: Type of database
            connection_string: Database connection string
            
        Returns:
            Created database adapter
        """
        adapter = DatabaseAdapter(name, db_type, connection_string)
        self.register_adapter(adapter)
        return adapter
    
    def configure_adapter(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Configure an adapter.
        
        Args:
            name: Adapter name
            config: Configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.adapters:
            logger.error(f"Adapter not found: {name}")
            return False
            
        # Configure adapter
        return self.adapters[name].configure(config)