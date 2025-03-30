"""
Event management system for hardware cycle events.

This module provides an event management system for monitoring and handling
hardware cycle-level events in the emulator. It allows registering event handlers
and triggering events when specific conditions are met.
"""

import logging
import time
from typing import Dict, List, Callable, Any, Optional, Tuple, Set, Union
from enum import Enum, auto
from collections import defaultdict

logger = logging.getLogger("QuantumSignalEmulator.EventManager")

class EventPriority(Enum):
    """Event handler priority levels."""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()

class EventType(Enum):
    """Types of hardware events."""
    # System events
    SYSTEM_RESET = auto()
    SYSTEM_SHUTDOWN = auto()
    
    # CPU events
    CPU_INSTRUCTION = auto()
    CPU_INTERRUPT = auto()
    CPU_DMA = auto()
    
    # Memory events
    MEMORY_READ = auto()
    MEMORY_WRITE = auto()
    
    # Video events
    VIDEO_SCANLINE = auto()
    VIDEO_FRAME = auto()
    VIDEO_VBLANK = auto()
    
    # Audio events
    AUDIO_SAMPLE = auto()
    
    # Register events
    REGISTER_READ = auto()
    REGISTER_WRITE = auto()
    REGISTER_CHANGE = auto()
    
    # Analysis events
    ANALYSIS_START = auto()
    ANALYSIS_COMPLETE = auto()
    TIMING_ANOMALY = auto()
    
    # Custom event (can be used for system-specific events)
    CUSTOM = auto()

class Event:
    """
    Event object for hardware events.
    
    This class represents an event in the emulator, including its type,
    payload, timestamp, and source.
    """
    
    def __init__(self, event_type: EventType, 
                source: str, 
                payload: Optional[Dict[str, Any]] = None,
                timestamp: Optional[float] = None):
        """
        Initialize event.
        
        Args:
            event_type: Type of event
            source: Source component of event
            payload: Additional event data
            timestamp: Event timestamp (None for auto)
        """
        self.type = event_type
        self.source = source
        self.payload = payload or {}
        self.timestamp = timestamp or time.time()
        self.handled = False
        
    def __str__(self) -> str:
        """String representation of event."""
        return f"Event({self.type.name}, source={self.source}, payload={self.payload})"

# Type alias for event handler functions
EventHandler = Callable[[Event], None]

class EventManager:
    """
    Event management system for hardware cycle events.
    
    This class provides an event bus for registering handlers and dispatching
    hardware events throughout the emulator.
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize event manager.
        
        Args:
            max_history: Maximum number of events to keep in history
        """
        # Event handlers by type and priority
        self.handlers: Dict[EventType, Dict[EventPriority, List[EventHandler]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Event history
        self.event_history: List[Event] = []
        self.max_history = max_history
        
        # Statistics
        self.stats = {
            "events_triggered": 0,
            "events_handled": 0,
            "handlers_called": 0
        }
        
        # Filters for history tracking
        self.history_filters: Set[EventType] = set()
        
        logger.debug("EventManager initialized")
    
    def register_handler(self, event_type: EventType, 
                        handler: EventHandler,
                        priority: EventPriority = EventPriority.NORMAL) -> None:
        """
        Register an event handler.
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
            priority: Handler priority
        """
        self.handlers[event_type][priority].append(handler)
        logger.debug(f"Registered handler for {event_type.name} with {priority.name} priority")
    
    def unregister_handler(self, event_type: EventType, 
                          handler: EventHandler) -> bool:
        """
        Unregister an event handler.
        
        Args:
            event_type: Type of event
            handler: Handler function to remove
            
        Returns:
            True if handler was removed, False if not found
        """
        for priority in self.handlers[event_type].values():
            if handler in priority:
                priority.remove(handler)
                logger.debug(f"Unregistered handler for {event_type.name}")
                return True
        
        return False
    
    def trigger_event(self, event: Event) -> bool:
        """
        Trigger an event and call handlers.
        
        Args:
            event: Event to trigger
            
        Returns:
            True if event was handled by at least one handler
        """
        self.stats["events_triggered"] += 1
        
        # Add to history if not filtered
        if not self.history_filters or event.type in self.history_filters:
            self.event_history.append(event)
            
            # Trim history if needed
            if len(self.event_history) > self.max_history:
                self.event_history = self.event_history[-self.max_history:]
        
        # Get handlers for this event type
        event_handlers = self.handlers[event.type]
        
        # No handlers registered
        if not event_handlers:
            return False
            
        # Call handlers in priority order
        handled = False
        
        # Process in order: CRITICAL, HIGH, NORMAL, LOW
        priorities = [
            EventPriority.CRITICAL,
            EventPriority.HIGH,
            EventPriority.NORMAL,
            EventPriority.LOW
        ]
        
        for priority in priorities:
            for handler in event_handlers[priority]:
                try:
                    handler(event)
                    self.stats["handlers_called"] += 1
                    handled = True
                except Exception as e:
                    logger.error(f"Error in event handler for {event.type.name}: {e}")
                
                # Stop if event was marked as handled (for handlers that set this)
                if event.handled:
                    break
                    
            # Stop if event was marked as handled
            if event.handled:
                break
                
        if handled:
            self.stats["events_handled"] += 1
            
        return handled
    
    def create_event(self, event_type: EventType, 
                    source: str, 
                    payload: Optional[Dict[str, Any]] = None) -> Event:
        """
        Create and trigger an event.
        
        Args:
            event_type: Type of event
            source: Source component of event
            payload: Additional event data
            
        Returns:
            The created and triggered event
        """
        event = Event(event_type, source, payload)
        self.trigger_event(event)
        return event
    
    def clear_handlers(self, event_type: Optional[EventType] = None) -> None:
        """
        Clear event handlers.
        
        Args:
            event_type: Type of event to clear (None for all)
        """
        if event_type is None:
            # Clear all handlers
            self.handlers.clear()
            logger.debug("Cleared all event handlers")
        else:
            # Clear handlers for specific event type
            self.handlers[event_type].clear()
            logger.debug(f"Cleared handlers for {event_type.name}")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get event statistics.
        
        Returns:
            Dictionary with event statistics
        """
        return self.stats.copy()
    
    def get_event_history(self, event_type: Optional[EventType] = None) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_type: Type of event to filter (None for all)
            
        Returns:
            List of events in history
        """
        if event_type is None:
            return self.event_history.copy()
        else:
            return [e for e in self.event_history if e.type == event_type]
    
    def set_history_filter(self, event_types: List[EventType]) -> None:
        """
        Set filter for which events are stored in history.
        
        Args:
            event_types: List of event types to track (empty for all)
        """
        self.history_filters = set(event_types)
        logger.debug(f"Set history filter to {[et.name for et in event_types]}")
    
    def clear_history(self) -> None:
        """Clear event history."""
        self.event_history.clear()
        logger.debug("Cleared event history")
    
    def add_timing_anomaly_detector(self, 
                                  threshold: float = 3.0, 
                                  window_size: int = 100) -> None:
        """
        Add a detector for timing anomalies.
        
        This registers handlers to detect abnormal timing patterns in
        CPU instructions, scanlines, or other timing-sensitive events.
        
        Args:
            threshold: Standard deviation threshold for anomaly detection
            window_size: Analysis window size
        """
        # Create timing anomaly detector for CPU instructions
        def cpu_timing_detector(event: Event) -> None:
            if "cycle_delta" not in event.payload:
                return
                
            # Simple detector: Get recent CPU events and check for outliers
            cpu_events = self.get_event_history(EventType.CPU_INSTRUCTION)[-window_size:]
            
            if len(cpu_events) < 10:  # Need enough data
                return
                
            # Calculate mean and std dev of cycle_delta
            deltas = [e.payload.get("cycle_delta", 0) for e in cpu_events if "cycle_delta" in e.payload]
            
            if not deltas:
                return
                
            mean_delta = sum(deltas) / len(deltas)
            variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
            std_dev = variance ** 0.5
            
            # Current delta
            current_delta = event.payload["cycle_delta"]
            
            # Check if current delta is an anomaly
            if abs(current_delta - mean_delta) > threshold * std_dev:
                # Create timing anomaly event
                anomaly_payload = {
                    "expected": mean_delta,
                    "actual": current_delta,
                    "deviation": current_delta - mean_delta,
                    "source_event": event
                }
                
                self.create_event(
                    EventType.TIMING_ANOMALY,
                    "anomaly_detector",
                    anomaly_payload
                )
        
        # Register handlers
        self.register_handler(EventType.CPU_INSTRUCTION, cpu_timing_detector, EventPriority.LOW)
        self.register_handler(EventType.VIDEO_SCANLINE, cpu_timing_detector, EventPriority.LOW)
        
        logger.info(f"Added timing anomaly detector (threshold={threshold}, window={window_size})")
    
    def register_register_change_monitor(self, register_names: List[str]) -> None:
        """
        Register handlers to monitor changes in specific registers.
        
        Args:
            register_names: List of register names to monitor
        """
        def register_monitor(event: Event) -> None:
            if "register" not in event.payload:
                return
                
            register_name = event.payload["register"]
            
            if register_name not in register_names:
                return
                
            # Create register change event
            change_payload = {
                "register": register_name,
                "old_value": event.payload.get("old_value"),
                "new_value": event.payload.get("new_value"),
                "source_event": event
            }
            
            self.create_event(
                EventType.REGISTER_CHANGE,
                "register_monitor",
                change_payload
            )
        
        # Register handler for register write events
        self.register_handler(EventType.REGISTER_WRITE, register_monitor, EventPriority.NORMAL)
        
        logger.info(f"Registered monitor for registers: {register_names}")
    
    def register_logger(self, 
                      event_types: List[EventType],
                      log_level: int = logging.INFO) -> None:
        """
        Register a handler to log specific event types.
        
        Args:
            event_types: List of event types to log
            log_level: Logging level for events
        """
        def event_logger(event: Event) -> None:
            logger.log(log_level, f"Event: {event}")
        
        # Register handler for each event type
        for event_type in event_types:
            self.register_handler(event_type, event_logger, EventPriority.LOW)
            
        logger.info(f"Registered logger for event types: {[et.name for et in event_types]}")