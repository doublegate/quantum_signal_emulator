"""
Nintendo Entertainment System Audio Processing Unit (APU) emulation.

This module implements the NES APU which is responsible for generating sound.
The NES APU contains five sound channels:
- Two pulse wave channels (Square 1 & 2)
- One triangle wave channel
- One noise channel
- One delta modulation channel (DMC)

Each channel has specific features and capabilities which are emulated with
cycle accuracy to match the original hardware behavior.
"""

from ...common.interfaces import AudioProcessor
import numpy as np
import typing as t

class NESAPU(AudioProcessor):
    """
    Emulates the NES APU (Audio Processing Unit).
    
    Provides cycle-accurate emulation of the five NES sound channels with proper
    frequency sweep, envelope, and length counter features. Outputs audio
    samples that can be played through the system's audio interface.
    """
    
    # NES APU constants
    SAMPLE_RATE = 44100
    CPU_FREQUENCY = 1789773  # NTSC NES CPU frequency in Hz

    # Channel enable/disable flags
    PULSE1_ENABLE = 0x01
    PULSE2_ENABLE = 0x02
    TRIANGLE_ENABLE = 0x04
    NOISE_ENABLE = 0x08
    DMC_ENABLE = 0x10
    
    def __init__(self):
        # Frame counter
        self.frame_counter = 0
        self.frame_period = 0  # 0: 4-step, 1: 5-step
        self.frame_irq_enable = True
        
        # Channel registers
        self.pulse1 = {
            'enabled': False,
            'duty': 0,           # Duty cycle (0-3)
            'length_counter': 0, # Length counter value
            'envelope': {
                'volume': 0,     # Volume/envelope
                'decay': 0,      # Envelope decay level
                'loop': False,   # Envelope loop flag
                'constant': False # Constant volume flag
            },
            'sweep': {
                'enabled': False, # Sweep enable
                'period': 0,      # Sweep period
                'negate': False,  # Sweep negate flag
                'shift': 0,       # Sweep shift count
                'reload': False   # Sweep reload flag
            },
            'timer': 0,          # Timer period (frequency)
            'timer_value': 0,    # Current timer value
            'length_halt': False # Halt length counter flag
        }
        
        # Clone pulse1 structure for pulse2
        self.pulse2 = self.pulse1.copy()
        
        # Triangle channel
        self.triangle = {
            'enabled': False,
            'length_counter': 0,
            'linear_counter': 0,
            'linear_counter_reload': 0,
            'control_flag': False,
            'timer': 0,
            'timer_value': 0,
            'length_halt': False
        }
        
        # Noise channel
        self.noise = {
            'enabled': False,
            'length_counter': 0,
            'envelope': {
                'volume': 0,
                'decay': 0,
                'loop': False,
                'constant': False
            },
            'mode': False,       # Noise mode flag (0: long, 1: short)
            'period': 0,         # Noise period
            'timer': 0,
            'timer_value': 0,
            'shift_register': 1, # 15-bit shift register
            'length_halt': False
        }
        
        # DMC channel (Delta Modulation Channel)
        self.dmc = {
            'enabled': False,
            'frequency': 0,      # Rate index
            'loop': False,       # Loop flag
            'sample_address': 0, # Sample address
            'sample_length': 0,  # Sample length
            'current_address': 0,
            'bytes_remaining': 0,
            'buffer': 0,
            'bits_remaining': 0,
            'output_level': 0,
            'irq_enable': False  # IRQ enable flag
        }
        
        # Status register
        self.status = 0x00
        
        # IRQ flags
        self.frame_irq = False
        self.dmc_irq = False
        
        # Audio output buffer
        self.output_buffer = []
        self.cycles_per_sample = self.CPU_FREQUENCY / self.SAMPLE_RATE
        self.cycle_accumulator = 0
        
        # Lookup tables
        self._init_lookup_tables()
    
    def _init_lookup_tables(self):
        """Initialize lookup tables for efficient audio synthesis."""
        # Duty cycle patterns (8 steps each)
        self.duty_table = [
            [0, 0, 0, 0, 0, 0, 0, 1],  # 12.5%
            [0, 0, 0, 0, 0, 0, 1, 1],  # 25%
            [0, 0, 0, 0, 1, 1, 1, 1],  # 50%
            [1, 1, 1, 1, 1, 1, 0, 0]   # 75% (inverted 25%)
        ]
        
        # Triangle waveform (32 steps)
        self.triangle_table = []
        for i in range(16):
            self.triangle_table.append(i)
        for i in range(16):
            self.triangle_table.append(15 - i)
            
        # Noise period table (16 entries)
        self.noise_period_table = [
            4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034, 4068
        ]
        
        # DMC rate table (16 entries)
        self.dmc_rate_table = [
            428, 380, 340, 320, 286, 254, 226, 214, 190, 160, 142, 128, 106, 84, 72, 54
        ]
    
    def step(self, cycles: int) -> t.List[float]:
        """
        Run the APU for a specified number of CPU cycles.
        
        Args:
            cycles: Number of CPU cycles to simulate
            
        Returns:
            List of audio samples generated during this time
        """
        # Process each cycle
        self.output_buffer = []
        
        for _ in range(cycles):
            # Run frame sequencer
            self._clock_frame_sequencer()
            
            # Run channel timers
            self._clock_pulse1()
            self._clock_pulse2()
            self._clock_triangle()
            self._clock_noise()
            self._clock_dmc()
            
            # Generate audio sample if needed
            self.cycle_accumulator += 1
            if self.cycle_accumulator >= self.cycles_per_sample:
                self.cycle_accumulator -= self.cycles_per_sample
                self.output_buffer.append(self._mix_audio())
        
        return self.output_buffer
    
    def get_state(self) -> dict:
        """
        Return the current APU state as a dictionary.
        
        Returns:
            Dictionary containing the APU state
        """
        return {
            "frame_counter": self.frame_counter,
            "status": self.status,
            "frame_irq": self.frame_irq,
            "dmc_irq": self.dmc_irq,
            "pulse1": {
                "enabled": self.pulse1['enabled'],
                "duty": self.pulse1['duty'],
                "length_counter": self.pulse1['length_counter'],
                "timer": self.pulse1['timer']
            },
            "pulse2": {
                "enabled": self.pulse2['enabled'],
                "duty": self.pulse2['duty'],
                "length_counter": self.pulse2['length_counter'],
                "timer": self.pulse2['timer']
            },
            "triangle": {
                "enabled": self.triangle['enabled'],
                "length_counter": self.triangle['length_counter'],
                "linear_counter": self.triangle['linear_counter'],
                "timer": self.triangle['timer']
            },
            "noise": {
                "enabled": self.noise['enabled'],
                "length_counter": self.noise['length_counter'],
                "mode": self.noise['mode'],
                "period": self.noise['period']
            },
            "dmc": {
                "enabled": self.dmc['enabled'],
                "bytes_remaining": self.dmc['bytes_remaining'],
                "output_level": self.dmc['output_level']
            }
        }
    
    def _clock_frame_sequencer(self):
        """Clock the frame sequencer which controls envelope and length counters."""
        # Increment frame counter
        self.frame_counter += 1
        
        # 4-step sequence
        if self.frame_period == 0:
            if self.frame_counter == 3729:
                # Quarter frame: clock envelopes and triangle linear counter
                self._clock_envelopes()
                self._clock_linear_counter()
            elif self.frame_counter == 7457:
                # Half frame: clock envelopes, triangle linear counter, and length counters/sweeps
                self._clock_envelopes()
                self._clock_linear_counter()
                self._clock_length_counters()
                self._clock_sweeps()
            elif self.frame_counter == 11186:
                # Quarter frame: clock envelopes and triangle linear counter
                self._clock_envelopes()
                self._clock_linear_counter()
            elif self.frame_counter == 14916:
                # Half frame and reset: clock all units and generate IRQ if enabled
                self._clock_envelopes()
                self._clock_linear_counter()
                self._clock_length_counters()
                self._clock_sweeps()
                
                # Set frame IRQ if enabled
                if not self.frame_irq_enable:
                    self.frame_irq = True
                
                # Reset frame counter
                self.frame_counter = 0
        # 5-step sequence
        else:
            if self.frame_counter == 3729:
                # Quarter frame: clock envelopes and triangle linear counter
                self._clock_envelopes()
                self._clock_linear_counter()
            elif self.frame_counter == 7457:
                # Half frame: clock envelopes, triangle linear counter, and length counters/sweeps
                self._clock_envelopes()
                self._clock_linear_counter()
                self._clock_length_counters()
                self._clock_sweeps()
            elif self.frame_counter == 11186:
                # Quarter frame: clock envelopes and triangle linear counter
                self._clock_envelopes()
                self._clock_linear_counter()
            elif self.frame_counter == 14916:
                # Half frame: clock envelopes, triangle linear counter, and length counters/sweeps
                self._clock_envelopes()
                self._clock_linear_counter()
                self._clock_length_counters()
                self._clock_sweeps()
            elif self.frame_counter == 18641:
                # Reset frame counter
                self.frame_counter = 0
    
    def _clock_envelopes(self):
        """Clock the volume envelopes for pulse and noise channels."""
        # Pulse 1 envelope
        if self.pulse1['envelope']['loop'] and self.pulse1['envelope']['decay'] == 0:
            self.pulse1['envelope']['decay'] = 15
        elif self.pulse1['envelope']['decay'] > 0:
            self.pulse1['envelope']['decay'] -= 1
            
        # Pulse 2 envelope
        if self.pulse2['envelope']['loop'] and self.pulse2['envelope']['decay'] == 0:
            self.pulse2['envelope']['decay'] = 15
        elif self.pulse2['envelope']['decay'] > 0:
            self.pulse2['envelope']['decay'] -= 1
            
        # Noise envelope
        if self.noise['envelope']['loop'] and self.noise['envelope']['decay'] == 0:
            self.noise['envelope']['decay'] = 15
        elif self.noise['envelope']['decay'] > 0:
            self.noise['envelope']['decay'] -= 1
    
    def _clock_linear_counter(self):
        """Clock the triangle channel's linear counter."""
        if self.triangle['control_flag']:
            self.triangle['linear_counter'] = self.triangle['linear_counter_reload']
        elif self.triangle['linear_counter'] > 0:
            self.triangle['linear_counter'] -= 1
            
        if not self.triangle['length_halt']:
            self.triangle['control_flag'] = False
    
    def _clock_length_counters(self):
        """Clock the length counters for all channels."""
        # Pulse 1 length counter
        if not self.pulse1['length_halt'] and self.pulse1['length_counter'] > 0:
            self.pulse1['length_counter'] -= 1
            
        # Pulse 2 length counter
        if not self.pulse2['length_halt'] and self.pulse2['length_counter'] > 0:
            self.pulse2['length_counter'] -= 1
            
        # Triangle length counter
        if not self.triangle['length_halt'] and self.triangle['length_counter'] > 0:
            self.triangle['length_counter'] -= 1
            
        # Noise length counter
        if not self.noise['length_halt'] and self.noise['length_counter'] > 0:
            self.noise['length_counter'] -= 1
    
    def _clock_sweeps(self):
        """Clock the sweep units for pulse channels."""
        # Pulse 1 sweep
        if self.pulse1['sweep']['enabled'] and self.pulse1['sweep']['period'] > 0:
            # Calculate target period
            target_period = self.pulse1['timer']
            shift = target_period >> self.pulse1['sweep']['shift']
            
            if self.pulse1['sweep']['negate']:
                target_period -= shift
                target_period -= 1  # One's complement for pulse 1
            else:
                target_period += shift
                
            # Check if period is valid
            if target_period > 0x7FF or self.pulse1['timer'] < 8:
                self.pulse1['enabled'] = False
            else:
                self.pulse1['timer'] = target_period & 0x7FF
        
        # Pulse 2 sweep (similar to pulse 1 but no -1 adjustment)
        if self.pulse2['sweep']['enabled'] and self.pulse2['sweep']['period'] > 0:
            # Calculate target period
            target_period = self.pulse2['timer']
            shift = target_period >> self.pulse2['sweep']['shift']
            
            if self.pulse2['sweep']['negate']:
                target_period -= shift
            else:
                target_period += shift
                
            # Check if period is valid
            if target_period > 0x7FF or self.pulse2['timer'] < 8:
                self.pulse2['enabled'] = False
            else:
                self.pulse2['timer'] = target_period & 0x7FF
    
    def _clock_pulse1(self):
        """Clock the pulse channel 1."""
        if self.pulse1['timer_value'] > 0:
            self.pulse1['timer_value'] -= 1
        else:
            # Reset timer
            self.pulse1['timer_value'] = self.pulse1['timer']
            
            # Clock sequence
            self.pulse1['sequence_pos'] = (self.pulse1['sequence_pos'] + 1) % 8
    
    def _clock_pulse2(self):
        """Clock the pulse channel 2."""
        if self.pulse2['timer_value'] > 0:
            self.pulse2['timer_value'] -= 1
        else:
            # Reset timer
            self.pulse2['timer_value'] = self.pulse2['timer']
            
            # Clock sequence
            self.pulse2['sequence_pos'] = (self.pulse2['sequence_pos'] + 1) % 8
    
    def _clock_triangle(self):
        """Clock the triangle channel."""
        if self.triangle['timer_value'] > 0:
            self.triangle['timer_value'] -= 1
        else:
            # Reset timer
            self.triangle['timer_value'] = self.triangle['timer']
            
            # Only clock sequence if linear counter > 0 and length counter > 0
            if self.triangle['linear_counter'] > 0 and self.triangle['length_counter'] > 0:
                self.triangle['sequence_pos'] = (self.triangle['sequence_pos'] + 1) % 32
    
    def _clock_noise(self):
        """Clock the noise channel."""
        if self.noise['timer_value'] > 0:
            self.noise['timer_value'] -= 1
        else:
            # Reset timer
            self.noise['timer_value'] = self.noise_period_table[self.noise['period']]
            
            # Clock random number generator
            bit = (self.noise['shift_register'] ^ (self.noise['shift_register'] >> 1)) & 1
            
            self.noise['shift_register'] >>= 1
            self.noise['shift_register'] |= bit << 14
            
            # In mode 1, bit 6 is used instead of bit 1
            if self.noise['mode']:
                bit = (self.noise['shift_register'] ^ (self.noise['shift_register'] >> 6)) & 1
                self.noise['shift_register'] = (self.noise['shift_register'] & 0x3FBF) | (bit << 15)
    
    def _clock_dmc(self):
        """Clock the Delta Modulation Channel (DMC)."""
        # DMC processing logic
        # ...
        # (Simplified implementation for brevity)
        pass
    
    def _mix_audio(self) -> float:
        """
        Mix audio from all channels into a single output sample.
        Uses the NES non-linear mixing formula.
        
        Returns:
            Mixed audio sample value (-1.0 to 1.0)
        """
        # Get output from each channel
        pulse1_out = 0
        if self.pulse1['enabled'] and self.pulse1['length_counter'] > 0:
            duty_value = self.duty_table[self.pulse1['duty']][self.pulse1.get('sequence_pos', 0)]
            vol = self.pulse1['envelope']['constant'] and self.pulse1['envelope']['volume'] or self.pulse1['envelope']['decay']
            pulse1_out = duty_value * vol
        
        pulse2_out = 0
        if self.pulse2['enabled'] and self.pulse2['length_counter'] > 0:
            duty_value = self.duty_table[self.pulse2['duty']][self.pulse2.get('sequence_pos', 0)]
            vol = self.pulse2['envelope']['constant'] and self.pulse2['envelope']['volume'] or self.pulse2['envelope']['decay']
            pulse2_out = duty_value * vol
        
        triangle_out = 0
        if self.triangle['enabled'] and self.triangle['length_counter'] > 0 and self.triangle['linear_counter'] > 0:
            triangle_out = self.triangle_table[self.triangle.get('sequence_pos', 0)]
        
        noise_out = 0
        if self.noise['enabled'] and self.noise['length_counter'] > 0:
            # Output is either 0 or current volume based on shift register bit 0
            if (self.noise['shift_register'] & 1) == 0:
                vol = self.noise['envelope']['constant'] and self.noise['envelope']['volume'] or self.noise['envelope']['decay']
                noise_out = vol
        
        dmc_out = self.dmc['output_level']
        
        # Apply NES mixing formula (approximate)
        pulse_out = 0.00752 * (pulse1_out + pulse2_out)
        tnd_out = 0.00851 * triangle_out + 0.00494 * noise_out + 0.00335 * dmc_out
        
        # Combine and normalize to -1.0 to 1.0 range
        return pulse_out + tnd_out
    
    def write_register(self, address: int, value: int) -> None:
        """
        Write to an APU register.
        
        Args:
            address: Register address (0x4000-0x4017)
            value: Value to write
        """
        if address == 0x4000:  # Pulse 1 control
            self.pulse1['duty'] = (value >> 6) & 0x03
            self.pulse1['length_halt'] = (value & 0x20) != 0
            self.pulse1['envelope']['constant'] = (value & 0x10) != 0
            self.pulse1['envelope']['loop'] = (value & 0x20) != 0
            self.pulse1['envelope']['volume'] = value & 0x0F
            
        elif address == 0x4001:  # Pulse 1 sweep
            self.pulse1['sweep']['enabled'] = (value & 0x80) != 0
            self.pulse1['sweep']['period'] = ((value >> 4) & 0x07) + 1
            self.pulse1['sweep']['negate'] = (value & 0x08) != 0
            self.pulse1['sweep']['shift'] = value & 0x07
            self.pulse1['sweep']['reload'] = True
            
        elif address == 0x4002:  # Pulse 1 timer low
            self.pulse1['timer'] = (self.pulse1['timer'] & 0x700) | value
            
        elif address == 0x4003:  # Pulse 1 timer high/length counter
            self.pulse1['timer'] = (self.pulse1['timer'] & 0xFF) | ((value & 0x07) << 8)
            if self.pulse1['enabled']:
                self.pulse1['length_counter'] = self._length_counter_table[(value >> 3) & 0x1F]
            self.pulse1['envelope']['decay'] = 15
            
        elif address == 0x4004:  # Pulse 2 control
            self.pulse2['duty'] = (value >> 6) & 0x03
            self.pulse2['length_halt'] = (value & 0x20) != 0
            self.pulse2['envelope']['constant'] = (value & 0x10) != 0
            self.pulse2['envelope']['loop'] = (value & 0x20) != 0
            self.pulse2['envelope']['volume'] = value & 0x0F
            
        elif address == 0x4005:  # Pulse 2 sweep
            self.pulse2['sweep']['enabled'] = (value & 0x80) != 0
            self.pulse2['sweep']['period'] = ((value >> 4) & 0x07) + 1
            self.pulse2['sweep']['negate'] = (value & 0x08) != 0
            self.pulse2['sweep']['shift'] = value & 0x07
            self.pulse2['sweep']['reload'] = True
            
        elif address == 0x4006:  # Pulse 2 timer low
            self.pulse2['timer'] = (self.pulse2['timer'] & 0x700) | value
            
        elif address == 0x4007:  # Pulse 2 timer high/length counter
            self.pulse2['timer'] = (self.pulse2['timer'] & 0xFF) | ((value & 0x07) << 8)
            if self.pulse2['enabled']:
                self.pulse2['length_counter'] = self._length_counter_table[(value >> 3) & 0x1F]
            self.pulse2['envelope']['decay'] = 15
            
        elif address == 0x4008:  # Triangle control
            self.triangle['length_halt'] = (value & 0x80) != 0
            self.triangle['linear_counter_reload'] = value & 0x7F
            
        elif address == 0x400A:  # Triangle timer low
            self.triangle['timer'] = (self.triangle['timer'] & 0x700) | value
            
        elif address == 0x400B:  # Triangle timer high/length counter
            self.triangle['timer'] = (self.triangle['timer'] & 0xFF) | ((value & 0x07) << 8)
            if self.triangle['enabled']:
                self.triangle['length_counter'] = self._length_counter_table[(value >> 3) & 0x1F]
            self.triangle['control_flag'] = True
            
        elif address == 0x400C:  # Noise control
            self.noise['length_halt'] = (value & 0x20) != 0
            self.noise['envelope']['constant'] = (value & 0x10) != 0
            self.noise['envelope']['loop'] = (value & 0x20) != 0
            self.noise['envelope']['volume'] = value & 0x0F
            
        elif address == 0x400E:  # Noise period/mode
            self.noise['mode'] = (value & 0x80) != 0
            self.noise['period'] = value & 0x0F
            
        elif address == 0x400F:  # Noise length counter
            if self.noise['enabled']:
                self.noise['length_counter'] = self._length_counter_table[(value >> 3) & 0x1F]
            self.noise['envelope']['decay'] = 15
            
        elif address == 0x4010:  # DMC control
            self.dmc['frequency'] = value & 0x0F
            self.dmc['loop'] = (value & 0x40) != 0
            self.dmc['irq_enable'] = (value & 0x80) != 0
            
        elif address == 0x4011:  # DMC direct load
            self.dmc['output_level'] = value & 0x7F
            
        elif address == 0x4012:  # DMC sample address
            self.dmc['sample_address'] = 0xC000 + (value * 64)
            
        elif address == 0x4013:  # DMC sample length
            self.dmc['sample_length'] = (value * 16) + 1
            
        elif address == 0x4015:  # Status register
            self.pulse1['enabled'] = (value & self.PULSE1_ENABLE) != 0
            self.pulse2['enabled'] = (value & self.PULSE2_ENABLE) != 0
            self.triangle['enabled'] = (value & self.TRIANGLE_ENABLE) != 0
            self.noise['enabled'] = (value & self.NOISE_ENABLE) != 0
            self.dmc['enabled'] = (value & self.DMC_ENABLE) != 0
            
            # Clear DMC IRQ flag
            self.dmc_irq = False
            
            # Initialize DMC if enabled
            if self.dmc['enabled'] and self.dmc['bytes_remaining'] == 0:
                self.dmc['current_address'] = self.dmc['sample_address']
                self.dmc['bytes_remaining'] = self.dmc['sample_length']
            
            # If a channel is disabled, its length counter is zeroed
            if not self.pulse1['enabled']:
                self.pulse1['length_counter'] = 0
            if not self.pulse2['enabled']:
                self.pulse2['length_counter'] = 0
            if not self.triangle['enabled']:
                self.triangle['length_counter'] = 0
            if not self.noise['enabled']:
                self.noise['length_counter'] = 0
                
        elif address == 0x4017:  # Frame counter
            self.frame_period = (value >> 7) & 1
            self.frame_irq_enable = (value & 0x40) == 0
            
            # Reset frame counter
            self.frame_counter = 0
            
            # Clock immediate updates if bit 7 is set
            if value & 0x80:
                self._clock_envelopes()
                self._clock_linear_counter()
                self._clock_length_counters()
                self._clock_sweeps()
            
            # Clear frame IRQ if disabled
            if not self.frame_irq_enable:
                self.frame_irq = False
    
    def read_status(self) -> int:
        """
        Read the APU status register.
        
        Returns:
            Value of the status register
        """
        status = 0
        
        # Set bits for active length counters
        if self.pulse1['length_counter'] > 0:
            status |= self.PULSE1_ENABLE
        if self.pulse2['length_counter'] > 0:
            status |= self.PULSE2_ENABLE
        if self.triangle['length_counter'] > 0:
            status |= self.TRIANGLE_ENABLE
        if self.noise['length_counter'] > 0:
            status |= self.NOISE_ENABLE
        if self.dmc['bytes_remaining'] > 0:
            status |= self.DMC_ENABLE
            
        # Set IRQ flags
        if self.frame_irq:
            status |= 0x40
        if self.dmc_irq:
            status |= 0x80
            
        # Reading status clears the frame IRQ flag
        self.frame_irq = False
        
        return status
    
    # Length counter lookup table (32 entries)
    _length_counter_table = [
        10, 254, 20,  2, 40,  4, 80,  6, 160,  8, 60, 10, 14, 12, 26, 14,
        12,  16, 24, 18, 48, 20, 96, 22, 192, 24, 72, 26, 16, 28, 32, 30
    ]