"""
SNES Audio DSP (Digital Signal Processor) emulation.

The SNES APU consists of an SPC700 processor and a DSP that generates sound.
This module focuses on emulating the DSP side, which handles sample playback,
ADSR envelopes, and audio mixing for the SNES's 8 audio channels.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import math

logger = logging.getLogger("QuantumSignalEmulator.SNES.DSP")

class SNESDSP:
    """
    Emulates the SNES Digital Signal Processor for audio.
    
    The SNES DSP provides 8 audio channels, each with configurable volume,
    pitch, ADSR envelope, and sample playback. This class implements the
    audio generation logic, register access, and sample output.
    """
    
    # DSP register addresses (per voice)
    VOL_L   = 0x00  # Left volume
    VOL_R   = 0x01  # Right volume
    PITCH_L = 0x02  # Pitch (low)
    PITCH_H = 0x03  # Pitch (high)
    SRC     = 0x04  # Source number
    ADSR_1  = 0x05  # ADSR setting 1
    ADSR_2  = 0x06  # ADSR setting 2
    GAIN    = 0x07  # Gain setting
    ENV_X   = 0x08  # Current envelope value
    OUT_X   = 0x09  # Current voice output
    
    # Global DSP registers
    MVOL_L  = 0x0C  # Main volume left
    MVOL_R  = 0x1C  # Main volume right
    EVOL_L  = 0x2C  # Echo volume left
    EVOL_R  = 0x3C  # Echo volume right
    KON     = 0x4C  # Key on
    KOFF    = 0x5C  # Key off
    FLG     = 0x6C  # DSP flags
    ENDX    = 0x7C  # End of sample
    EFB     = 0x0D  # Echo feedback
    PMON    = 0x2D  # Pitch modulation
    NON     = 0x3D  # Noise enable
    EON     = 0x4D  # Echo enable
    DIR     = 0x5D  # BRR directory base address
    ESA     = 0x6D  # Echo buffer base address
    EDL     = 0x7D  # Echo delay
    FIR_0   = 0x0F  # Echo FIR filter coefficient 0
    FIR_1   = 0x1F  # Echo FIR filter coefficient 1
    FIR_2   = 0x2F  # Echo FIR filter coefficient 2
    FIR_3   = 0x3F  # Echo FIR filter coefficient 3
    FIR_4   = 0x4F  # Echo FIR filter coefficient 4
    FIR_5   = 0x5F  # Echo FIR filter coefficient 5
    FIR_6   = 0x6F  # Echo FIR filter coefficient 6
    FIR_7   = 0x7F  # Echo FIR filter coefficient 7
    
    # Constants
    NUM_VOICES = 8
    SAMPLE_RATE = 32000  # SPC700 runs at 32kHz
    
    def __init__(self):
        """Initialize the SNES DSP."""
        # Register storage
        self.registers = bytearray(128)
        
        # Internal state for each voice
        self.voices = []
        for i in range(self.NUM_VOICES):
            self.voices.append({
                "sample_position": 0.0,  # Current sample position (fractional)
                "decoded_samples": [0] * 16,  # Decoded BRR samples buffer
                "last_sample_1": 0,  # Previous sample (for interpolation)
                "last_sample_2": 0,  # Second previous sample (for interpolation)
                "envelope": 0,        # Current envelope value
                "envelope_mode": 0,   # Current envelope mode (ADSR phase)
                "envelope_level": 0,  # Internal envelope level
                "sample_address": 0,  # Current sample address
                "brr_offset": 0,      # Offset within current BRR block
                "brr_header": 0,      # Current BRR header
                "key_on": False,      # Key on state
                "echo_enable": False  # Echo enable state
            })
        
        # RAM access (normally provided by SPC700)
        self.ram = bytearray(64 * 1024)  # Full 64KB address space
        
        # Echo buffer
        self.echo_buffer = [0] * 128 * 1024  # Large enough for max delay
        self.echo_pos = 0
        self.echo_length = 0
        
        # Output buffer
        self.output_buffer = []
        self.output_position = 0
        
        # FIR filter history buffer for echo
        self.fir_buffer = [0] * 8
        
        # Sample counters
        self.sample_counter = 0
        self.dsp_counter = 0
        
        # Internal noise generator
        self.noise_level = 0
        self.noise_counter = 0
        
        # For CPU cycle counting
        self.cycles_per_sample = 32  # ~1MHz CPU / 32kHz sample rate
        self.cycle_counter = 0
        
        logger.info("SNES DSP initialized")
    
    def read_register(self, address: int) -> int:
        """
        Read a DSP register.
        
        Args:
            address: Register address (0-127)
            
        Returns:
            Register value
        """
        return self.registers[address & 0x7F]
    
    def write_register(self, address: int, value: int) -> None:
        """
        Write to a DSP register.
        
        Args:
            address: Register address (0-127)
            value: Value to write
        """
        reg = address & 0x7F
        self.registers[reg] = value & 0xFF
        
        # Handle special register writes
        if reg == self.KON:
            # Key on
            for voice in range(self.NUM_VOICES):
                if value & (1 << voice):
                    self._key_on_voice(voice)
                    
        elif reg == self.KOFF:
            # Key off
            for voice in range(self.NUM_VOICES):
                if value & (1 << voice):
                    self._key_off_voice(voice)
                    
        elif reg == self.FLG:
            # Reset noise generator if bit 7 is set
            if value & 0x80:
                self.noise_level = 0
                self.noise_counter = 0
                
        elif reg == self.EDL:
            # Echo delay
            delay = value & 0x0F
            if delay == 0:
                self.echo_length = 4
            else:
                self.echo_length = delay * 2048
            
            # Reset echo position
            self.echo_pos = 0
    
    def _key_on_voice(self, voice: int) -> None:
        """
        Key on a voice.
        
        Args:
            voice: Voice number (0-7)
        """
        if voice < 0 or voice >= self.NUM_VOICES:
            return
            
        # Get sample address from DIR register and source number
        dir_addr = self.registers[self.DIR] << 8
        source = self.registers[voice * 16 + self.SRC]
        self.voices[voice]["sample_address"] = dir_addr + (source << 2)
        
        # Reset envelope
        self.voices[voice]["envelope_mode"] = 1  # Attack
        self.voices[voice]["envelope_level"] = 0
        self.voices[voice]["envelope"] = 0
        
        # Reset sample position and history
        self.voices[voice]["sample_position"] = 0.0
        self.voices[voice]["decoded_samples"] = [0] * 16
        self.voices[voice]["last_sample_1"] = 0
        self.voices[voice]["last_sample_2"] = 0
        
        # Reset BRR decoder
        self.voices[voice]["brr_offset"] = 0
        self.voices[voice]["brr_header"] = 0
        
        # Set key on flag
        self.voices[voice]["key_on"] = True
        
        logger.debug(f"Key on voice {voice}, source={source}, addr=${self.voices[voice]['sample_address']:04X}")
    
    def _key_off_voice(self, voice: int) -> None:
        """
        Key off a voice.
        
        Args:
            voice: Voice number (0-7)
        """
        if voice < 0 or voice >= self.NUM_VOICES:
            return
            
        # Set envelope mode to release
        self.voices[voice]["envelope_mode"] = 4  # Release
        
        # Clear key on flag
        self.voices[voice]["key_on"] = False
        
        logger.debug(f"Key off voice {voice}")
    
    def step(self, cycles: int) -> List[float]:
        """
        Run the DSP for the specified number of CPU cycles.
        
        Args:
            cycles: Number of CPU cycles to simulate
            
        Returns:
            List of audio samples generated during this time
        """
        # Add cycles to counter
        self.cycle_counter += cycles
        
        # Generate samples based on cycles
        output_samples = []
        
        while self.cycle_counter >= self.cycles_per_sample:
            # Generate one audio sample
            left, right = self._generate_sample()
            output_samples.append((left + right) / 2)  # Mono mix for simplicity
            
            # Decrement cycle counter
            self.cycle_counter -= self.cycles_per_sample
        
        return output_samples
    
    def _generate_sample(self) -> Tuple[float, float]:
        """
        Generate one stereo sample.
        
        Returns:
            Tuple of (left, right) sample values
        """
        # Clear output accumulators
        left_out = 0
        right_out = 0
        echo_left = 0
        echo_right = 0
        
        # Process all voices
        for voice in range(self.NUM_VOICES):
            # Update envelope
            self._update_envelope(voice)
            
            # Get current sample
            sample = self._get_voice_sample(voice)
            
            # Apply envelope
            envelope = self.voices[voice]["envelope"]
            sample = (sample * envelope) // 128
            
            # Set voice output register
            self.registers[voice * 16 + self.OUT_X] = max(-128, min(127, sample)) & 0xFF
            
            # Get voice volumes
            vol_left = self._sign_extend_8bit(self.registers[voice * 16 + self.VOL_L])
            vol_right = self._sign_extend_8bit(self.registers[voice * 16 + self.VOL_R])
            
            # Add to output accumulators
            left_out += (sample * vol_left) // 128
            right_out += (sample * vol_right) // 128
            
            # Add to echo accumulators if echo enabled for this voice
            if (self.registers[self.EON] & (1 << voice)) != 0:
                echo_left += (sample * vol_left) // 128
                echo_right += (sample * vol_right) // 128
        
        # Process echo
        echo_out_left, echo_out_right = self._process_echo(echo_left, echo_right)
        
        # Apply master volume
        mvol_left = self._sign_extend_8bit(self.registers[self.MVOL_L])
        mvol_right = self._sign_extend_8bit(self.registers[self.MVOL_R])
        
        left_out = (left_out * mvol_left) // 128
        right_out = (right_out * mvol_right) // 128
        
        # Add echo to output
        left_out += echo_out_left
        right_out += echo_out_right
        
        # Convert to float in range [-1, 1]
        left_float = max(-32768, min(32767, left_out)) / 32768.0
        right_float = max(-32768, min(32767, right_out)) / 32768.0
        
        # Increment DSP counter
        self.dsp_counter += 1
        
        return left_float, right_float
    
    def _update_envelope(self, voice: int) -> None:
        """
        Update the envelope for a voice.
        
        Args:
            voice: Voice number (0-7)
        """
        # Get ADSR parameters
        adsr1 = self.registers[voice * 16 + self.ADSR_1]
        adsr2 = self.registers[voice * 16 + self.ADSR_2]
        
        # Use ADSR or GAIN mode
        use_adsr = (adsr1 & 0x80) != 0
        
        if use_adsr:
            # ADSR mode
            mode = self.voices[voice]["envelope_mode"]
            level = self.voices[voice]["envelope_level"]
            
            if mode == 1:
                # Attack mode
                attack_rate = adsr1 & 0x0F
                
                # Calculate attack increment
                if attack_rate == 0:
                    increment = 4  # Minimum rate
                elif attack_rate == 15:
                    increment = 1024  # Maximum rate
                else:
                    increment = (2 << (attack_rate >> 1)) & (attack_rate & 1)
                
                # Apply attack
                level += increment
                
                if level >= 0x7800:
                    # Attack complete, move to decay
                    level = 0x7800
                    mode = 2  # Decay
            
            elif mode == 2:
                # Decay mode
                decay_rate = (adsr1 >> 4) & 0x07
                
                # Calculate decay decrement
                decrement = 2 << decay_rate
                
                # Apply decay
                level -= decrement
                
                if level <= 0:
                    level = 0
                
                # Check if sustain level reached
                sustain_level = (adsr2 >> 5) & 0x07
                if level <= (sustain_level << 8):
                    mode = 3  # Sustain
            
            elif mode == 3:
                # Sustain mode
                sustain_rate = adsr2 & 0x1F
                
                # Calculate sustain decrement
                if sustain_rate == 0:
                    decrement = 0
                else:
                    decrement = 2 << (sustain_rate >> 1)
                
                # Apply sustain
                level -= decrement
                
                if level <= 0:
                    level = 0
            
            elif mode == 4:
                # Release mode
                # Fixed release rate
                level -= 8
                
                if level <= 0:
                    level = 0
            
            # Save updated values
            self.voices[voice]["envelope_mode"] = mode
            self.voices[voice]["envelope_level"] = level
            
            # Convert level to 0-127 range for envelope
            self.voices[voice]["envelope"] = level >> 4
            
        else:
            # GAIN mode
            gain = self.registers[voice * 16 + self.GAIN]
            
            # Simplified GAIN mode (direct level setting)
            if (gain & 0x80) == 0:
                # Direct setting
                self.voices[voice]["envelope"] = gain & 0x7F
            else:
                # Increase/decrease modes not implemented in this simplified version
                pass
        
        # Update envelope register
        self.registers[voice * 16 + self.ENV_X] = min(0x7F, self.voices[voice]["envelope"]) & 0xFF
    
    def _get_voice_sample(self, voice: int) -> int:
        """
        Get the current sample for a voice.
        
        Args:
            voice: Voice number (0-7)
            
        Returns:
            Sample value
        """
        # Check if noise is enabled for this voice
        if (self.registers[self.NON] & (1 << voice)) != 0:
            # Use noise generator instead of sample
            return self._get_noise_sample()
        
        # Get current fractional sample position
        pos = self.voices[voice]["sample_position"]
        
        # Calculate integer and fractional parts
        int_pos = int(pos)
        frac_pos = pos - int_pos
        
        # Check if we need to decode more samples
        if int_pos >= 12:  # Need 3 samples for cubic interpolation plus margin
            self._decode_brr_block(voice)
            
            # Reset position
            pos -= int_pos
            int_pos = 0
            self.voices[voice]["sample_position"] = pos
        
        # Get samples for interpolation
        s0 = self.voices[voice]["last_sample_2"]
        s1 = self.voices[voice]["last_sample_1"]
        s2 = self.voices[voice]["decoded_samples"][int_pos]
        s3 = self.voices[voice]["decoded_samples"][int_pos + 1]
        
        # Perform cubic interpolation
        sample = self._cubic_interpolate(s0, s1, s2, s3, frac_pos)
        
        # Get pitch value
        pitch_l = self.registers[voice * 16 + self.PITCH_L]
        pitch_h = self.registers[voice * 16 + self.PITCH_H]
        pitch = (pitch_h << 8) | pitch_l
        
        # Implement pitch modulation if enabled
        if voice > 0 and ((self.registers[self.PMON] & (1 << voice)) != 0):
            # Modulate with previous voice's output
            prev_out = self._sign_extend_8bit(self.registers[(voice - 1) * 16 + self.OUT_X])
            pitch = (pitch * (prev_out + 0x80)) >> 7
        
        # Advance sample position
        self.voices[voice]["sample_position"] += pitch / 4096.0
        
        return sample
    
    def _decode_brr_block(self, voice: int) -> None:
        """
        Decode a BRR block for a voice.
        
        Args:
            voice: Voice number (0-7)
        """
        # Shift samples in buffer
        self.voices[voice]["last_sample_2"] = self.voices[voice]["last_sample_1"]
        self.voices[voice]["last_sample_1"] = self.voices[voice]["decoded_samples"][0]
        
        for i in range(15):
            self.voices[voice]["decoded_samples"][i] = self.voices[voice]["decoded_samples"][i + 1]
        
        # Get current BRR block address
        addr = self.voices[voice]["sample_address"]
        offset = self.voices[voice]["brr_offset"]
        
        # Read BRR header if at the start of a block
        if offset == 0:
            self.voices[voice]["brr_header"] = self.ram[addr]
            offset = 1
            
            # Check for end flag
            if (self.voices[voice]["brr_header"] & 0x01) != 0:
                # End of sample
                # Set ENDX bit
                self.registers[self.ENDX] |= (1 << voice)
                
                # Load next block address from directory
                dir_addr = self.registers[self.DIR] << 8
                source = self.registers[voice * 16 + self.SRC]
                next_addr = dir_addr + (source << 2) + 2
                
                # Read next block address
                self.voices[voice]["sample_address"] = (self.ram[next_addr] << 8) | self.ram[next_addr + 1]
        
        # Get shift and filter from header
        shift = (self.voices[voice]["brr_header"] >> 4) & 0x0F
        filter_mode = (self.voices[voice]["brr_header"] >> 2) & 0x03
        
        # Read next BRR byte
        brr_byte = self.ram[addr + offset]
        offset += 1
        
        # Decode 2 samples from byte
        sample1 = ((brr_byte >> 4) & 0x0F)
        sample2 = (brr_byte & 0x0F)
        
        # Sign extend
        if sample1 >= 8:
            sample1 -= 16
        if sample2 >= 8:
            sample2 -= 16
        
        # Scale samples based on shift
        if shift <= 12:
            sample1 = (sample1 << shift) >> 1
            sample2 = (sample2 << shift) >> 1
        else:
            sample1 = (sample1 >> 3) << 12
            sample2 = (sample2 >> 3) << 12
        
        # Apply IIR filter
        s1 = self.voices[voice]["last_sample_1"]
        s2 = self.voices[voice]["last_sample_2"]
        
        if filter_mode == 1:
            # Filter 1: sample += s1 * 15/16
            sample1 += (s1 * 15) >> 4
            sample2 += (sample1 * 15) >> 4
        elif filter_mode == 2:
            # Filter 2: sample += s1 * 61/32 - s2 * 15/16
            sample1 += (s1 * 61) >> 5
            sample1 -= (s2 * 15) >> 4
            sample2 += (sample1 * 61) >> 5
            sample2 -= (s1 * 15) >> 4
        elif filter_mode == 3:
            # Filter 3: sample += s1 * 115/64 - s2 * 13/16
            sample1 += (s1 * 115) >> 6
            sample1 -= (s2 * 13) >> 4
            sample2 += (sample1 * 115) >> 6
            sample2 -= (s1 * 13) >> 4
        
        # Add decoded samples to buffer
        self.voices[voice]["decoded_samples"][15] = sample1
        self.voices[voice]["decoded_samples"][16] = sample2
        
        # Update BRR offset
        if offset >= 9:
            # End of BRR block
            offset = 0
            self.voices[voice]["sample_address"] += 9
        
        self.voices[voice]["brr_offset"] = offset
    
    def _get_noise_sample(self) -> int:
        """
        Get a sample from the noise generator.
        
        Returns:
            Noise sample value
        """
        # Update noise generator
        self.noise_counter += 1
        
        if self.noise_counter >= 32:
            self.noise_counter = 0
            
            # LFSR noise algorithm
            bit0 = (self.noise_level & 0x0001) ^ ((self.noise_level & 0x0002) >> 1)
            self.noise_level = (self.noise_level >> 1) | (bit0 << 14)
        
        # Convert noise level to sample (-128 to 127)
        return ((self.noise_level & 0x4000) >> 6) - 128
    
    def _process_echo(self, echo_in_left: int, echo_in_right: int) -> Tuple[int, int]:
        """
        Process echo effect.
        
        Args:
            echo_in_left: Left channel echo input
            echo_in_right: Right channel echo input
            
        Returns:
            Tuple of (left, right) echo output
        """
        # Check if echo is disabled
        if (self.registers[self.FLG] & 0x20) != 0:
            return 0, 0
            
        # Get echo buffer address
        echo_addr = self.registers[self.ESA] << 8
        
        # Determine echo buffer size
        echo_size = self.echo_length
        
        # Read echo samples from buffer
        echo_pos = (echo_addr + self.echo_pos) & 0xFFFF
        echo_left = self.ram[echo_pos]
        echo_right = self.ram[echo_pos + 1]
        
        # Sign extend
        echo_left = self._sign_extend_8bit(echo_left)
        echo_right = self._sign_extend_8bit(echo_right)
        
        # Apply FIR filter
        # (Simplified version - actual SNES has 8 taps)
        filtered_left = 0
        filtered_right = 0
        
        for i in range(8):
            coef = self._sign_extend_8bit(self.registers[self.FIR_0 + i * 16])
            filtered_left += (echo_left * coef) // 128
            filtered_right += (echo_right * coef) // 128
        
        # Get echo volume
        evol_left = self._sign_extend_8bit(self.registers[self.EVOL_L])
        evol_right = self._sign_extend_8bit(self.registers[self.EVOL_R])
        
        # Calculate output
        out_left = (filtered_left * evol_left) // 128
        out_right = (filtered_right * evol_right) // 128
        
        # Write new samples to echo buffer
        # Apply feedback
        efb = self._sign_extend_8bit(self.registers[self.EFB])
        feedback_left = (echo_in_left + (filtered_left * efb) // 128) >> 7
        feedback_right = (echo_in_right + (filtered_right * efb) // 128) >> 7
        
        # Clamp to 8-bit
        feedback_left = max(-128, min(127, feedback_left)) & 0xFF
        feedback_right = max(-128, min(127, feedback_right)) & 0xFF
        
        # Write to echo buffer
        self.ram[echo_pos] = feedback_left
        self.ram[echo_pos + 1] = feedback_right
        
        # Advance echo position
        self.echo_pos = (self.echo_pos + 2) % echo_size
        
        return out_left, out_right
    
    def _cubic_interpolate(self, s0: int, s1: int, s2: int, s3: int, mu: float) -> int:
        """
        Perform cubic interpolation on 4 samples.
        
        Args:
            s0: Sample t-1
            s1: Sample t
            s2: Sample t+1
            s3: Sample t+2
            mu: Fractional position between s1 and s2 (0.0 to 1.0)
            
        Returns:
            Interpolated sample value
        """
        # Cubic interpolation formula
        mu2 = mu * mu
        a0 = s3 - s2 - s0 + s1
        a1 = s0 - s1 - a0
        a2 = s2 - s0
        a3 = s1
        
        return int(a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3)
    
    def _sign_extend_8bit(self, value: int) -> int:
        """
        Sign extend an 8-bit value to 16-bit.
        
        Args:
            value: 8-bit value
            
        Returns:
            Sign-extended 16-bit value
        """
        if value & 0x80:
            return value - 256
        return value
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current DSP state.
        
        Returns:
            Dictionary with DSP state
        """
        voice_states = []
        for i in range(self.NUM_VOICES):
            voice_states.append({
                "envelope": self.voices[i]["envelope"],
                "key_on": self.voices[i]["key_on"],
                "sample_addr": self.voices[i]["sample_address"]
            })
            
        return {
            "voices": voice_states,
            "registers": {i: self.registers[i] for i in range(0, 128, 16)},  # Just show global regs
            "dsp_counter": self.dsp_counter,
            "echo_pos": self.echo_pos,
            "noise_level": self.noise_level
        }