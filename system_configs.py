"""
Configuration data for supported video game systems.
"""

SYSTEM_CONFIGS = {
    "nes": {
        "cpu_freq_mhz": 1.789773,  # NTSC NES
        "ppu_freq_mhz": 5.369318,  # PPU runs at 3× CPU clock
        "cpu_type": "6502",
        "memory_map": {
            "ram": {"start": 0x0000, "end": 0x07FF, "mirror": True},
            "ppu_registers": {"start": 0x2000, "end": 0x2007, "mirror": True},
            "apu_io_registers": {"start": 0x4000, "end": 0x4017},
            "cartridge": {"start": 0x4020, "end": 0xFFFF},
        },
        "registers": [
            # PPU registers (complete set)
            "PPUCTRL", "PPUMASK", "PPUSTATUS", "OAMADDR", "OAMDATA", 
            "PPUSCROLL", "PPUADDR", "PPUDATA",
            # APU registers
            "APUPULSE1_1", "APUPULSE1_2", "APUPULSE1_3", "APUPULSE1_4",
            "APUPULSE2_1", "APUPULSE2_2", "APUPULSE2_3", "APUPULSE2_4",
            # CPU registers
            "PC", "A", "X", "Y", "SP", "P"
        ],
        "resolution": (256, 240),
        "cycles_per_scanline": 341,
        "total_scanlines": 262,  # NTSC
        "visible_scanlines": 240,
        "vblank_scanlines": 20,
        "quantum_mapping": "direct",
    },
    "snes": {
        "cpu_freq_mhz": 3.58,  # 3.58 MHz (with slowdown in some modes)
        "cpu_type": "65C816",
        "memory_map": {
            "wram": {"start": 0x7E0000, "end": 0x7FFFFF},
            "sram": {"start": 0x700000, "end": 0x7DFFFF},
            "registers": {"start": 0x2100, "end": 0x21FF},
            "vram_access": {"start": 0x2118, "end": 0x2119},
        },
        "registers": [
            # PPU registers (partial list)
            "INIDISP", "OBSEL", "BGMODE", "MOSAIC", "BG1SC", "BG2SC", 
            "BG3SC", "BG4SC", "BG12NBA", "BG34NBA", "BG1HOFS", "BG1VOFS",
            # CPU registers
            "PC", "A", "X", "Y", "SP", "P", "D", "DB", "PBR"
        ],
        "resolution": (256, 224),  # Standard resolution mode
        "high_resolution": (512, 448),
        "cycles_per_scanline": 1364,
        "total_scanlines": 262,  # NTSC
        "visible_scanlines": 224,
        "vblank_scanlines": 38,
        "quantum_mapping": "superposition",
    },
    "genesis": {
        "cpu_freq_mhz": 7.67,  # Motorola 68000 main CPU
        "sound_cpu_freq_mhz": 3.58,  # Z80 co-processor
        "cpu_type": "68000",
        "secondary_cpu": "Z80",
        "vdp_freq_mhz": 13.423,  # Video Display Processor clock
        "memory_map": {
            "boot_rom": {"start": 0x000000, "end": 0x0003FF},
            "work_ram": {"start": 0xFF0000, "end": 0xFFFFFF},
            "z80_ram": {"start": 0xA00000, "end": 0xA01FFF},
            "vdp_registers": {"start": 0xC00000, "end": 0xC0001F},
            "vdp_data": {"start": 0xC00000, "end": 0xC00003},
            "cartridge": {"start": 0x400000, "end": 0x9FFFFF},
        },
        "registers": [
            # 68000 CPU registers
            "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7",
            "A0", "A1", "A2", "A3", "A4", "A5", "A6", "SP", "PC", "SR",
            # VDP registers
            "VDPCTRL", "VDPDATA", "VDPHV", "VDPDMA",
            # Z80 CPU registers
            "Z80_A", "Z80_F", "Z80_B", "Z80_C", "Z80_D", "Z80_E", "Z80_H", "Z80_L",
            "Z80_SP", "Z80_PC", "Z80_IX", "Z80_IY", "Z80_I", "Z80_R"
        ],
        "resolution": (320, 224),  # Standard Mode 4 (most common)
        "alt_resolution": (256, 224),  # Mode 5 H40 cells
        "interlaced_resolution": (320, 448),  # Interlaced Mode 4
        "cycles_per_scanline": 3420,  # 68000 cycles
        "total_scanlines": 262,  # NTSC
        "visible_scanlines": 224,
        "vblank_scanlines": 38,
        "color_depth": 512,  # 9-bit color (512 colors)
        "sprite_limits": {
            "per_scanline": 20,
            "per_frame": 80,
            "width": [8, 16, 24, 32],
            "height": [8, 16, 24, 32]
        },
        "sound_chips": ["YM2612", "SN76489"],
        "quantum_mapping": "vector",
    },
    "atari2600": {
        "cpu_freq_mhz": 1.19,  # MOS Technology 6507 (limited 6502)
        "cpu_type": "6507",
        "memory_map": {
            "tia_registers": {"start": 0x00, "end": 0x7F},
            "riot_registers": {"start": 0x280, "end": 0x297},
            "ram": {"start": 0x80, "end": 0xFF},  # 128 bytes of RAM
            "rom": {"start": 0xF000, "end": 0xFFFF},  # 4KB ROM space
        },
        "registers": [
            # CPU registers
            "A", "X", "Y", "SP", "PC", "P",  # 6507 CPU registers
            # TIA registers
            "VSYNC", "VBLANK", "WSYNC", "RSYNC", "NUSIZ0", "NUSIZ1",
            "COLUP0", "COLUP1", "COLUPF", "COLUBK", "CTRLPF", "REFP0",
            "REFP1", "PF0", "PF1", "PF2", "RESP0", "RESP1", "RESM0", 
            "RESM1", "RESBL", "AUDC0", "AUDC1", "AUDF0", "AUDF1", 
            "AUDV0", "AUDV1", "GRP0", "GRP1", "ENAM0", "ENAM1", "ENABL",
            "HMP0", "HMP1", "HMM0", "HMM1", "HMBL", "VDELP0", "VDELP1",
            "VDELBL", "RESMP0", "RESMP1", "HMOVE", "HMCLR", "CXCLR",
            # RIOT registers
            "SWCHA", "SWCHB", "INTIM", "TIM1T", "TIM8T", "TIM64T", "T1024T"
        ],
        "resolution": (160, 192),  # Approximation - TIA doesn't work like modern systems
        "color_system": "NTSC",  # Also supports PAL with different timings
        "cycles_per_scanline": 228,  # 76 TIA color clocks (3 CPU cycles per color clock)
        "total_scanlines": 262,  # NTSC (312 for PAL)
        "visible_scanlines": 192,  # Typical - programmable by the game
        "vblank_scanlines": 40,
        "overscan_scanlines": 30,
        "colors": 128,  # NTSC palette
        "sprite_types": {
            "player": 2,  # Two player sprites
            "missile": 2,  # Two missile sprites
            "ball": 1,     # One ball sprite
            "playfield": 1  # One playfield (background)
        },
        "quantum_mapping": "deterministic",
        "horizontal_positioning": "cycle_precise",  # Requires precise CPU cycle timing
    }
}