# Impulse Response Generator with CUDA Acceleration

A Python tool for generating room impulse responses (RIR) based on acoustic wave simulation. This tool uses the Image Source Method to simulate sound reflections in a rectangular room and can leverage CUDA 13.0 for GPU-accelerated parallel computation.

## Features

- 🚀 **CUDA Acceleration**: Leverages NVIDIA CUDA 13.0 for fast parallel computation
- 🏠 **Room Acoustic Simulation**: Uses Image Source Method for realistic reflections
- 🎧 **Binaural Output (Default)**: Generates stereo WAV files with realistic spatial audio for left and right ears
- 🎵 **WAV Output**: Generates standard WAV files compatible with audio software
- 📍 **Flexible Positioning**: Configure room dimensions, source, and listener positions
- 🎛️ **Customizable Parameters**: Control absorption, reflection order, sample rate, duration, and ear separation
- 🔊 **Custom Source Signal**: Bipolar impulse with g(1)=1, g(3)=-1 for realistic sound wave representation
- 💻 **CPU Fallback**: Automatically falls back to CPU if CUDA is unavailable

## Requirements

- Python 3.7+
- NumPy
- SciPy
- soundfile
- PyCUDA (optional, for GPU acceleration)
- NVIDIA GPU with CUDA 13.0+ (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/azet24/impulse_response.git
cd impulse_response
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install CUDA toolkit 13.0 or later for GPU acceleration:
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Follow the installation instructions for your platform

## Usage

### Command Line Interface

Generate a binaural (stereo) impulse response by specifying room dimensions, source position, and listener position:

```bash
python impulse_response.py --room 5 4 3 --source 1 2 1.5 --listener 4 2 1.5 -o room.wav
```

This creates a stereo WAV file with separate left and right ear channels for realistic spatial audio.

#### Parameters:

- `--room LENGTH WIDTH HEIGHT`: Room dimensions in meters (required)
- `--source X Y Z`: Sound source position in meters (required)
- `--listener X Y Z`: Listener head center position in meters (required)
- `-o, --output`: Output WAV filename (default: impulse_response.wav)
- `--sample-rate`: Sample rate in Hz (default: 44100)
- `--duration`: Duration in seconds (default: 1.0)
- `--max-order`: Maximum reflection order (default: 10)
- `--absorption`: Wall absorption coefficient 0-1 (default: 0.2)
- `--speed-of-sound`: Speed of sound in m/s (default: 343.0)
- `--mono`: Generate mono output instead of binaural (stereo is default)
- `--ear-separation`: Distance between ears in meters for binaural (default: 0.15)

#### Examples:

**Small bedroom (binaural, default):**
```bash
python impulse_response.py \
  --room 4 3 2.5 \
  --source 1 1 1.2 \
  --listener 2 1.5 1.2 \
  --absorption 0.3 \
  -o bedroom_stereo.wav
```

**Large concert hall (binaural):**
```bash
python impulse_response.py \
  --room 20 15 8 \
  --source 10 3 2 \
  --listener 10 10 1.5 \
  --max-order 15 \
  --duration 2.0 \
  --absorption 0.15 \
  -o concert_hall_stereo.wav
```

**Mono output:**
```bash
python impulse_response.py \
  --room 6 5 3 \
  --source 3 2 1.5 \
  --listener 3 4 1.5 \
  --mono \
  -o mono_output.wav
```

**Custom ear separation:**
```bash
python impulse_response.py \
  --room 5 4 3 \
  --source 2 2 1.5 \
  --listener 3 2 1.5 \
  --ear-separation 0.18 \
  -o wide_stereo.wav
```

**High-resolution binaural recording:**
```bash
python impulse_response.py \
  --room 6 5 3 \
  --source 3 2 1.5 \
  --listener 3 4 1.5 \
  --sample-rate 96000 \
  --max-order 12 \
  -o hires_stereo.wav
```

### Python API

You can also use the impulse response generator programmatically:

```python
from impulse_response import ImpulseResponseGenerator
import numpy as np

# Create generator (binaural by default)
generator = ImpulseResponseGenerator(
    room_dimensions=(5.0, 4.0, 3.0),  # 5m x 4m x 3m room
    sample_rate=44100,
    binaural=True,  # Default, generates stereo output
    ear_separation=0.15  # Default, 15cm between ears
)

# Generate binaural impulse response
ir = generator.generate(
    source_position=(1.0, 2.0, 1.5),
    listener_position=(4.0, 2.0, 1.5),
    max_order=10,
    duration=1.0,
    absorption=0.2
)

# ir shape: (samples, 2) for stereo - left channel: ir[:, 0], right channel: ir[:, 1]
generator.save_wav(ir, "binaural_output.wav")

# For mono output
generator_mono = ImpulseResponseGenerator(
    room_dimensions=(5.0, 4.0, 3.0),
    sample_rate=44100,
    binaural=False  # Mono output
)

ir_mono = generator_mono.generate(
    source_position=(1.0, 2.0, 1.5),
    listener_position=(4.0, 2.0, 1.5),
    max_order=10,
    duration=1.0,
    absorption=0.2
)

# ir_mono shape: (samples,) for mono
generator_mono.save_wav(ir_mono, "mono_output.wav")

# Custom source signal (default is g(1)=1, g(3)=-1)
custom_signal = np.array([0, 1, 0, -1], dtype=np.float32)
ir_custom = generator.generate(
    source_position=(1.0, 2.0, 1.5),
    listener_position=(4.0, 2.0, 1.5),
    max_order=10,
    duration=1.0,
    absorption=0.2,
    source_signal=custom_signal
)
```

### Running Examples

The repository includes several examples demonstrating different use cases:

```bash
python example.py
```

This will generate:
- `example_small_room.wav` - Small bedroom impulse response
- `example_large_room.wav` - Large concert hall impulse response
- `example_stereo.wav` - Binaural stereo impulse response
- `example_absorption_*.wav` - Comparison of different absorption values

## How It Works

### Image Source Method

The Image Source Method simulates room acoustics by creating virtual "image sources" that represent reflections from room walls. Each reflection:

1. **Position**: Mirrored across room boundaries
2. **Attenuation**: Reduced by distance and wall absorption
3. **Delay**: Based on sound travel time (distance / speed of sound)

### Binaural Audio Generation

For binaural (stereo) output (default mode):
- The listener position represents the center of the head
- Left and right ear positions are calculated along the X-axis
- Separate impulse responses are generated for each ear
- Inter-aural Time Difference (ITD) and Inter-aural Level Difference (ILD) are naturally captured
- Creates realistic 3D spatial audio suitable for headphone listening

### Custom Source Signal

The default source impulse is a bipolar signal where:
- g(0) = 0
- g(1) = 1 (positive peak)
- g(2) = 0
- g(3) = -1 (negative peak)

This creates realistic sound wave propagation with both positive and negative components, mimicking natural acoustic phenomena.

### CUDA Acceleration

When CUDA is available, the tool:
- Parallelizes computation across thousands of image sources
- Uses GPU atomic operations for thread-safe accumulation
- Achieves significant speedup for high reflection orders
- Works identically for both mono and binaural output

### Parameters Explained

- **Room Dimensions**: Physical size of the rectangular room in meters
- **Source Position**: Where the sound impulse originates (e.g., speaker location)
- **Listener Position**: Head center position for binaural, or microphone position for mono
- **Max Order**: Number of wall bounces to simulate (higher = more realistic but slower)
- **Absorption**: How much energy walls absorb (0 = perfect reflection, 1 = complete absorption)
- **Duration**: Length of the impulse response recording
- **Ear Separation**: Distance between left and right ears for binaural mode (default: 0.15m)
- **Binaural**: Whether to generate stereo (True, default) or mono (False) output

## Performance

Performance varies based on parameters:

| Configuration | CPU Time | CUDA Time | Speedup |
|---------------|----------|-----------|---------|
| Max order 10  | ~2-5s    | ~0.1-0.3s | 10-20x  |
| Max order 15  | ~30-60s  | ~1-2s     | 20-30x  |
| Max order 20  | ~5-10min | ~5-15s    | 30-40x  |

*Note: Performance depends on CPU/GPU models and room parameters*

## Troubleshooting

### CUDA Not Available

If you see "Warning: PyCUDA not available", the tool will use CPU fallback. To enable CUDA:

1. Install NVIDIA CUDA Toolkit 13.0+
2. Install PyCUDA: `pip install pycuda`
3. Verify GPU access: `nvidia-smi`

### Position Validation Errors

Ensure source and listener positions are within room boundaries:
- All coordinates must be ≥ 0
- X coordinate must be ≤ room length
- Y coordinate must be ≤ room width  
- Z coordinate must be ≤ room height

For binaural mode, ensure that ear positions (listener position ± ear_separation/2 along X-axis) are also within room boundaries.

### Mono vs Binaural

- **Binaural (default)**: Generates stereo WAV files with separate left and right ear channels. Best for headphone listening and spatial audio applications.
- **Mono**: Use `--mono` flag or set `binaural=False` in the API. Generates single-channel output. Useful for non-spatial applications or when file size is a concern.

### Memory Issues

For very large rooms or high reflection orders:
- Reduce `--max-order`
- Reduce `--duration`
- Use smaller room dimensions
- Note: Binaural mode requires approximately 2x the memory of mono mode

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Based on the Image Source Method for room acoustics simulation
- CUDA acceleration implementation inspired by GPU computing best practices
- Uses the excellent libraries: NumPy, SciPy, soundfile, and PyCUDA