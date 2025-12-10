# Impulse Response Generator with CUDA Acceleration

A Python tool for generating room impulse responses (RIR) based on acoustic wave simulation. This tool uses the Image Source Method to simulate sound reflections in a rectangular room and can leverage CUDA 13.0 for GPU-accelerated parallel computation.

## Features

- 🚀 **CUDA Acceleration**: Leverages NVIDIA CUDA 13.0 for fast parallel computation
- 🏠 **Room Acoustic Simulation**: Uses Image Source Method for realistic reflections
- 🎵 **WAV Output**: Generates standard WAV files compatible with audio software
- 📍 **Flexible Positioning**: Configure room dimensions, source, and listener positions
- 🎛️ **Customizable Parameters**: Control absorption, reflection order, sample rate, and duration
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

Generate an impulse response by specifying room dimensions, source position, and listener position:

```bash
python impulse_response.py --room 5 4 3 --source 1 2 1.5 --listener 4 2 1.5 -o room.wav
```

#### Parameters:

- `--room LENGTH WIDTH HEIGHT`: Room dimensions in meters (required)
- `--source X Y Z`: Sound source position in meters (required)
- `--listener X Y Z`: Listener position in meters (required)
- `-o, --output`: Output WAV filename (default: impulse_response.wav)
- `--sample-rate`: Sample rate in Hz (default: 48000)
- `--duration`: Duration in seconds (default: 1.0)
- `--max-order`: Maximum reflection order (default: 10)
- `--absorption`: Wall absorption coefficient 0-1 (default: 0.2)
- `--speed-of-sound`: Speed of sound in m/s (default: 343.0)

#### Examples:

**Small bedroom:**
```bash
python impulse_response.py \
  --room 4 3 2.5 \
  --source 1 1 1.2 \
  --listener 2 1.5 1.2 \
  --absorption 0.3 \
  -o bedroom.wav
```

**Large concert hall:**
```bash
python impulse_response.py \
  --room 20 15 8 \
  --source 10 3 2 \
  --listener 10 10 1.5 \
  --max-order 15 \
  --duration 2.0 \
  --absorption 0.15 \
  -o concert_hall.wav
```

**High-resolution recording:**
```bash
python impulse_response.py \
  --room 6 5 3 \
  --source 3 2 1.5 \
  --listener 3 4 1.5 \
  --sample-rate 96000 \
  --max-order 12 \
  -o hires.wav
```

### Python API

You can also use the impulse response generator programmatically:

```python
from impulse_response import ImpulseResponseGenerator

# Create generator
generator = ImpulseResponseGenerator(
    room_dimensions=(5.0, 4.0, 3.0),  # 5m x 4m x 3m room
    sample_rate=48000
)

# Generate impulse response
ir = generator.generate(
    source_position=(1.0, 2.0, 1.5),
    listener_position=(4.0, 2.0, 1.5),
    max_order=10,
    duration=1.0,
    absorption=0.2
)

# Save to WAV file
generator.save_wav(ir, "my_impulse_response.wav")
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

### CUDA Acceleration

When CUDA is available, the tool:
- Parallelizes computation across thousands of image sources
- Uses GPU atomic operations for thread-safe accumulation
- Achieves significant speedup for high reflection orders

### Parameters Explained

- **Room Dimensions**: Physical size of the rectangular room in meters
- **Source Position**: Where the sound impulse originates (e.g., speaker location)
- **Listener Position**: Where the sound is received (e.g., microphone or ear position)
- **Max Order**: Number of wall bounces to simulate (higher = more realistic but slower)
- **Absorption**: How much energy walls absorb (0 = perfect reflection, 1 = complete absorption)
- **Duration**: Length of the impulse response recording

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

### Memory Issues

For very large rooms or high reflection orders:
- Reduce `--max-order`
- Reduce `--duration`
- Use smaller room dimensions

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Based on the Image Source Method for room acoustics simulation
- CUDA acceleration implementation inspired by GPU computing best practices
- Uses the excellent libraries: NumPy, SciPy, soundfile, and PyCUDA