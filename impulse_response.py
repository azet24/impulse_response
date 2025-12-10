#!/usr/bin/env python3
"""
Impulse Response Generator with CUDA Acceleration

This script generates room impulse responses (RIR) using the Image Source Method
accelerated with CUDA for parallel computation of reflections.
"""

import numpy as np
import argparse
import sys
from typing import Tuple, List

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: PyCUDA not available. Falling back to CPU implementation.")

import soundfile as sf


# CUDA kernel for computing image source contributions
CUDA_KERNEL = """
__global__ void compute_reflections(
    float *output,
    float source_x, float source_y, float source_z,
    float listener_x, float listener_y, float listener_z,
    float room_x, float room_y, float room_z,
    float absorption,
    int max_order,
    int sample_rate,
    int num_samples,
    float speed_of_sound
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Iterate through image source orders
    for (int nx = -max_order; nx <= max_order; nx++) {
        for (int ny = -max_order; ny <= max_order; ny++) {
            for (int nz = -max_order; nz <= max_order; nz++) {
                // Skip if thread index doesn't match this reflection
                int reflection_idx = (nx + max_order) * (2*max_order+1) * (2*max_order+1) + 
                                    (ny + max_order) * (2*max_order+1) + 
                                    (nz + max_order);
                
                if (idx != reflection_idx) continue;
                
                // Calculate image source position
                float img_x = source_x;
                float img_y = source_y;
                float img_z = source_z;
                
                if (nx % 2 == 0) {
                    img_x = nx * room_x + source_x;
                } else {
                    img_x = nx * room_x + (room_x - source_x);
                }
                
                if (ny % 2 == 0) {
                    img_y = ny * room_y + source_y;
                } else {
                    img_y = ny * room_y + (room_y - source_y);
                }
                
                if (nz % 2 == 0) {
                    img_z = nz * room_z + source_z;
                } else {
                    img_z = nz * room_z + (room_z - source_z);
                }
                
                // Calculate distance to listener
                float dx = img_x - listener_x;
                float dy = img_y - listener_y;
                float dz = img_z - listener_z;
                float distance = sqrtf(dx*dx + dy*dy + dz*dz);
                
                if (distance < 0.01f) continue; // Skip if too close
                
                // Calculate reflection order (number of wall bounces)
                int order = abs(nx) + abs(ny) + abs(nz);
                
                // Calculate attenuation due to distance and absorption
                float attenuation = powf(1.0f - absorption, (float)order) / distance;
                
                // Calculate arrival time in samples
                float arrival_time = distance / speed_of_sound;
                int sample_idx = (int)(arrival_time * sample_rate);
                
                if (sample_idx >= 0 && sample_idx < num_samples) {
                    // Use atomic add to prevent race conditions
                    atomicAdd(&output[sample_idx], attenuation);
                }
            }
        }
    }
}
"""


class ImpulseResponseGenerator:
    """Generate room impulse responses using Image Source Method with CUDA acceleration."""
    
    def __init__(self, 
                 room_dimensions: Tuple[float, float, float],
                 sample_rate: int = 44100,
                 speed_of_sound: float = 343.0):
        """
        Initialize the impulse response generator.
        
        Args:
            room_dimensions: Tuple of (length, width, height) in meters
            sample_rate: Audio sample rate in Hz
            speed_of_sound: Speed of sound in m/s (default 343 m/s at 20°C)
        """
        self.room_dimensions = np.array(room_dimensions, dtype=np.float32)
        self.sample_rate = sample_rate
        self.speed_of_sound = speed_of_sound
        self.cuda_available = CUDA_AVAILABLE
        
        if self.cuda_available:
            self.cuda_module = SourceModule(CUDA_KERNEL)
            self.cuda_kernel = self.cuda_module.get_function("compute_reflections")
    
    def generate(self,
                 source_position: Tuple[float, float, float],
                 listener_position: Tuple[float, float, float],
                 max_order: int = 10,
                 duration: float = 1.0,
                 absorption: float = 0.2,
                 source_signal: np.ndarray = None) -> np.ndarray:
        """
        Generate impulse response for given source and listener positions.
        
        Args:
            source_position: (x, y, z) position of sound source in meters
            listener_position: (x, y, z) position of listener in meters
            max_order: Maximum reflection order to compute
            duration: Duration of impulse response in seconds
            absorption: Wall absorption coefficient (0-1, 0=no absorption, 1=full absorption)
            source_signal: Source impulse signal to emit (default: bipolar pulse with g(1)=1, g(3)=-1)
        
        Returns:
            Impulse response as numpy array
        """
        source_pos = np.array(source_position, dtype=np.float32)
        listener_pos = np.array(listener_position, dtype=np.float32)
        
        # Validate positions are within room
        if not self._validate_position(source_pos):
            raise ValueError(f"Source position {source_pos} is outside room {self.room_dimensions}")
        if not self._validate_position(listener_pos):
            raise ValueError(f"Listener position {listener_pos} is outside room {self.room_dimensions}")
        
        # Default source signal: g(1)=1, g(3)=-1, zero elsewhere
        if source_signal is None:
            source_signal = np.zeros(4, dtype=np.float32)
            source_signal[1] = 1.0   # g(1) = 1
            source_signal[3] = -1.0  # g(3) = -1
        
        num_samples = int(duration * self.sample_rate)
        
        if self.cuda_available:
            rir = self._generate_cuda(source_pos, listener_pos, max_order, 
                                      num_samples, absorption)
        else:
            rir = self._generate_cpu(source_pos, listener_pos, max_order, 
                                     num_samples, absorption)
        
        # Convolve room impulse response with source signal
        output = np.convolve(rir, source_signal, mode='same')
        
        # Normalize
        if np.max(np.abs(output)) > 0:
            output = output / np.max(np.abs(output)) * 0.9
        
        return output.astype(np.float32)
    
    def _validate_position(self, position: np.ndarray) -> bool:
        """Check if position is within room boundaries."""
        return np.all(position >= 0) and np.all(position <= self.room_dimensions)
    
    def _generate_cuda(self,
                      source_pos: np.ndarray,
                      listener_pos: np.ndarray,
                      max_order: int,
                      num_samples: int,
                      absorption: float) -> np.ndarray:
        """Generate impulse response using CUDA acceleration."""
        # Allocate output array
        output = np.zeros(num_samples, dtype=np.float32)
        
        # Allocate GPU memory
        output_gpu = cuda.mem_alloc(output.nbytes)
        cuda.memcpy_htod(output_gpu, output)
        
        # Calculate grid dimensions
        total_reflections = (2 * max_order + 1) ** 3
        block_size = 256
        grid_size = (total_reflections + block_size - 1) // block_size
        
        # Launch kernel
        self.cuda_kernel(
            output_gpu,
            np.float32(source_pos[0]), np.float32(source_pos[1]), np.float32(source_pos[2]),
            np.float32(listener_pos[0]), np.float32(listener_pos[1]), np.float32(listener_pos[2]),
            np.float32(self.room_dimensions[0]), np.float32(self.room_dimensions[1]), 
            np.float32(self.room_dimensions[2]),
            np.float32(absorption),
            np.int32(max_order),
            np.int32(self.sample_rate),
            np.int32(num_samples),
            np.float32(self.speed_of_sound),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )
        
        # Copy result back
        cuda.memcpy_dtoh(output, output_gpu)
        
        return output
    
    def _generate_cpu(self,
                     source_pos: np.ndarray,
                     listener_pos: np.ndarray,
                     max_order: int,
                     num_samples: int,
                     absorption: float) -> np.ndarray:
        """Generate impulse response using CPU (fallback when CUDA unavailable)."""
        output = np.zeros(num_samples, dtype=np.float32)
        
        # Iterate through all image sources
        for nx in range(-max_order, max_order + 1):
            for ny in range(-max_order, max_order + 1):
                for nz in range(-max_order, max_order + 1):
                    # Calculate image source position
                    img_pos = np.copy(source_pos)
                    
                    if nx % 2 == 0:
                        img_pos[0] = nx * self.room_dimensions[0] + source_pos[0]
                    else:
                        img_pos[0] = nx * self.room_dimensions[0] + (self.room_dimensions[0] - source_pos[0])
                    
                    if ny % 2 == 0:
                        img_pos[1] = ny * self.room_dimensions[1] + source_pos[1]
                    else:
                        img_pos[1] = ny * self.room_dimensions[1] + (self.room_dimensions[1] - source_pos[1])
                    
                    if nz % 2 == 0:
                        img_pos[2] = nz * self.room_dimensions[2] + source_pos[2]
                    else:
                        img_pos[2] = nz * self.room_dimensions[2] + (self.room_dimensions[2] - source_pos[2])
                    
                    # Calculate distance to listener
                    distance = np.linalg.norm(img_pos - listener_pos)
                    
                    if distance < 0.01:  # Skip if too close
                        continue
                    
                    # Calculate reflection order
                    order = abs(nx) + abs(ny) + abs(nz)
                    
                    # Calculate attenuation
                    attenuation = ((1.0 - absorption) ** order) / distance
                    
                    # Calculate arrival time
                    arrival_time = distance / self.speed_of_sound
                    sample_idx = int(arrival_time * self.sample_rate)
                    
                    if 0 <= sample_idx < num_samples:
                        output[sample_idx] += attenuation
        
        return output
    
    def save_wav(self, impulse_response: np.ndarray, filename: str):
        """
        Save impulse response to WAV file.
        
        Args:
            impulse_response: The impulse response array
            filename: Output filename
        """
        sf.write(filename, impulse_response, self.sample_rate)
        print(f"Impulse response saved to {filename}")


def main():
    """Command-line interface for impulse response generator."""
    parser = argparse.ArgumentParser(
        description="Generate room impulse response with CUDA acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate IR for a 5x4x3m room
  python impulse_response.py --room 5 4 3 --source 1 2 1.5 --listener 4 2 1.5 -o room.wav
  
  # Generate IR with higher reflection order and custom duration
  python impulse_response.py --room 10 8 3 --source 2 2 1.5 --listener 8 6 1.5 \\
                             --max-order 15 --duration 2.0 -o large_room.wav
        """
    )
    
    parser.add_argument('--room', type=float, nargs=3, required=True,
                       metavar=('LENGTH', 'WIDTH', 'HEIGHT'),
                       help='Room dimensions in meters (length width height)')
    parser.add_argument('--source', type=float, nargs=3, required=True,
                       metavar=('X', 'Y', 'Z'),
                       help='Sound source position in meters (x y z)')
    parser.add_argument('--listener', type=float, nargs=3, required=True,
                       metavar=('X', 'Y', 'Z'),
                       help='Listener position in meters (x y z)')
    parser.add_argument('-o', '--output', type=str, default='impulse_response.wav',
                       help='Output WAV filename (default: impulse_response.wav)')
    parser.add_argument('--sample-rate', type=int, default=44100,
                       help='Sample rate in Hz (default: 44100)')
    parser.add_argument('--duration', type=float, default=1.0,
                       help='Duration in seconds (default: 1.0)')
    parser.add_argument('--max-order', type=int, default=10,
                       help='Maximum reflection order (default: 10)')
    parser.add_argument('--absorption', type=float, default=0.2,
                       help='Wall absorption coefficient 0-1 (default: 0.2)')
    parser.add_argument('--speed-of-sound', type=float, default=343.0,
                       help='Speed of sound in m/s (default: 343.0)')
    
    args = parser.parse_args()
    
    # Validate absorption coefficient
    if not 0 <= args.absorption <= 1:
        parser.error("Absorption coefficient must be between 0 and 1")
    
    # Print configuration
    print("=" * 60)
    print("Impulse Response Generator")
    print("=" * 60)
    print(f"Room dimensions: {args.room[0]}m x {args.room[1]}m x {args.room[2]}m")
    print(f"Source position: ({args.source[0]}, {args.source[1]}, {args.source[2]}) m")
    print(f"Listener position: ({args.listener[0]}, {args.listener[1]}, {args.listener[2]}) m")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Duration: {args.duration} s")
    print(f"Max reflection order: {args.max_order}")
    print(f"Wall absorption: {args.absorption}")
    print(f"CUDA acceleration: {'Enabled' if CUDA_AVAILABLE else 'Disabled (CPU fallback)'}")
    print("=" * 60)
    
    try:
        # Create generator
        generator = ImpulseResponseGenerator(
            room_dimensions=tuple(args.room),
            sample_rate=args.sample_rate,
            speed_of_sound=args.speed_of_sound
        )
        
        # Generate impulse response
        print("Generating impulse response...")
        ir = generator.generate(
            source_position=tuple(args.source),
            listener_position=tuple(args.listener),
            max_order=args.max_order,
            duration=args.duration,
            absorption=args.absorption
        )
        
        # Save to file
        generator.save_wav(ir, args.output)
        
        # Print statistics
        print("\nImpulse Response Statistics:")
        print(f"  Samples: {len(ir)}")
        print(f"  Peak amplitude: {np.max(np.abs(ir)):.4f}")
        print(f"  Non-zero samples: {np.count_nonzero(ir)}")
        print("\nSuccess!")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
