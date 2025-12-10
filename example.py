#!/usr/bin/env python3
"""
Example usage of the Impulse Response Generator

This script demonstrates how to use the impulse response generator
with various room configurations.
"""

from impulse_response import ImpulseResponseGenerator
import numpy as np


def example_small_room():
    """Generate impulse response for a small room (bedroom)."""
    print("\n" + "="*60)
    print("Example 1: Small Room (Bedroom)")
    print("="*60)
    
    # Room: 4m x 3m x 2.5m
    generator = ImpulseResponseGenerator(
        room_dimensions=(4.0, 3.0, 2.5),
        sample_rate=48000
    )
    
    # Source in one corner, listener in center
    ir = generator.generate(
        source_position=(1.0, 1.0, 1.2),
        listener_position=(2.0, 1.5, 1.2),
        max_order=12,
        duration=0.5,
        absorption=0.3  # Moderate absorption (furnished room)
    )
    
    generator.save_wav(ir, "example_small_room.wav")
    print(f"Generated {len(ir)} samples")


def example_large_room():
    """Generate impulse response for a large room (concert hall)."""
    print("\n" + "="*60)
    print("Example 2: Large Room (Concert Hall)")
    print("="*60)
    
    # Room: 20m x 15m x 8m
    generator = ImpulseResponseGenerator(
        room_dimensions=(20.0, 15.0, 8.0),
        sample_rate=48000
    )
    
    # Source at front, listener in middle
    ir = generator.generate(
        source_position=(10.0, 3.0, 2.0),
        listener_position=(10.0, 10.0, 1.5),
        max_order=15,
        duration=2.0,
        absorption=0.15  # Low absorption (reverberant space)
    )
    
    generator.save_wav(ir, "example_large_room.wav")
    print(f"Generated {len(ir)} samples")


def example_stereo():
    """Generate stereo impulse response (binaural)."""
    print("\n" + "="*60)
    print("Example 3: Stereo (Binaural) Impulse Response")
    print("="*60)
    
    # Room: 6m x 5m x 3m
    generator = ImpulseResponseGenerator(
        room_dimensions=(6.0, 5.0, 3.0),
        sample_rate=48000
    )
    
    # Source in front, two listeners (left and right ear)
    # Ears are approximately 0.15m apart
    source_pos = (3.0, 1.5, 1.5)
    head_center = (3.0, 3.5, 1.5)
    ear_separation = 0.15
    
    # Left ear
    left_ear = (head_center[0] - ear_separation/2, head_center[1], head_center[2])
    ir_left = generator.generate(
        source_position=source_pos,
        listener_position=left_ear,
        max_order=10,
        duration=1.0,
        absorption=0.25
    )
    
    # Right ear
    right_ear = (head_center[0] + ear_separation/2, head_center[1], head_center[2])
    ir_right = generator.generate(
        source_position=source_pos,
        listener_position=right_ear,
        max_order=10,
        duration=1.0,
        absorption=0.25
    )
    
    # Combine into stereo
    import soundfile as sf
    stereo_ir = np.column_stack([ir_left, ir_right])
    sf.write("example_stereo.wav", stereo_ir, generator.sample_rate)
    print(f"Generated stereo impulse response: {stereo_ir.shape}")
    print("Saved to example_stereo.wav")


def example_comparison():
    """Generate impulse responses with different absorption coefficients."""
    print("\n" + "="*60)
    print("Example 4: Absorption Comparison")
    print("="*60)
    
    # Same room, different absorption values
    generator = ImpulseResponseGenerator(
        room_dimensions=(5.0, 4.0, 3.0),
        sample_rate=48000
    )
    
    source_pos = (2.5, 1.0, 1.5)
    listener_pos = (2.5, 3.0, 1.5)
    
    for absorption in [0.1, 0.3, 0.5, 0.7]:
        ir = generator.generate(
            source_position=source_pos,
            listener_position=listener_pos,
            max_order=12,
            duration=1.0,
            absorption=absorption
        )
        
        filename = f"example_absorption_{int(absorption*100):02d}.wav"
        generator.save_wav(ir, filename)
        print(f"  Absorption {absorption:.1f}: {filename}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Impulse Response Generator - Examples")
    print("="*60)
    print("\nGenerating example impulse responses...")
    print("This may take a moment depending on CUDA availability.")
    
    try:
        # Run examples
        example_small_room()
        example_large_room()
        example_stereo()
        example_comparison()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        print("\nGenerated files:")
        print("  - example_small_room.wav")
        print("  - example_large_room.wav")
        print("  - example_stereo.wav")
        print("  - example_absorption_10.wav")
        print("  - example_absorption_30.wav")
        print("  - example_absorption_50.wav")
        print("  - example_absorption_70.wav")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
