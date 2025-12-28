"""
Coconet: 4-Part Harmony Generator

Train a neural network to harmonize melodies in the style of Bach.
"""

import torch


def check_setup():
    """Verify the setup is working."""
    print("=" * 50)
    print("Coconet Setup Check")
    print("=" * 50)

    # Check CUDA
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Check music libraries
    try:
        import music21
        print(f"music21 version: {music21.__version__}")
    except ImportError:
        print("music21: NOT INSTALLED - run 'uv sync'")

    try:
        import pretty_midi
        print("pretty_midi: OK")
    except ImportError:
        print("pretty_midi: NOT INSTALLED - run 'uv sync'")

    # Check Coconet module
    try:
        from coconet import Coconet, Harmonizer
        print("\nCoconet module: OK")

        # Create a small test model
        model = Coconet(num_layers=4, num_filters=32)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"\nCoconet module error: {e}")

    print("\n" + "=" * 50)


def demo():
    """Run a simple demo with an untrained model."""
    from coconet import Harmonizer
    from coconet.data import MIN_PITCH

    print("\nDemo: Generating harmony for a C major scale")
    print("(Note: Using untrained model - results will be random)")
    print("-" * 50)

    # C major scale
    melody = [60, 60, 62, 62, 64, 64, 65, 65, 67, 67, 69, 69, 71, 71, 72, 72]

    harmonizer = Harmonizer(num_layers=8, num_filters=64)
    result = harmonizer.harmonize_melody(melody, temperature=1.0, num_iterations=50)

    voice_names = ["Soprano", "Alto", "Tenor", "Bass"]
    print("\nGenerated voices (MIDI pitches):")
    for voice_idx, name in enumerate(voice_names):
        pitches = []
        for t in range(result.shape[1]):
            if result[voice_idx, t].max() > 0.5:
                pitch = result[voice_idx, t].argmax() + MIN_PITCH
                pitches.append(pitch)
            else:
                pitches.append("-")
        print(f"  {name:8}: {pitches}")


if __name__ == "__main__":
    check_setup()
    demo()
