"""
Harmonization utilities for Coconet.

Generate 4-part harmonizations from melodies.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union

from .model import Coconet, create_model
from .data import MIN_PITCH, MAX_PITCH, NUM_PITCHES, SOPRANO, ALTO, TENOR, BASS


class Harmonizer:
    """
    High-level interface for generating 4-part harmonizations.

    Takes a melody and generates SATB (Soprano, Alto, Tenor, Bass) harmony.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = None,
        num_layers: int = 32,
        num_filters: int = 128,
    ):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run on (cuda/cpu)
            num_layers: Number of model layers (if no checkpoint)
            num_filters: Number of filters (if no checkpoint)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = create_model(
            num_layers=num_layers,
            num_filters=num_filters,
            device=self.device
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)
            print(f"Loaded model from {checkpoint_path}")
        else:
            print("Warning: Using untrained model")

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def harmonize_melody(
        self,
        melody: Union[list[int], np.ndarray],
        temperature: float = 0.8,
        num_iterations: int = 100,
        melody_voice: int = SOPRANO,
    ) -> np.ndarray:
        """
        Generate 4-part harmony from a melody.

        Args:
            melody: List of MIDI pitch values (one per time step, 0 for rest)
            temperature: Sampling temperature (lower = more deterministic)
            num_iterations: Gibbs sampling iterations
            melody_voice: Which voice the melody is (0=soprano, 3=bass)

        Returns:
            Piano roll of shape (4, time_steps, num_pitches)
        """
        melody = np.array(melody)
        time_steps = len(melody)

        # Create input piano roll with just the melody
        pianoroll = np.zeros((1, 4, time_steps, NUM_PITCHES), dtype=np.float32)

        for t, pitch in enumerate(melody):
            if pitch > 0 and MIN_PITCH <= pitch <= MAX_PITCH:
                pitch_idx = pitch - MIN_PITCH
                pianoroll[0, melody_voice, t, pitch_idx] = 1.0

        # Create mask: melody voice is known (1), others are unknown (0)
        mask = np.zeros((1, 4, time_steps), dtype=np.float32)
        mask[0, melody_voice] = 1.0

        # Convert to tensors
        pianoroll_t = torch.from_numpy(pianoroll)
        mask_t = torch.from_numpy(mask)

        # Generate harmonization
        result = self.model.sample(
            pianoroll_t,
            mask_t,
            temperature=temperature,
            num_iterations=num_iterations,
        )

        return result[0].cpu().numpy()

    def harmonize_midi_file(
        self,
        input_path: str,
        output_path: str,
        temperature: float = 0.8,
        melody_voice: int = SOPRANO,
    ):
        """
        Harmonize a melody from a MIDI file.

        Args:
            input_path: Path to input MIDI file (single melody)
            output_path: Path to save harmonized MIDI
            temperature: Sampling temperature
            melody_voice: Which voice the melody should be
        """
        import pretty_midi

        # Load input MIDI
        midi = pretty_midi.PrettyMIDI(input_path)
        if len(midi.instruments) == 0:
            raise ValueError("No instruments in MIDI file")

        # Extract melody (take first instrument)
        instrument = midi.instruments[0]
        notes = sorted(instrument.notes, key=lambda n: n.start)

        if len(notes) == 0:
            raise ValueError("No notes in MIDI file")

        # Quantize to 16th notes
        tempo = midi.get_tempo_changes()[1][0] if midi.get_tempo_changes()[1].size else 120
        step_duration = 60 / tempo / 4  # 16th note duration in seconds

        # Find total duration
        end_time = max(n.end for n in notes)
        num_steps = int(np.ceil(end_time / step_duration))

        # Create melody array
        melody = np.zeros(num_steps, dtype=np.int32)
        for note in notes:
            start_step = int(note.start / step_duration)
            end_step = int(note.end / step_duration)
            for t in range(start_step, min(end_step, num_steps)):
                melody[t] = note.pitch

        # Harmonize
        pianoroll = self.harmonize_melody(
            melody,
            temperature=temperature,
            melody_voice=melody_voice,
        )

        # Convert back to MIDI
        self._pianoroll_to_midi(pianoroll, output_path, step_duration)
        print(f"Saved harmonization to {output_path}")

    def _pianoroll_to_midi(
        self,
        pianoroll: np.ndarray,
        output_path: str,
        step_duration: float = 0.125,  # 16th note at 120 BPM
    ):
        """Convert piano roll to MIDI file."""
        import pretty_midi

        midi = pretty_midi.PrettyMIDI()
        voice_names = ["Soprano", "Alto", "Tenor", "Bass"]
        programs = [52, 52, 52, 52]  # Choir Aahs for all voices

        for voice_idx in range(4):
            instrument = pretty_midi.Instrument(
                program=programs[voice_idx],
                name=voice_names[voice_idx]
            )

            # Find notes in this voice
            voice_roll = pianoroll[voice_idx]  # (time, pitches)
            time_steps, num_pitches = voice_roll.shape

            current_pitch = None
            note_start = None

            for t in range(time_steps):
                # Find active pitch at this time step
                active = voice_roll[t].argmax() if voice_roll[t].max() > 0.5 else None

                if active != current_pitch:
                    # End previous note
                    if current_pitch is not None and note_start is not None:
                        midi_pitch = current_pitch + MIN_PITCH
                        note = pretty_midi.Note(
                            velocity=80,
                            pitch=midi_pitch,
                            start=note_start * step_duration,
                            end=t * step_duration
                        )
                        instrument.notes.append(note)

                    # Start new note
                    if active is not None:
                        note_start = t
                    else:
                        note_start = None

                    current_pitch = active

            # Handle last note
            if current_pitch is not None and note_start is not None:
                midi_pitch = current_pitch + MIN_PITCH
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=midi_pitch,
                    start=note_start * step_duration,
                    end=time_steps * step_duration
                )
                instrument.notes.append(note)

            midi.instruments.append(instrument)

        midi.write(output_path)

    def harmonize_from_music21(
        self,
        score,
        temperature: float = 0.8,
        melody_voice: int = SOPRANO,
    ):
        """
        Harmonize a melody from a music21 stream.

        Args:
            score: music21 Stream containing a melody
            temperature: Sampling temperature
            melody_voice: Which voice the melody is

        Returns:
            music21 Score with 4-part harmony
        """
        from music21 import note, stream, chord

        # Extract melody pitches at 16th note resolution
        time_step = 0.25
        total_duration = float(score.highestTime)
        num_steps = int(np.ceil(total_duration / time_step))

        melody = np.zeros(num_steps, dtype=np.int32)

        for element in score.flatten().notesAndRests:
            if isinstance(element, note.Rest):
                continue

            start_step = int(float(element.offset) / time_step)
            end_step = int((float(element.offset) + float(element.quarterLength)) / time_step)

            if isinstance(element, chord.Chord):
                pitch = element.pitches[-1].midi  # Highest note
            else:
                pitch = element.pitch.midi

            for t in range(start_step, min(end_step, num_steps)):
                melody[t] = pitch

        # Harmonize
        pianoroll = self.harmonize_melody(
            melody,
            temperature=temperature,
            melody_voice=melody_voice,
        )

        # Convert to music21 score
        return self._pianoroll_to_music21(pianoroll, time_step)

    def _pianoroll_to_music21(
        self,
        pianoroll: np.ndarray,
        time_step: float = 0.25,
    ):
        """Convert piano roll to music21 Score."""
        from music21 import note, stream, clef

        score = stream.Score()
        voice_names = ["Soprano", "Alto", "Tenor", "Bass"]
        clefs = [clef.TrebleClef(), clef.TrebleClef(), clef.TrebleClef(), clef.BassClef()]

        for voice_idx in range(4):
            part = stream.Part()
            part.partName = voice_names[voice_idx]
            part.insert(0, clefs[voice_idx])

            voice_roll = pianoroll[voice_idx]
            time_steps, num_pitches = voice_roll.shape

            current_pitch = None
            note_start = None

            for t in range(time_steps + 1):
                if t < time_steps:
                    active = voice_roll[t].argmax() if voice_roll[t].max() > 0.5 else None
                else:
                    active = None

                if active != current_pitch:
                    if current_pitch is not None and note_start is not None:
                        midi_pitch = current_pitch + MIN_PITCH
                        duration = (t - note_start) * time_step
                        n = note.Note(midi_pitch)
                        n.quarterLength = duration
                        n.offset = note_start * time_step
                        part.insert(n.offset, n)

                    if active is not None:
                        note_start = t
                    else:
                        note_start = None

                    current_pitch = active

            score.insert(0, part)

        return score


def demo_harmonize():
    """Demo: harmonize a simple melody."""
    # Simple C major scale melody
    melody = [
        60, 60,  # C
        62, 62,  # D
        64, 64,  # E
        65, 65,  # F
        67, 67,  # G
        69, 69,  # A
        71, 71,  # B
        72, 72,  # C
    ]

    harmonizer = Harmonizer()
    result = harmonizer.harmonize_melody(melody, temperature=0.8)

    print("Generated harmonization:")
    voice_names = ["Soprano", "Alto", "Tenor", "Bass"]
    for voice_idx, name in enumerate(voice_names):
        pitches = []
        for t in range(result.shape[1]):
            if result[voice_idx, t].max() > 0.5:
                pitch = result[voice_idx, t].argmax() + MIN_PITCH
                pitches.append(pitch)
            else:
                pitches.append(0)
        print(f"{name}: {pitches}")


if __name__ == "__main__":
    demo_harmonize()
