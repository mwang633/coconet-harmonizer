"""
Data loading utilities for Bach Chorales.

Uses music21 to load and process Bach chorales into piano roll format.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional
import pickle


# MIDI pitch range for SATB voices
# Bass can go as low as ~36 (C2), Soprano as high as ~84 (C6)
MIN_PITCH = 36
MAX_PITCH = 97
NUM_PITCHES = MAX_PITCH - MIN_PITCH + 1  # 62 pitches

# Standard voice indices
SOPRANO = 0
ALTO = 1
TENOR = 2
BASS = 3


def load_bach_chorales(
    data_dir: Optional[Path] = None,
    use_music21: bool = True,
    max_chorales: Optional[int] = None,
) -> list[np.ndarray]:
    """
    Load Bach chorales dataset.

    Args:
        data_dir: Directory to save/load cached data
        use_music21: If True, load from music21's corpus
        max_chorales: Maximum number of chorales to load

    Returns:
        List of piano roll arrays, each shape (4, time_steps, num_pitches)
    """
    if data_dir is None:
        data_dir = Path("./data")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    cache_path = data_dir / "bach_chorales.pkl"

    if cache_path.exists():
        print(f"Loading cached chorales from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if not use_music21:
        raise ValueError("music21 required for initial data loading")

    print("Loading Bach chorales from music21 corpus...")
    chorales = _load_from_music21(max_chorales)

    # Cache for faster future loading
    with open(cache_path, "wb") as f:
        pickle.dump(chorales, f)
    print(f"Cached {len(chorales)} chorales to {cache_path}")

    return chorales


def _load_from_music21(max_chorales: Optional[int] = None) -> list[np.ndarray]:
    """Load chorales using music21 library."""
    from music21 import corpus

    chorales = []
    bach_bundle = corpus.search("bach", "composer")
    chorale_paths = [
        b.sourcePath for b in bach_bundle
        if "riemenschneider" in str(b.sourcePath).lower() or "bwv" in str(b.sourcePath).lower()
    ]

    # Fallback: get all Bach works and filter
    if len(chorale_paths) < 10:
        chorale_paths = corpus.getComposer("bach")

    if max_chorales:
        chorale_paths = chorale_paths[:max_chorales]

    for i, path in enumerate(chorale_paths):
        try:
            score = corpus.parse(path)
            pianoroll = _score_to_pianoroll(score)
            if pianoroll is not None:
                chorales.append(pianoroll)
                print(f"  Loaded {i+1}/{len(chorale_paths)}: {Path(path).stem}")
        except Exception as e:
            print(f"  Skipping {path}: {e}")
            continue

    print(f"Successfully loaded {len(chorales)} chorales")
    return chorales


def _score_to_pianoroll(
    score,
    time_step: float = 0.25,  # 16th note resolution
) -> Optional[np.ndarray]:
    """
    Convert a music21 score to piano roll format.

    Args:
        score: music21 Score object
        time_step: Time resolution in quarter notes (0.25 = 16th note)

    Returns:
        Piano roll array of shape (4, time_steps, num_pitches) or None if invalid
    """
    from music21 import chord, note

    # Get the 4 voice parts
    parts = list(score.parts)
    if len(parts) != 4:
        return None

    # Find total duration
    total_duration = float(score.highestTime)
    num_steps = int(np.ceil(total_duration / time_step))

    if num_steps < 16:  # Skip very short pieces
        return None

    # Initialize piano roll
    pianoroll = np.zeros((4, num_steps, NUM_PITCHES), dtype=np.float32)

    for voice_idx, part in enumerate(parts):
        for element in part.recurse().notesAndRests:
            if isinstance(element, note.Rest):
                continue

            start_time = float(element.offset)
            duration = float(element.quarterLength)
            start_step = int(start_time / time_step)
            end_step = int((start_time + duration) / time_step)

            # Handle chords
            if isinstance(element, chord.Chord):
                pitches = [p.midi for p in element.pitches]
            else:
                pitches = [element.pitch.midi]

            for pitch in pitches:
                if MIN_PITCH <= pitch <= MAX_PITCH:
                    pitch_idx = pitch - MIN_PITCH
                    for t in range(start_step, min(end_step, num_steps)):
                        pianoroll[voice_idx, t, pitch_idx] = 1.0

    return pianoroll


class BachChoraleDataset(Dataset):
    """
    PyTorch Dataset for Bach chorales.

    Provides random segments of chorales with random masking
    for Coconet-style training.
    """

    def __init__(
        self,
        chorales: list[np.ndarray],
        segment_length: int = 32,
        mask_prob: float = 0.5,
        augment: bool = True,
    ):
        """
        Args:
            chorales: List of piano roll arrays
            segment_length: Length of segments to extract
            mask_prob: Probability of masking each voice
            augment: Whether to apply data augmentation (transposition)
        """
        self.chorales = chorales
        self.segment_length = segment_length
        self.mask_prob = mask_prob
        self.augment = augment

        # Build index of valid segments
        self.segments = []
        for chorale_idx, chorale in enumerate(chorales):
            num_steps = chorale.shape[1]
            for start in range(0, num_steps - segment_length + 1, segment_length // 2):
                self.segments.append((chorale_idx, start))

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pianoroll: (4, segment_length, num_pitches) input
            mask: (4, segment_length) which voices are known
            target: (4, segment_length, num_pitches) full ground truth
        """
        chorale_idx, start = self.segments[idx]
        chorale = self.chorales[chorale_idx]

        # Extract segment
        segment = chorale[:, start:start + self.segment_length].copy()

        # Data augmentation: random transposition
        if self.augment:
            segment = self._transpose(segment)

        # Create random mask
        mask = np.random.random((4, self.segment_length)) > self.mask_prob
        mask = mask.astype(np.float32)

        # Ensure at least one voice is masked and one is known
        if mask.sum() == 0:
            # Keep soprano
            mask[0] = 1.0
        if mask.sum() == mask.size:
            # Mask a random voice
            mask[np.random.randint(4)] = 0.0

        target = torch.from_numpy(segment)
        pianoroll = torch.from_numpy(segment)
        mask = torch.from_numpy(mask)

        return pianoroll, mask, target

    def _transpose(self, segment: np.ndarray, max_shift: int = 3) -> np.ndarray:
        """Randomly transpose the segment."""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return segment

        new_segment = np.zeros_like(segment)
        if shift > 0:
            new_segment[:, :, shift:] = segment[:, :, :-shift]
        else:
            new_segment[:, :, :shift] = segment[:, :, -shift:]

        return new_segment


def create_dataloader(
    chorales: list[np.ndarray],
    batch_size: int = 32,
    segment_length: int = 32,
    mask_prob: float = 0.5,
    shuffle: bool = True,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for training."""
    dataset = BachChoraleDataset(
        chorales=chorales,
        segment_length=segment_length,
        mask_prob=mask_prob,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
