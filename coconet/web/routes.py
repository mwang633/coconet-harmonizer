"""API routes for Coconet harmonizer."""

import base64
import io
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..data import MIN_PITCH, MAX_PITCH

router = APIRouter()


class HarmonizeRequest(BaseModel):
    """Request body for harmonization."""
    melody: list[int] = Field(
        ...,
        description="List of MIDI pitch values (0 for rest, 36-97 for notes)",
        min_length=1,
        max_length=512,
    )
    temperature: float = Field(
        default=0.8,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (lower = more deterministic)",
    )
    num_iterations: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Gibbs sampling iterations",
    )


class VoiceData(BaseModel):
    """MIDI pitches for one voice."""
    pitches: list[int]


class HarmonizeResponse(BaseModel):
    """Response with harmonized voices."""
    soprano: list[int]
    alto: list[int]
    tenor: list[int]
    bass: list[int]
    midi_base64: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    from .app import harmonizer

    if harmonizer is None:
        return HealthResponse(
            status="initializing",
            model_loaded=False,
            device="unknown",
        )

    return HealthResponse(
        status="ok",
        model_loaded=True,
        device=harmonizer.device,
    )


@router.post("/harmonize", response_model=HarmonizeResponse)
async def harmonize(request: HarmonizeRequest):
    """Generate 4-part harmony from a melody."""
    from .app import get_harmonizer

    try:
        harmonizer = get_harmonizer()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate melody pitches
    for i, pitch in enumerate(request.melody):
        if pitch != 0 and (pitch < MIN_PITCH or pitch > MAX_PITCH):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid pitch {pitch} at position {i}. "
                       f"Must be 0 (rest) or {MIN_PITCH}-{MAX_PITCH}",
            )

    # Generate harmonization
    try:
        pianoroll = harmonizer.harmonize_melody(
            melody=request.melody,
            temperature=request.temperature,
            num_iterations=request.num_iterations,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Harmonization failed: {e}")

    # Extract pitches for each voice
    voices = extract_pitches_from_pianoroll(pianoroll)

    # Generate MIDI
    midi_base64 = generate_midi_base64(pianoroll)

    return HarmonizeResponse(
        soprano=voices[0],
        alto=voices[1],
        tenor=voices[2],
        bass=voices[3],
        midi_base64=midi_base64,
    )


def extract_pitches_from_pianoroll(pianoroll: np.ndarray) -> list[list[int]]:
    """Extract MIDI pitch values from piano roll.

    Args:
        pianoroll: Shape (4, time_steps, num_pitches)

    Returns:
        List of 4 lists of MIDI pitches (one per voice)
    """
    voices = []
    for voice_idx in range(4):
        pitches = []
        for t in range(pianoroll.shape[1]):
            if pianoroll[voice_idx, t].max() > 0.5:
                pitch_idx = pianoroll[voice_idx, t].argmax()
                pitch = int(pitch_idx + MIN_PITCH)
                pitches.append(pitch)
            else:
                pitches.append(0)
        voices.append(pitches)
    return voices


def generate_midi_base64(pianoroll: np.ndarray, step_duration: float = 0.125) -> str:
    """Generate MIDI file and return as base64 string."""
    import pretty_midi

    midi = pretty_midi.PrettyMIDI()
    voice_names = ["Soprano", "Alto", "Tenor", "Bass"]

    for voice_idx in range(4):
        instrument = pretty_midi.Instrument(
            program=52,  # Choir Aahs
            name=voice_names[voice_idx],
        )

        voice_roll = pianoroll[voice_idx]
        time_steps = voice_roll.shape[0]

        current_pitch = None
        note_start = None

        for t in range(time_steps + 1):
            if t < time_steps:
                active = voice_roll[t].argmax() if voice_roll[t].max() > 0.5 else None
            else:
                active = None

            if active != current_pitch:
                if current_pitch is not None and note_start is not None:
                    midi_pitch = int(current_pitch + MIN_PITCH)
                    note = pretty_midi.Note(
                        velocity=80,
                        pitch=midi_pitch,
                        start=note_start * step_duration,
                        end=t * step_duration,
                    )
                    instrument.notes.append(note)

                if active is not None:
                    note_start = t
                else:
                    note_start = None

                current_pitch = active

        midi.instruments.append(instrument)

    # Write to bytes buffer
    buffer = io.BytesIO()
    midi.write(buffer)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")
