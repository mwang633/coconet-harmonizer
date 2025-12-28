# Coconet: 4-Part Harmony Generator

A PyTorch implementation of Coconet for generating Bach-style SATB (Soprano, Alto, Tenor, Bass) harmonizations from melodies.

## Setup

```bash
# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

## Web UI

A React-based web interface for interactive harmonization.

### Running the Web UI

**Terminal 1 - Start the backend:**
```bash
.venv/bin/uvicorn coconet.web.app:app --reload --port 8000
```

**Terminal 2 - Start the frontend:**
```bash
cd web-ui
npm install  # first time only
npm run dev
```

Open http://localhost:5173 in your browser.

### Features
- Interactive piano keyboard to input melodies
- Real-time audio playback with soundfont samples
- Temperature control for creativity
- Download harmonized MIDI files
- Visual display of all 4 voices

## Python API

### Quick Start - Harmonize a Melody

```python
from coconet import Harmonizer

# Load the trained model
harmonizer = Harmonizer(checkpoint_path="checkpoints/best_model.pt")

# Define a melody as MIDI pitches (60 = Middle C)
melody = [60, 62, 64, 65, 67, 65, 64, 62, 60]

# Generate 4-part harmony
result = harmonizer.harmonize_melody(melody, temperature=0.8)
```

### Harmonize a MIDI File

```python
from coconet import Harmonizer

harmonizer = Harmonizer(checkpoint_path="checkpoints/best_model.pt")

# Input: single melody MIDI file
# Output: 4-part SATB harmonization
harmonizer.harmonize_midi_file(
    input_path="my_melody.mid",
    output_path="harmonized.mid",
    temperature=0.8
)
```

### Using music21

```python
from music21 import converter
from coconet import Harmonizer

harmonizer = Harmonizer(checkpoint_path="checkpoints/best_model.pt")

# Load a melody
melody = converter.parse("my_melody.xml")

# Generate harmonization
score = harmonizer.harmonize_from_music21(melody, temperature=0.8)

# Show or save
score.show()  # Opens in MuseScore/notation software
score.write("musicxml", "harmonized.xml")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.8 | Sampling temperature (lower = more conservative, higher = more creative) |
| `num_iterations` | 100 | Gibbs sampling iterations (more = better quality, slower) |
| `melody_voice` | 0 | Which voice has the melody (0=Soprano, 1=Alto, 2=Tenor, 3=Bass) |

## REST API

The FastAPI backend provides these endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check and model status |
| POST | `/api/harmonize` | Harmonize a melody |

### POST /api/harmonize

```bash
curl -X POST http://localhost:8000/api/harmonize \
  -H "Content-Type: application/json" \
  -d '{"melody": [60, 62, 64, 65, 67], "temperature": 0.8}'
```

Response:
```json
{
  "soprano": [60, 62, 64, 65, 67],
  "alto": [55, 57, 60, 60, 62],
  "tenor": [48, 50, 52, 53, 55],
  "bass": [36, 38, 40, 41, 43],
  "midi_base64": "..."
}
```

## Training

To train your own model:

```bash
# Basic training
.venv/bin/python -m coconet.train

# With custom options
.venv/bin/python -m coconet.train \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --layers 32 \
    --filters 128
```

Training uses the Bach chorales corpus from music21 (~323 chorales).

## Project Structure

```
coconet/
├── __init__.py      # Package exports
├── model.py         # Coconet neural network architecture
├── data.py          # Data loading and preprocessing
├── train.py         # Training script
├── harmonize.py     # Inference and harmonization utilities
└── web/             # FastAPI backend
    ├── app.py       # Application setup
    └── routes.py    # API endpoints

web-ui/              # React frontend
├── src/
│   ├── components/  # Piano, controls, display
│   ├── hooks/       # Audio playback
│   └── api/         # Backend client
└── package.json
```

## How It Works

Coconet uses a convolutional neural network trained on Bach chorales. It learns to fill in missing voices by:

1. Taking a partial score (e.g., just the melody)
2. Using Gibbs sampling to iteratively generate the missing voices
3. Producing musically coherent 4-part harmony in Bach's style

The model uses dilated convolutions to capture both local (chord) and global (phrase) musical structure.

## References

- [Counterpoint by Convolution](https://arxiv.org/abs/1903.07227) - Original Coconet paper
- [Bach Doodle](https://magenta.tensorflow.org/coconet) - Google's implementation
