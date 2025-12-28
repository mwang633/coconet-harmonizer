interface HarmonyDisplayProps {
  melody: number[];
  harmony: {
    soprano: number[];
    alto: number[];
    tenor: number[];
    bass: number[];
  } | null;
}

const VOICE_COLORS = {
  soprano: '#e74c3c',  // Red
  alto: '#3498db',     // Blue
  tenor: '#2ecc71',    // Green
  bass: '#9b59b6',     // Purple
};

function midiToNoteName(midi: number): string {
  if (midi === 0) return '-';
  const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const octave = Math.floor(midi / 12) - 1;
  const note = noteNames[midi % 12];
  return `${note}${octave}`;
}

export function HarmonyDisplay({ melody, harmony }: HarmonyDisplayProps) {
  if (melody.length === 0) {
    return (
      <div className="harmony-display empty">
        <p>Add notes using the piano above</p>
      </div>
    );
  }

  const voices = harmony
    ? [
        { name: 'Soprano', notes: harmony.soprano, color: VOICE_COLORS.soprano },
        { name: 'Alto', notes: harmony.alto, color: VOICE_COLORS.alto },
        { name: 'Tenor', notes: harmony.tenor, color: VOICE_COLORS.tenor },
        { name: 'Bass', notes: harmony.bass, color: VOICE_COLORS.bass },
      ]
    : [
        { name: 'Melody', notes: melody, color: VOICE_COLORS.soprano },
      ];

  return (
    <div className="harmony-display">
      <div className="voice-grid">
        {voices.map(({ name, notes, color }) => (
          <div key={name} className="voice-row">
            <span className="voice-name" style={{ color }}>
              {name}
            </span>
            <div className="notes">
              {notes.map((note, i) => (
                <span
                  key={i}
                  className={`note ${note === 0 ? 'rest' : ''}`}
                  style={{ backgroundColor: note > 0 ? color : 'transparent' }}
                >
                  {midiToNoteName(note)}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
