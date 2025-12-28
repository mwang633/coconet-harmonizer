import { Piano, KeyboardShortcuts, MidiNumbers } from 'react-piano';
import 'react-piano/dist/styles.css';

interface PianoInputProps {
  onNoteClick: (midiNumber: number) => void;
  playNote: (midiNumber: number) => void;
  melody: number[];
}

export function PianoInput({ onNoteClick, playNote, melody }: PianoInputProps) {
  const firstNote = MidiNumbers.fromNote('c3'); // 48
  const lastNote = MidiNumbers.fromNote('c6');  // 84

  const keyboardShortcuts = KeyboardShortcuts.create({
    firstNote: firstNote,
    lastNote: lastNote,
    keyboardConfig: KeyboardShortcuts.HOME_ROW,
  });

  // Highlight notes that are in the melody
  const activeNotes = melody.filter(n => n > 0);

  return (
    <div className="piano-container">
      <Piano
        noteRange={{ first: firstNote, last: lastNote }}
        playNote={(midiNumber: number) => {
          playNote(midiNumber);
          onNoteClick(midiNumber);
        }}
        stopNote={() => {
          // Notes are added on press, no action needed on release
        }}
        width={800}
        keyboardShortcuts={keyboardShortcuts}
        activeNotes={activeNotes}
      />
      <p className="piano-hint">
        Click keys or use keyboard (A-L, W-P) to add notes
      </p>
    </div>
  );
}
