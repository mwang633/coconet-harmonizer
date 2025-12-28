declare module 'react-piano' {
  import { Component } from 'react';

  export interface NoteRange {
    first: number;
    last: number;
  }

  export interface KeyboardShortcut {
    key: string;
    midiNumber: number;
  }

  export interface PianoProps {
    noteRange: NoteRange;
    playNote: (midiNumber: number) => void;
    stopNote: (midiNumber: number) => void;
    width?: number;
    keyboardShortcuts?: KeyboardShortcut[];
    activeNotes?: number[];
    disabled?: boolean;
    className?: string;
  }

  export class Piano extends Component<PianoProps> {}

  export const MidiNumbers: {
    fromNote: (note: string) => number;
    getAttributes: (midiNumber: number) => {
      note: string;
      pitchName: string;
      octave: number;
      isAccidental: boolean;
    };
    NATURAL_MIDI_NUMBERS: number[];
    VALID_MIDI_NUMBERS: number[];
    MIN_MIDI_NUMBER: number;
    MAX_MIDI_NUMBER: number;
  };

  export const KeyboardShortcuts: {
    create: (config: {
      firstNote: number;
      lastNote: number;
      keyboardConfig: KeyboardShortcut[];
    }) => KeyboardShortcut[];
    HOME_ROW: KeyboardShortcut[];
    BOTTOM_ROW: KeyboardShortcut[];
    QWERTY_ROW: KeyboardShortcut[];
  };
}
