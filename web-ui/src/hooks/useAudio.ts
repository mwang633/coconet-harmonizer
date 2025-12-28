import { useRef, useState, useCallback, useEffect } from 'react';
import Soundfont, { type Player } from 'soundfont-player';

interface VoiceNotes {
  soprano: number[];
  alto: number[];
  tenor: number[];
  bass: number[];
}

export function useAudio() {
  const audioContextRef = useRef<AudioContext | null>(null);
  const instrumentRef = useRef<Player | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const scheduledNotesRef = useRef<Player[]>([]);

  // Initialize audio context and load instrument
  useEffect(() => {
    const init = async () => {
      try {
        const ac = new AudioContext();
        audioContextRef.current = ac;

        const instrument = await Soundfont.instrument(ac, 'acoustic_grand_piano');
        instrumentRef.current = instrument;
        setIsLoading(false);
      } catch (error) {
        console.error('Failed to initialize audio:', error);
        setIsLoading(false);
      }
    };

    init();

    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  // Play a single note (for piano input feedback)
  const playNote = useCallback((midiNote: number, duration: number = 0.5) => {
    if (!instrumentRef.current || !audioContextRef.current) return;

    // Resume audio context if suspended (browser autoplay policy)
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }

    instrumentRef.current.play(midiNote.toString(), audioContextRef.current.currentTime, {
      duration,
      gain: 0.8,
    });
  }, []);

  // Stop all currently playing notes
  const stopAll = useCallback(() => {
    scheduledNotesRef.current.forEach(player => {
      try {
        player.stop();
      } catch {
        // Ignore errors from already stopped notes
      }
    });
    scheduledNotesRef.current = [];
    setIsPlaying(false);
  }, []);

  // Play all 4 voices simultaneously
  const playVoices = useCallback((voices: VoiceNotes, tempo: number = 120) => {
    if (!instrumentRef.current || !audioContextRef.current) return;

    // Resume audio context if suspended
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }

    stopAll();
    setIsPlaying(true);

    const stepDuration = 60 / tempo / 4; // 16th note duration
    const ac = audioContextRef.current;
    const instrument = instrumentRef.current;
    const startTime = ac.currentTime + 0.1;

    const voiceNames = ['soprano', 'alto', 'tenor', 'bass'] as const;
    const gains = [0.9, 0.7, 0.7, 0.8]; // Slightly louder soprano/bass

    voiceNames.forEach((voice, voiceIdx) => {
      const notes = voices[voice];

      // Find note boundaries and play each note
      let i = 0;
      while (i < notes.length) {
        const pitch = notes[i];
        if (pitch > 0) {
          // Find end of this note
          let end = i + 1;
          while (end < notes.length && notes[end] === pitch) {
            end++;
          }

          const duration = (end - i) * stepDuration;
          const player = instrument.play(
            String(pitch),
            startTime + i * stepDuration,
            { duration, gain: gains[voiceIdx] }
          );
          if (player) scheduledNotesRef.current.push(player);

          i = end;
        } else {
          i++;
        }
      }
    });

    // Set timeout to reset playing state when done
    const totalDuration = Math.max(
      voices.soprano.length,
      voices.alto.length,
      voices.tenor.length,
      voices.bass.length
    ) * stepDuration;

    setTimeout(() => {
      setIsPlaying(false);
    }, (totalDuration + 0.5) * 1000);
  }, [stopAll]);

  return {
    isLoading,
    isPlaying,
    playNote,
    playVoices,
    stopAll,
  };
}
