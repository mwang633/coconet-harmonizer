import { useState, useCallback } from 'react';
import { PianoInput } from './components/PianoInput';
import { HarmonyDisplay } from './components/HarmonyDisplay';
import { Controls } from './components/Controls';
import { useAudio } from './hooks/useAudio';
import { harmonize, downloadMidi, type HarmonizeResponse } from './api/harmonize';
import './App.css';

function App() {
  const [melody, setMelody] = useState<number[]>([]);
  const [harmony, setHarmony] = useState<HarmonizeResponse | null>(null);
  const [temperature, setTemperature] = useState(0.8);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { isLoading: audioLoading, isPlaying, playNote, playVoices, stopAll } = useAudio();

  const handleNoteClick = useCallback((midiNumber: number) => {
    setMelody((prev) => [...prev, midiNumber]);
    setHarmony(null); // Clear harmony when melody changes
    setError(null);
  }, []);

  const handleHarmonize = useCallback(async () => {
    if (melody.length === 0) return;

    setIsLoading(true);
    setError(null);

    try {
      const result = await harmonize({
        melody,
        temperature,
        num_iterations: 100,
      });
      setHarmony(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to harmonize');
    } finally {
      setIsLoading(false);
    }
  }, [melody, temperature]);

  const handlePlay = useCallback(() => {
    if (!harmony) return;
    playVoices({
      soprano: harmony.soprano,
      alto: harmony.alto,
      tenor: harmony.tenor,
      bass: harmony.bass,
    });
  }, [harmony, playVoices]);

  const handleDownload = useCallback(() => {
    if (harmony?.midi_base64) {
      downloadMidi(harmony.midi_base64, 'bach-harmony.mid');
    }
  }, [harmony]);

  const handleClear = useCallback(() => {
    setMelody([]);
    setHarmony(null);
    setError(null);
    stopAll();
  }, [stopAll]);

  return (
    <div className="app">
      <header>
        <h1>Coconet Harmonizer</h1>
        <p>Generate Bach-style 4-part harmonies from your melody</p>
      </header>

      <main>
        {audioLoading && (
          <div className="loading-audio">Loading audio samples...</div>
        )}

        <PianoInput
          onNoteClick={handleNoteClick}
          playNote={playNote}
          melody={melody}
        />

        <Controls
          temperature={temperature}
          onTemperatureChange={setTemperature}
          onHarmonize={handleHarmonize}
          onPlay={handlePlay}
          onStop={stopAll}
          onClear={handleClear}
          onDownload={handleDownload}
          isLoading={isLoading}
          isPlaying={isPlaying}
          hasHarmony={harmony !== null}
          hasMelody={melody.length > 0}
        />

        {error && <div className="error">{error}</div>}

        <HarmonyDisplay melody={melody} harmony={harmony} />
      </main>

      <footer>
        <p>
          Based on <a href="https://arxiv.org/abs/1903.07227">Coconet</a> -
          Counterpoint by Convolution
        </p>
      </footer>
    </div>
  );
}

export default App;
