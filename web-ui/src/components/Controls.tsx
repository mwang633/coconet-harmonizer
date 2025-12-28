interface ControlsProps {
  temperature: number;
  onTemperatureChange: (value: number) => void;
  onHarmonize: () => void;
  onPlay: () => void;
  onStop: () => void;
  onClear: () => void;
  onDownload: () => void;
  isLoading: boolean;
  isPlaying: boolean;
  hasHarmony: boolean;
  hasMelody: boolean;
}

export function Controls({
  temperature,
  onTemperatureChange,
  onHarmonize,
  onPlay,
  onStop,
  onClear,
  onDownload,
  isLoading,
  isPlaying,
  hasHarmony,
  hasMelody,
}: ControlsProps) {
  return (
    <div className="controls">
      <div className="control-group">
        <label htmlFor="temperature">
          Temperature: {temperature.toFixed(1)}
        </label>
        <input
          id="temperature"
          type="range"
          min="0.1"
          max="2.0"
          step="0.1"
          value={temperature}
          onChange={(e) => onTemperatureChange(parseFloat(e.target.value))}
        />
        <span className="temperature-hint">
          Lower = conservative, Higher = creative
        </span>
      </div>

      <div className="button-group">
        <button
          onClick={onHarmonize}
          disabled={!hasMelody || isLoading}
          className="primary"
        >
          {isLoading ? 'Harmonizing...' : 'Harmonize'}
        </button>

        {isPlaying ? (
          <button onClick={onStop}>Stop</button>
        ) : (
          <button onClick={onPlay} disabled={!hasHarmony}>
            Play
          </button>
        )}

        <button onClick={onDownload} disabled={!hasHarmony}>
          Download MIDI
        </button>

        <button onClick={onClear} className="secondary">
          Clear
        </button>
      </div>
    </div>
  );
}
