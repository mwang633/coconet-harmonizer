declare module 'soundfont-player' {
  export interface Player {
    play: (
      note: string | number,
      when?: number,
      options?: {
        duration?: number;
        gain?: number;
        attack?: number;
        decay?: number;
        sustain?: number;
        release?: number;
      }
    ) => Player;
    stop: (when?: number) => void;
    schedule: (
      when: number,
      events: Array<{
        time: number;
        note: string | number;
        duration?: number;
      }>
    ) => Player[];
  }

  export function instrument(
    audioContext: AudioContext,
    name: string,
    options?: {
      soundfont?: string;
      format?: 'mp3' | 'ogg';
      nameToUrl?: (name: string, soundfont: string, format: string) => string;
    }
  ): Promise<Player>;

  export default {
    instrument,
  };
}
