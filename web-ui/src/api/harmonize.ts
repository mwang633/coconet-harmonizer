const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface HarmonizeRequest {
  melody: number[];
  temperature: number;
  num_iterations: number;
}

export interface HarmonizeResponse {
  soprano: number[];
  alto: number[];
  tenor: number[];
  bass: number[];
  midi_base64: string | null;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
}

export async function harmonize(request: HarmonizeRequest): Promise<HarmonizeResponse> {
  const response = await fetch(`${API_BASE}/api/harmonize`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/api/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }
  return response.json();
}

export function downloadMidi(base64: string, filename: string = 'harmony.mid') {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: 'audio/midi' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
