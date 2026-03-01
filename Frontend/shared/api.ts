/**
 * Shared types for Flask backend API integration
 */

export interface SystemStatus {
  alphabet_model: boolean;
  word_model: boolean;
  hand_present: boolean;
  gesture_locked: boolean;
  last_detection: string | null;
  llm_enabled: boolean;
  llm_cache_size: number;
  current_mode: "WORD" | "ALPHABET";
  word_labels: string[];
  alphabet_labels: string[];
}

export interface ModeChangeResponse {
  status: "ok" | "error";
  mode?: "WORD" | "ALPHABET";
  message?: string;
}

export interface HealthCheckResponse {
  status: "healthy" | "unhealthy";
  models: {
    alphabet: boolean;
    word: boolean;
  };
}
