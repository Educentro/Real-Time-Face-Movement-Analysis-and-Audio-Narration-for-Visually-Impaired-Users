/**
 * Runtime configuration for the application
 * Uses Vite environment variables (prefixed with VITE_)
 */

const envFlaskApiUrl = import.meta.env.VITE_FLASK_API_URL?.trim();
const currentOrigin = typeof window !== "undefined" ? window.location.origin : "";
const defaultFlaskApiUrl = import.meta.env.DEV ? "http://localhost:5000" : currentOrigin;
const flaskApiUrl = (envFlaskApiUrl || defaultFlaskApiUrl).replace(/\/+$/, "");

export const config = {
  flaskApiUrl,
  usingEnvApiUrl: Boolean(envFlaskApiUrl),
} as const;

export const API_ENDPOINTS = {
  VIDEO_STREAM: `${config.flaskApiUrl}/video`,
  STATUS: `${config.flaskApiUrl}/status`,
  SET_MODE: (mode: string) => `${config.flaskApiUrl}/set_mode/${mode}`,
  HEALTH: `${config.flaskApiUrl}/api/health`,
  INFER_FRAME: `${config.flaskApiUrl}/api/infer_frame`,
} as const;
