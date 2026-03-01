/**
 * Runtime configuration for the application
 * Uses Vite environment variables (prefixed with VITE_)
 */

export const config = {
  flaskApiUrl: import.meta.env.VITE_FLASK_API_URL || "http://localhost:5000",
} as const;

export const API_ENDPOINTS = {
  VIDEO_STREAM: `${config.flaskApiUrl}/video`,
  STATUS: `${config.flaskApiUrl}/status`,
  SET_MODE: (mode: string) => `${config.flaskApiUrl}/set_mode/${mode}`,
  HEALTH: `${config.flaskApiUrl}/api/health`,
} as const;
