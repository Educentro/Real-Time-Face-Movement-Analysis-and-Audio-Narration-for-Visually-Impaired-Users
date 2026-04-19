import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { API_ENDPOINTS } from "@/lib/config";
import type {
  SystemStatus,
  ModeChangeResponse,
  HealthCheckResponse,
  InferFrameRequest,
  InferFrameResponse,
} from "@shared/api";

export function useSystemStatus(enabled = true) {
  return useQuery<SystemStatus>({
    queryKey: ["system-status"],
    queryFn: async () => {
      const response = await fetch(API_ENDPOINTS.STATUS);
      if (!response.ok) {
        throw new Error("Failed to fetch system status");
      }
      return response.json();
    },
    refetchInterval: 500,
    enabled,
    retry: 3,
    retryDelay: 1000,
  });
}

export function useHealthCheck() {
  return useQuery<HealthCheckResponse>({
    queryKey: ["health-check"],
    queryFn: async () => {
      const response = await fetch(API_ENDPOINTS.HEALTH);
      if (!response.ok) {
        throw new Error("Backend is not healthy");
      }
      return response.json();
    },
    refetchInterval: 5000,
    retry: 2,
  });
}

export function useSetMode() {
  const queryClient = useQueryClient();

  return useMutation<ModeChangeResponse, Error, "WORD" | "ALPHABET">({
    mutationFn: async (mode) => {
      const response = await fetch(API_ENDPOINTS.SET_MODE(mode));
      if (!response.ok) {
        throw new Error("Failed to change mode");
      }
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["system-status"] });
    },
  });
}

export function useInferFrame() {
  return useMutation<InferFrameResponse, Error, InferFrameRequest>({
    mutationFn: async (payload) => {
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 7000);
      let response: Response;
      try {
        response = await fetch(API_ENDPOINTS.INFER_FRAME, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
      } finally {
        window.clearTimeout(timeout);
      }

      if (!response.ok) {
        throw new Error("Failed to run frame inference");
      }

      return response.json();
    },
  });
}
