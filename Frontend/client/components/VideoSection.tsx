import { useState, useEffect, useCallback } from "react";
import { API_ENDPOINTS } from "@/lib/config";
import { useHealthCheck, useSetMode, useSystemStatus } from "@/hooks/use-flask-api";
import { Button } from "@/components/ui/button";
import { Loader2, Video, VideoOff, RefreshCw } from "lucide-react";

export default function VideoSection() {
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const { data: health, isError: healthError } = useHealthCheck();
  const { data: status } = useSystemStatus(isConnected);
  const setModeMutation = useSetMode();

  const handleImageLoad = useCallback(() => {
    setIsConnected(true);
    setIsLoading(false);
    setError(null);
  }, []);

  const handleImageError = useCallback(() => {
    setIsConnected(false);
    setIsLoading(false);
    setError("Cannot connect to video stream. Make sure the Flask backend is running.");
  }, []);

  const handleRetry = useCallback(() => {
    setIsLoading(true);
    setError(null);
    const img = document.getElementById("video-stream") as HTMLImageElement;
    if (img) {
      img.src = `${API_ENDPOINTS.VIDEO_STREAM}?t=${Date.now()}`;
    }
  }, []);

  const handleModeChange = (mode: "WORD" | "ALPHABET") => {
    setModeMutation.mutate(mode);
  };

  useEffect(() => {
    const timeout = setTimeout(() => {
      if (isLoading) {
        setIsLoading(false);
        setError("Connection timeout. Please check if the Flask backend is running on port 5000.");
      }
    }, 10000);

    return () => clearTimeout(timeout);
  }, [isLoading]);

  const backendOnline = health?.status === "healthy";

  return (
    <div className="bg-white rounded-2xl p-6 shadow-xl">
      {/* Mode Controls */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          {isConnected ? (
            <Video className="h-5 w-5 text-green-500" />
          ) : (
            <VideoOff className="h-5 w-5 text-red-500" />
          )}
          <span className="font-semibold text-gray-700">
            {isConnected ? "Live Feed" : "Disconnected"}
          </span>
        </div>

        <div className="flex gap-2">
          <Button
            variant={status?.current_mode === "ALPHABET" ? "default" : "outline"}
            size="sm"
            onClick={() => handleModeChange("ALPHABET")}
            disabled={setModeMutation.isPending || !isConnected}
            className={status?.current_mode === "ALPHABET" ? "bg-purple-600 hover:bg-purple-700" : ""}
          >
            🔤 Alphabet
          </Button>
          <Button
            variant={status?.current_mode === "WORD" ? "default" : "outline"}
            size="sm"
            onClick={() => handleModeChange("WORD")}
            disabled={setModeMutation.isPending || !isConnected}
            className={status?.current_mode === "WORD" ? "bg-green-600 hover:bg-green-700" : ""}
          >
            📝 Word
          </Button>
        </div>
      </div>

      {/* Video Container */}
      <div className="relative w-full bg-black rounded-xl overflow-hidden">
        <div className="aspect-video relative">
          {isLoading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-white z-10 bg-black/80">
              <Loader2 className="h-12 w-12 animate-spin mb-4" />
              <p className="text-lg">Connecting to video stream...</p>
              <p className="text-sm text-gray-400 mt-2">
                Make sure Flask backend is running on port 5000
              </p>
            </div>
          )}

          {error && !isLoading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-white z-10 bg-black">
              <VideoOff className="h-16 w-16 text-red-500 mb-4" />
              <p className="text-lg text-red-400 mb-2">Connection Failed</p>
              <p className="text-sm text-gray-400 text-center max-w-md mb-6 px-4">
                {error}
              </p>
              <Button onClick={handleRetry} variant="outline" className="gap-2">
                <RefreshCw className="h-4 w-4" />
                Retry Connection
              </Button>
            </div>
          )}

          <img
            id="video-stream"
            src={API_ENDPOINTS.VIDEO_STREAM}
            alt="Sign Language Recognition Video Feed"
            className={`w-full h-full object-contain ${isConnected ? "opacity-100" : "opacity-0"}`}
            onLoad={handleImageLoad}
            onError={handleImageError}
          />
        </div>
      </div>

      {/* Status Bar */}
      <div className="mt-4 flex items-center justify-between text-sm">
        <div className="flex items-center gap-4">
          <span className={`flex items-center gap-1 ${backendOnline ? "text-green-600" : "text-red-500"}`}>
            <span className={`w-2 h-2 rounded-full ${backendOnline ? "bg-green-500 animate-pulse" : "bg-red-500"}`} />
            Backend: {backendOnline ? "Online" : "Offline"}
          </span>
          {status?.hand_present && (
            <span className="text-blue-600 flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
              Hand Detected
            </span>
          )}
        </div>
        {status?.last_detection && (
          <span className="font-semibold text-purple-600">
            Last: {status.last_detection}
          </span>
        )}
      </div>
    </div>
  );
}
