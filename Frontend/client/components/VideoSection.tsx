import { useState, useEffect, useCallback, useRef } from "react";
import { useHealthCheck, useInferFrame, useSetMode, useSystemStatus } from "@/hooks/use-flask-api";
import { Button } from "@/components/ui/button";
import { Loader2, Video, VideoOff, RefreshCw } from "lucide-react";

const INFERENCE_INTERVAL_MS = 220;
const FRAME_WIDTH = 512;
const FRAME_HEIGHT = 384;
const CAMERA_INIT_TIMEOUT_MS = 12000;

const CAMERA_PROFILES: MediaStreamConstraints[] = [
  {
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user",
    },
    audio: false,
  },
  {
    video: {
      width: { ideal: FRAME_WIDTH },
      height: { ideal: FRAME_HEIGHT },
      facingMode: "user",
    },
    audio: false,
  },
  { video: true, audio: false },
];

export default function VideoSection() {
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const inFlightRef = useRef(false);
  const inferenceTimerRef = useRef<number | null>(null);
  const cameraErrorRef = useRef<string | null>(null);
  const loopActiveRef = useRef(false);

  const { data: health } = useHealthCheck();
  const { data: status } = useSystemStatus(true);
  const setModeMutation = useSetMode();
  const inferFrameMutation = useInferFrame();

  const backendOnline = health?.status === "healthy";
  const alphabetAvailable = status?.alphabet_model ?? false;

  const stopInferenceLoop = useCallback(() => {
    loopActiveRef.current = false;
    if (inferenceTimerRef.current) {
      window.clearTimeout(inferenceTimerRef.current);
      inferenceTimerRef.current = null;
    }
    inFlightRef.current = false;
  }, []);

  const stopCamera = useCallback(() => {
    stopInferenceLoop();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    const video = videoRef.current;
    if (video) {
      video.srcObject = null;
    }
  }, [stopInferenceLoop]);

  const captureFrameAsDataUrl = useCallback(
    (canvas: HTMLCanvasElement, video: HTMLVideoElement): Promise<string> =>
      new Promise((resolve, reject) => {
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          reject(new Error("Canvas context unavailable"));
          return;
        }

        canvas.width = FRAME_WIDTH;
        canvas.height = FRAME_HEIGHT;
        ctx.save();
        ctx.translate(FRAME_WIDTH, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
        ctx.restore();

        canvas.toBlob(
          (blob) => {
            if (!blob) {
              reject(new Error("Frame encoding failed"));
              return;
            }
            const reader = new FileReader();
            reader.onloadend = () => resolve(String(reader.result || ""));
            reader.onerror = () => reject(new Error("Failed to read encoded frame"));
            reader.readAsDataURL(blob);
          },
          "image/jpeg",
          0.72,
        );
      }),
    [],
  );

  const captureAndInfer = useCallback(async () => {
    if (!backendOnline || inFlightRef.current) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) {
      return;
    }

    inFlightRef.current = true;
    try {
      const image = await captureFrameAsDataUrl(canvas, video);

      await inferFrameMutation.mutateAsync({ image });
      if (!cameraErrorRef.current) {
        setError(null);
      }
    } catch (e) {
      if (!cameraErrorRef.current) {
        setError("Inference is reconnecting. Please keep your hand in frame.");
      }
    } finally {
      inFlightRef.current = false;
    }
  }, [backendOnline, captureFrameAsDataUrl, inferFrameMutation]);

  const startInferenceLoop = useCallback(() => {
    stopInferenceLoop();
    loopActiveRef.current = true;
    const run = async () => {
      if (!loopActiveRef.current) {
        return;
      }
      await captureAndInfer();
      if (!loopActiveRef.current) {
        return;
      }
      inferenceTimerRef.current = window.setTimeout(run, INFERENCE_INTERVAL_MS);
    };
    inferenceTimerRef.current = window.setTimeout(run, INFERENCE_INTERVAL_MS);
  }, [captureAndInfer, stopInferenceLoop]);

  const startCamera = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    cameraErrorRef.current = null;

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Browser does not support camera access");
      }

      const getUserMediaWithTimeout = (constraints: MediaStreamConstraints) =>
        Promise.race([
          navigator.mediaDevices.getUserMedia(constraints),
          new Promise<never>((_, reject) =>
            window.setTimeout(() => reject(new Error("Camera request timed out")), CAMERA_INIT_TIMEOUT_MS),
          ),
        ]);

      let stream: MediaStream | null = null;
      let cameraErr: unknown = null;
      for (const profile of CAMERA_PROFILES) {
        try {
          stream = await getUserMediaWithTimeout(profile);
          break;
        } catch (e) {
          cameraErr = e;
        }
      }

      if (!stream) {
        throw cameraErr || new Error("No camera profile succeeded");
      }

      const video = videoRef.current;
      if (!video) {
        throw new Error("Video element not mounted");
      }

      streamRef.current = stream;
      video.srcObject = stream;
      try {
        await video.play();
      } catch {
        // Some browsers block autoplay even for muted streams. Keep stream active and continue.
      }

      setIsConnected(true);
      setIsLoading(false);
      startInferenceLoop();
    } catch (e) {
      setIsConnected(false);
      setIsLoading(false);
      const maybeErr = e as { name?: string; message?: string };
      if (maybeErr?.name === "NotAllowedError") {
        cameraErrorRef.current = "Camera blocked. Allow camera permission in browser site settings.";
      } else if (maybeErr?.name === "NotFoundError") {
        cameraErrorRef.current = "No camera device found. Connect a webcam and retry.";
      } else if (maybeErr?.name === "NotReadableError") {
        cameraErrorRef.current = "Camera is busy in another app/tab. Close other camera apps and retry.";
      } else if (maybeErr?.message?.includes("timed out")) {
        cameraErrorRef.current = "Camera request timed out. Close other camera apps and click Retry Camera.";
      } else if (maybeErr?.message?.includes("No camera device found")) {
        cameraErrorRef.current = "No camera device found. Connect a webcam and retry.";
      } else {
        cameraErrorRef.current = "Camera access failed. Allow camera permission and retry.";
      }
      setError(cameraErrorRef.current);
    }
  }, [startInferenceLoop]);

  const handleRetry = useCallback(() => {
    stopCamera();
    void startCamera();
  }, [startCamera, stopCamera]);

  const handleModeChange = (mode: "WORD" | "ALPHABET") => {
    setModeMutation.mutate(mode);
  };

  useEffect(() => {
    void startCamera();
    return () => {
      stopCamera();
    };
    // Keep camera startup stable; don't restart on every render.
  }, []);

  useEffect(() => {
    if (!backendOnline) {
      setError("Backend is offline. Check deployment and API URL.");
      return;
    }

    if (isConnected && !inferenceTimerRef.current) {
      startInferenceLoop();
    }
  }, [backendOnline, isConnected, startInferenceLoop]);

  return (
    <div className="bg-white rounded-2xl p-6 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          {isConnected ? (
            <Video className="h-5 w-5 text-green-500" />
          ) : (
            <VideoOff className="h-5 w-5 text-red-500" />
          )}
          <span className="font-semibold text-gray-700">{isConnected ? "Live Camera" : "Disconnected"}</span>
        </div>

        <div className="flex gap-2">
          <Button
            variant={status?.current_mode === "ALPHABET" ? "default" : "outline"}
            size="sm"
            onClick={() => handleModeChange("ALPHABET")}
            disabled={setModeMutation.isPending || !isConnected || !alphabetAvailable}
            className={status?.current_mode === "ALPHABET" ? "bg-purple-600 hover:bg-purple-700" : ""}
          >
            Alphabet
          </Button>
          <Button
            variant={status?.current_mode === "WORD" ? "default" : "outline"}
            size="sm"
            onClick={() => handleModeChange("WORD")}
            disabled={setModeMutation.isPending || !isConnected}
            className={status?.current_mode === "WORD" ? "bg-green-600 hover:bg-green-700" : ""}
          >
            Word
          </Button>
        </div>
      </div>

      <div className="relative w-full bg-black rounded-xl overflow-hidden">
        <div className="aspect-video relative">
          {isLoading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-white z-10 bg-black/80">
              <Loader2 className="h-12 w-12 animate-spin mb-4" />
              <p className="text-lg">Starting camera...</p>
              <p className="text-sm text-gray-400 mt-2">Allow browser camera access</p>
            </div>
          )}

          {error && !isLoading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-white z-10 bg-black/80 px-4">
              <VideoOff className="h-16 w-16 text-red-500 mb-4" />
              <p className="text-lg text-red-400 mb-2">Connection Failed</p>
              <p className="text-sm text-gray-300 text-center max-w-md mb-6">{error}</p>
              <Button onClick={handleRetry} variant="outline" className="gap-2">
                <RefreshCw className="h-4 w-4" />
                Retry Camera
              </Button>
            </div>
          )}

          <video
            ref={videoRef}
            className={`w-full h-full object-cover ${isConnected ? "opacity-100" : "opacity-0"} scale-x-[-1]`}
            autoPlay
            playsInline
            muted
          />
          <canvas ref={canvasRef} className="hidden" />
        </div>
      </div>

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
        {status?.last_detection && <span className="font-semibold text-purple-600">Last: {status.last_detection}</span>}
      </div>
    </div>
  );
}
