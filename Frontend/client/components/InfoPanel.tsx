import { useSystemStatus, useHealthCheck } from "@/hooks/use-flask-api";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";

interface Feature {
  icon: string;
  title: string;
  description: string;
}

const FEATURES: Feature[] = [
  {
    icon: "🔤",
    title: "Alphabet Recognition",
    description: "Detects A-Z letters with high accuracy",
  },
  {
    icon: "📝",
    title: "Word Recognition",
    description: "Recognizes complete sign language words",
  },
  {
    icon: "🔊",
    title: "Audio Narration",
    description: "Speaks detected gestures aloud",
  },
  {
    icon: "🧠",
    title: "AI Fallback",
    description: "LLM generates context when needed",
  },
];

export default function InfoPanel() {
  const { data: status, isLoading, isError } = useSystemStatus();
  const { data: health } = useHealthCheck();

  const backendOnline = health?.status === "healthy";

  const getStatusColor = (active: boolean) => {
    return active ? "bg-green-500 text-white" : "bg-red-500 text-white";
  };

  const getNeutralStatusColor = (active: boolean) => {
    return active
      ? "bg-green-500 text-white"
      : "bg-gray-400 text-white";
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-2xl p-8 shadow-xl h-fit">
        <h2 className="text-2xl font-bold text-primary mb-6">📊 System Status</h2>
        <div className="space-y-3 mb-8">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-2xl p-8 shadow-xl h-fit">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-primary">📊 System Status</h2>
        <Badge variant={backendOnline ? "default" : "destructive"} className="text-xs">
          {backendOnline ? "Connected" : "Disconnected"}
        </Badge>
      </div>

      {/* Backend Connection Status */}
      {isError && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-600">
            Cannot connect to backend. Make sure Flask server is running on port 5000.
          </p>
        </div>
      )}

      {/* Status Items */}
      <div className="space-y-3 mb-8">
        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border-l-4 border-primary">
          <span className="font-semibold text-gray-800">Alphabet Model</span>
          <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getStatusColor(status?.alphabet_model ?? false)}`}>
            {status?.alphabet_model ? "Active" : "Inactive"}
          </span>
        </div>

        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border-l-4 border-primary">
          <span className="font-semibold text-gray-800">Word Model</span>
          <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getStatusColor(status?.word_model ?? false)}`}>
            {status?.word_model ? "Active" : "Inactive"}
          </span>
        </div>

        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border-l-4 border-primary">
          <span className="font-semibold text-gray-800">Hand Detection</span>
          <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getNeutralStatusColor(status?.hand_present ?? false)} ${status?.hand_present ? "" : "pulse-animation"}`}>
            {status?.hand_present ? "Detected ✓" : "Waiting..."}
          </span>
        </div>

        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border-l-4 border-primary">
          <span className="font-semibold text-gray-800">Detection Mode</span>
          <span className={`px-3 py-1 rounded-full text-xs font-semibold ${status?.current_mode === "WORD" ? "bg-green-500" : "bg-purple-500"} text-white`}>
            {status?.current_mode || "WORD"}
          </span>
        </div>

        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border-l-4 border-primary">
          <span className="font-semibold text-gray-800">LLM Fallback</span>
          <span className={`px-3 py-1 rounded-full text-xs font-semibold ${status?.llm_enabled ? "bg-green-500 text-white" : "bg-gray-400 text-white"}`}>
            {status?.llm_enabled ? "Enabled" : "Disabled"}
          </span>
        </div>

        {status?.gesture_locked && (
          <div className="flex items-center justify-between p-3 bg-orange-50 rounded-lg border-l-4 border-orange-400">
            <span className="font-semibold text-gray-800">Gesture Status</span>
            <span className="px-3 py-1 rounded-full text-xs font-semibold bg-orange-500 text-white animate-pulse">
              LOCKED
            </span>
          </div>
        )}
      </div>

      {/* Last Detection */}
      {status?.last_detection && (
        <div className="mb-8 p-4 bg-gradient-to-r from-purple-100 to-blue-100 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">Last Detection:</p>
          <p className="text-xl font-bold text-purple-700">{status.last_detection}</p>
        </div>
      )}

      {/* Model Statistics */}
      {status && (
        <div className="mb-8 p-4 bg-blue-50 rounded-lg">
          <p className="text-sm font-semibold text-gray-700 mb-2">Available Labels</p>
          <div className="flex gap-4 text-sm">
            <span className="text-purple-600">
              🔤 Alphabet: {status.alphabet_labels?.length || 26}
            </span>
            <span className="text-green-600">
              📝 Words: {status.word_labels?.length || 0}
            </span>
          </div>
          {status.llm_cache_size > 0 && (
            <p className="text-xs text-gray-500 mt-2">
              LLM Cache: {status.llm_cache_size} responses
            </p>
          )}
        </div>
      )}

      {/* Features Section */}
      <div className="pt-8 border-t border-gray-200">
        <h3 className="text-xl font-bold text-gray-900 mb-4">✨ Features</h3>
        <div className="space-y-3">
          {FEATURES.map((feature) => (
            <div key={feature.title} className="flex gap-3 p-3 bg-blue-50 rounded-lg">
              <div className="text-2xl flex-shrink-0">{feature.icon}</div>
              <div className="flex-1">
                <div className="font-semibold text-primary">{feature.title}</div>
                <div className="text-sm text-gray-600">{feature.description}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
