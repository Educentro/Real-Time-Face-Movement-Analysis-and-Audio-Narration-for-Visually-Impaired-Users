import { useState, useEffect } from "react";

interface StatusValue {
  label: string;
  value: string;
  status: "active" | "inactive" | "neutral";
}

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
  const [statuses, setStatuses] = useState<StatusValue[]>([
    { label: "Alphabet Model", value: "Active", status: "active" },
    { label: "Word Model", value: "Active", status: "active" },
    { label: "Hand Detection", value: "Waiting...", status: "neutral" },
    { label: "LLM Fallback", value: "Enabled", status: "active" },
  ]);

  useEffect(() => {
    const timer = setInterval(() => {
      setStatuses((prev) =>
        prev.map((status) => {
          if (status.label === "Hand Detection") {
            return {
              ...status,
              value: status.value === "Waiting..." ? "Detected ✓" : "Waiting...",
              status: status.value === "Waiting..." ? "active" : "neutral",
            };
          }
          return status;
        }),
      );
    }, 3000);

    return () => clearInterval(timer);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active":
        return "bg-green-500 text-white";
      case "inactive":
        return "bg-red-500 text-white";
      case "neutral":
        return "bg-gray-500 text-white";
      default:
        return "bg-gray-500 text-white";
    }
  };

  return (
    <div className="bg-white rounded-2xl p-8 shadow-xl h-fit">
      <h2 className="text-2xl font-bold text-primary mb-6">📊 System Status</h2>

      {/* Status Items */}
      <div className="space-y-3 mb-8">
        {statuses.map((status) => (
          <div
            key={status.label}
            className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border-l-4 border-primary"
          >
            <span className="font-semibold text-gray-800">{status.label}</span>
            <span
              className={`px-3 py-1 rounded-full text-xs font-semibold ${getStatusColor(status.status)} ${
                status.status === "neutral" ? "pulse-animation" : ""
              }`}
            >
              {status.value}
            </span>
          </div>
        ))}
      </div>

      {/* Features Section */}
      <div className="pt-8 border-t border-gray-200">
        <h3 className="text-xl font-bold text-gray-900 mb-4">✨ Features</h3>
        <div className="space-y-3">
          {FEATURES.map((feature) => (
            <div
              key={feature.title}
              className="flex gap-3 p-3 bg-blue-50 rounded-lg"
            >
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
