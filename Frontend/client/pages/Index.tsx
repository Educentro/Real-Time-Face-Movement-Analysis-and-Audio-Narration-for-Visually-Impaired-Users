import VideoSection from "@/components/VideoSection";
import InfoPanel from "@/components/InfoPanel";
import InstructionsSection from "@/components/InstructionsSection";

export default function Index() {
  return (
    <div className="min-h-screen gradient-purple flex flex-col">
      {/* Header */}
      <header className="text-center text-white py-12 px-4 md:py-16">
        <h1 className="text-4xl md:text-5xl font-bold mb-3 drop-shadow-lg">
          🤖 Dual-Model Sign Language Recognition
        </h1>
        <p className="text-lg md:text-xl opacity-90">
          Real-time Alphabet & Word Detection with AI Narration
        </p>
      </header>

      {/* Main Content */}
      <div className="flex-1 px-4 md:px-8 pb-12 md:pb-16">
        <div className="max-w-7xl mx-auto">
          {/* Video and Info Panel */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
            <div className="lg:col-span-2">
              <VideoSection />
            </div>
            <div className="lg:col-span-1">
              <InfoPanel />
            </div>
          </div>

          {/* Instructions Section */}
          <InstructionsSection />
        </div>
      </div>

      {/* Footer */}
      <footer className="text-center text-white py-8 opacity-80">
        <p className="text-base">Powered by MediaPipe • TensorFlow • Flask</p>
        <p className="text-sm mt-2">Company-Grade Live Demo System</p>
      </footer>
    </div>
  );
}
