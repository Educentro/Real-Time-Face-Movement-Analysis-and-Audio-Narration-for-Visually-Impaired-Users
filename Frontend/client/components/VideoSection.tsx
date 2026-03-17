export default function VideoSection() {
  return (
    <div className="bg-white rounded-2xl p-6 shadow-xl">
      <div className="relative w-full bg-black rounded-xl overflow-hidden">
        {/* Placeholder for video feed - in production would be actual video feed */}
        <div className="aspect-video flex items-center justify-center text-white">
          <div className="text-center">
            <div className="text-6xl mb-4">📹</div>
            <p className="text-lg">Video Feed</p>
            <p className="text-sm text-gray-400 mt-2">Camera stream will appear here</p>
          </div>
        </div>
      </div>
    </div>
  );
}
