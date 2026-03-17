interface Instruction {
  step: number;
  text: string;
}

const INSTRUCTIONS: Instruction[] = [
  {
    step: 1,
    text: "Position your hand clearly in front of the camera",
  },
  {
    step: 2,
    text: "Show an alphabet letter (A-Z) or word sign",
  },
  {
    step: 3,
    text: "Hold steady for 0.5-1 second",
  },
  {
    step: 4,
    text: "System detects, displays, and speaks the result",
  },
];

export default function InstructionsSection() {
  return (
    <div className="bg-white rounded-2xl p-8 shadow-xl">
      <h2 className="text-2xl font-bold text-primary mb-8">📖 How to Use</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {INSTRUCTIONS.map((instruction) => (
          <div
            key={instruction.step}
            className="gradient-purple text-white rounded-xl p-6 text-center"
          >
            <div className="text-4xl font-bold mb-3 opacity-90">
              {instruction.step}
            </div>
            <p className="text-base leading-relaxed">{instruction.text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
