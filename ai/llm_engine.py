from transformers import pipeline

class LLMNarrator:
    def __init__(self):
        self.generator = pipeline(
            task="text-generation",
            model="gpt2"    
        )

    def generate_sentence(self, gesture_sequence):
        """
        gesture_sequence: list like ["HELLO"] or ["HELLO", "HOW_ARE_YOU"]
        """

        prompt = f"""
You are an assistive narration system for visually impaired users.

Rules:
- Describe ONLY the gestures provided
- Do NOT assume emotions or intent
- Do NOT add extra actions
- Generate ONE short, neutral sentence

Detected gestures: {gesture_sequence}

Sentence:
"""
        output = self.generator(prompt, num_return_sequences=1)
        text = output[0]["generated_text"]

        if "Sentence:" in text:
            text = text.split("Sentence:")[-1]

        return text.strip()
