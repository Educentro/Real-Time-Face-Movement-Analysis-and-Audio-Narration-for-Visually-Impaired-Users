# Real-Time Face Movement Analysis Backend

This backend system detects face and hand movements in real time and generates audio narration to assist visually impaired users. The focus is on reliability, low latency, and accessibility during live usage.

## Tech Stack
- Python
- Flask
- OpenCV
- MediaPipe

## Optional LLM Enhancement

The system includes an optional Large Language Model (LLM) module to enhance narration quality.
When enabled, detected gesture meanings are refined into more natural sentences.
For real-time reliability and accessibility, the system falls back to predefined static narration if the LLM is unavailable or disabled.
This ensures stable performance during live usage and prevents latency or dependency issues.
