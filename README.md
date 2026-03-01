# Real-Time Sign Language Recognition System

A dual-model sign language recognition system with real-time video streaming, alphabet/word detection, and audio narration for visually impaired users.

## Tech Stack

**Backend (Python/Flask)**
- Flask + Flask-CORS
- OpenCV + MediaPipe
- TensorFlow/Keras
- Real-time MJPEG streaming

**Frontend (React/TypeScript)**
- Vite + React 18
- TanStack Query
- Tailwind CSS + shadcn/ui
- Real-time status polling

## Local Development

### Prerequisites
- Python 3.9+
- Node.js 18+
- Webcam/Camera

### 1. Start Flask Backend

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run Flask server (port 5000)
python app.py
```

### 2. Start React Frontend

```bash
cd Frontend

# Install dependencies
npm install

# Run dev server (port 8080)
npm run dev
```

Open http://localhost:8080 in your browser.

## Deployment

### Frontend Deployment (Vercel)

The React frontend can be deployed to Vercel:

1. **Push to GitHub** - Push the `Frontend` folder to a GitHub repository

2. **Import to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Import your repository
   - Set root directory to `Frontend`

3. **Configure Environment Variable**
   ```
   VITE_FLASK_API_URL=https://your-backend-url.com
   ```

4. **Deploy** - Vercel will automatically build and deploy

### Backend Deployment

> **Important:** The Flask backend requires camera access for video streaming. For cloud deployment, you have two options:

#### Option A: Local Backend + Cloud Frontend (Recommended for Demo)

Run the Flask backend locally with camera access, and point the Vercel frontend to your local machine using a tunnel service like [ngrok](https://ngrok.com):

```bash
# Start Flask locally
python app.py

# In another terminal, expose with ngrok
ngrok http 5000
```

Then set `VITE_FLASK_API_URL` in Vercel to your ngrok URL.

#### Option B: Full Cloud Deployment (No Camera)

For a fully cloud-hosted solution, you would need to modify the architecture to:
- Use client-side MediaPipe (runs in browser)
- Send frames to backend for ML inference only
- This requires significant code changes

#### Option C: VPS with Camera/GPU

Deploy the backend to a VPS with camera hardware or GPU:
- Railway, Render (limited - no camera)
- AWS EC2, Google Cloud VM
- Self-hosted server with webcam

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/video` | GET | MJPEG video stream |
| `/status` | GET | System status JSON |
| `/set_mode/<mode>` | GET | Switch ALPHABET/WORD mode |
| `/api/health` | GET | Health check |

## Environment Variables

### Frontend (.env.local)
```
VITE_FLASK_API_URL=http://localhost:5000
```

### Backend
```
PORT=5000
HOST=0.0.0.0
FLASK_DEBUG=false
```

## Features

- **Dual Model Detection** - Alphabet (A-Z) and Word recognition
- **Real-time Video Streaming** - MJPEG stream with hand tracking overlay
- **Audio Narration** - Text-to-speech for detected signs
- **Mode Switching** - Toggle between alphabet and word detection
- **Gesture Locking** - Prevents duplicate detections
- **LLM Fallback** - Optional natural language enhancement

## License

MIT
