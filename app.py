"""
PRODUCTION-GRADE DUAL-MODEL SIGN LANGUAGE RECOGNITION SYSTEM
Company-grade live demo system with:
- Intelligent alphabet/word model switching
- Temporal smoothing and stability
- LLM fallback for unknown gestures
- Gesture locking and cooldown
- Flask video streaming
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
from tensorflow.keras.models import load_model
import os
import time
import threading
import subprocess
import requests
from queue import Queue
from dataclasses import dataclass
from typing import Optional, Tuple
import base64

DETECTION_MODE = "WORD"

# CONFIGURATION
@dataclass
class Config:
    # Model confidence thresholds
    ALPHABET_CONFIDENCE_THRESHOLD = 0.55  # Lowered for cloud frame-based inference
    WORD_CONFIDENCE_THRESHOLD = 0.35      # Lowered for cloud frame-based inference
    
    # Stability requirements
    ALPHABET_STABLE_FRAMES = 3   # Reduced for lower-latency API inference
    WORD_STABLE_FRAMES = 4       # Reduced for lower-latency API inference
    BUFFER_SIZE = 15
    
    # Gesture locking
    COOLDOWN_TIME = 1.5          
    GESTURE_LOCK_TIME = 2.0      
    ALPHABET_COOLDOWN = 1.0
    WORD_COOLDOWN = 1.5

    # Model selection strategy
    CONFIDENCE_DELTA_THRESHOLD = 0.15  
    
    # LLM settings
    USE_LLM = False
    LLM_TIMEOUT = 1.5
    LLM_MODEL = "mistral"
    LLM_URL = "http://localhost:11434/api/generate"
    
    # Display
    DISPLAY_HOLD_TIME = 1.5
     
config = Config()


# FLASK APP
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
print("Loading models...")

try:
    word_model = load_model("frame_model.keras")
    print("[OK] Word model loaded")
except Exception as e:
    print(f"[ERROR] Word model failed: {e}")
    word_model = None

try:
    alphabet_model = load_model("alphabet_model.keras")
    print("[OK] Alphabet model loaded")
except Exception as e:
    print(f"[WARN] Alphabet model (alphabet_model.keras) failed: {e}")
    try:
        alphabet_model = load_model("model/asl_mlp.h5")
        print("[OK] Alphabet fallback model loaded from model/asl_mlp.h5")
    except Exception as fallback_error:
        print(f"[ERROR] Alphabet fallback model failed: {fallback_error}")
        alphabet_model = None

# Load labels
try:
    if os.path.exists("dataset"):
        WORD_LABELS = sorted([
            d for d in os.listdir("dataset")
            if os.path.isdir(os.path.join("dataset", d))
        ])
    else:
        WORD_LABELS = [
            "HELLO", "YOU", "NAME", "THANK_YOU", "WELCOME", "GOODBYE",
            "MORNING", "AFTERNOON", "NIGHT", "PLEASE", "SORRY", "YES",
            "NO", "STOP", "WAIT", "HELP", "I'M_HUNGRY", "I'M_THIRSTY",
            "I'M_TIRED", "I'M_BUSY", "I'M_BORED", "EAT", "WHAT", "WHEN",
            "FATHER", "MOTHER", "FRIEND", "MY"
        ]
        print("[INFO] Using default word labels (dataset folder not found)")
except Exception as e:
    WORD_LABELS = ["HELLO", "THANK_YOU", "YES", "NO", "PLEASE", "SORRY"]
    print(f"[WARN] Error loading labels: {e}")

ALPHABET_LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
if alphabet_model is not None and os.path.exists("model/labels.npy"):
    try:
        loaded_alphabet_labels = np.load("model/labels.npy", allow_pickle=True).tolist()
        if loaded_alphabet_labels and isinstance(loaded_alphabet_labels, list):
            ALPHABET_LABELS = [str(label) for label in loaded_alphabet_labels]
            print(f"[INFO] Using alphabet labels from model/labels.npy ({len(ALPHABET_LABELS)})")
    except Exception as e:
        print(f"[WARN] Could not load model/labels.npy: {e}")

print(f"[INFO] Loaded {len(WORD_LABELS)} word labels, {len(ALPHABET_LABELS)} alphabet labels")


# MEDIAPIPE SETUP (SHARED)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# STATE MANAGEMENT

class SystemState:
    def __init__(self):
        self.alphabet_buffer = deque(maxlen=config.BUFFER_SIZE)
        self.word_buffer = deque(maxlen=config.BUFFER_SIZE)

        self.last_detection = None
        self.last_detection_time = 0

        self.gesture_locked = False
        self.gesture_lock_time = 0

        self.hand_present = False
        self.lock = threading.Lock()

    def reset_buffers(self):
        with self.lock:
            self.alphabet_buffer.clear()
            self.word_buffer.clear()

    def clear_other_buffer(self, model_type: str):
        if model_type == "alphabet":
            self.word_buffer.clear()
        else:
            self.alphabet_buffer.clear()

    def lock_gesture(self):
        self.gesture_locked = True
        self.gesture_lock_time = time.time()

    def update_lock(self):
        if self.gesture_locked and (time.time() - self.gesture_lock_time) > config.GESTURE_LOCK_TIME:
            self.gesture_locked = False

    def is_locked(self) -> bool:
        return self.gesture_locked

    def can_detect_new(self, model_type: str) -> bool:
        cooldown = (
            config.ALPHABET_COOLDOWN
            if model_type == "alphabet"
            else config.WORD_COOLDOWN
        )
        return (time.time() - self.last_detection_time) > cooldown

state = SystemState()


# WORD RESPONSES (FOR NARRATION)

WORD_RESPONSES = {
    "HELLO": "The person is greeting you.",
    "YOU": "The person is referring to you.",
    "NAME": "The person is introducing their name.",
    "THANK_YOU": "The person is thanking you.",
    "WELCOME": "The person is responding politely.",
    "GOODBYE": "The person is saying goodbye.",
    "MORNING": "The person is wishing you good morning.",
    "AFTERNOON": "The person is wishing you good afternoon.",
    "NIGHT": "The person is wishing you good night.",
    "PLEASE": "The person is requesting politely.",
    "SORRY": "The person is apologizing.",
    "YES": "The person is agreeing.",
    "NO": "The person is disagreeing.",
    "STOP": "The person is asking to stop.",
    "WAIT": "The person is asking to wait.",
    "HELP": "The person is asking for help.",
    "I'M_HUNGRY": "The person is saying they are hungry.",
    "I'M_THIRSTY": "The person is saying they are thirsty.",
    "I'M_TIRED": "The person is saying they are tired.",
    "I'M_BUSY": "The person is saying they are busy.",
    "I'M_BORED": "The person is saying they are bored.",
    "EAT": "The person is referring to eating.",
    "WHAT": "The person is asking what.",
    "WHEN": "The person is asking when.",
    "FATHER": "The person is referring to their father.",
    "MOTHER": "The person is referring to their mother.",
    "FRIEND": "The person is referring to a friend.",
    "MY": "The person is saying my.",
}

# AUDIO NARRATION SYSTEM

narration_queue = Queue()
llm_cache = {}

def speak_async(text: str):
    def _run():
        try:
            # Escape quotes for PowerShell
            safe_text = text.replace('"', '""')
            command = f'''
            Add-Type -AssemblyName System.Speech;
            $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;
            $speak.Rate = 1;
            $speak.Speak("{safe_text}");
            '''
            subprocess.run(
                ["powershell", "-Command", command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10
            )
        except Exception as e:
            print(f"[WARN] TTS error: {e}")
    
    threading.Thread(target=_run, daemon=True).start()

def llm_polish(sentence: str) -> str:
    if not config.USE_LLM:
        return sentence
    
    # Return cached version instantly
    if sentence in llm_cache:
        return llm_cache[sentence]
    
    try:
        response = requests.post(
            config.LLM_URL,
            json={
                "model": config.LLM_MODEL,
                "prompt": (
                    "Rewrite this sign language detection into a natural narration sentence.\n"
                    "Keep it brief and conversational. One sentence only.\n\n"
                    f"Detection: {sentence}\n"
                    "Natural sentence:"
                ),
                "stream": False
            },
            timeout=config.LLM_TIMEOUT
        )
        
        if response.status_code == 200:
            polished = response.json().get("response", sentence).strip()
            # Remove common LLM artifacts
            polished = polished.replace("Here's the rewritten sentence:", "").strip()
            polished = polished.strip('"\'')
            
            # Cache it
            llm_cache[sentence] = polished
            return polished
        else:
            return sentence
            
    except Exception as e:
        print(f"[WARN] LLM error: {e}")
        return sentence

def narration_worker():
    while True:
        item = narration_queue.get()
        if item is None:
            break
        
        sentence, use_llm = item
        
        # 1️⃣ Speak immediately (ZERO delay)
        speak_async(sentence)

        # 2️⃣ If LLM is enabled, polish in background for NEXT time
        if use_llm:
            threading.Thread(
                target=llm_polish,
                args=(sentence,),
                daemon=True
            ).start()

# Start narration worker
threading.Thread(target=narration_worker, daemon=True).start()


# DUAL MODEL PREDICTION & DECISION LOGIC

@dataclass
class Prediction:
    label: str
    confidence: float
    model_type: str  # "alphabet" or "word"

def predict_alphabet(landmarks: np.ndarray) -> Optional[Prediction]:
    if alphabet_model is None:
        return None
    
    try:
        expected_dim = int(alphabet_model.input_shape[-1])
        if landmarks.shape[1] == expected_dim:
            model_input = landmarks
        elif landmarks.shape[1] == 63 and expected_dim == 42:
            # Fallback model expects x/y only, but live extraction has x/y/z.
            model_input = landmarks.reshape(1, 21, 3)[:, :, :2].reshape(1, 42)
        else:
            return None

        pred = alphabet_model.predict(model_input, verbose=0)[0]
        confidence = float(np.max(pred))
        label_id = int(np.argmax(pred))
        
        if confidence >= config.ALPHABET_CONFIDENCE_THRESHOLD and label_id < len(ALPHABET_LABELS):
            return Prediction(
                label=ALPHABET_LABELS[label_id],
                confidence=confidence,
                model_type="alphabet"
            )
    except Exception as e:
        print(f"[WARN] Alphabet prediction error: {e}")
    
    return None

def predict_word(landmarks: np.ndarray) -> Optional[Prediction]:
    if word_model is None:
        return None
    
    try:
        pred = word_model.predict(landmarks, verbose=0)[0]
        confidence = float(np.max(pred))
        label_id = int(np.argmax(pred))
        
        if confidence >= config.WORD_CONFIDENCE_THRESHOLD:
            return Prediction(
                label=WORD_LABELS[label_id],
                confidence=confidence,
                model_type="word"
            )
    except Exception as e:
        print(f"[WARN] Word prediction error: {e}")
    
    return None

def decide_best_prediction(
    alphabet_pred: Optional[Prediction],
    word_pred: Optional[Prediction]
) -> Optional[Prediction]:
    
    # Case 1: Only alphabet confident
    if alphabet_pred and not word_pred:
        return alphabet_pred
    
    # Case 2: Only word confident
    if word_pred and not alphabet_pred:
        return word_pred
    
    # Case 3: Both confident - need to decide
    if alphabet_pred and word_pred:
        confidence_diff = abs(alphabet_pred.confidence - word_pred.confidence)
        
        # If one is significantly more confident, use it
        if confidence_diff > config.CONFIDENCE_DELTA_THRESHOLD:
            if alphabet_pred.confidence > word_pred.confidence:
                return alphabet_pred
            else:
                return word_pred
        
        # If similar confidence, prefer WORD (better for temporal gestures)
        return word_pred
    
    # Case 4: Neither confident
    return None

def get_stable_prediction(
    prediction: Prediction,
    buffer: deque,
    required_frames: int
) -> Optional[str]:
    
    # Add to buffer
    buffer.append(prediction.label)
    
    # Need minimum frames
    if len(buffer) < required_frames:
        return None
    
    # Get most common prediction
    counter = Counter(buffer)
    most_common_label, count = counter.most_common(1)[0]
    
    # Need strong consensus (at least 60% agreement)
    if count >= int(required_frames * 0.6):
        return most_common_label
    
    return None


# VIDEO PROCESSING PIPELINE

def process_frame(frame: np.ndarray) -> Tuple[np.ndarray, Optional[str], Optional[str]]:
    
    global state
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    detected_label = None
    narration_text = None
    now = time.time()
    state.update_lock()

    # HAND DETECTION
    
    if result.multi_hand_landmarks:
        state.hand_present = True
        hand = result.multi_hand_landmarks[0]
        
        # Draw landmarks
        mp_draw.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
        )
        
        # EXTRACT LANDMARKS
        
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        
        if len(landmarks) != 63:
            return frame, None, None
        
        landmarks_array = np.array(landmarks, dtype=np.float32).reshape(1, -1)
        
        
        # DUAL MODEL PREDICTION
        
        # Get predictions from both models
        if DETECTION_MODE == "ALPHABET":
            chosen_pred = predict_alphabet(landmarks_array)
            state.word_buffer.clear()

        elif DETECTION_MODE == "WORD":
            chosen_pred = predict_word(landmarks_array)
            state.alphabet_buffer.clear()


        # TEMPORAL STABILITY CHECK
        
        if chosen_pred:
            if chosen_pred.model_type == "alphabet":
              state.clear_other_buffer("alphabet")
              stable_label = get_stable_prediction(
              chosen_pred,
              state.alphabet_buffer,
              config.ALPHABET_STABLE_FRAMES
              )
            else:  # word
              state.clear_other_buffer("word")
              stable_label = get_stable_prediction(
              chosen_pred,
              state.word_buffer,
              config.WORD_STABLE_FRAMES
              )
    
            # GESTURE LOCKING & COOLDOWN
            
            if stable_label and not state.is_locked() and state.can_detect_new(chosen_pred.model_type):
                if chosen_pred.model_type == "alphabet":
                     detected_label = stable_label
                     narration_text = stable_label
                     use_llm = False

                else:  # WORD
                     narration_text = WORD_RESPONSES.get(
                     stable_label,
                    stable_label.replace("_", " ").title()
                    )
                     detected_label = narration_text 
                     use_llm = False

                # Prepare narration
                if chosen_pred.model_type == "alphabet":
                    # For alphabet: just speak the letter
                    narration_text = stable_label
                    use_llm = False
                else:
                    # For words: speak full sentence
                    narration_text = WORD_RESPONSES.get(
                        stable_label,
                        f"The person is signing {stable_label.replace('_', ' ').lower()}"
                    )
                    use_llm = False
                
                # Update state
                state.last_detection = detected_label
                state.last_detection_time = now
                state.lock_gesture()
                
                # Clear buffers after successful detection
                state.reset_buffers()
                
                # Queue narration
                narration_queue.put((narration_text, use_llm))
                
                print(f"[DETECTED] {detected_label} ({chosen_pred.model_type}, conf={chosen_pred.confidence:.2f})")
    
    else:
        # NO HAND DETECTED
        if state.hand_present:
            # Hand just disappeared - reset everything
            state.reset_buffers()
            state.hand_present = False
    
    return frame, detected_label, narration_text

# FLASK VIDEO STREAM

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    current_display_label = None
    time.sleep(0.04)
    display_start_time = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Mirror frame
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Process frame
        small = cv2.resize(frame, (640, 480))
        small, detected_label, narration_text = process_frame(small)
        frame = cv2.resize(small, (w, h))

        # Update display label
        if detected_label:
            current_display_label = detected_label
            display_start_time = time.time()
        
        # Clear display label after hold time
        if current_display_label:
            if (time.time() - display_start_time) > config.DISPLAY_HOLD_TIME:
                current_display_label = None
        
        # UI OVERLAY
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # System status
        status_color = (0, 255, 0) if state.hand_present else (100, 100, 100)
        status_text = "✓ Hand Detected" if state.hand_present else "⨯ No Hand"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)
        
        # Model status
        model_text = "Models: "
        if alphabet_model:
            model_text += "Alphabet✓ "
        if word_model:
            model_text += "Word✓"
        cv2.putText(frame, model_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Gesture lock indicator
        if state.is_locked():
            cv2.putText(frame, " LOCKED", (w - 110, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
        
        # DETECTED GESTURE DISPLAY (CENTER)
        
        if current_display_label:
            if len(current_display_label) == 1:
                # Alphabet
                display_color = (255, 0, 255)  
                bg_color = (255, 0, 255)
            else:
                # Word
                display_color = (0, 255, 0)  
                bg_color = (0, 255, 0)
            
            # Clean display text
            display_text = current_display_label.replace("_", " ")
            
            # Calculate text size and position
            font_scale = 2.0 if len(display_text) == 1 else 1.2
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 5)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            
            # Background box
            padding = 18
            cv2.rectangle(frame,
                         (text_x - padding, text_y - text_size[1] - padding),
                         (text_x + text_size[0] + padding, text_y + padding),
                         bg_color, -1)
            
            # Text
            cv2.putText(frame, display_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3)
            
            # Type indicator
            type_text = "ALPHABET" if len(current_display_label) == 1 else "WORD"
            cv2.putText(frame, type_text, (text_x, text_y + text_size[1] + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        
        # ENCODE & STREAM
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

# FLASK ROUTES

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/status')
def status():
    """API endpoint for system status"""
    return jsonify({
        'alphabet_model': alphabet_model is not None,
        'word_model': word_model is not None,
        'hand_present': state.hand_present,
        'gesture_locked': state.is_locked(),
        'last_detection': state.last_detection,
        'llm_enabled': config.USE_LLM,
        'llm_cache_size': len(llm_cache),
        'current_mode': DETECTION_MODE,
        'word_labels': WORD_LABELS,
        'alphabet_labels': ALPHABET_LABELS
    })

@app.route("/set_mode/<mode>")
def set_mode(mode):
    global DETECTION_MODE
    if mode.upper() in ["WORD", "ALPHABET"]:
        DETECTION_MODE = mode.upper()
        state.reset_buffers()
        return jsonify({"status": "ok", "mode": DETECTION_MODE})
    return jsonify({"status": "error", "message": "Invalid mode"}), 400

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'alphabet': alphabet_model is not None,
            'word': word_model is not None
        }
    })


@app.route('/api/infer_frame', methods=['POST'])
def infer_frame():
    """Run inference on a single client-provided frame."""
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        print(f"[DEBUG] Missing image payload. Raw data length: {len(request.data)}")
        return jsonify({"status": "error", "message": "Missing image payload"}), 400

    try:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        frame_bytes = base64.b64decode(image_data)
        np_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

        if frame is None:
            print("[DEBUG] cv2.imdecode returned None")
            return jsonify({"status": "error", "message": "Invalid image data"}), 400

        # Match training/inference orientation used in local webcam path.
        frame = cv2.flip(frame, 1)

        small = cv2.resize(frame, (640, 480))
        _, detected_label, narration_text = process_frame(small)

        return jsonify({
            "status": "ok",
            "detected_label": detected_label,
            "narration_text": narration_text,
            "hand_present": state.hand_present,
            "gesture_locked": state.is_locked(),
            "last_detection": state.last_detection,
            "current_mode": DETECTION_MODE
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Inference failed: {e}"}), 500


# FOR RUNNING APPLICATION

if __name__ == "__main__":
    import os as _os
    
    port = int(_os.environ.get("PORT", 5000))
    host = _os.environ.get("HOST", "0.0.0.0")
    debug = _os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    
    print("\n" + "="*70)
    print("DUAL-MODEL SIGN LANGUAGE SYSTEM")
    print("="*70)
    print(f"Alphabet model: {'[OK]' if alphabet_model else '[ERROR]'}")
    print(f"Word model: {'[OK]' if word_model else '[ERROR]'}")
    print(f"LLM enabled: {'[OK]' if config.USE_LLM else '[OFF]'}")
    print(f"Word labels: {len(WORD_LABELS)}")
    print(f"Alphabet labels: {len(ALPHABET_LABELS)}")
    print(f"Running on: {host}:{port}")
    print("="*70)
    
    app.run(host=host, port=port, debug=debug, threaded=True)
