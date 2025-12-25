from flask import Flask, Response
import cv2
from ai.face_detector import detect_face

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def gen_frames():
    import mediapipe as mp
    mp_draw = mp.solutions.drawing_utils

    while True:
        success, frame = camera.read()
        if not success:
            break

        results = detect_face(frame)

        if results.multi_face_landmarks:
            face_count = len(results.multi_face_landmarks)
            for face_landmarks in results.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp.solutions.face_mesh.FACEMESH_CONTOURS
                )

            cv2.putText(frame,f"Face Detected:{face_count}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            h, w, _ = frame.shape

            for face_landmarks in results.multi_face_landmarks:
    # Get nose tip (landmark 1 is reliable)
                nose_x = int(face_landmarks.landmark[1].x * w)
                if nose_x < w / 3:
                    position = "Left"
                elif nose_x < 2 * w / 3:
                    position = "Center"
                else:
                    position = "Right"
                mp_draw.draw_landmarks(
                     frame,
                    face_landmarks,
                    mp.solutions.face_mesh.FACEMESH_CONTOURS
    )

                cv2.putText(
        frame,
        f"Position: {position}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )


        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    app.run(debug=True)
