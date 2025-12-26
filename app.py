from flask import Flask, Response
import cv2
import mediapipe as mp
from ai.face_detector import detect_face

app = Flask(__name__)
camera = cv2.VideoCapture(0)

mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


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
            h, w, _ = frame.shape

            for face_landmarks in results.multi_face_landmarks:

                # ---------- DRAW FACE MESH ----------
                mp_draw.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS
                )

                # ---------- POSITION DETECTION ----------
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

                # ---------- MOUTH OPEN DETECTION (STABLE) ----------
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]

            lip_distance = abs(upper_lip.y - lower_lip.y)

            if lip_distance > 0.03:
                  mouth_open_frames += 1
            else:
                    mouth_open_frames = 0

            if mouth_open_frames > 5:
                  cv2.putText(
        frame,
        "Mouth Open",
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

            # ---------- BLINK DETECTION (BOTH EYES) ----------
        right_top = face_landmarks.landmark[159]
        right_bottom = face_landmarks.landmark[145]

        left_top = face_landmarks.landmark[386]
        left_bottom = face_landmarks.landmark[374]

        right_eye_dist = abs(right_top.y - right_bottom.y)
        left_eye_dist = abs(left_top.y - left_bottom.y)

        if right_eye_dist < 0.008 or left_eye_dist < 0.008:
          cv2.putText(
        frame,
        "Blink",
        (30, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )


            # ---------- FACE COUNT ----------
        cv2.putText(
                frame,
                f"Face Detected: {face_count}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
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
