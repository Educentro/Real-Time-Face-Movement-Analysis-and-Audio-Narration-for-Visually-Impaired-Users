import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def detect_face(frame):
    rgb = frame[:, :, ::-1]  # BGR → RGB
    results = face_mesh.process(rgb)
    return results
