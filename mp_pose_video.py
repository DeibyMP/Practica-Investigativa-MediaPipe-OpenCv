import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("media/5262396-uhd_3840_2160_25fps.mp4")

#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

window_width = 1270
window_height = 720

with mp_pose.Pose(static_image_mode=False) as pose:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame, (window_width, window_height))
        #frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(128,0,250), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
            )

        cv2.imshow("Frame", frame) 
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()