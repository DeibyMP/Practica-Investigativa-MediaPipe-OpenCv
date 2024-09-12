import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("media/5262396-uhd_3840_2160_25fps.mp4")

window_width = 1270
window_height = 720

def tracking_especifico(frame, results, width, height):
    x1 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)
    y1 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)

    x2 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * width)
    y2 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * height)

    x3 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width)
    y3 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)

    x4 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)
    y4 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)

    x5 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * width)
    y5 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * height)

    x6 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width)
    y6 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)

    cv2.circle(frame, (x1, y1), 10, (225, 0, 0), -1)
    cv2.circle(frame, (x2, y2), 10, (225, 0, 0), -1)
    cv2.circle(frame, (x3, y3), 10, (225, 0, 0), -1)

    cv2.line(frame, (x1, y1), (x2, y2), (225, 225, 225), 3)
    cv2.line(frame, (x2, y2), (x3, y3), (225, 225, 225), 3)

    cv2.circle(frame, (x4, y4), 10, (225, 0, 0), -1)
    cv2.circle(frame, (x5, y5), 10, (225, 0, 0), -1)
    cv2.circle(frame, (x6, y6), 10, (225, 0, 0), -1)

    cv2.line(frame, (x4, y4), (x5, y5), (225, 225, 225), 3)
    cv2.line(frame, (x5, y5), (x6, y6), (225, 225, 225), 3)

def main():
    
    with mp_pose.Pose(static_image_mode=False) as pose:
            
        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            frame = cv2.resize(frame, (window_width, window_height))
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks is not None:
                tracking_especifico(frame, results, width, height)
            cv2.imshow("Frame", frame) 
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

        # if results.pose_landmarks is not None:
        #     mp_drawing.draw_landmarks(
        #         frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #         mp_drawing.DrawingSpec(color=(128,0,250), thickness=2, circle_radius=3),
        #         mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
        #     )



if __name__ == "__main__":
    main()