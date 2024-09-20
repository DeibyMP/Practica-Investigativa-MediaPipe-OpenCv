import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

window_width = 1600
window_height = 1050

up = False
down = False
count = 0

#Deteccion y Dibujo de las marcas

def tracking_especifico(frame, results, width, height):

    global count, up, down

    x1 = int(results.pose_landmarks.landmark[11].x * width)
    y1 = int(results.pose_landmarks.landmark[11].y * height)

    x2 = int(results.pose_landmarks.landmark[13].x * width)
    y2 = int(results.pose_landmarks.landmark[13].y * height)

    x3 = int(results.pose_landmarks.landmark[15].x * width)
    y3 = int(results.pose_landmarks.landmark[15].y * height)

    x4 = int(results.pose_landmarks.landmark[12].x * width)
    y4 = int(results.pose_landmarks.landmark[12].y * height)

    x5 = int(results.pose_landmarks.landmark[14].x * width)
    y5 = int(results.pose_landmarks.landmark[14].y * height)

    x6 = int(results.pose_landmarks.landmark[16].x * width)
    y6 = int(results.pose_landmarks.landmark[16].y * height)

    #cv2.circle(frame, (x1, y1), 10, (225, 0, 0), -1)
    #cv2.circle(frame, (x2, y2), 10, (225, 0, 0), -1)
    #cv2.circle(frame, (x3, y3), 10, (225, 0, 0), -1)

    #cv2.circle(frame, (x4, y4), 10, (252, 252, 252), -1)
    #cv2.circle(frame, (x5, y5), 10, (252, 252, 252), -1)
    #cv2.circle(frame, (x6, y6), 10, (252, 252, 252), -1)

    #cv2.line(frame, (x1, y1), (x2, y2), (255, 34, 2), 3)
    #cv2.line(frame, (x2, y2), (x3, y3), (255, 34, 2), 3)
    #cv2.line(frame, (x3, y3), (x1, y1), (255, 34, 2), 3)
    #cv2.line(frame, (x4, y4), (x5, y5), (255, 242, 204), 3)
    #cv2.line(frame, (x5, y5), (x6, y6), (255, 242, 204), 3)
    #cv2.line(frame, (x6, y6), (x4, y4), (252, 252, 252), 3)

    point1 = np.array([x1, y1])
    point2 = np.array([x2, y2])
    point3 = np.array([x3, y3])

    point4 = np.array([x4, y4])
    point5 = np.array([x5, y5])
    point6 = np.array([x6, y6])

    line1 = np.linalg.norm(point2 - point3)
    line2 = np.linalg.norm(point1 - point3)
    line3 = np.linalg.norm(point1 - point2)
    line4 = np.linalg.norm(point5 - point6)
    line5 = np.linalg.norm(point4 - point6)
    line6 = np.linalg.norm(point4 - point5)

    angle1 = degrees(acos((line1**2 + line3**2 - line2**2) / (2 * line1 * line3)))
    angle2 = degrees(acos((line4**2 + line6**2 - line5**2) / (2 * line4 * line6)))

    #cv2.putText(frame, str(int(angle1)), (x2 +30, y2), 1, 1.5, (128,0,250), 2)
    #cv2.putText(frame, str(int(angle2)), (x5 +30, y5), 1, 1.5, (128,0,250), 2)

    if angle1 and angle2 >= 160:
        up = True
    if up == True and down == False and angle1 and angle2 <= 70:
        down = True
    if up == True and down == True and angle1 and angle2 >= 160:
        count += 1
        up = False
        down = False
    
    if count < 3:
        cv2.putText(frame, "Repeticiones: {}".format(count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Ejercicio Completado', (500, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.putText(frame, 'Buen trabajo hasta la proxima', (330, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.putText(frame, 'Pulsa la tecla Esc para salir', (380, 600), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)

    

def main():
    
    with mp_pose.Pose(static_image_mode=False) as pose:
            
        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (window_width, window_height))
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks is not None:
                if tracking_especifico(frame, results, width, height):
                    break

            cv2.imshow("Frame", frame) 
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()