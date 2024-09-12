import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

window_width = 1270
window_height = 720

#Nuestra funcion calcular_angulo, por medio de la libreria math, resta de los angulos formados 
#por los segmentos (RIGHT_SHOULDER, RIGHT_ELBOW) y (RIGHT_ELBOW, RIGHT_WRIST), 
#calculando el angulo que forma la conexion total del brazo derecho en el eje x 
#convirtiendo la resta de radianes a grados con math.degress.
#math.atan2 nos permite calcular el angulo formado por cada segmento en radianes.


def calcular_angulo(point1, point2, point3):
    angulo = math.degrees(math.atan2(point3[1] - point2[1], point3[0] - point2[0]) -
                         math.atan2(point1[1] - point2[1], point1[0] - point2[0]))
    return angulo + 180 if angulo < 0 else angulo #Con esta linea nos aseguramos de ajustar que el calculo del angulo sea positivo.


def tracking_especifico(frame, results, width, height):
    right_shoulder = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width),
                      int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height))

    right_elbow = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * width),
                   int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * height))

    right_wrist = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width),
                   int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height))

    left_shoulder = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width),
                     int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height))

    left_elbow = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * width),
                  int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * height))

    left_wrist = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width),
                  int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height))

    cv2.circle(frame, right_shoulder, 10, (225, 0, 0), -1)
    cv2.circle(frame, right_elbow, 10, (225, 0, 0), -1)
    cv2.circle(frame, right_wrist, 10, (225, 0, 0), -1)

    cv2.circle(frame, left_shoulder, 10, (225, 0, 0), -1)
    cv2.circle(frame, left_elbow, 10, (225, 0, 0), -1)
    cv2.circle(frame, left_wrist, 10, (225, 0, 0), -1)

    cv2.line(frame, right_shoulder, right_elbow, (225, 225, 225), 3)
    cv2.line(frame, right_elbow, right_wrist, (225, 225, 225), 3)
    cv2.line(frame, left_shoulder, left_elbow, (225, 225, 225), 3)
    cv2.line(frame, left_elbow, left_wrist, (225, 225, 225), 3)


    #Con right_arm_angle y left_arm_angle, invocamos a la funcion calcular_angulo 
    # para que realice el respectivo calculo de estos.
    right_arm_angle = calcular_angulo(right_shoulder, right_elbow, right_wrist)
    left_arm_angle = calcular_angulo(left_shoulder, left_elbow, left_wrist)

    #Con las dos siguientes lineas imprimimos en la ventana un pequeño mensaje 
    # que nos permita saber el angulo que se esta calculando mientras se desarrolla el ejercicio.
    cv2.putText(frame, str(int(right_arm_angle)), right_elbow, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, str(int(left_arm_angle)), left_elbow, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    #Con esta sentencia If-else, validamos cual de los dos mensajes imprimir en 
    #pantalla basandose en el calculo continuo de los angulos.
    if -30 <= right_arm_angle <= 10 and -30 <= left_arm_angle <= 10:
        cv2.putText(frame, "Good Form", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Bad Form", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)


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