import cv2 #Importacion de OpecCv
import mediapipe as mp #Importacion mediaPipe
import math

mp_drawing = mp.solutions.drawing_utils #Esta linea nos permite acceder a todas las funciones que ofrece MediaPipe 
                                        #para dibujar nuestros landmarks y conexciones de estas
                                        #Opciones de configuracion
                                        #STATIC_IMAGE_MODE: TRUE/FALSE  si le asignamos False este interactua con las imagenes de entrada como un video
                                        #todo esto en los primeros fotogramas, pero si se le asigna TRUE, este hara el tracking a cada imagen por separado
                                        #lo cual es ideal para imagenes que no tengan una relacion.

mp_pose = mp.solutions.pose #Esta linea nos permite hacer uso del pose estimation de MediaPipe, que fue diseñada para 
                            #detectar y estimar la postura del cuerpo humano en imagenes o videos

window_width = 1270
window_height = 720

def calcular_angulo(point1, point2, point3):
    angulo = math.degrees(math.atan2(point3[1] - point2[1], point3[0] - point2[0]) -
                         math.atan2(point1[1] - point2[1], point1[0] - point2[0]))
    return angulo + 180 if angulo < 0 else angulo

def tracking_especifico(image, results, width, height):

        print(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * width))
        right_elbow = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * width),
                       int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * height))#Coordenada en Y)
        
        print(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width))
        right_wrist = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width),
                       int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height))

        #Aqui vamos a dibujar el punto en la mano izquierda
        right_shoulder = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width),
                          int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height))


        cv2.circle(image, right_elbow, 10, (225, 0, 0), -1)#Dibujamos el punto despues de obtener sus coordenadas
        cv2.circle(image, right_wrist, 10, (225, 40, 57), -1)#Dibujamos el punto despues de obtener sus coordenadas
        cv2.circle(image, right_shoulder, 10, (225, 40, 57), -1)
        cv2.line(image, right_shoulder, right_elbow, (225, 225, 225), 3)#Dibujamos las conexiones entre los puntos
        cv2.line(image, right_elbow, right_wrist, (225, 225, 225), 3)

        right_arm_angle = calcular_angulo(right_shoulder, right_elbow, right_wrist)
        #left_arm_angle = calcular_angulo(left_shoulder, left_elbow, left_wrist)

        cv2.putText(image, str(int(right_arm_angle)), right_elbow, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #cv2.putText(image, str(int(left_arm_angle)), left_elbow, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if -120 <= right_arm_angle <= -110: #and -110 <= left_arm_angle <= -120:
            cv2.putText(image, "Good Form", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
        else:
            cv2.putText(image, "Bad Form", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)


def tracking_global(image, results):

        #con mp_drawing.draw_landmarks dibujamos todos los punto detectables de la imagen
        #con mp_drawing.draw_landmarks dibujamos los puntos de nuestra imagen
        #con mp_pose.POSE_CONNECTIONS esto podemos dibujar las conexiones de los puntos de referencia
        #con mp_drawing.DrawingSpec podemos cambiar el color, grosor y radio de nuestros puntos o conexiones

        mp_drawing.draw_landmarks(image, results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(225, 0, 0), thickness=3 ,circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))


def main():
    #static_image_mode=True es una de las tantas configuraciones de mediapipe para la detecion de estos puntos,
    #en este caso se le asigno TRUE para que tome la imagen individualmente y haga la detecion de los vertices, 
    #mientras que si se le asignara FALSE esta tomaria las fotos como un video y detctaria la posicion de los 
    #puntos en los primeros fotogramas.
    with mp_pose.Pose(static_image_mode=True) as pose:
        while True:
            image = cv2.imread("media\ORIGINAL_0107d4748c9cb833a3b1874ab0927372.jpg")
            image = cv2.resize(image, (window_width, window_height))
            height, width, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = pose.process(image_rgb)
            print("Pose landmarks:", results.pose_landmarks)
            if results.pose_landmarks is not None:
                print("Seleccione el modo de tracking:")
                print("1. Tracking específico")
                print("2. Tracking global")
                print("3. Salir")
                choice = input("Ingrese el número de su elección: ")

                if choice == '1':
                    tracking_especifico(image, results, width, height)
                    cv2.imshow("image", image)
                    cv2.waitKey(0)
                    cv2.destroyWindow("image")
                elif choice == '2':
                    tracking_global(image, results)
                    cv2.imshow("image", image)
                    cv2.waitKey(0)
                    cv2.destroyWindow("image")

                elif choice == '3':
                    print("Saliendo de la aplicación...")
                    break
                else:
                    print("Opción no válida")


if __name__ == "__main__":
    main()