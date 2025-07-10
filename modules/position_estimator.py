import math  # Importa el módulo math para operaciones matemáticas

class PositionEstimator:
    def __init__(self, mp_pose):
        # Ancho promedio de hombros en centímetros (usado como referencia)
        self.reference_shoulder_width = 40  # cm, average shoulder width
        # Altura estimada de la cámara en centímetros
        self.camera_height = 100  # cm, estimated camera height
        # Lista para almacenar posiciones anteriores (para suavizado)
        self.prev_positions = []
        # Máximo número de posiciones a recordar para el suavizado
        self.max_history = 10
        # Referencia al módulo de poses de MediaPipe
        self.mp_pose = mp_pose
        
    def estimate_distance_from_camera(self, landmarks, frame_width, frame_height):
        """Estima la distancia desde la cámara basándose en el ancho de los hombros"""
        # Obtiene el punto de referencia del hombro izquierdo
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        # Obtiene el punto de referencia del hombro derecho
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # Calcula la distancia en píxeles entre los hombros
        # .x da coordenadas normalizadas (0-1), se multiplica por ancho del frame
        shoulder_pixel_width = abs(left_shoulder.x - right_shoulder.x) * frame_width
        
        # Evita división por cero si los hombros están en la misma posición
        if shoulder_pixel_width == 0:
            return None
            
        # Estimación de la distancia focal (necesitaría calibración para mayor precisión)
        # Se usa 0.8 como factor empírico basado en cámaras web típicas
        focal_length = frame_width * 0.8  # Rough estimation
        
        # Cálculo de distancia usando triángulos similares
        # Distancia = (Tamaño_real * Distancia_focal) / Tamaño_en_píxeles
        distance = (self.reference_shoulder_width * focal_length) / shoulder_pixel_width
        return distance
    
    def estimate_3d_position(self, landmarks, frame_width, frame_height):
        """Estima la posición 3D relativa a la cámara"""
        # Primero calcula la distancia desde la cámara
        distance = self.estimate_distance_from_camera(landmarks, frame_width, frame_height)
        
        # Si no se puede calcular la distancia, retorna None
        if distance is None:
            return None
            
        # Obtiene puntos de referencia del torso para calcular el centro de masa
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calcula el centro horizontal promediando las 4 coordenadas x
        center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
        # Calcula el centro vertical promediando las 4 coordenadas y
        center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
        
        # Convierte coordenadas normalizadas a coordenadas del mundo real
        # (center_x - 0.5) centra la coordenada en 0, luego escala por distancia
        world_x = (center_x - 0.5) * distance * 0.8  # Horizontal offset
        # Similar para y, pero con factor de escala diferente (aspecto de imagen)
        world_y = (center_y - 0.5) * distance * 0.6  # Vertical offset
        # La coordenada z es simplemente la distancia calculada
        world_z = distance
        
        # Crea diccionario con la posición 3D
        position = {
            'x': world_x,    # Posición horizontal en cm
            'y': world_y,    # Posición vertical en cm
            'z': world_z,    # Distancia de la cámara en cm
            'distance': distance  # Distancia calculada en cm
        }
        
        # Añade la nueva posición al historial para suavizado
        self.prev_positions.append(position)
        # Mantiene solo las últimas max_history posiciones
        if len(self.prev_positions) > self.max_history:
            self.prev_positions.pop(0)  # Elimina la posición más antigua
            
        # Retorna la posición suavizada
        return self.smooth_position()
    
    def smooth_position(self):
        """Aplica suavizado para reducir el ruido en las mediciones"""
        # Si no hay posiciones previas, retorna None
        if not self.prev_positions:
            return None
            
        # Calcula el promedio de todas las coordenadas x guardadas
        avg_x = sum(p['x'] for p in self.prev_positions) / len(self.prev_positions)
        # Calcula el promedio de todas las coordenadas y guardadas
        avg_y = sum(p['y'] for p in self.prev_positions) / len(self.prev_positions)
        # Calcula el promedio de todas las coordenadas z guardadas
        avg_z = sum(p['z'] for p in self.prev_positions) / len(self.prev_positions)
        # Calcula el promedio de todas las distancias guardadas
        avg_distance = sum(p['distance'] for p in self.prev_positions) / len(self.prev_positions)
        
        # Retorna la posición promediada (suavizada)
        return {
            'x': avg_x,
            'y': avg_y,
            'z': avg_z,
            'distance': avg_distance
        }
    
    def estimate_body_orientation(self, landmarks):
        """Estima la orientación del cuerpo (dirección hacia donde mira)"""
        # Obtiene los puntos de referencia de ambos hombros
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # Obtiene puntos adicionales para mejor detección
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE.value]
        
        # Calcula el ángulo de la línea que conecta los hombros
        shoulder_angle = math.atan2(
            right_shoulder.y - left_shoulder.y,  # Diferencia vertical
            right_shoulder.x - left_shoulder.x   # Diferencia horizontal
        )
        
        # Convierte de radianes a grados para facilitar interpretación
        shoulder_angle_deg = math.degrees(shoulder_angle)
        
        # Verifica la visibilidad de los ojos y nariz para distinguir frente/espalda
        nose_visible = nose.visibility > 0.5 if hasattr(nose, 'visibility') else True
        eyes_visible = (left_eye.visibility > 0.5 and right_eye.visibility > 0.5) if hasattr(left_eye, 'visibility') else True
        
        # Calcula el centro de la cara
        face_center_x = (nose.x + left_eye.x + right_eye.x) / 3
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        
        # Determina si está de frente o de espalda basándose en la visibilidad facial
        is_facing_camera = nose_visible and eyes_visible
        
        # Determina la dirección hacia donde mira basándose en el ángulo y visibilidad
        if abs(shoulder_angle_deg) < 30:          # Hombros casi horizontales
            if is_facing_camera:
                facing = "Front"
            else:
                facing = "Back"
        elif abs(shoulder_angle_deg) > 150:       # Hombros muy inclinados
            if is_facing_camera:
                facing = "Front"  # Podría estar inclinado pero de frente
            else:
                facing = "Back"
        elif shoulder_angle_deg > 0:              # Hombro derecho más bajo
            # Verifica qué lado del cuerpo es más visible
            if abs(face_center_x - shoulder_center_x) < 0.1:  # Cara centrada
                facing = "Front" if is_facing_camera else "Back"
            else:
                facing = "Right Side"
        else:                                     # Hombro izquierdo más bajo
            if abs(face_center_x - shoulder_center_x) < 0.1:  # Cara centrada
                facing = "Front" if is_facing_camera else "Back"
            else:
                facing = "Left Side"
        
        # Retorna la dirección y el ángulo calculado
        return facing, shoulder_angle_deg

# Bloque que se ejecuta solo cuando se ejecuta este archivo directamente
if __name__ == "__main__":
    import mediapipe as mp  # Importa MediaPipe para detección de poses
    import cv2              # Importa OpenCV para manejo de video

    # Inicializa los módulos de MediaPipe para detección de poses
    mp_pose = mp.solutions.pose           # Módulo de poses
    mp_draw = mp.solutions.drawing_utils  # Utilidades para dibujar
    
    # Crea el objeto detector de poses con configuraciones específicas
    pose = mp_pose.Pose(
        static_image_mode=False,           # False = modo video, True = imagen estática
        min_detection_confidence=0.5,      # Confianza mínima para detectar persona
        min_tracking_confidence=0.5        # Confianza mínima para seguir persona
    )
    
    # Crea una instancia del estimador de posición
    estimator = PositionEstimator(mp_pose)
    
    # Inicializa la captura de video desde la cámara web (índice 0)
    cap = cv2.VideoCapture(0)  # Use webcam

    # Bucle principal para procesar video en tiempo real
    while True:
        # Lee un frame de la cámara
        ret, frame = cap.read()
        # Si no se pudo leer el frame, sale del bucle
        if not ret:
            break
            
        # Define las dimensiones de visualización
        display_width, display_height = 800, 450
        # Redimensiona el frame a las dimensiones deseadas
        frame = cv2.resize(frame, (display_width, display_height))
        # Voltea horizontalmente para efecto espejo (más natural)
        frame = cv2.flip(frame, 1)
        # Obtiene las dimensiones del frame procesado
        h, w, c = frame.shape

        # Convierte de BGR (formato OpenCV) a RGB (formato MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Procesa el frame para detectar poses
        results = pose.process(rgb_frame)

        # Prueba de uso del método estimate_distance_from_camera
        if results.pose_landmarks:  # Si se detectaron poses
            # Dibuja los puntos de referencia y conexiones en el frame
            mp_draw.draw_landmarks(
                frame,                          # Imagen donde dibujar
                results.pose_landmarks,         # Puntos detectados
                mp_pose.POSE_CONNECTIONS,       # Conexiones entre puntos
                # Especificaciones para dibujar puntos (verde)
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                # Especificaciones para dibujar líneas (rojo)
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

            # Extrae los puntos de referencia detectados
            landmarks = results.pose_landmarks.landmark
            
            opcion = 3

            if opcion == 1:
                # Estima la distancia desde la cámara
                position = estimator.estimate_distance_from_camera(landmarks, w, h)
                if position is not None:
                    # Dibuja el texto con la distancia en la pantalla
                    cv2.putText(
                        frame,                                   # Imagen donde escribir
                        f"Distance: {position:.2f} cm",          # Texto a mostrar
                        (10, 30),                                # Posición del texto
                        cv2.FONT_HERSHEY_SIMPLEX,                # Tipo de fuente
                        1,                                       # Tamaño de fuente
                        (255, 255, 255),                         # Color blanco
                        2                                        # Grosor del texto
                    )

            if opcion == 2:
                # Estima la posición 3D relativa a la cámara
                position = estimator.estimate_3d_position(landmarks, w, h)
                if position is not None:
                    # Dibuja el texto con la posición 3D en la pantalla
                    cv2.putText(
                        frame,                                   # Imagen donde escribir
                        f"Position: ({position['x']:.2f}, {position['y']:.2f}, {position['z']:.2f}) cm",  # Texto a mostrar
                        (10, 60),                                # Posición del texto
                        cv2.FONT_HERSHEY_SIMPLEX,                # Tipo de fuente
                        1,                                       # Tamaño de fuente
                        (255, 255, 255),                         # Color blanco
                        2                                        # Grosor del texto
                    )

            if opcion == 3:
                # Estima la orientación del cuerpo
                facing, angle = estimator.estimate_body_orientation(landmarks)
                # Dibuja el texto con la orientación en la pantalla
                cv2.putText(
                    frame,                                   # Imagen donde escribir
                    f"Facing: {facing} (Angle: {angle:.2f}°)",  # Texto a mostrar
                    (10, 90),                                # Posición del texto
                    cv2.FONT_HERSHEY_SIMPLEX,                # Tipo de fuente
                    1,                                       # Tamaño de fuente
                    (255, 255, 255),                         # Color blanco
                    2                                        # Grosor del texto
                )

        # Muestra la imagen procesada en una ventana
        cv2.imshow('Position Estimator', frame)
        
        # Espera por una tecla presionada (1ms de timeout)
        # Si se presiona 'q', sale del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera los recursos de la cámara
    cap.release()
    # Cierra todas las ventanas de OpenCV
    cv2.destroyAllWindows()