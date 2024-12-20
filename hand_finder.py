import cv2
import mediapipe as mp

# Inicjalizacja kamery
cam = cv2.VideoCapture(0)

# Inicjalizacja MediaPipe dla dłoni
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Konfiguracja detektora dłoni
with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Odbicie lustrzane ramki dla naturalnego wyświetlania
        frame = cv2.flip(frame, 1)

        # Konwersja obrazu na RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wykrycie dłoni i landmarków
        results = hands.process(rgb_frame)

        # Rysowanie landmarków na obrazie
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Pobierz nazwę wykrytej dłoni (lewa/prawa)
                hand_label = "Unknown"
                if results.multi_handedness:
                    hand_label = results.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'

                # Rysowanie etykiety (lewa/prawa dłoń) nad pierwszym landmarkiem (nad nadgarstkiem)
                h, w, _ = frame.shape
                landmark_0 = hand_landmarks.landmark[0]
                cx, cy = int(landmark_0.x * w), int(landmark_0.y * h)
                cv2.putText(frame, hand_label, (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)



                # Rysowanie połączeń landmarków
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Wyświetlanie przechwyconej klatki z nałożonymi landmarkami
        cv2.imshow('Camera', frame)

        # Wciśnięcie 'q' kończy działanie pętli
        if cv2.waitKey(1) == ord('q'):
            break

# Zwolnienie zasobów
cam.release()
cv2.destroyAllWindows()
