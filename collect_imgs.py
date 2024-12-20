import os
import cv2
import mediapipe as mp

DATA_DIR = 'photos_datasets/data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

cap = cv2.VideoCapture(0)  # Domyślna kamera


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

print("Instrukcje:")
print("1. Naciśnij 'q', aby rozpocząć/zatrzymać zbieranie danych.")
print("2. Naciśnij 's', aby włączyć/wyłączyć wyświetlanie landmarków na obrazie.")
print("3. Naciśnij ponownie 'q', aby zakończyć zbieranie danych i zapisać dane.")

collecting = False  # Flaga oznaczająca stan zbierania danych
show_landmarks = False  # Flaga oznaczająca włączenie/wyłączenie landmarków
frames = []  # Lista do przechowywania zebranych klatek

while True:
    ret, frame = cap.read()
    if not ret:
        print("Nie można uzyskać obrazu z kamery.")
        break


    frame_flipped = cv2.flip(frame, 1)
    frame_display = frame_flipped.copy()  # Kopia obrazu do wyświetlania landmarków

    # Wyświetlanie landmarków, jeśli jest włączone
    if show_landmarks:
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_display,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

    # Wyświetlanie obrazu
    if not collecting:
        cv2.putText(frame_display, "Press 'q' to start collecting data", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame_display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        if not collecting:
            # Rozpoczęcie zbierania danych
            collecting = True
            frames = []  # Zresetuj listę klatek
            print("Rozpoczęto zbieranie danych.")
        else:
            # Zatrzymanie zbierania danych
            print(f"\nZatrzymano zbieranie danych. Zebrano {len(frames)} zdjęć.")
            cap.release()
            cv2.destroyAllWindows()

            # Poproś o nazwę folderu na dane
            class_name = input("Podaj nazwę dla nowej klasy danych: ")
            class_dir = os.path.join(DATA_DIR, class_name)

            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
                print(f"Utworzono katalog: {class_dir}")
            else:
                print(f"Katalog '{class_dir}' już istnieje. Dane zostaną dodane do tego katalogu.")

            # Zapis klatek do katalogu
            for i, frame in enumerate(frames):
                cv2.imwrite(os.path.join(class_dir, f'{i}.jpg'), frame)
            print(f"Zapisano {len(frames)} obrazów w katalogu '{class_name}'.")
            break

    elif key == ord('s'):
        # Przełącz włączanie/wyłączanie landmarków
        show_landmarks = not show_landmarks
        state = "ON" if show_landmarks else "OFF"
        print(f"Landmarki: {state}")

    if collecting:
        # Dodaj czysty obraz (bez landmarków) do listy klatek
        frames.append(frame_flipped)
        print(f"Liczba zebranych zdjęć: {len(frames)}", end='\r')
