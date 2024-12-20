import os
import cv2
import mediapipe as mp
import pickle
import numpy as np

CLASSIFIERS_DIR = 'classifiers'

# Wyświetlenie dostępnych modeli
print("Dostępne modele w katalogu './classifiers':")
classifier_files = os.listdir(CLASSIFIERS_DIR)  # Pobranie wszystkich plików w katalogu
for idx, file in enumerate(classifier_files, start=1):
    print(f"{idx}. {file}")

# Wybór modelu
try:
    selected_index = int(input("Wybierz numer modelu do użycia: ")) - 1
    if selected_index < 0 or selected_index >= len(classifier_files):
        raise ValueError("Wybrano nieprawidłowy numer modelu.")
    selected_model = classifier_files[selected_index]
except ValueError as e:
    print(f"Błąd: {e}. Kończę program.")
    exit()

model_path = os.path.join(CLASSIFIERS_DIR, selected_model)
print(f"Wczytuję model: {model_path}")

# Wczytanie modelu
with open(model_path, 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

# Generowanie labels_dict na podstawie model.classes_
try:
    classes = model.classes_
    labels_dict = {i: cls for i, cls in enumerate(classes)}
    print(f"Używane etykiety: {labels_dict}")
except AttributeError:
    print("Nie można wyciągnąć klas z modelu. Używam domyślnych etykiet (liczb).")
    labels_dict = {}

# Sprawdzenie, ile cech oczekuje model
expected_features = model.n_features_in_
print(f"Model oczekuje {expected_features} cech.")

# Inicjalizacja kamery
cam = cv2.VideoCapture(0)

# Inicjalizacja MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

    while True:
        data_aux = []

        ret, frame = cam.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        predicted_character = None
        probabilities = None

        if results.multi_hand_landmarks:  # Jeśli wykryto dłoń
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    if expected_features == 63:  # Jeśli model oczekuje wymiaru 'z'
                        z = hand_landmarks.landmark[i].z
                        data_aux.append(z)

            # Przewidywanie i wyznaczanie prawdopodobieństw
            if len(data_aux) == expected_features:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict.get(prediction[0], str(prediction[0]))
                probabilities = model.predict_proba([np.asarray(data_aux)])[0]

                # Sortowanie prawdopodobieństw i wybór top 5
                top_indices = np.argsort(probabilities)[-5:][::-1]
                top_probabilities = [(labels_dict.get(i, str(i)), probabilities[i]) for i in top_indices]
            else:
                print(f"Błąd: liczba cech ({len(data_aux)}) nie zgadza się z oczekiwaniami modelu ({expected_features}).")

        # Odbicie obrazu w poziomie
        frame = cv2.flip(frame, 1)

        # Wyświetlenie przewidywanego znaku na ekranie
        if predicted_character and probabilities is not None:
            cv2.putText(frame,
                        f"Predicted: {predicted_character}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA)

            # Wyświetlenie top 5 najbardziej prawdopodobnych znaków
            y_offset = 100
            for class_label, prob in top_probabilities:
                cv2.putText(frame,
                            f"{class_label}: {prob:.2f}",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA)
                y_offset += 20

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
