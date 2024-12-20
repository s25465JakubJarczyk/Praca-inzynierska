import time
import pickle
import os
import random
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

PHOTOS_DIR = 'photos_datasets'
PICKLE_DIR = 'datasets_pickle'

# Upewniamy się, że katalogi istnieją
os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(PICKLE_DIR, exist_ok=True)

# Inicjalizacja MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Słownik do przechowywania statystyk
class_stats = {}

# Funkcja do przetwarzania katalogów i zbierania danych
def process_directory(dir_name, data, labels, use_z_dimension=False, limit=None):
    """
    Funkcja przetwarza obrazy z podkatalogów w katalogu głównym (każdy podkatalog to klasa).
    Użytkownik może ograniczyć liczbę zdjęć do przetworzenia.
    """
    class_dir = os.path.join(PHOTOS_DIR, dir_name)

    if not os.path.exists(class_dir):
        print(f"Katalog {class_dir} nie istnieje. Pomijam.")
        return

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    # Iteracja po podkatalogach (klasach)
    for sub_dir in os.listdir(class_dir):
        sub_dir_path = os.path.join(class_dir, sub_dir)

        if not os.path.isdir(sub_dir_path):
            print(f"Pomijam {sub_dir}, nie jest katalogiem klasy.")
            continue

        print(f"\nPrzetwarzam klasę: {sub_dir}")
        total_images = 0
        rejected_images = 0
        processed_images = 0

        images = [img for img in os.listdir(sub_dir_path) if img.lower().endswith(valid_extensions)]
        random.shuffle(images)  # Losowe przetasowanie obrazów

        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.10) as hands:
            for img_path in images:
                if limit and processed_images >= limit:
                    break  # Przetworzono wystarczającą liczbę zdjęć

                total_images += 1
                data_aux = []
                img = cv2.imread(os.path.join(sub_dir_path, img_path))

                if img is None:
                    print(f"Błąd: Nie można załadować obrazu {img_path}. Pomijam.")
                    rejected_images += 1
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Wykrycie dłoni i landmarków
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:  # Jeśli wykryto dłoń
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x)
                            data_aux.append(y)
                            if use_z_dimension:
                                z = hand_landmarks.landmark[i].z
                                data_aux.append(z)

                    data.append(data_aux)
                    labels.append(sub_dir)  # Nazwa podkatalogu jako etykieta
                    processed_images += 1
                    print(f"Przetworzono obraz {img_path} ({processed_images}/{limit or '∞'})")
                else:
                    rejected_images += 1
                    print(f"Odrzucono obraz {img_path}, brak wykrytej dłoni.")

        # Zapis statystyk dla klasy
        class_stats[sub_dir] = {
            "total_images": total_images,
            "rejected_images": rejected_images,
            "processed_images": processed_images,
            "rejected_percentage": (rejected_images / total_images) * 100 if total_images > 0 else 0
        }


# Wybór trybu działania
print("Wybierz tryb działania:")
print(f"1 - Wybierz katalog i stwórz nową bazę danych.")
print("2 - Dodaj dane tylko dla nowej klasy.")
mode = input("Podaj tryb działania (1/2): ")

# Wybór, czy uwzględniać wymiar 'z'
use_z = input("Czy chcesz uwzględniać wymiar 'z' (tak/nie)? ").lower() == 'tak'

# Wybór liczby danych do przetworzenia
use_all = input("Czy chcesz użyć wszystkich zdjęć? (tak/nie): ").lower() == 'tak'
data_limit = None if use_all else int(input("Podaj liczbę zdjęć do przetworzenia: "))

# Inicjalizacja bazy danych
if mode == "1":
    # Tryb: przetwarzanie wybranego katalogu
    print("\nDostępne katalogi w './photos_datasets':")
    directories = os.listdir(PHOTOS_DIR)
    for idx, directory in enumerate(directories, start=1):
        print(f"{idx}. {directory}")

    try:
        selected_index = int(input("Wybierz numer katalogu do przetworzenia: ")) - 1
        if selected_index < 0 or selected_index >= len(directories):
            raise ValueError("Wybrano nieprawidłowy numer katalogu.")
        selected_directory = directories[selected_index]
    except ValueError as e:
        print(f"Błąd: {e}. Kończę program.")
        exit()

    new_pickle_name = input("Podaj nazwę pliku pickle (bez rozszerzenia): ")
    PICKLE_FILE = os.path.join(PICKLE_DIR, f"{new_pickle_name}.pickle")

    data = []
    labels = []
    print(f"\nPrzetwarzam katalog '{selected_directory}' w './photos_datasets'.")
    process_directory(selected_directory, data, labels, use_z_dimension=use_z, limit=data_limit)

elif mode == "2":
    # Tryb: dodanie nowej klasy
    existing_pickle_name = input("Podaj nazwę istniejącego pliku pickle (bez rozszerzenia): ")
    PICKLE_FILE = os.path.join(PICKLE_DIR, f"{existing_pickle_name}.pickle")

    if os.path.exists(PICKLE_FILE):
        with open(PICKLE_FILE, 'rb') as f:
            database = pickle.load(f)
            data = database['data']
            labels = database['labels']
        print("Wczytano istniejącą bazę danych.")
    else:
        print(f"Plik {PICKLE_FILE} nie istnieje. Tworzę nową bazę danych.")
        data = []
        labels = []

    new_class = input("Podaj nazwę nowej klasy do przetworzenia: ")
    process_directory(new_class, data, labels, use_z_dimension=use_z, limit=data_limit)

else:
    print("Nieprawidłowy tryb działania. Kończę program.")
    exit()

# Wyświetlenie statystyk dla wszystkich klas
print("\nPodsumowanie statystyk dla wszystkich klas:")
for class_name, stats in class_stats.items():
    print(f"Kategoria '{class_name}':")
    print(f"  Łączna liczba zdjęć: {stats['total_images']}")
    print(f"  Przetworzone zdjęcia: {stats['processed_images']}")
    print(f"  Odrzucone zdjęcia: {stats['rejected_images']}")
    print(f"  Procent odrzuconych: {stats['rejected_percentage']:.2f}%")

# Zapisanie bazy danych do pliku
with open(PICKLE_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
print(f"\nZaktualizowano bazę danych. Plik zapisano jako: {PICKLE_FILE}")
