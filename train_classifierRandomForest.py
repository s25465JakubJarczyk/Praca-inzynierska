import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

PICKLE_DIR = 'datasets_pickle'
CLASSIFIERS_DIR = 'classifiers'

# Upewnij się, że katalog na klasyfikatory istnieje
os.makedirs(CLASSIFIERS_DIR, exist_ok=True)

# Wyświetlenie wszystkich plików w katalogu
print("Dostępne pliki w katalogu './datasets_pickle':")
pickle_files = os.listdir(PICKLE_DIR)  # Pobranie wszystkich plików z katalogu
for idx, file in enumerate(pickle_files, start=1):
    print(f"{idx}. {file}")

# Wybór pliku
try:
    selected_index = int(input("Wybierz numer pliku do użycia w trenowaniu: ")) - 1
    if selected_index < 0 or selected_index >= len(pickle_files):
        raise ValueError("Wybrano nieprawidłowy numer pliku.")
    selected_pickle = pickle_files[selected_index]
except ValueError as e:
    print(f"Błąd: {e}. Kończę program.")
    exit()

# Wczytanie danych z wybranego pliku
pickle_path = os.path.join(PICKLE_DIR, selected_pickle)
print(f"Trenuję model danymi z pliku: {pickle_path}")

with open(pickle_path, 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Wykrywanie liczby wymiarów
num_features_per_landmark = len(data[0]) // 21  # Liczba cech na jeden punkt dłoni (21 punktów w MediaPipe)
if num_features_per_landmark == 2:
    print("Dane zawierają tylko współrzędne 'x' i 'y'.")
    train_with_z = False
elif num_features_per_landmark == 3:
    print("Dane zawierają współrzędne 'x', 'y' oraz 'z'.")
    train_with_z = True
else:
    print("Nieprawidłowy format danych. Kończę program.")
    exit()

# Funkcja do trenowania i zapisywania modelu
def train_and_save_model(x_train, x_test, y_train, y_test, model_name):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # Predykcja na danych testowych
    y_predict = model.predict(x_test)

    # Obliczenie dokładności i raportu klasyfikacji
    score = accuracy_score(y_test, y_predict)
    print(f"\nModel '{model_name}':")
    print(classification_report(y_test, y_predict))
    print(f'{score * 100:.2f}% of samples were classified correctly!')

    # Zapisanie modelu
    model_path = os.path.join(CLASSIFIERS_DIR, f"{model_name}.p")
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'labels': list(set(labels))}, f)
    print(f"Model zapisano jako: {model_path}")

# Trenowanie modeli
if train_with_z:
    # Podział danych na zestaw z i bez 'z'
    print("\nRozdzielam dane na zestawy z i bez współrzędnej 'z'...")
    data_with_z = data
    data_without_z = np.delete(data, np.arange(2, data.shape[1], 3), axis=1)  # Usuwamy każdy trzeci element ('z')

    # Model z 'z'
    x_train, x_test, y_train, y_test = train_test_split(data_with_z, labels, test_size=0.2, stratify=labels)
    train_and_save_model(x_train, x_test, y_train, y_test, f"{selected_pickle}_with_z")

    # Model bez 'z'
    x_train, x_test, y_train, y_test = train_test_split(data_without_z, labels, test_size=0.2, stratify=labels)
    train_and_save_model(x_train, x_test, y_train, y_test, f"{selected_pickle}_without_z")
else:
    # Model bez 'z'
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)
    train_and_save_model(x_train, x_test, y_train, y_test, f"{selected_pickle}_without_z")
