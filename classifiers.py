import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Katalog z plikami danych
PICKLE_DIR = 'datasets_pickle'

# Wyświetlanie dostępnych plików w katalogu
print("Dostępne pliki w katalogu './datasets_pickle':")
pickle_files = os.listdir(PICKLE_DIR)
for idx, file in enumerate(pickle_files, start=1):
    print(f"{idx}. {file}")

# Wybór pliku
try:
    selected_index = int(input("Wybierz numer pliku do użycia w klasyfikacji: ")) - 1
    if selected_index < 0 or selected_index >= len(pickle_files):
        raise ValueError("Wybrano nieprawidłowy numer pliku.")
    selected_pickle = pickle_files[selected_index]
except ValueError as e:
    print(f"Błąd: {e}. Kończę program.")
    exit()

pickle_path = os.path.join(PICKLE_DIR, selected_pickle)
print(f"Wczytuję dane z pliku: {pickle_path}")

# Wczytanie danych
with open(pickle_path, 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Wykrywanie liczby cech na punkt dłoni (21 punktów w MediaPipe)
num_features_per_landmark = len(data[0]) // 21
print(f"\nLiczba cech na punkt dłoni: {num_features_per_landmark}")

# Jeśli dane mają wymiar 'z', przygotuj zestaw bez wymiaru 'z'
if num_features_per_landmark == 3:
    print("Dane zawierają współrzędne 'x', 'y' oraz 'z'.")
    data_without_z = np.delete(data, np.arange(2, data.shape[1], 3), axis=1)
    test_variants = {
        "Dane z wymiarem 'z'": data,
        "Dane bez wymiaru 'z'": data_without_z
    }
elif num_features_per_landmark == 2:
    print("Dane zawierają tylko współrzędne 'x' i 'y'.")
    test_variants = {
        "Dane bez wymiaru 'z'": data
    }
else:
    print("Nieprawidłowy format danych. Kończę program.")
    exit()

# Lista klasyfikatorów do przetestowania
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Testowanie klasyfikatorów dla każdej wersji danych
for variant_name, variant_data in test_variants.items():
    print(f"\nTestowanie klasyfikatorów dla: {variant_name}")

    # Podział danych na treningowe i testowe
    x_train, x_test, y_train, y_test = train_test_split(variant_data, labels, test_size=0.2, shuffle=True, stratify=labels)

    results = {}
    for name, clf in classifiers.items():
        # Trening klasyfikatora
        clf.fit(x_train, y_train)

        # Predykcja na danych testowych
        y_predict = clf.predict(x_test)

        # Obliczanie metryk
        accuracy = accuracy_score(y_test, y_predict)
        print(f"\n{name}:")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(classification_report(y_test, y_predict))

        # Zapisywanie wyników
        results[name] = {
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_predict, output_dict=True)
        }

    # Wyświetlanie najlepszego klasyfikatora dla tej wersji danych
    best_model = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\nNajlepszy klasyfikator dla {variant_name}:")
    print(f"{best_model} z dokładnością {results[best_model]['accuracy'] * 100:.2f}%")
