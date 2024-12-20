import pickle
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
import os

PICKLE_DIR = 'datasets_pickle'
CLASSIFIERS_DIR = 'photos_datasets/classifiers_autogluon'

# Upewnij się, że katalog na klasyfikatory istnieje
os.makedirs(CLASSIFIERS_DIR, exist_ok=True)

# Wyświetlenie dostępnych plików pickle
print("Dostępne pliki w katalogu './datasets_pickle':")
pickle_files = os.listdir(PICKLE_DIR)
for idx, file in enumerate(pickle_files, start=1):
    print(f"{idx}. {file}")

# Wybór pliku
try:
    selected_index = int(input("Wybierz numer pliku do użycia w AutoML: ")) - 1
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

# Przygotowanie danych w formacie Pandas DataFrame
df = pd.DataFrame(data)
df['label'] = labels  # Dodanie etykiet jako kolumny

# Podział na dane treningowe i testowe
train_data = df.sample(frac=0.8, random_state=42)
test_data = df.drop(train_data.index)

# Trening z AutoGluon
print("Rozpoczynam trening z AutoGluon...")
predictor = TabularPredictor(label='label', path='autogluon_models').fit(train_data)

# Testowanie najlepszego modelu
print("\nOcena na danych testowych:")
results = predictor.evaluate(test_data)

# Zapisanie najlepszego modelu
model_name = input("Podaj nazwę dla najlepszego modelu (bez rozszerzenia): ")
model_path = os.path.join(CLASSIFIERS_DIR, f"{model_name}.p")

with open(model_path, 'wb') as f:
    pickle.dump({'model': predictor, 'labels': list(set(labels))}, f)

print(f"Najlepszy model zapisano jako: {model_path}")
