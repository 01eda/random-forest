import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

# Veri klasörünüzün yolu
veri_klasoru = r'/content/sample_data/data'

actions = ['bas_parmak', 'isaret_parmak', 'orta_parmak', 'yuzuk_parmak', 'serce_parmak']
sensor_columns = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor6', 'sensor7', 'sensor8']

allDataFrame = pd.DataFrame()

for action in actions:
    for j in range(1, 31):
        file_path = os.path.join(veri_klasoru, f'{action}-{j}.csv')
        try:
            actionData = pd.read_csv(file_path)
            actionData = actionData[sensor_columns + ['label']]  # Sadece istediğimiz sütunları al
            allDataFrame = pd.concat([allDataFrame, actionData], ignore_index=True)
        except ValueError as e:
            print(f"Hata: {e}, Dosya: {file_path}")

X = allDataFrame[sensor_columns]
y = allDataFrame['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, y_train)

model_kayit_adi = "egitilen_model_random_forest_farklı_el_hareketleri.pkl"
joblib.dump(model, model_kayit_adi)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted')
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1 = f1_score(y_train, y_train_pred, average='weighted')

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

conf_matrix = confusion_matrix(y_test, y_test_pred)

# Sonuçları yazdırma
print("Eğitim Seti Sonuçları:")
print(f"Doğruluk (Accuracy): {train_accuracy:.4f}")
print(f"Kesinlik (Precision): {train_precision:.4f}")
print(f"Hassasiyet (Recall): {train_recall:.4f}")
print(f"F1-Skor: {train_f1:.4f}")

print("\nTest Seti Sonuçları:")
print(f"Doğruluk (Accuracy): {test_accuracy:.4f}")
print(f"Kesinlik (Precision): {test_precision:.4f}")
print(f"Hassasiyet (Recall): {test_recall:.4f}")
print(f"F1-Skor: {test_f1:.4f}")

print("\nKarışıklık Matrisi:")
print(conf_matrix)