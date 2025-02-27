import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Author: Antonio Cersuo
# Creation date: 23/02/2025

# 📌 1️⃣ Carica il dataset pulito
df = pd.read_csv("dataset_pulito.csv")

# 📌 2️⃣ Codificare le variabili categoriche
label_encoders = {}
categorical_columns = ['Card', 'Pilot', 'Event', 'Most Recent Printing', 'Rarity']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Salviamo il LabelEncoder per decodificare più avanti

# 📌 3️⃣ Salva il LabelEncoder di "Archetype" separatamente
# Codifica solo Archetype per l'uso futuro ma mantieni la colonna come testo nel dataset
archetype_encoder = LabelEncoder()
df['Archetype_encoded'] = archetype_encoder.fit_transform(df['Archetype'])

# Salva l'encoder per Archetype
joblib.dump(archetype_encoder, "label_encoder_archetype.pkl")
print("✅ LabelEncoder di 'Archetype' salvato correttamente!")

# 📌 4️⃣ Normalizzare le colonne numeriche
scaler = StandardScaler()
numeric_columns = ['Quantity', 'Price EUR', 'Price USD']
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# 📌 5️⃣ Salva il dataset pronto per il Machine Learning
df.to_csv("dataset_ml.csv", index=False)
print("✅ Preparazione dati completata!")
