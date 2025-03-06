import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Author: Antonio Cersuo
# Creation date: 23/02/2025

# 📌 1️⃣ Carica il dataset pulito
df = pd.read_csv("Datasets/dataset_pulito.csv")

# 📌 2️⃣ Codificare le variabili categoriche
label_encoders = {}
categorical_columns = ['Card', 'Pilot', 'Event', 'Rarity']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Salviamo il LabelEncoder per decodificare più avanti

# 📌 3️⃣ Salva il LabelEncoder di "Archetype" separatamente
# Codifica solo Archetype per l'uso futuro ma mantieni la colonna come testo nel dataset
archetype_encoder = LabelEncoder()
df['Archetype_encoded'] = archetype_encoder.fit_transform(df['Archetype'])

# Salva l'encoder per Archetype
joblib.dump(archetype_encoder, "Encoders/label_encoder_archetype.pkl")
print("✅ LabelEncoder di 'Archetype' salvato correttamente!")

# 📌 4️⃣ Normalizzare le colonne numeriche
scaler = StandardScaler()
numeric_columns = ['Quantity', 'Price EUR', 'Price USD']
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# 📌 5️⃣ Split del dataset in training e testing (80/20)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# 📌 6️⃣ Salva i dataset splittati
df_train.to_csv("Datasets/dataset_train.csv", index=False)
df_test.to_csv("Datasets/dataset_test.csv", index=False)

print("✅ Dataset splittato in training e testing e salvato con successo!")
