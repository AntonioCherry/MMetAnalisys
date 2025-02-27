import joblib
import pandas as pd

# Author: Antonio Cersuo
# Creation date: 23/02/2025

# 📌 1️⃣ Carica il modello e gli encoder salvati
model = joblib.load("modello_meta.pkl")
encoder = joblib.load("one_hot_encoder.pkl")
label_encoder = joblib.load("label_encoder_archetype.pkl")  # Carica il LabelEncoder di Archetype

# 📌 2️⃣ Carica i nuovi dati per le previsioni
future_data = pd.read_csv("dataset_ml.csv")

# 📌 3️⃣ Preprocessare i dati futuri come quelli di addestramento
categorical_columns = ['Main/Sideboard', 'Mana Cost', 'Type Line', 'Rarity']
for column in categorical_columns:
    future_data[column] = future_data[column].fillna('Unknown')  # Gestisci i valori mancanti

# 📌 4️⃣ Rimuovi la colonna "Archetype" se presente
future_data = future_data.drop(columns=['Archetype'], errors='ignore')

# 📌 5️⃣ Codifica le nuove colonne categoriche con l'encoder salvato
encoded_future_data = encoder.transform(future_data[categorical_columns])
encoded_future_df = pd.DataFrame(encoded_future_data, columns=encoder.get_feature_names_out(categorical_columns))

# 📌 6️⃣ Aggiungi le colonne codificate ai dati futuri e rimuovi le originali
future_data = pd.concat([future_data, encoded_future_df], axis=1)
future_data = future_data.drop(columns=categorical_columns)

# 📌 7️⃣ Aggiungi la colonna "Archetype_encoded" (per evitare il mismatch delle colonne)
# Se non hai dati per Archetype_encoded, puoi aggiungere una colonna con valori di default (es. 0)
future_data['Archetype_encoded'] = 0  # Aggiungi la colonna con valori fittizi

# 📌 8️⃣ Assicurati che future_data abbia le stesse colonne del training
df_train = pd.read_csv("dataset_ml.csv")
df_encoded_train = encoder.transform(df_train[categorical_columns])
encoded_train_df = pd.DataFrame(df_encoded_train, columns=encoder.get_feature_names_out(categorical_columns))

X_train = df_train.drop(columns=categorical_columns + ['Archetype'], errors='ignore')
X_train = pd.concat([X_train, encoded_train_df], axis=1)
training_columns = X_train.columns.tolist()

# Aggiungi le colonne mancanti con 0 e riordina
missing_cols = set(training_columns) - set(future_data.columns)
for col in missing_cols:
    future_data[col] = 0
future_data = future_data[training_columns]

# 📌 9️⃣ Fai le previsioni sui dati futuri
future_predictions = model.predict(future_data)

# 📌 🔟 Decodifica le previsioni numeriche in archetipi testuali
future_predictions_text = label_encoder.inverse_transform(future_predictions)

# 📌 🔟 Visualizza le previsioni
print("Etichette numeriche previste:", future_predictions)
print("Etichette testuali corrispondenti:", future_predictions_text)

# 📌 🔟 Visualizza le previsioni testuali
print("Previsioni sui dati futuri (in formato testuale):")
print(future_predictions_text)
