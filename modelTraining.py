import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import joblib

# 📌 1️⃣ Carica il dataset
df = pd.read_csv("dataset_ml.csv")
print(df['Archetype'].unique())

# 📌 2️⃣ Creare un encoder per le colonne categoriche
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Usa sparse_output=False

# 📌 3️⃣ Trasforma le colonne categoriche con l'encoder
categorical_columns = ['Main/Sideboard', 'Mana Cost', 'Type Line', 'Rarity']
df_encoded = encoder.fit_transform(df[categorical_columns])

# 📌 4️⃣ Converti il risultato in un DataFrame
df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# 📌 5️⃣ Sostituisci le colonne originali nel DataFrame con quelle codificate
df = df.drop(columns=categorical_columns)
df = pd.concat([df, df_encoded], axis=1)

# 📌 6️⃣ Codifica la colonna "Archetype" con LabelEncoder
label_encoder = LabelEncoder()
df['Archetype'] = label_encoder.fit_transform(df['Archetype'])  # Converti il testo in numeri

# 📌 7️⃣ Definiamo feature (X) e target (y)
X = df.drop(columns=['Archetype'])
y = df['Archetype']

# 📌 8️⃣ Suddividere in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 🔟 Creare e allenare il modello
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 1️⃣1️⃣ Salvare il modello e gli encoder
joblib.dump(model, "modello_meta.pkl")  # Salva il modello
joblib.dump(encoder, "one_hot_encoder.pkl")  # Salva l'encoder per le colonne categoriche
joblib.dump(label_encoder, "label_encoder_archetype.pkl")  # Salva il LabelEncoder per l'archetipo

print("✅ Modello e encoder salvati con successo!")
