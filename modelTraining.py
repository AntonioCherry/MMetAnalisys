import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
import joblib

# 📌 1️⃣ Carica il dataset
df = pd.read_csv("dataset_ml.csv")
print("\n📊 Archetipi unici nel dataset:")
print(df['Archetype'].value_counts())

# 📌 2️⃣ Creare un encoder per le colonne categoriche
encoder = OneHotEncoder(sparse_output=False, drop='first')

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
df['Archetype'] = label_encoder.fit_transform(df['Archetype'])

# 📌 7️⃣ Definiamo feature (X) e target (y)
X = df.drop(columns=['Archetype'])
y = df['Archetype']

# 📌 8️⃣ Suddividere in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 🔟 Creare e allenare il modello
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 1️⃣1️⃣ Valutazione del modello
y_pred = model.predict(X_test)

# 📌 🔹 Accuracy tradizionale
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy del modello: {accuracy:.4f}")

# 📌 🔹 Balanced Accuracy (più affidabile se il dataset è sbilanciato)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"⚖️ Balanced Accuracy: {balanced_acc:.4f}")

# 📌 🔹 Classification Report migliorato
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 📌 🔹 Confusion Matrix con Heatmap
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.title("🔀 Confusion Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# 📌 1️⃣2️⃣ Salvare il modello e gli encoder
joblib.dump(model, "modello_meta.pkl")
joblib.dump(encoder, "one_hot_encoder.pkl")
joblib.dump(label_encoder, "label_encoder_archetype.pkl")

print("\n✅ Modello e encoder salvati con successo!")
