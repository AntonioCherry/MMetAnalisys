import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 📌 1️⃣ Carica il dataset
print("\n📊 Caricamento dataset...")
df = pd.read_csv("dataset_ml.csv")

# 📌 2️⃣ Selezioniamo le feature per la previsione del prezzo
features = ['Mana Cost', 'Type Line', 'Rarity', 'Main/Sideboard', 'Quantity']
target = 'Price EUR'  # Puoi cambiare con 'Price USD'

# 📌 3️⃣ Separiamo le variabili categoriche e numeriche
categorical_columns = ['Mana Cost', 'Type Line', 'Rarity', 'Main/Sideboard']
numeric_columns = ['Quantity']

# 📌 4️⃣ One-Hot Encoding per variabili categoriche
encoder = OneHotEncoder(sparse_output=False, drop='first')
df_encoded = encoder.fit_transform(df[categorical_columns])
df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# 📌 5️⃣ Normalizziamo le feature numeriche
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numeric_columns])
df_scaled = pd.DataFrame(df_scaled, columns=numeric_columns)

# 📌 6️⃣ Creiamo il dataset finale per il modello
df_final = pd.concat([df_encoded, df_scaled, df[target]], axis=1)

# 📌 7️⃣ Separiamo feature (X) e target (y)
X = df_final.drop(columns=[target])
y = df_final[target]

# 📌 8️⃣ Suddividiamo in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 🔟 Creiamo e alleniamo il modello
print("\n🚀 Allenamento del modello...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 1️⃣1️⃣ Facciamo le previsioni
y_pred = model.predict(X_test)

# 📌 1️⃣2️⃣ Valutiamo il modello
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Valutazione del modello:")
print(f"📉 MAE: {mae:.4f}")
print(f"📉 MSE: {mse:.4f}")
print(f"📈 R² Score: {r2:.4f}")

# 📌 1️⃣3️⃣ Grafico per confrontare valori reali vs. predetti
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Prezzo Reale")
plt.ylabel("Prezzo Predetto")
plt.title("📊 Confronto tra Prezzi Reali e Predetti")
plt.show()

# 📌 1️⃣4️⃣ Salviamo il modello e gli encoder
joblib.dump(model, "modello_regressione_prezzi.pkl")
joblib.dump(encoder, "one_hot_encoder_prezzi.pkl")
joblib.dump(scaler, "scaler_prezzi.pkl")
print("\n✅ Modello di regressione salvato con successo!")