import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Carica i dataset di training e test
print("\n Caricamento dataset...")
df_train = pd.read_csv("Datasets/dataset_train.csv")
df_test = pd.read_csv("Datasets/dataset_test.csv")

# Selezioniamo le feature per la previsione del prezzo
features = ['Mana Cost', 'Type Line', 'Rarity', 'Main/Sideboard', 'Quantity']
target = 'Price EUR'  # Puoi cambiare con 'Price USD'

# Separiamo le variabili categoriche e numeriche
categorical_columns = ['Mana Cost', 'Type Line', 'Rarity', 'Main/Sideboard']
numeric_columns = ['Quantity']

# One-Hot Encoding per variabili categoriche
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
df_train_encoded = encoder.fit_transform(df_train[categorical_columns])
df_train_encoded = pd.DataFrame(df_train_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_test_encoded = encoder.transform(df_test[categorical_columns])
df_test_encoded = pd.DataFrame(df_test_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Normalizziamo le feature numeriche
scaler = StandardScaler()
df_train_scaled = scaler.fit_transform(df_train[numeric_columns])
df_train_scaled = pd.DataFrame(df_train_scaled, columns=numeric_columns)

df_test_scaled = scaler.transform(df_test[numeric_columns])
df_test_scaled = pd.DataFrame(df_test_scaled, columns=numeric_columns)

# Creiamo il dataset finale per il modello
df_train_final = pd.concat([df_train_encoded, df_train_scaled, df_train[target]], axis=1)
df_test_final = pd.concat([df_test_encoded, df_test_scaled, df_test[target]], axis=1)

# Separiamo feature (X) e target (y)
X_train = df_train_final.drop(columns=[target])
y_train = df_train_final[target]

X_test = df_test_final.drop(columns=[target])
y_test = df_test_final[target]

# Creiamo e alleniamo il modello
print("\nAllenamento del modello...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Facciamo le previsioni
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Valutazione del modello
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("\n Valutazione del modello (Training):")
print(f" MAE: {mae_train:.4f}")
print(f" MSE: {mse_train:.4f}")
print(f" R² Score: {r2_train:.4f}")

print("\n Valutazione del modello (Testing):")
print(f" MAE: {mae_test:.4f}")
print(f" MSE: {mse_test:.4f}")
print(f" R² Score: {r2_test:.4f}")

# Grafici di confronto tra Training e Testing

plt.figure(figsize=(15, 6))

# Scatterplot: Prezzi reali vs Predetti (Training)
plt.subplot(2, 2, 1)
sns.scatterplot(x=y_train, y=y_train_pred, alpha=0.6)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--r')
plt.xlabel("Prezzo Reale")
plt.ylabel("Prezzo Predetto")
plt.title("Training - Prezzo Reale vs Predetto")

# Scatterplot: Prezzi reali vs Predetti (Testing)
plt.subplot(2, 2, 2)
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Prezzo Reale")
plt.ylabel("Prezzo Predetto")
plt.title("Testing - Prezzo Reale vs Predetto")

# Istogramma degli errori residui (Training)
plt.subplot(2, 2, 3)
sns.histplot(y_train - y_train_pred, bins=30, kde=True, color='blue')
plt.axvline((y_train - y_train_pred).mean(), color='red', linestyle='dashed')
plt.xlabel("Errore Residuo")
plt.title("Training - Distribuzione Errori Residui")

# Istogramma degli errori residui (Testing)
plt.subplot(2, 2, 4)
sns.histplot(y_test - y_test_pred, bins=30, kde=True, color='green')
plt.axvline((y_test - y_test_pred).mean(), color='red', linestyle='dashed')
plt.xlabel("Errore Residuo")
plt.title("Testing - Distribuzione Errori Residui")

plt.tight_layout()
plt.show()

# Salviamo il modello e gli encoder
joblib.dump(model, "Models/modello_regressione_prezzi.pkl")
joblib.dump(encoder, "Encoders/one_hot_encoder_prezzi.pkl")
joblib.dump(scaler, "Encoders/scaler_prezzi.pkl")
print("\nModello di regressione salvato con successo!")
