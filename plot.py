import pickle
from importlib.metadata import distribution

import pandas as pd
import  seaborn as sns
import joblib
import matplotlib.pyplot as plt

# Caricamento pkl dati mancanti
with open("PlotData/missing_data.pkl", "rb") as f:
    missing_df = pickle.load(f)

# Plot distribuzione valori mancanti
missing_df.sort_values(by='Valori Mancanti').plot(
    kind='barh',
    stacked=True,
    color=['#ff7f0e', '#1f77b4'],
    title='Distribuzione Valori Mancanti per Colonna'
)

plt.xlabel('Conteggio')
plt.ylabel('Colonne')
plt.legend(loc='lower right')
plt.show()

# Caricamento pkl dati mancanti
with open("PlotData/oversampling.pkl", "rb") as f:
    entry_aggiunte = pickle.load(f)

# Crea il grafico a barre orizzontali
plt.figure(figsize=(10, 6))
entry_aggiunte.sort_values().plot(
    kind='barh',
    color='#2ca02c',
    title='Entry Aggiunte per Archetipo dopo Oversampling con ADASYN'
)
plt.xlabel('Numero di Entry Aggiunte')
plt.ylabel('Archetipo')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

with open("PlotData/distribution_post_oversampling.pkl","rb") as f:
    distribution_data = pickle.load(f)

# Generazione del grafico della distribuzione degli archetipi post-oversampling
plt.figure(figsize=(12, 6))
sns.barplot(x=distribution_data.index, y=distribution_data.values, palette='viridis')
plt.xticks(rotation=90)
plt.xlabel("Archetype")
plt.ylabel("Count")
plt.title("Distribuzione degli archetipi dopo il data cleaning e ADASYN")
plt.show()