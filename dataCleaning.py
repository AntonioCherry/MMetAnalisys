import pandas as pd
import matplotlib.pyplot as plt  # Correzione dell'import
import seaborn as sns
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Carica il dataset
df = pd.read_csv("dataset.csv", sep=",")
pd.set_option('display.max_columns', None)

# Rimozione colonne non necessarie
df.drop(columns=['Card Text', 'Date Posted', 'Mana Value', 'Colours'], inplace=True)

# 1. Visualizzazione valori mancanti iniziali
missing = df.isna().sum()
present = len(df) - missing
missing_df = pd.DataFrame({'Valori Mancanti': missing, 'Valori Presenti': present})
missing_df.sort_values(by='Valori Mancanti').plot(
    kind='barh',
    stacked=True,
    color=['#ff7f0e', '#1f77b4'],
    title='Distribuzione Valori Mancanti per Colonnina'
)
plt.xlabel('Conteggio')
plt.ylabel('Colonne')
plt.legend(loc='lower right')
plt.show()

# Gestione valori mancanti
colonne_numeriche = df.select_dtypes(include=['float64', 'int64']).columns
df[colonne_numeriche] = df[colonne_numeriche].apply(lambda x: x.fillna(x.mean()), axis=0)

colonne_non_numeriche = df.select_dtypes(include=['object', 'category']).columns
df[colonne_non_numeriche] = df[colonne_non_numeriche].apply(lambda x: x.fillna(x.mode()[0]), axis=0)

# 2. Calcolo percentuale duplicati
duplicati_percent = (df.duplicated().sum() / len(df)) * 100
print(f"\nPercentuale di duplicati nel dataset: {duplicati_percent:.2f}%\n")

# Rimuovere duplicati
df.drop_duplicates(inplace=True)

# 📌 Stampa ogni archetipo e il numero di entry associate
print("\n📊 Conteggio entry per ogni Archetipo prima dell'oversampling:")
print(df['Archetype'].value_counts())

# Identificare le classi con meno di 40 occorrenze
conteggio_archetipi = df['Archetype'].value_counts()
classi_da_oversamplare = conteggio_archetipi[conteggio_archetipi < 50].index.tolist()

# Identifica le colonne categoriche
colonne_categoriche = df.select_dtypes(include=['object', 'category']).columns.tolist()
colonne_categoriche.remove('Archetype')  # Rimuovere la colonna target

# Convertire le feature categoriche in numerico
for col in colonne_categoriche:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Separazione feature e target
X = df.drop(columns=['Archetype'])
y = df['Archetype']

# Divisione train-test (ADASYN deve essere applicato solo sul train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Applicare ADASYN solo alle classi selezionate
adasyn = ADASYN(sampling_strategy={cls: 50 for cls in classi_da_oversamplare}, random_state=42, n_neighbors=5)
X_res, y_res = adasyn.fit_resample(X_train, y_train)

# Ricostruire il DataFrame con i dati sintetici
df_resampled = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.DataFrame(y_res, columns=['Archetype'])], axis=1)

# Unire il dataset sintetico con il test set originale
df_finale = pd.concat([df_resampled, pd.concat([X_test, y_test], axis=1)])

# Salva il dataset pulito e bilanciato
df_finale.to_csv("dataset_pulito.csv", index=False)

# Calcola il numero di entry aggiunte per ogni archetipo
conteggio_pre_oversampling = y_train.value_counts()
conteggio_post_oversampling = df_finale['Archetype'].value_counts()

# Inizializza un dizionario per memorizzare le entry aggiunte
entry_aggiunte = {}

# Calcola le entry aggiunte solo per le classi oversampliate
for cls in classi_da_oversamplare:
    entry_aggiunte[cls] = conteggio_post_oversampling[cls] - conteggio_pre_oversampling[cls]

# Per le altre classi, imposta le entry aggiunte a 0
for cls in conteggio_post_oversampling.index:
    if cls not in classi_da_oversamplare:
        entry_aggiunte[cls] = 0

# Converti il dizionario in una Serie per il plotting
entry_aggiunte = pd.Series(entry_aggiunte)

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

print("\n✅ Pulizia dati e oversampling con ADASYN completati!")


# Generazione del grafico della distribuzione degli archetipi post-oversampling
plt.figure(figsize=(12, 6))
sns.barplot(x=df_finale['Archetype'].value_counts().index, y=df_finale['Archetype'].value_counts().values, palette='viridis')
plt.xticks(rotation=90)
plt.xlabel("Archetype")
plt.ylabel("Count")
plt.title("Distribuzione degli archetipi dopo il data cleaning e ADASYN")
plt.show()