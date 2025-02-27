import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Author: Antonio Cersuo
# Creation date: 23/02/2025

# Carica il dataset
df = pd.read_csv("dataset.csv", sep=",")
pd.set_option('display.max_columns', None)

# Rimozione colonne non necessarie
df.drop(columns=['Card Text', 'Date Posted', 'Mana Value', 'Colours'], inplace=True)

# Gestione valori mancanti
colonne_numeriche = df.select_dtypes(include=['float64', 'int64']).columns
df[colonne_numeriche] = df[colonne_numeriche].apply(lambda x: x.fillna(x.mean()), axis=0)

colonne_non_numeriche = df.select_dtypes(include=['object', 'category']).columns
df[colonne_non_numeriche] = df[colonne_non_numeriche].apply(lambda x: x.fillna(x.mode()[0]), axis=0)

# Rimuovere duplicati
df.drop_duplicates(inplace=True)

# 📌 Stampa ogni archetipo e il numero di entry associate
print("\n📊 Conteggio entry per ogni Archetipo prima dell'oversampling:")
print(df['Archetype'].value_counts())

# Identificare le classi con meno di 40 occorrenze
conteggio_archetipi = df['Archetype'].value_counts()
classi_da_oversamplare = conteggio_archetipi[conteggio_archetipi < 40].index.tolist()

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
adasyn = ADASYN(sampling_strategy={cls: 40 for cls in classi_da_oversamplare}, random_state=42, n_neighbors=5)
X_res, y_res = adasyn.fit_resample(X_train, y_train)

# Ricostruire il DataFrame con i dati sintetici
df_resampled = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.DataFrame(y_res, columns=['Archetype'])], axis=1)

# Unire il dataset sintetico con il test set originale
df_finale = pd.concat([df_resampled, pd.concat([X_test, y_test], axis=1)])

# Salva il dataset pulito e bilanciato
df_finale.to_csv("dataset_pulito.csv", index=False)

# 📌 Stampa il conteggio degli archetipi dopo l'oversampling
print("\n📊 Conteggio entry per ogni Archetipo dopo l'oversampling:")
print(df_finale['Archetype'].value_counts())

print("\n✅ Pulizia dati e oversampling con ADASYN completati!")