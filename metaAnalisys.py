import pandas as pd

df = pd.read_csv("dataset.csv", sep=",")  # Usa sep=";" se i dati non si separano beneù
pd.set_option('display.max_columns', None)

# Rimozione colonna Card Text
df.drop(columns=['Card Text'], inplace=True)

# Visualizza le prime righe del dataset
print(df.head().to_string(index=False))

# Selezionare solo le colonne numeriche
colonne_numeriche = df.select_dtypes(include=['float64', 'int64']).columns

# Imputare i valori mancanti con la media per ogni colonna numerica
df[colonne_numeriche] = df[colonne_numeriche].apply(lambda x: x.fillna(x.mean()), axis=0)

# Selezionare solo le colonne non numeriche (di tipo 'object' o 'category')
colonne_non_numeriche = df.select_dtypes(include=['object', 'category']).columns

# Imputare i valori mancanti con la moda per ogni colonna non numerica
df[colonne_non_numeriche] = df[colonne_non_numeriche].apply(lambda x: x.fillna(x.mode()[0]), axis=0)

# Stampare il DataFrame per verificare l'imputazione
print(df.head())

# Identificare le colonne con valori mancanti
colonne_con_mancanti = df.columns[df.isna().sum() > 0]

# Stampare il nome delle colonne con valori mancanti
print("Colonne con valori mancanti:")
print(colonne_con_mancanti)