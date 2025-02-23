import pandas as pd

# Carica il dataset
df = pd.read_csv("dataset.csv", sep=",")
pd.set_option('display.max_columns', None)

# Rimozione colonna "Card Text"
df.drop(columns=['Card Text'], inplace=True)
# Rimozione colonna "Date Posted"
df.drop(columns=['Date Posted'], inplace=True)
# Rimozione colonna "Date Posted"
df.drop(columns=['Mana Value'], inplace=True)
# Rimozione colonna "Date Posted"
df.drop(columns=['Colours'], inplace=True)
#

# Gestione valori mancanti
colonne_numeriche = df.select_dtypes(include=['float64', 'int64']).columns
df[colonne_numeriche] = df[colonne_numeriche].apply(lambda x: x.fillna(x.mean()), axis=0)

colonne_non_numeriche = df.select_dtypes(include=['object', 'category']).columns
df[colonne_non_numeriche] = df[colonne_non_numeriche].apply(lambda x: x.fillna(x.mode()[0]), axis=0)

# Rimuovere duplicati
df.drop_duplicates(inplace=True)

# Salva il dataset pulito
df.to_csv("dataset_pulito.csv", index=False)
print("✅ Pulizia dati completata!")
