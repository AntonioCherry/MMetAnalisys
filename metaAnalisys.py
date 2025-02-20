import pandas as pd

df = pd.read_csv("dataset.csv", sep=",")  # Usa sep=";" se i dati non si separano beneù
# Mostra tutte le colonne senza troncamenti
pd.set_option('display.max_columns', None)
print(df.head().to_string(index=False))

