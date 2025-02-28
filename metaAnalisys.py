import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.metrics import balanced_accuracy_score

# 📌 Carica i dati di training
df_train = pd.read_csv("dataset_ml.csv")

# 📌 Identifica le colonne categoriche e la colonna target
categorical_columns = ['Main/Sideboard', 'Mana Cost', 'Type Line', 'Rarity']
target_column = 'Archetype'

# 📌 Carica il LabelEncoder per Archetype
label_encoder = joblib.load("label_encoder_archetype.pkl")

# 📌 Codifica le colonne categoriche
encoder = joblib.load("one_hot_encoder.pkl")
df_encoded_train = encoder.transform(df_train[categorical_columns])
encoded_train_df = pd.DataFrame(df_encoded_train, columns=encoder.get_feature_names_out(categorical_columns))

# 📌 Prepara il dataset
X_train = df_train.drop(columns=categorical_columns + [target_column], errors='ignore')
X_train = pd.concat([X_train, encoded_train_df], axis=1)
y_train = label_encoder.transform(df_train[target_column])  # Trasforma il target in numerico

# 📌 Imposta il numero di fold
k = 10  # Puoi modificare il numero di fold
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# 📌 Inizializza il classificatore
classifier = RandomForestClassifier(random_state=42)

# 📌 Analizza la distribuzione delle classi nei fold
fold_distributions = []
y_preds = []
y_true = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train), start=1):
    X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
    y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]

    # Allena il modello
    classifier.fit(X_train_fold, y_train_fold)

    # Predizioni sui dati di test
    y_pred_fold = classifier.predict(X_test_fold)

    # Memorizza le etichette vere e le predizioni per la confusion matrix
    y_true.extend(y_test_fold)
    y_preds.extend(y_pred_fold)

    # Calcola la distribuzione delle classi nel fold
    class_counts = np.bincount(y_test_fold, minlength=len(label_encoder.classes_))
    fold_distributions.append(class_counts)

# 📌 Calcola e visualizza la confusion matrix
cm = confusion_matrix(y_true, y_preds, labels=range(len(label_encoder.classes_)))
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

# 📌 Stampa le etichette predette nel terminale
print("\n📊 Etichette predette per ogni fold:")
predicted_labels = label_encoder.inverse_transform(y_preds)
for i, label in enumerate(predicted_labels[:10]):  # Mostra solo le prime 10 per brevità
    print(f"Predizione {i+1}: {label}")

# 📌 Stampa la Confusion Matrix nel terminale
print("\n📊 Confusion Matrix:")
print(cm)

# 📌 Converti in DataFrame per la visualizzazione della distribuzione dei fold
fold_distribution_df = pd.DataFrame(fold_distributions, columns=label_encoder.classes_)
fold_distribution_df.index = [f"Fold {i+1}" for i in range(k)]

# 📌 Stampa la distribuzione delle classi nei vari fold
print("\n📊 Distribuzione delle classi in ciascun Fold:")
print(fold_distribution_df)

# 📌 Visualizza la distribuzione tramite heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(fold_distribution_df.T, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
plt.xlabel("Fold")
plt.ylabel("Archetype")
plt.title("Distribuzione delle classi nei diversi Fold (Stratified K-Fold)")
plt.show()

# 📌 Visualizza la Confusion Matrix
cm_display.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.xticks(rotation=45, ha='right')  # Ruota le etichette sull'asse x per evitare sovrapposizioni
plt.yticks(rotation=0)  # Mantieni le etichette sull'asse y orizzontali
plt.show()

# 📌 Calcola le metriche
accuracy = accuracy_score(y_true, y_preds)
recall = recall_score(y_true, y_preds, average='weighted')
precision = precision_score(y_true, y_preds, average='weighted')
f1 = f1_score(y_true, y_preds, average='weighted')

# 📌 Stampa le metriche
print("\n📊 Metriche di valutazione:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")

# 📌 Stampa il report di classificazione
print("\n📊 Report di Classificazione:")
print(classification_report(y_true, y_preds, target_names=label_encoder.classes_))

# 📌 Converti il report di classificazione in un DataFrame
report = classification_report(y_true, y_preds, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# 📌 Stampa il report di classificazione come tabella
print("\n📊 Report di Classificazione (Tabella):")
print(report_df)

# 📌 Visualizza il report di classificazione come heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :].astype(float), annot=True, fmt=".2f", cmap="Blues", cbar=True)
plt.title("Report di Classificazione (Heatmap)")
plt.xlabel("Metriche")
plt.ylabel("Classi")
plt.show()

balanced_accuracy = balanced_accuracy_score(y_true, y_preds)
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
