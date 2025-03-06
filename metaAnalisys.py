import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score, f1_score, classification_report, roc_curve, auc, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# 📌 1️⃣ Carica i dati di test
df_test = pd.read_csv("Datasets/dataset_test.csv")  # Assicurati di avere un file di test separato

# 📌 2️⃣ Identifica le colonne categoriche e la colonna target
categorical_columns = ['Main/Sideboard', 'Mana Cost', 'Type Line', 'Rarity']
target_column = 'Archetype'

# 📌 3️⃣ Carica il LabelEncoder per Archetype
label_encoder = joblib.load("Encoders/label_encoder_archetype.pkl")

# 📌 4️⃣ Carica l'OneHotEncoder per le colonne categoriche
encoder = joblib.load("Encoders/one_hot_encoder.pkl")

# 📌 5️⃣ Codifica le colonne categoriche nel dataset di test
df_encoded_test = encoder.transform(df_test[categorical_columns])
encoded_test_df = pd.DataFrame(df_encoded_test, columns=encoder.get_feature_names_out(categorical_columns))

# 📌 6️⃣ Prepara il dataset di test
X_test = df_test.drop(columns=categorical_columns + [target_column], errors='ignore')
X_test = pd.concat([X_test, encoded_test_df], axis=1)
y_test = label_encoder.transform(df_test[target_column])  # Trasforma il target in numerico

# 📌 7️⃣ Carica il modello RandomForest addestrato
model = joblib.load("Models/modello_meta.pkl")

# 📌 8️⃣ Fai predizioni sul dataset di test
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 📌 9️⃣ Calcola e visualizza la confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_)))
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
cm_display.plot(cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()

# 📌 🔟 Calcola le metriche
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

# 📌 Stampa le metriche
print("\n📊 Metriche di valutazione sul test set:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

# 📌 Report di Classificazione
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# 📌 Visualizza il report di classificazione come heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :].astype(float), annot=True, fmt=".2f", cmap="Blues", cbar=True)
plt.title("Report di Classificazione")
plt.xlabel("Metriche")
plt.ylabel("Classi")
plt.show()

# 📌 Generazione della ROC Curve multi-classe con miglior scaling
y_test_bin = label_binarize(y_test, classes=range(len(label_encoder.classes_)))
n_classes = y_test_bin.shape[1]

# Calcola la macro-media delle curve ROC
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calcola la macro-media delle curve ROC
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes

# Aggiungi manualmente il punto (0, 0) all'inizio della curva
all_fpr = np.insert(all_fpr, 0, 0.0)
mean_tpr = np.insert(mean_tpr, 0, 0.0)

# Calcola l'AUC della macro-media
macro_auc = auc(all_fpr, mean_tpr)

# 📌 Plot della singola curva ROC
plt.figure(figsize=(10, 8))
plt.plot(all_fpr, mean_tpr, color="blue", lw=2, label=f"Macro-Average ROC Curve (AUC = {macro_auc:.2f})")

# 📌 Linea diagonale di riferimento
plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6)

# 📌 Miglioramenti al grafico
plt.xlim([-0.02, 1.02])  # Un piccolo margine a sinistra e a destra
plt.ylim([-0.02, 1.1])  # Evita che la curva tocchi il bordo superiore
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("Singola Curva ROC (Macro-Average)", fontsize=14)
plt.legend(loc="lower right", fontsize=12, frameon=True)  # Legenda
plt.grid(True, linestyle="--", alpha=0.5)

# 📌 Mostra il grafico
plt.show()

# 📌 Generazione della ROC Curve multi-classe con miglior scaling
y_test_bin = label_binarize(y_test, classes=range(len(label_encoder.classes_)))
n_classes = y_test_bin.shape[1]
colors = sns.color_palette("Set2", n_classes)  # Palette chiara e leggibile

plt.figure(figsize=(12, 8))

for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, alpha=0.8, label=f"{label_encoder.classes_[i]} (AUC = {roc_auc:.2f})")

# 📌 Linea diagonale di riferimento
plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6)

# 📌 Miglioramenti al grafico
plt.xlim([-0.02, 1.02])  # Un piccolo margine a sinistra e a destra
plt.ylim([-0.02, 1.1])  # Evita che la curva tocchi il bordo superiore
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("Curva ROC Multi-Classe", fontsize=14)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, frameon=True)  # Legenda fuori dal grafico
plt.grid(True, linestyle="--", alpha=0.5)

# 📌 Mostra il grafico
plt.show()

