import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
import joblib

# Carica il dataset
df = pd.read_csv("Datasets/dataset_train.csv")

encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
categorical_columns = ['Main/Sideboard', 'Mana Cost', 'Type Line', 'Rarity']
df_encoded = encoder.fit_transform(df[categorical_columns])
df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out(categorical_columns))
df = df.drop(columns=categorical_columns)
df = pd.concat([df, df_encoded], axis=1)

label_encoder = LabelEncoder()
df['Archetype'] = label_encoder.fit_transform(df['Archetype'])

X = df.drop(columns=['Archetype'])
y = df['Archetype']

k = 10  # Numero di fold
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

y_true = []
y_preds = []

for train_idx, test_idx in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_fold, y_train_fold)
    y_pred_fold = model.predict(X_test_fold)

    y_true.extend(y_test_fold)
    y_preds.extend(y_pred_fold)

# Classification Report
y_true = np.array(y_true)
y_preds = np.array(y_preds)
report = classification_report(y_true, y_preds, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :].astype(float), annot=True, fmt=".2f", cmap="Blues", cbar=True)
plt.title("Report di Classificazione (Training Set)")
plt.xlabel("Metriche")
plt.ylabel("Classi")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_true, y_preds, labels=range(len(label_encoder.classes_)))
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix (Training Set)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()

# Salvataggio del modello
joblib.dump(model, "Models/modello_meta.pkl")
joblib.dump(encoder, "Encoders/one_hot_encoder.pkl")
joblib.dump(label_encoder, "Encoders/label_encoder_archetype.pkl")
print("\n✅ Modello e encoder salvati con successo!")