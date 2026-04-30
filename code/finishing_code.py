####################################################
# Finalizing code
####################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)


# LOAD DATA


train_data = pd.read_csv(
    'data/TRAINING_SET_GSE62944_subsample_log2TPM.csv',
    index_col=0
)

train_metadata = pd.read_csv(
    'data/TRAINING_SET_GSE62944_metadata.csv',
    index_col=0
)

val_data = pd.read_csv(
    'data/VALIDATION_SET_GSE62944_subsample_log2TPM.csv',
    index_col=0
)

val_metadata = pd.read_csv(
    'data/VALIDATION_SET_GSE62944_metadata.csv',
    index_col=0
)

print("Train shape:", train_data.shape)
print("Validation shape:", val_data.shape)

# LOAD HALLMARKS

hallmark_df = pd.read_csv(
    'data/Menyhart_JPA_CancerHallmarks_core.txt',
    sep="\t",
    header=None
)

hallmark_dict = {}
for _, row in hallmark_df.iterrows():
    hallmark_dict[row[0]] = row[1:].dropna().tolist()

angiogenesis_genes_raw = set(hallmark_dict["SUSTAINED ANGIOGENESIS"])
growth_genes_raw = set(hallmark_dict["EVADING GROWTH SUPPRESSORS"])

print("\nRaw angiogenesis genes:", len(angiogenesis_genes_raw))
print("Raw growth suppression genes:", len(growth_genes_raw))

# FILTER CANCER TYPES

cancer_types = ['PRAD', 'OV']

train_samples = train_metadata[
    train_metadata['cancer_type'].isin(cancer_types)
].index

val_samples = val_metadata[
    val_metadata['cancer_type'].isin(cancer_types)
].index

train_data = train_data[train_samples]
train_metadata = train_metadata.loc[train_samples]

val_data = val_data[val_samples]
val_metadata = val_metadata.loc[val_samples]


# DEFINE GENE UNIVERSE 


gene_universe = set(train_data.index)

# FILTER HALLMARKS CONSISTENTLY

angiogenesis_genes = sorted(list(angiogenesis_genes_raw & gene_universe))
growth_genes = sorted(list(growth_genes_raw & gene_universe))

print("\nFiltered angiogenesis genes:", len(angiogenesis_genes))
print("Filtered growth suppression genes:", len(growth_genes))

print(
    "Ratio (angiogenesis/growth):",
    len(angiogenesis_genes) / max(len(growth_genes), 1)
)

# ALIGN GENES BETWEEN TRAIN AND VAL

common_genes = list(set(train_data.index).intersection(val_data.index))

print("\nCommon genes train/val:", len(common_genes))

train_data = train_data.loc[common_genes]
val_data = val_data.loc[common_genes]

# FUNCTION: SCORE COMPUTATION

def compute_scores(data, metadata):

    ang_valid = [g for g in angiogenesis_genes if g in data.index]
    gro_valid = [g for g in growth_genes if g in data.index]

    metadata = metadata.copy()

    metadata["Angiogenesis_Score"] = data.loc[ang_valid].mean()
    metadata["Growth_Suppression_Score"] = data.loc[gro_valid].mean()

    return metadata

train_metadata = compute_scores(train_data, train_metadata)
val_metadata = compute_scores(val_data, val_metadata)

# PCA GENE SET

gene_list = list(set(angiogenesis_genes + growth_genes))

print("\nGenes used in PCA:", len(gene_list))

X = train_data.loc[gene_list].T

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"],
    index=X.index
).merge(train_metadata, left_index=True, right_index=True)

plt.figure()
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="cancer_type")
plt.title("PCA (Training Set)")
plt.show()

# MODEL FEATURES

features = ["Angiogenesis_Score", "Growth_Suppression_Score"]

X_train = train_metadata[features]
y_train = train_metadata["cancer_type"].map({"PRAD": 0, "OV": 1})

X_val = val_metadata[features]
y_val = val_metadata["cancer_type"].map({"PRAD": 0, "OV": 1})

# SCALE MODEL INPUTS

scaler_model = StandardScaler()

X_train_scaled = scaler_model.fit_transform(X_train)
X_val_scaled = scaler_model.transform(X_val)

# TRAIN MODEL

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# PREDICTIONS

y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)
y_val_probs = model.predict_proba(X_val_scaled)[:, 1]

# METRICS

print("\n===== TRAINING =====")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred))
print("Recall:", recall_score(y_train, y_train_pred))
print("F1:", f1_score(y_train, y_train_pred))

print("\n===== VALIDATION =====")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Precision:", precision_score(y_val, y_val_pred))
print("Recall:", recall_score(y_val, y_val_pred))
print("F1:", f1_score(y_val, y_val_pred))

# ERROR

train_error = 1 - accuracy_score(y_train, y_train_pred)
val_error = 1 - accuracy_score(y_val, y_val_pred)

print("\nTraining Error:", train_error)
print("Validation Error:", val_error)

plt.figure()
plt.bar(["Train Error","Validation Error"], [train_error, val_error])
plt.ylabel("Error")
plt.title("In-sample vs Out-of-sample Error")
plt.show()

# CONFUSION MATRIX

cm = confusion_matrix(y_val, y_val_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

# ROC CURVE

fpr, tpr, _ = roc_curve(y_val, y_val_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.legend()
plt.title("ROC Curve")
plt.show()

# CLUSTERING (K-MEANS)

from sklearn.cluster import KMeans

# Clustering using your original parameters
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
pca_df["Cluster"] = clusters

# PCA Plot 4 — Clusters (Your original plotting style)
plt.figure()
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="Set2"
)
plt.title(
    "PCA Colored by K-Means Clusters"
)
plt.show()