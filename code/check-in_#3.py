####################################################
# Check-In #2
# PRAD vs OV Classification Pipeline
####################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

data = pd.read_csv('data/TRAINING_SET_GSE62944_subsample_log2TPM.csv',index_col=0)
metadata_df = pd.read_csv('data/TRAINING_SET_GSE62944_metadata.csv',index_col=0)

print("Data shape:", data.shape)
print("Metadata shape:", metadata_df.shape)


hallmark_df = pd.read_csv('data/Menyhart_JPA_CancerHallmarks_core.txt',sep="\t",header=None)



# Converting hallmark table to dictionary

hallmark_dict = {}

for i, row in hallmark_df.iterrows():
    hallmark_name = row[0]
    genes = row[1:].dropna().tolist()
    hallmark_dict[hallmark_name] = genes


angiogenesis_genes = hallmark_dict["SUSTAINED ANGIOGENESIS"]
growth_suppression_genes = hallmark_dict["EVADING GROWTH SUPPRESSORS"]
combined_genes = (
    angiogenesis_genes +
    growth_suppression_genes
)

# cancer types

cancer_types = ['PRAD', 'OV']
selected_samples = metadata_df[metadata_df['cancer_type'].isin(cancer_types)].index
selected_data = data[selected_samples]
selected_metadata = metadata_df.loc[selected_samples]

# Keeping valid genes

gene_list = [
    gene for gene in combined_genes
    if gene in selected_data.index
]

hallmark_data = selected_data.loc[gene_list]
print("\nNumber of genes used:", len(gene_list))

# Calculating hallmark scores

angiogenesis_valid = [
    gene for gene in angiogenesis_genes
    if gene in selected_data.index
]

growth_valid = [
    gene for gene in growth_suppression_genes
    if gene in selected_data.index
]

angiogenesis_score = selected_data.loc[angiogenesis_valid].mean()
growth_score = selected_data.loc[growth_valid].mean()
selected_metadata["Angiogenesis_Score"] = angiogenesis_score
selected_metadata["Growth_Suppression_Score"] = growth_score

selected_metadata[
    ["Angiogenesis_Score",
     "Growth_Suppression_Score"]
].to_csv("hallmark_scores_PRAD_OV.csv")

# preparing PCA

X = hallmark_data.T
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Running PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Creating PCA dataframe
pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"],
    index=X.index
)
pca_df = pca_df.merge(
    selected_metadata,
    left_index=True,
    right_index=True
)

# PCA Plot 1 — Cancer Type
plt.figure()
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="cancer_type"
)
plt.title("PCA Colored by Cancer Type (PRAD vs OV)")
plt.show()

# PCA Plot 2 — Angiogenesis Score
plt.figure()

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Angiogenesis_Score",
    palette="viridis"
)
plt.title(
    "PCA Colored by Angiogenesis Score"
)
plt.show()

# Clustering
kmeans = KMeans(n_clusters=2,random_state=42)
clusters = kmeans.fit_predict(X_scaled)
pca_df["Cluster"] = clusters

# PCA Plot 3 — Growth Suppression Score

plt.figure()

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Growth_Suppression_Score",
    palette="magma"
)

plt.title(
    "PCA Colored by Growth Suppression Score"
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.show()

# PCA Plot 4 — Clusters
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


# SUPERVISED MODEL (LOGISTIC REGRESSION)

feature_cols = [
    "Angiogenesis_Score",
    "Growth_Suppression_Score"
]

X_model = selected_metadata[feature_cols]

y = selected_metadata["cancer_type"].map({
    "PRAD": 0,
    "OV": 1
})

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_model, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)


# PREDICTIONS

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

y_val_probs = model.predict_proba(X_val)[:, 1]

# METRICS


print("\n===== TRAINING (IN-SAMPLE) =====")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred))
print("Recall:", recall_score(y_train, y_train_pred))
print("F1:", f1_score(y_train, y_train_pred))

print("\n===== VALIDATION (OUT-OF-SAMPLE) =====")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Precision:", precision_score(y_val, y_val_pred))
print("Recall:", recall_score(y_val, y_val_pred))
print("F1:", f1_score(y_val, y_val_pred))


train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

train_error = 1 - train_accuracy
val_error = 1 - val_accuracy

print("\n===== IN-SAMPLE ERROR =====")
print("Training Error:", train_error)

print("\n===== OUT-OF-SAMPLE ERROR =====")
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
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC CURVE

fpr, tpr, _ = roc_curve(y_val, y_val_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="AUC = " + str(roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()