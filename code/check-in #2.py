####################################################
# Check-In #2
# PRAD vs OV
# Hallmarks:
#   - Induce Angiogenesis
#   - Evade Growth Suppression
####################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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
