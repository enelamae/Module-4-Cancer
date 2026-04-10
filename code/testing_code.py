# Exploratory data analysis (EDA) on a cancer dataset
# Loading the files and exploring the data with pandas
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
# Load the data
####################################################
data = pd.read_csv(
    'data/TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0, header=0)  # can also use larger dataset with more genes
metadata_df = pd.read_csv(
    'data/TRAINING_SET_GSE62944_metadata.csv', index_col=0, header=0)
print(data.head())

# %%
# Explore the data
####################################################
print(data.shape)
print(data.info())
print(data.describe())

# %%
# Explore the metadata
####################################################
print(metadata_df.info())
print(metadata_df.describe())

# %%
# Subset the data for a specific cancer type
####################################################
cancer_types = ['PRAD', 'OV']  # Breast Invasive Carcinoma

# From metadata, get the rows where "cancer_type" is equal to the specified cancer type
# Then grab the index of this subset (these are the sample IDs)
selected_samples = metadata_df[ metadata_df['cancer_type'].isin(cancer_types)].index
print(selected_samples) 
# Subset the main data to include only these samples
# When you want a subset of columns, you can pass a list of column names to the data frame in []
selected_data = data[selected_samples]
selected_metadata = metadata_df.loc[selected_samples]
# %%
# Subset by index (genes)
####################################################
# Induce Angiogenesis genes
angiogenesis_genes = ['VEGFA', 'HIF1A', 'ANGPT1', 'ANGPT2', 'FLT1']
# Evade Growth Suppression genes
growth_suppression_genes = ['TP53', 'RB1', 'CDKN2A', 'PTEN', 'SMAD4']
desired_gene_list = angiogenesis_genes + growth_suppression_genes
gene_list = [gene for gene in desired_gene_list if gene in selected_data.index]
hallmark_data = selected_data.loc[gene_list]
for gene in desired_gene_list:
    if gene not in gene_list:
        print(f"Warning: {gene} not found in the dataset.")

# .loc[] is the method to subset by index labels
# .iloc[] will subset by index position (integer location) instead
print(hallmark_data.head())

# %%
# Basic statistics on the subsetted data
####################################################
print(hallmark_data.describe())
print(hallmark_data.var(axis=1))  # Variance of each gene across samples
# Mean expression of each gene across samples
print(hallmark_data.mean(axis=1))
# Median expression of each gene across samples
print(hallmark_data.median(axis=1))

# %%
# Explore categorical variables in metadata
####################################################
# groupby allows you to group on a specific column in the dataset,
# and then print out summary stats or counts for other columns within those groups
print(metadata_df.groupby('cancer_type')["gender"].value_counts())

# Explore average age at diagnosis by cancer type
metadata_df['age_at_diagnosis'] = pd.to_numeric(
    metadata_df['age_at_diagnosis'], errors='coerce')
print(metadata_df.groupby(
    'cancer_type')["age_at_diagnosis"].mean())
# %%
# Merging datasets
####################################################
merged_data = hallmark_data.T.merge(
    selected_metadata,
    left_index=True,
    right_index=True
)

print("\nMerged Data Preview:")
print(merged_data.head())

# %%
# Calculate hallmark scores
####################################################
merged_data['Angiogenesis_Score'] = merged_data[
    [g for g in angiogenesis_genes if g in merged_data.columns]
].mean(axis=1)

merged_data['Growth_Suppression_Score'] = merged_data[
    [g for g in growth_suppression_genes if g in merged_data.columns]
].mean(axis=1)

# %%
# Summary statistics
####################################################
print("\nAverage Hallmark Scores by Cancer Type:")
print(
    merged_data.groupby('cancer_type')[
        ['Angiogenesis_Score', 'Growth_Suppression_Score']
    ].mean()
)

# %%
# Plotting
####################################################
# Plot 1: VEGFA comparison
####################################################
sns.boxplot(data=merged_data, x='cancer_type', y='VEGFA')
plt.title("VEGFA Expression: PRAD vs OV")
plt.show()

# %%
# Plot 2: TP53 comparison
####################################################
sns.boxplot(data=merged_data, x='cancer_type', y='TP53')
plt.title("TP53 Expression: PRAD vs OV")
plt.show()

# %%
# Plot 3: Angiogenesis hallmark score
####################################################
sns.boxplot(data=merged_data, x='cancer_type', y='Angiogenesis_Score')
plt.title("Angiogenesis Hallmark Score: PRAD vs OV")
plt.show()

# %%
# Plot 4: Growth suppression hallmark score
####################################################
sns.boxplot(data=merged_data, x='cancer_type', y='Growth_Suppression_Score')
plt.title("Growth Suppression Hallmark Score: PRAD vs OV")
plt.show()

# %%
