#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing neccessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


# In[2]:


# Loading the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names) # Creates a table with feature data as columns
df['species'] = iris.target  # Add species as a column
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})  # Map target to species names


# In[3]:


# Exploring the dataset
print("Dataset preview:")
print(df.head())  # First few rows
print("\nBasic info about the dataset:")
print(df.info())  # Data types, non-null counts
print("\nMissing values:")
print(df.isnull().sum())  # Checking for missing data


# In[4]:


# Cleaning and preprocessing (handle duplicates or missing data)
print("\nDuplicate rows:")
duplicates = df.duplicated().sum()
print(f"Number of duplicates: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)


# In[5]:


# Generate basic descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())


# In[6]:


# Additional stats: median and standard deviation for numerical columns
print("\nAdditional statistics:")
for col in df.columns[:-1]:  # Skip the species column
    print(f"{col}:")
    print(f"  Median: {df[col].median()}")
    print(f"  Standard Deviation: {df[col].std()}")


# In[7]:


# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Filters numeric columns

# Compute correlation matrix
correlation_matrix = numeric_df.corr()

# Plot correlation heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))  # Set the figure size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')  # Heatmap with annotations
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[8]:


# Group summary statistics
species_summary = df.groupby('species').describe()
print(species_summary)


# In[9]:


# Task summary
print("\nDataset exploration and cleaning completed successfully.")


# In[10]:


# Importing libraries for visualization
import matplotlib.pyplot as plt  # For basic plotting
import seaborn as sns  # For advanced visualizations


# In[11]:


# Load the Iris dataset as a DataFrame
from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)  # Creating DataFrame from Iris data
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)  # Adding species as a column


# In[12]:


# 1. Histogram for each feature
plt.figure(figsize=(10, 6))  # Set the figure size for better readability
df.hist(bins=20, figsize=(10, 8), color='skyblue', edgecolor='black')  # Plot histograms for all numeric columns
plt.suptitle('Histograms of Features', fontsize=16)  # Add a title to the entire figure
plt.show()  # Display the plot


# In[13]:


# 2. Scatter plot between petal length and petal width
plt.figure(figsize=(8, 6))  # Set the figure size
sns.scatterplot(x=df['petal length (cm)'], y=df['petal width (cm)'], hue=df['species'], palette='viridis')  # Plot scatterplot
plt.title('Petal Length vs Petal Width')  # Add a title
plt.xlabel('Petal Length (cm)')  # Label for x-axis
plt.ylabel('Petal Width (cm)')  # Label for y-axis
plt.legend(title='Species')  # Add a legend with a title
plt.show()  # Display the plot


# In[14]:


# Box plot for feature distribution by species
plt.figure(figsize=(10, 6))  # Set the figure size
sns.boxplot(data=df, x='species', y='sepal length (cm)', palette='pastel')  # Plot boxplot for sepal length by species
plt.title('Box Plot of Sepal Length by Species')  # Add a title
plt.xlabel('Species')  # Label for x-axis
plt.ylabel('Sepal Length (cm)')  # Label for y-axis
plt.show()  # Display the plot


# In[15]:


sns.pairplot(df, hue='species', diag_kind='kde', palette='husl')
plt.show()


# In[16]:


sns.violinplot(data=df, x='species', y='sepal length (cm)', palette='muted')
plt.title('Violin Plot of Sepal Length by Species')
plt.show()


# In[17]:


# 1. Correlation Matrix
correlation_matrix = df.iloc[:, :-1].corr()  # Compute correlation matrix (ignoring 'species' column)
print("Correlation Matrix:")
print(correlation_matrix)  # Display the correlation matrix


# In[18]:


# Heatmap for correlation matrix
plt.figure(figsize=(8, 6))  # Set figure size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')  # Plot heatmap with annotations
plt.title('Correlation Heatmap')  # Add title
plt.show()  # Display the heatmap


# In[19]:


# 2. Pairplot (Relationships between variables)
sns.pairplot(df, hue='species', palette='husl', diag_kind='kde')  # Pairplot with kernel density estimation on diagonal
plt.suptitle('Pairplot of Features Colored by Species', y=1.02, fontsize=16)  # Add title above the plots
plt.show()  # Display the pairplot


# In[20]:


# 3. Detecting Outliers using Box Plots
plt.figure(figsize=(10, 8))  # Set figure size
for i, column in enumerate(df.columns[:-1], 1):  # Iterate through feature columns (ignoring 'species')
    plt.subplot(2, 2, i)  # Create 2x2 subplot grid
    sns.boxplot(data=df, y=column, x='species', palette='pastel')  # Box plot for each feature
    plt.title(f'Boxplot of {column} by Species')  # Title for each subplot
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()  # Display all boxplots


# In[21]:


# Calculate Z-scores
from scipy.stats import zscore
z_scores = np.abs(zscore(df.iloc[:, :-1]))  # Exclude species column
outliers = (z_scores > 3).any(axis=1)  # Threshold for outlier detection
print(f"Number of outliers: {outliers.sum()}")


# In[22]:


from sklearn.ensemble import RandomForestClassifier
X = df.iloc[:, :-1]  # Features
y = df['species']  # Target
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
print(feature_importance.sort_values(by='Importance', ascending=False))


# In[23]:


# Step 4: Project Presentation Enhancements

# 1. Interactive Visualizations
import plotly.express as px  # Import Plotly for interactive visualizations
fig = px.scatter_matrix(
    df, 
    dimensions=df.columns[:-1],  # Exclude species column
    color='species', 
    title='Interactive Scatter Matrix'
)
fig.show()


# In[24]:


# 2. Correlation Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = df.iloc[:, :-1].corr()  # Exclude species column for numeric correlation
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()


# In[25]:


# 3. Clustering: Unsupervised learning with K-means
from sklearn.cluster import KMeans  # Importing KMeans for clustering


# In[26]:


# Select only numeric columns (excluding 'species')
X = df.drop('species', axis=1)  # Drop the 'species' column to focus only on features


# In[27]:


# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Set number of clusters to 3
df['cluster'] = kmeans.fit_predict(X)  # Perform clustering on numeric feature data



# In[28]:


# Compare actual species and predicted clusters
print("Comparison of actual species and predicted clusters:")
print(df[['species', 'cluster']].head())

