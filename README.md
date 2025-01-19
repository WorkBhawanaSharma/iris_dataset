Iris Dataset Project

This project involves the analysis of the famous Iris dataset using Python. The Iris dataset is widely used in machine learning and statistical analysis and contains information about different species of Iris flowers, with measurements of various flower characteristics.

Overview

The Iris dataset consists of four numerical features:
Sepal Length
Sepal Width
Petal Length
Petal Width

It also includes a categorical target feature, species, which classifies each flower as one of three species:
Setosa
Versicolor
Virginica

This project includes various steps of data exploration, visualization, and machine learning to better understand the dataset and the relationships between the features.

Steps Included in the Project

1. Dataset Exploration
   
->Loaded the Iris dataset and converted it into a DataFrame for easy analysis.

->Explored the dataset's basic characteristics including:

->Descriptive statistics (mean, median, standard deviation, etc.).

->Checking for missing values.

->Cleaning the data by handling duplicates.

->Basic analysis of numerical and categorical features.

2. Data Visualization

->Histograms: Displayed the distribution of features across the dataset.

->Scatter Plots: Investigated the relationships between petal length, petal width, and other features.

->Box Plots: Visualized the distribution of features across species.

->Pairplots: Provided insight into the correlations between all features.

->Violin Plots: Showed feature distributions with an emphasis on differences between species.

3. Enhanced Data Analysis (EDA)

->Computed the correlation matrix to check feature relationships.

->Used heatmaps and pairplots for visual exploration.

->Detected and analyzed outliers in the dataset using z-scores and boxplots.

4. Clustering Insights

->Applied K-means clustering to explore unsupervised learning techniques.

->Validated clustering results by comparing predicted clusters with the actual species.

->Cluster analysis reinforced the idea that petal features are crucial for distinguishing between species.

5. Feature Importance

->Used a Random Forest Classifier to evaluate feature importance, highlighting the critical role of petal length and width in species classification.

6. Interactive Visualizations

->Created an interactive scatter matrix using Plotly to visualize relationships among features.
Technologies Used

->Python (Jupyter Notebook)

->Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, plotly

->Machine Learning: K-Means, Random Forest Classifier

#Key Findings

Petal Features: Petal length and petal width are key differentiators between species.

Clustering: K-means clustering correctly grouped the dataset into three clusters, closely aligning with actual species.

Visualization: Data visualizations clearly showed the separability of Setosa and the overlap between Versicolor and Virginica, especially for the sepal features.
