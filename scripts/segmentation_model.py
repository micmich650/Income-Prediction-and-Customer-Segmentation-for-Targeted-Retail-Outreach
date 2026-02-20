#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA


def main():
    df = pd.read_csv('data/processed/cleaned_dataset.csv')


    df = df.drop(columns = [
        'Unnamed: 0',
        'label'
    ])

    categorical = [ #Create the list of categorical variables
        'class_of_worker_grouped',
        'education_grouped',
        'marital_stat_grouped',
        'major_industry_code',
        'major_occupation_code',
        'race',
        'sex',
        'ft_or_pt_grouped',
        'citizenship_grouped',
        'veterans_benefits',
        'has_financial_activity',
        'num_persons_worked_for_employer'
    ]

    numerical = [ #Create the list of numerical variables
        'age',
        'weeks_worked_in_year'
    ]

    ca = ColumnTransformer(transformers = [
        ("cat", OneHotEncoder(handle_unknown = 'ignore', sparse_output = False), categorical), # One hot encoder object, if there is an unknown value the function will raise an error
        ("num", StandardScaler(), numerical) # Standard Scaler object with the standard inputs
        ],
        remainder = 'drop'
    )

    data = ca.fit_transform(df)
    df = pd.DataFrame(data, columns = ca.get_feature_names_out())


    df_tst = df.sample(n = 15000, random_state = 42)

    pca = PCA(n_components = 12, random_state = 42)
    X_pca = pca.fit_transform(df_tst)
    kmeans = MiniBatchKMeans(n_clusters=3, init = 'k-means++', batch_size = 4500, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    pca_2 = PCA(n_components = 2, random_state = 42)
    X_pca_2 = pca_2.fit_transform(df_tst)
    df_pca = pd.DataFrame(X_pca_2, columns = ['pc1', 'pc2'])
    df_pca['cluster'] = labels

    X_full_pca = pca.transform(df)

    # Predict cluster for full dataset
    full_labels = kmeans.predict(X_full_pca)

    df['cluster'] = full_labels

    cluster_output = pd.DataFrame({
        "index": df.index,
        "cluster": full_labels
    })

    print("Saving Cluster Labels to outputs/segmentation_outputs/cluster_labels.csv")
    cluster_output.to_csv("outputs/segmentation_outputs/cluster_labels.csv", index = False)


    pd.set_option('display.max_columns', None)


    df_centroids = df.groupby('cluster').mean()

    print("Saving Cluster Averages to outputs/segmentation_outputs/cluster_averages.csv")
    df_centroids.to_csv("outputs/segmentation_outputs/cluster_averages.csv")

if __name__ == '__main__':
   main()
