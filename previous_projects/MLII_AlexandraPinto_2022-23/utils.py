import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import ast
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import seaborn as sns


#Function that will be used in 4.1.3. Missing values
def missing_reporter(df, use_inf_as_na=True):
    """
    Generates a report of missing values in a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame to analyze.
        use_inf_as_na (bool, optional): Whether to treat infinity values as missing values. Defaults to True.

    Returns:
        DataFrame or None: If there are missing values, returns a DataFrame containing the count and percentage of missing values for each column. If there are no missing values, prints a message indicating so and returns None.
    """
    pd.options.mode.use_inf_as_na = use_inf_as_na
    df_na = df.isna().sum()
    features_na = df_na[df_na > 0]
    if df_na[0] == 0:
        print('There are no missing values!')
        return
    else:
        df_na = pd.DataFrame.from_dict(
            {"Nmissings": features_na,  # Nmissings is the total number of missing values.
             "Pmissings": features_na.divide(df.shape[0])})  # Pmissings is the percentage of null values that exist in the column.
        return df_na

    
#Function that will be used in 4.5. Correlation Matrix
def correlation_heatmap(cor):
    """
    Generate a correlation heatmap based on the provided correlation matrix.

    Parameters:
    cor (numpy.ndarray): Correlation matrix to visualize.

    Returns:
    None (displays the correlation heatmap)

    """

    # Set the figure size
    plt.figure(figsize=(40, 40))

    # Create a mask to hide the upper triangular part of the heatmap
    mask = np.triu(np.ones_like(cor, dtype=bool))

    # Generate the heatmap with annotations
    heatmap = sns.heatmap(data=cor, annot=True, alpha=0.7, mask=mask, cmap=plt.cm.Oranges, fmt='.1',
                          annot_kws={"size": 26})

    # Set the font size of x-axis and y-axis tick labels
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=30)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=30)

    # Display the heatmap
    plt.show()

    
    
#Function that will be used in 5.2. K-Means 
def compute_silhouette_scores(X, range_n_clusters):
    """
    Computes the silhouette scores for a range of cluster numbers using the KMeans algorithm.

    Parameters:
        X (array-like or sparse matrix): The input data.
        range_n_clusters (iterable): The range of cluster numbers to consider.

    Returns:
        list: A list of silhouette scores for each cluster number in the range.

    Notes:
        The silhouette score measures the compactness and separation of clusters. Higher scores indicate better-defined clusters.
    """
    silhouette_scores = []

    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters, "The average silhouette_score is:", silhouette_avg)
        silhouette_scores.append(silhouette_avg)

    return silhouette_scores


#Function that will be used in 6.1. T-SNE and 6.2.UMAP  - Taken from practical classes
def visualize_dimensionality_reduction(transformation, targets):
    """
    Visualizes the dimensionality reduction output using a scatter plot.

    Args:
        transformation (ndarray): The transformed data with reduced dimensions.
        targets (ndarray): The class labels or targets corresponding to each data point.

    Returns:
        None
    """
    # Get unique class labels and corresponding colors
    unique_labels = np.unique(targets)
    num_labels = len(unique_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, num_labels))

    # Create a scatter plot of the transformation output
    plt.scatter(transformation[:, 0], transformation[:, 1], 
                c=targets.astype(int), cmap=plt.cm.tab10)

    # Create a legend with the class labels and colors
    handles = [plt.scatter([],[], c=colors[i], label=label) for i, label in enumerate(unique_labels)]
    plt.legend(handles=handles, title='Classes')

    plt.show()
    
    
    
#Function that will be used in 7.1. Exploring Cluster Profiles
def calculate_cluster_statistics(data, features_categorical, features_numerical, cluster):
    """
    Calculates the mode and median values for each cluster based on the provided features.

    Args:
        data (DataFrame): The dataframe containing cluster data.
        features_categorical (list): A list of categorical features.
        features_numerical (list): A list of numerical features.
        cluster (str): The name of the cluster.

    Returns:
        DataFrame: A dataframe containing the cluster statistics.
                   The columns represent the features, and the index is the cluster name.
    """
    cluster_stats = {}  # Dictionary to store the cluster-wise statistics
    
    for col in features_categorical:
        cluster_stats[col] = data[col].mode().iloc[0]
        
    for col in features_numerical:
        cluster_stats[col] = data[col].median()
    
    cluster_df = pd.DataFrame(cluster_stats, index=[cluster])
    
    return cluster_df.T


#Function that will be used in 7.2. Association Rules
def create_cluster_dataframes(cust_info, basket, num_clusters):
    """
    Creates dataframes for each cluster based on the cluster ID.

    Args:
        cust_info (DataFrame): The customer information dataframe.
        basket (DataFrame): The basket dataframe.
        num_clusters (int): The number of clusters.

    Returns:
        dict: A dictionary containing the cluster dataframes.
              The keys are in the format 'dfX' where X represents the cluster ID.
              The values are the respective dataframes.
    """
    cluster_dataframes = {}
    
    for cluster_id in range(num_clusters):
        cluster_name = f'df{cluster_id}'
        cluster_dataframes[cluster_name] = cust_info[cust_info['cluster'] == cluster_id].merge(basket, on="customer_id")
    
    return cluster_dataframes


#Function that will be used in 7.2. Association Rules
def generate_association_rules(df):
    """
    Generate association rules from a dataframe.

    Parameters:
    - df (DataFrame): Input dataframe containing the 'list_of_goods' column.

    Returns:
    - rules (DataFrame): DataFrame containing the generated association rules.
    """

    # Convert 'list_of_goods' column to a list of lists
    list_of_goods_lists = df['list_of_goods'].apply(ast.literal_eval).tolist()

    # Transform the lists into a transaction matrix
    te = TransactionEncoder()
    te_fit = te.fit(list_of_goods_lists).transform(list_of_goods_lists)
    transactions_items = pd.DataFrame(te_fit, columns=te.columns_)

    # Generate frequent itemsets
    frequent_itemsets = apriori(
        transactions_items, min_support=0.02, use_colnames=True
    )

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    return rules
