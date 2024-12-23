import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
# from minisom import MiniSom

"""
FUNCTIONS USED IN DM2425_Part2_10_01.ipynb
"""


# The following function will be used 4.3. Missing Values
def plot_grouped_distributions(
    df, group_by_variable, color_palette, exclude_prefixes=["CUI", "DOW", "HR"]
):
    """
    Plots distributions of variables grouped by a specific categorical variable.

    Parameters:
    df (DataFrame): The input dataframe.
    group_by_variable (str): The categorical variable to group by (e.g., 'customer_region', 'last_promo').
    color_palette (list): List of colors for the plots.
    exclude_prefixes (list): List of prefixes for variables to exclude from plotting (default: ["CUI", "DOW", "HR"]).
    """
    # Set seaborn style and color palette
    sns.set_theme()
    sns.set_style("whitegrid")
    sns.set_palette(color_palette)

    # Define the order of the group_by_variable categories based on frequency
    category_order = df[group_by_variable].value_counts().index.tolist()

    # Special handling for '-' in 'customer_region'
    if group_by_variable == "customer_region" and "-" in category_order:
        category_order = ["-"] + sorted(
            [region for region in category_order if region != "-"],
            key=lambda x: int(x),
            reverse=True,
        )

    # Separate numeric and categorical variables
    numeric_vars = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_vars = df.select_dtypes(exclude=["number"]).columns.tolist()

    # Select variables to plot, excluding the group_by_variable and prefixed columns
    variables_to_plot = [
        col
        for col in df.columns
        if col != group_by_variable
        and not any(col.startswith(prefix) for prefix in exclude_prefixes)
    ]

    # Calculate grid layout for subplots
    total_features = (
        len(variables_to_plot) + 2
    )  # +2 for the main plot and additional plots
    sp_rows = math.ceil(math.sqrt(total_features))
    sp_cols = math.ceil(total_features / sp_rows)

    # Prepare figure with specified grid layout
    fig, axes = plt.subplots(sp_rows, sp_cols, figsize=(20, 20))
    axes = axes.flatten()

    # Plot the main group_by_variable countplot
    sns.countplot(
        data=df,
        x=group_by_variable,
        ax=axes[0],
        order=category_order,
        palette=color_palette,
    )
    axes[0].set(xlabel=group_by_variable, ylabel="Count")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0, ha="center")

    # Loop through variables to plot each one
    for ax, feat in zip(axes[1:], variables_to_plot):
        if feat == "customer_region" and group_by_variable == "last_promo":
            sns.countplot(
                data=df,
                x=feat,
                hue=group_by_variable,
                ax=ax,
                hue_order=category_order,
                palette=color_palette,
            )
            ax.set(xlabel="customer_region", ylabel="Count")
            ax.legend(title=group_by_variable, loc="upper right")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
        elif feat == "total_orders":
            sns.histplot(
                data=df,
                x=feat,
                ax=ax,
                bins=10,
                hue=group_by_variable,
                kde=True,
                hue_order=category_order,
            )
            ax.set(xlabel="Total Orders", ylabel="Density")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
        elif feat in numeric_vars:
            sns.histplot(
                data=df,
                x=feat,
                ax=ax,
                bins=10,
                hue=group_by_variable,
                kde=True,
                hue_order=category_order,
            )
            ax.set(ylabel="Density")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
        else:
            # For other categorical variables, plot stacked bar chart
            stacked_data = (
                df.groupby([group_by_variable, feat]).size().unstack(fill_value=0)
            )
            stacked_data = stacked_data.loc[
                category_order
            ]  # Reorder rows to match group_by_variable order
            stacked_data.div(stacked_data.sum(axis=1), axis=0).plot(
                kind="bar", stacked=True, ax=ax
            )
            ax.set(xlabel=group_by_variable, ylabel="Proportion")
            ax.legend(title=feat, bbox_to_anchor=(1, 1), loc="upper left")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")

    # Plot cuisine variables separately
    cuisine_columns = [col for col in df.columns if col.startswith("CUI_")]
    if cuisine_columns:
        cuisine_data = df.groupby(group_by_variable)[cuisine_columns].sum()
        cuisine_data = cuisine_data.loc[
            category_order
        ]  # Ensure the order is maintained
        cuisine_proportion = cuisine_data.div(cuisine_data.sum(axis=1), axis=0)

        # Select top 5 cuisines, others combined as 'Not Top 5'
        top_cuisines = cuisine_proportion.sum().nlargest(5).index
        combined_cuisines = cuisine_proportion[top_cuisines].copy()
        combined_cuisines["Not Top 5"] = cuisine_proportion.drop(
            top_cuisines, axis=1
        ).sum(axis=1)

        # Add the cuisine plot explicitly as the last plot
        last_ax = len(axes) - 1  # Explicitly assign the last axis
        combined_cuisines.plot(kind="bar", stacked=True, ax=axes[last_ax])
        axes[last_ax].set(xlabel=group_by_variable, ylabel="Proportion")
        axes[last_ax].legend(
            title="Top Cuisines", bbox_to_anchor=(1, 1), loc="upper left"
        )
        axes[last_ax].set_xticklabels(
            axes[last_ax].get_xticklabels(), rotation=0, ha="center"
        )

    # # Remove empty subplots
    # for ax in axes[len(variables_to_plot) + 1:]:
    #     fig.delaxes(ax)

    # Layout and display
    plt.suptitle(
        f"Distribution of Variables by {group_by_variable.title()}",
        y=1.01,
        fontsize=16,
        weight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


# The following function will be used 4.3. Missing Values
def plot_customer_distributions(
    df,
    age_column,
    spending_column,
    orders_column,
    vendor_column,
    product_column,
    color_palette,
):
    """
    Plots distributions and relationships for customer data including age, spending, orders, vendors, and products.

    Parameters:
    df (DataFrame): The input dataframe.
    age_column (str): Column name for customer age.
    spending_column (str): Column name for total spending.
    orders_column (str): Column name for total orders.
    vendor_column (str): Column name for vendor count.
    product_column (str): Column name for product count.
    """
    # Set seaborn style
    sns.set_theme()
    sns.set_style("whitegrid")
    sns.set_palette(color_palette)

    # Create a new dataframe with age bins
    spending_behavoiur = pd.DataFrame(
        {
            "age_group": pd.cut(
                df[age_column],
                bins=range(15, 66, 10),  # Bins from 15 to 65 in 10-year increments
                right=False,  # Exclude the rightmost edge for bin intervals
            )
        }
    )

    # Add other columns to the new dataframe
    spending_behavoiur[spending_column] = df[spending_column]
    spending_behavoiur[orders_column] = df[orders_column]
    spending_behavoiur[vendor_column] = df[vendor_column]
    spending_behavoiur[product_column] = df[product_column]

    # Drop rows where `age_group` or other columns might be NaN
    spending_behavoiur = spending_behavoiur.dropna()

    # Calculate the 95th percentiles
    percentiles = {
        "spending": spending_behavoiur[spending_column].quantile(0.95),
        "orders": spending_behavoiur[orders_column].quantile(0.95),
        "vendor_count": spending_behavoiur[vendor_column].quantile(0.95),
        "product_count": spending_behavoiur[product_column].quantile(0.95),
    }

    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(21, 21))

    # Plot 1: Distribution of customer age
    sns.histplot(df[age_column], bins=10, kde=True, alpha=0.6, ax=axes[0, 0])
    median_age = df[age_column].median()
    axes[0, 0].axvline(
        median_age,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median Age: {round(median_age)}",
    )
    axes[0, 0].set_title("Distribution of Customer Age", fontsize=16, weight="bold")
    axes[0, 0].legend()

    # Plot 2: Customer age vs total spending
    sns.scatterplot(data=df, x=age_column, y=spending_column, alpha=0.6, ax=axes[0, 1])
    axes[0, 1].set_title("Customer Age vs Total Spending", fontsize=16, weight="bold")

    # Plot 3: Customer age vs total orders
    sns.scatterplot(data=df, x=age_column, y=orders_column, alpha=0.6, ax=axes[0, 2])
    axes[0, 2].set_title("Customer Age vs Total Orders", fontsize=16, weight="bold")

    # Plot 4: Distribution of spending grouped by age
    sns.histplot(
        data=spending_behavoiur,
        x=spending_column,
        ax=axes[1, 0],
        bins=30,
        hue="age_group",
        kde=True,
    )
    axes[1, 0].set_title(
        "Distribution of Total Spending by Age Group", fontsize=16, weight="bold"
    )

    # Plot 5: Boxplot of spending by age
    sns.boxplot(
        data=spending_behavoiur, x="age_group", y=spending_column, ax=axes[1, 1]
    )
    axes[1, 1].set_title("Spending by Age Group", fontsize=16, weight="bold")
    axes[1, 1].set_ylim(-1, percentiles["spending"])

    # Plot 6: Boxplot of orders by age
    sns.boxplot(data=spending_behavoiur, x="age_group", y=orders_column, ax=axes[1, 2])
    axes[1, 2].set_title("Orders by Age Group", fontsize=16, weight="bold")
    axes[1, 2].set_ylim(-0.2, percentiles["orders"])

    # Plot 7: Boxplot of vendor count by age
    sns.boxplot(data=spending_behavoiur, x="age_group", y=vendor_column, ax=axes[2, 0])
    axes[2, 0].set_title("Vendor Count by Age Group", fontsize=16, weight="bold")
    axes[2, 0].set_ylim(-0.2, percentiles["vendor_count"])

    # Plot 8: Boxplot of product count by age
    sns.boxplot(data=spending_behavoiur, x="age_group", y=product_column, ax=axes[2, 1])
    axes[2, 1].set_title("Product Count by Age Group", fontsize=16, weight="bold")
    axes[2, 1].set_ylim(-0.2, percentiles["product_count"])

    # Remove empty subplot (last one in 3x3 grid)
    fig.delaxes(axes[2, 2])

    # Layout and display
    plt.suptitle("Customer Data Distribution", y=1.01, fontsize=20, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


# The following function will be used in 4.5.3. CUI_Asian vs Chinese, Indian, Japanese, etc, 4-7. Correlation Matrix and 7. Feature Selecion
def plot_matrix(data, title="Correlation Matrix", threshold=0.5, figsize=(24, 16)):
    """
    Plot a heatmap of the correlation matrix with annotations based on a threshold.

    Parameters:
        data (DataFrame): Correlation matrix to be visualized.
        title (str): Title of the plot.
        threshold (float): Minimum absolute value of correlations to annotate.
        figsize (tuple): Figure size for the plot.
    """
    # Annotate only values above the threshold
    mask_annot = np.absolute(data.values) >= threshold
    annot = np.where(mask_annot, np.round(data.values, 2), "")

    # Create a mask for the upper triangle
    upper_triangle_mask = np.triu(np.ones_like(data, dtype=bool))

    # Prepare figure
    plt.figure(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        data=data,
        mask=upper_triangle_mask,  # Apply mask
        annot=annot,  # Custom annotation
        fmt="",  # Avoid conflicting formats
        vmin=-1,
        vmax=1,
        center=0,  # Adjust color scale
        square=True,
        linewidths=0.5,  # Aesthetics
        cmap="PiYG",  # Colormap
    )

    # Set plot title
    plt.title(title, fontsize=16, weight="bold")

    # Show plot
    plt.show()




# The following function will be used in 5. Feature Engineering
def plot_distribution(
    data,
    x,
    plot_type="count",
    title=None,
    xlabel=None,
    ylabel=None,
    kde=False,
    bins=None,
    xlim=None,
    show_counts=False,
    figsize=(8, 6),
    order=None,
    color_palette=None,
):
    """
    Plots a distribution with an optional parameter to show counts above bars.

    Parameters:
    data (DataFrame): The DataFrame containing the data.
    x (str): The column to plot.
    plot_type (str): Type of plot ("count" or "hist").
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    kde (bool): Whether to include KDE in histograms (default: False).
    bins (int): Number of bins for histograms (default: None).
    xlim (tuple): Limit for the x-axis (default: None).
    show_counts (bool): Whether to display counts above the bars (default: False).
    figsize (tuple): Size of the figure (default: (8, 6)).
    order (list): Order for categorical variables (default: None).
    """
    # Set the figure size
    sns.set_style("whitegrid")
    sns.set_palette(color_palette)
    plt.figure(figsize=figsize)

    # Plot based on type
    if plot_type == "count":
        ax = sns.countplot(data=data, x=x, order=order)
    elif plot_type == "hist":
        ax = sns.histplot(data[x], kde=kde, bins=bins)
    else:
        raise ValueError("Invalid plot_type. Use 'count' or 'hist'.")

    # Set the plot titles and labels
    plt.title(
        title if title else f"Distribution of {x.title()}", fontsize=16, weight="bold"
    )
    plt.xlabel(xlabel if xlabel else x.title())
    plt.ylabel(ylabel if ylabel else "Frequency")

    # Set x-axis limits if provided
    if xlim:
        plt.xlim(xlim)

    # Show counts above bars if enabled
    if show_counts:
        if plot_type == "count":
            for container in ax.containers:
                ax.bar_label(container, label_type="edge", padding=3)
        elif plot_type == "hist":
            for patch in ax.patches:
                height = patch.get_height()
                if height > 0:  # Only annotate bars with height > 0
                    ax.annotate(
                        f"{int(height)}",
                        xy=(patch.get_x() + patch.get_width() / 2, height),
                        xytext=(0, 5),  # Offset label position
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                    )

    # Display the plot
    plt.tight_layout()
    plt.show()


# The following function will be used in 7. Outliers
def plot_boxplots_iqr_outliers(
    features, df, title, cols=5, figsize=(18, 12), sort_by="outliers"
):
    """
    Plots boxplots for the given features and calculates outlier counts (and percentages),
    sorted by the specified method (either by 'outliers' or 'alphabet').

    Parameters:
    features (list): List of feature names to plot.
    df (DataFrame): DataFrame containing the features.
    title (str): Title for the plot grid.
    cols (int): Number of columns in the grid layout (default: 5).
    figsize (tuple): Size of the figure (default: (18, 12)).
    sort_by (str): Method to sort the plots. Use 'outliers' to sort by outlier percentage,
                   or 'alphabet' to sort alphabetically (default: 'outliers').
    """
    # Calculate IQR outliers for each feature and their percentages
    outlier_info = []
    for feature in features:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100
        outlier_info.append((feature, outlier_count, outlier_percentage))

    # Sort features based on the specified method
    if sort_by == "outliers":
        sorted_features = sorted(
            outlier_info, key=lambda x: x[2], reverse=True
        )  # Sort by outlier percentage
    elif sort_by == "alphabet":
        sorted_features = sorted(
            outlier_info, key=lambda x: x[0]
        )  # Sort alphabetically by feature name
    else:
        raise ValueError("Invalid value for 'sort_by'. Use 'outliers' or 'alphabet'.")

    # Extract sorted feature names
    sorted_feature_names = [item[0] for item in sorted_features]

    # Calculate rows needed for the grid
    rows = (len(sorted_feature_names) // cols) + (len(sorted_feature_names) % cols > 0)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    axes = axes.flatten()

    # Generate box plots for each feature
    for i, (feature, outlier_count, outlier_percentage) in enumerate(sorted_features):
        sns.boxplot(x=df[feature], ax=axes[i])
        axes[i].set_title(f"Outliers: {outlier_count} ({outlier_percentage:.2f}%)")
        axes[i].set_xlabel(feature)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Set a main title for the grid
    plt.suptitle(title, fontsize=16, weight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def get_ss(df, feats):
    """
    Calculate the sum of squares (SS) for the given DataFrame.

    The sum of squares is computed as the sum of the variances of each column
    multiplied by the number of non-NA/null observations minus one.

    Parameters:
    df (pandas.DataFrame): The input DataFrame for which the sum of squares is to be calculated.
    feats (list of str): A list of feature column names to be used in the calculation.

    Returns:
    float: The sum of squares of the DataFrame.
    """
    df_ = df[feats]
    ss = np.sum(df_.var() * (df_.count() - 1))

    return ss


def get_ssb(df, feats, label_col):
    """
    Calculate the between-group sum of squares (SSB) for the given DataFrame.
    The between-group sum of squares is computed as the sum of the squared differences
    between the mean of each group and the overall mean, weighted by the number of observations
    in each group.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    feats (list of str): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column in the DataFrame that contains the group labels.

    Returns
    float: The between-group sum of squares of the DataFrame.
    """

    ssb_i = 0
    for i in np.unique(df[label_col]):
        df_ = df.loc[:, feats]
        X_ = df_.values
        X_k = df_.loc[df[label_col] == i].values

        ssb_i += X_k.shape[0] * (np.square(X_k.mean(axis=0) - X_.mean(axis=0)))

    ssb = np.sum(ssb_i)

    return ssb


def get_ssw(df, feats, label_col):
    """
    Calculate the sum of squared within-cluster distances (SSW) for a given DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    feats (list of str): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column containing cluster labels.

    Returns:
    float: The sum of squared within-cluster distances (SSW).
    """
    feats_label = feats + [label_col]

    df_k = (
        df[feats_label]
        .groupby(by=label_col)
        .apply(lambda col: get_ss(col, feats), include_groups=False)
    )

    return df_k.sum()


def get_rsq(df, feats, label_col):
    """
    Calculate the R-squared value for a given DataFrame and features.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    feats (list): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column containing the labels or cluster assignments.

    Returns:
    float: The R-squared value, representing the proportion of variance explained by the clustering.
    """

    df_sst_ = get_ss(df, feats)  # get total sum of squares
    df_ssw_ = get_ssw(df, feats, label_col)  # get ss within
    df_ssb_ = df_sst_ - df_ssw_  # get ss between

    # r2 = ssb/sst
    return df_ssb_ / df_sst_


def get_r2_hc(df, link_method, max_nclus, min_nclus=1, dist="euclidean"):
    """This function computes the R2 for a set of cluster solutions given by the application of a hierarchical method.
    The R2 is a measure of the homogenity of a cluster solution. It is based on SSt = SSw + SSb and R2 = SSb/SSt.

    Parameters:
    df (DataFrame): Dataset to apply clustering
    link_method (str): either "ward", "complete", "average", "single"
    max_nclus (int): maximum number of clusters to compare the methods
    min_nclus (int): minimum number of clusters to compare the methods. Defaults to 1.
    dist (str): distance to use to compute the clustering solution. Must be a valid distance. Defaults to "euclidean".

    Returns:
    ndarray: R2 values for the range of cluster solutions
    """

    r2 = []  # where we will store the R2 metrics for each cluster solution
    feats = df.columns.tolist()

    for i in range(min_nclus, max_nclus + 1):  # iterate over desired ncluster range
        cluster = AgglomerativeClustering(
            n_clusters=i, metric=dist, linkage=link_method
        )

        # get cluster labels
        hclabels = cluster.fit_predict(df)

        # concat df with labels
        df_concat = pd.concat(
            [df, pd.Series(hclabels, name="labels", index=df.index)], axis=1
        )

        # append the R2 of the given cluster solution
        r2.append(get_rsq(df_concat, feats, "labels"))

    return np.array(r2)


# The following function will be used in 7. Data Normalization
def scaled_dataframe(columns: list, df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales the specified columns in the DataFrame using StandardScaler.

    Parameters:
    columns (list): List of column names to scale.
    df (DataFrame): DataFrame containing the columns to scale.

    Returns:
    DataFrame: A DataFrame with the specified columns scaled.
    """
        # Ensure columns is a list
    if isinstance(columns, str):
        columns = [columns]
    
    # from list to pandas index
    if not isinstance(columns, pd.Index):
        columns = pd.Index(columns)

    # Check if all columns exist in the DataFrame
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")


    # Create a copy of the DataFrame
    df_scaled = df.copy()

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform the data
    df_scaled[columns + '_normalized'] = scaler.fit_transform(df_scaled[columns])

    # drop columns with original values
    df_scaled.drop(columns=columns, inplace=True)

    return df_scaled
