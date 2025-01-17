import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom

#Visualizing the data
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA

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
        cmap="YlGnBu",  # Colormap
    )

    # Set plot title
    plt.title(title, fontsize=16, weight="bold")

    # Show plot
    plt.show()



# The following functions will be used in 5. Feature Engineering
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


def plot_boxplots(df, column_label_pairs, figsize=(12, 8), cols=3):
    """
    Plots side-by-side boxplots for specified columns in the dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe containing the data.
        column_label_pairs (list of tuples): List of tuples where each tuple contains:
            - The column name (str).
            - The x-axis label for the plot (str).
        figsize (tuple): Overall figure size for the plots.
        cols (int): Number of columns of plots in the grid.
    """
    if len(column_label_pairs) == 1:
        # If only one feature, create a single boxplot
        col, label = column_label_pairs[0]
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[col])
        plt.title(label)
        plt.xlabel("Value")
        plt.ylabel("")
        plt.show()
    else:
        # For multiple features, create a grid
        rows = (len(column_label_pairs) + cols - 1) // cols  # Calculate the number of rows
        fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        for i, (col, label) in enumerate(column_label_pairs):
            sns.boxplot(x=df[col], ax=axes[i])
            axes[i].set_title(label)
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("")
        
        # Hide unused subplots if columns < total slots
        for j in range(len(column_label_pairs), len(axes)):
            axes[j].axis("off")
        
        plt.show()
    
def plot_scatter(data, pairs, xlim=None):
    """
    Plots scatter plots for given pairs of columns.

    Parameters:
    - data: DataFrame containing the data.
    - pairs: List of tuples with (x_column, y_column, title).
    - xlim: Tuple specifying the limits for the x-axis (optional).
    """
    for x_col, y_col, title in pairs:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=x_col, y=y_col, data=data)
        plt.title(title)
        plt.xlabel(f"{x_col.replace('_', ' ').title()}")
        plt.ylabel(f"{y_col.replace('_', ' ').title()}")
        if xlim:
            plt.xlim(xlim)
        plt.show()
        

def plot_kde_plots(df, column_label_pairs, figsize=(14, 10), cols=3, xlim=None, row_spacing=1.0, col_spacing=0.5):
    """
    Plots KDE plots for specified columns in the dataframe, either individually or side-by-side in a grid.
    
    Args:
        df (pd.DataFrame): Dataframe containing the data.
        column_label_pairs (list of tuples): List of tuples where each tuple contains:
            - The column name (str).
            - The x-axis label for the plot (str).
        figsize (tuple): Overall figure size for the grid layout.
        cols (int): Number of columns for the grid layout.
        xlim (tuple): Range for the x-axis, e.g., (0, 1). Default is None (automatic scaling).
        row_spacing (float): Space between rows in the grid.
        col_spacing (float): Space between columns in the grid.
    """
    if len(column_label_pairs) == 1:
        # Single KDE plot
        col, label = column_label_pairs[0]
        plt.figure(figsize=figsize)
        sns.kdeplot(df[col], fill=True)
        plt.title(f"KDE Plot of {label}")
        plt.xlabel(label)
        plt.ylabel("Density")
        if xlim:
            plt.xlim(xlim)
        plt.show()
    else:
        # Grid of KDE plots
        rows = (len(column_label_pairs) + cols - 1) // cols  # Calculate number of rows
        fig, axes = plt.subplots(
            rows, cols, figsize=figsize,
            gridspec_kw={"hspace": row_spacing, "wspace": col_spacing}  # Adjust spacing
        )
        axes = axes.flatten()  # Flatten axes for easy iteration

        for i, (col, label) in enumerate(column_label_pairs):
            sns.kdeplot(df[col], fill=True, ax=axes[i])
            axes[i].set_title(f"KDE Plot of {label}")
            axes[i].set_xlabel(label)
            axes[i].set_ylabel("Density")
            if xlim:
                axes[i].set_xlim(xlim)
        
        # Hide unused subplots if columns < total slots
        for j in range(len(column_label_pairs), len(axes)):
            axes[j].axis("off")
        
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

"""
FUNCTIONS USED IN DM2425_Part2_10_02.ipynb
"""
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

def get_ssw_with_medoids(df, feats, medoids, label_col):
    ssw = 0
    for medoid, cluster_id in zip(medoids, df[label_col].unique()):
        cluster_points = df[df[label_col] == cluster_id][feats].values
        distances = np.sum(np.square(cluster_points - medoid), axis=1)
        ssw += distances.sum()
    return ssw


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

def get_rsq_with_medoids(df, feats, label_col, medoids):
    """
    Calculate the R-squared value for k-medoids clustering using medoids.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    feats (list): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column containing the cluster labels.
    medoids (numpy.ndarray): Array of medoids for each cluster (shape: [n_clusters, n_features]).

    Returns:
    float: The R-squared value, representing the proportion of variance explained by the clustering.
    """
    # Total sum of squares (SSt)
    df_sst_ = get_ss(df, feats)

    # Within-cluster sum of squares (SSw) using medoids
    df_ssw_ = get_ssw_with_medoids(df, feats, medoids, label_col)

    # Between-cluster sum of squares (SSb)
    df_ssb_ = df_sst_ - df_ssw_

    # Calculate R^2
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

# function that assigns the cluster labels from the nodes to the corresponding datapoints.
def cluster_labels_som(df, features, M, N, sm, node_weights, node_labels):
    """
    This function returns cluster labels array based on the data points from the SOM grid. 

    Parameters:
        df (pd.DataFrame): The dataset containing the features for SOM training.
        features (list): List of feature column names to be used for training.
        M (int): Number of rows in the SOM grid.
        N (int): Number of columns in the SOM grid.
        sm (Mini Som): Trained MiniSom instance.
        node_weights (np.array): Weights of the nodes as a 2D array.
        node_labels (np.array): Cluster labels assigned to the nodes.

    Returns:
        np.array: Cluster labels assigned to the data points based on the SOM grid.

    
    """

    # get the labels out of the prediction

    # create a dataframe out of the node weights and their cluster label


    ## This gets BMU nodes, e.g. (4,4) for each data point
    bmu_index = np.array([sm.winner(x) for x in df[features].values])
    
    # get the cluster label for each data point
    kmeans_matrix = node_labels.reshape((M,N))

    som_final_labels = [kmeans_matrix[i[0]][i[1]] for i in bmu_index]

    return som_final_labels

def plot_cluster_means_and_frequencies(*solutions):
    """
    Plots cluster means as separate subplots for each solution and cluster frequencies as a bar chart.
    
    Parameters:
        solutions (tuple): Each solution is a tuple of (cluster_means, cluster_frequencies),
                           where cluster_means is a DataFrame and cluster_frequencies is a dictionary.
                           The solutions should be provided in the desired order of plotting.
    """
    num_solutions = len(solutions)
    cluster_means_list = [sol[0] for sol in solutions]
    cluster_frequencies_list = [sol[1] for sol in solutions]
    
    # Ensure all solutions share the same feature set
    features = cluster_means_list[0].columns

    # Create a figure with subplots: one for each cluster means and one for the bar chart
    fig, axs = plt.subplots(1, num_solutions + 1, 
                            figsize=(6 * (num_solutions + 1), 6), 
                            gridspec_kw={'width_ratios': [1] * num_solutions + [1.5]})
    
    # Plot cluster means for each solution
    for i, cluster_means in enumerate(cluster_means_list):
        for cluster in cluster_means.index:
            axs[i].plot(features, cluster_means.loc[cluster], marker='o', label=f"Cluster {cluster}")
        axs[i].set_title(f"Cluster Means: Solution {i + 1}")
        axs[i].set_xlabel("Features")
        axs[i].set_ylabel("Standardized Feature Mean")
        axs[i].grid(True)
        axs[i].legend(title="Clusters", loc='best', bbox_to_anchor=(1, 1))
        
        # Set shared x-axis labels for line charts
        axs[i].set_xticks(range(len(features)))
        axs[i].set_xticklabels(features, rotation=45, ha='right')

    # Bar chart for cluster frequencies
    all_clusters = sorted(set().union(*[freq.keys() for freq in cluster_frequencies_list]))
    bar_positions = np.arange(len(all_clusters))  # Positions for clusters
    bar_width = 0.8 / num_solutions  # Dynamic bar width based on the number of solutions

    # Plot frequencies for each solution
    for i, cluster_frequencies in enumerate(cluster_frequencies_list):
        frequencies = [cluster_frequencies.get(cluster, 0) for cluster in all_clusters]
        bars = axs[-1].bar(bar_positions + (i - num_solutions / 2) * bar_width, 
                           frequencies, bar_width, label=f"Solution {i + 1}")
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            axs[-1].text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{int(height)}", ha='center', va='bottom')

    # X-axis labels for bar chart
    axs[-1].set_xticks(bar_positions)
    axs[-1].set_xticklabels([f"Cluster {cluster}" for cluster in all_clusters], rotation=45)
    axs[-1].set_title("Cluster Frequencies")
    axs[-1].set_xlabel("Clusters")
    axs[-1].set_ylabel("Frequency")
    axs[-1].legend(title="Solution", loc='best', bbox_to_anchor=(1, 1))
    axs[-1].grid(True)

    # Adjust layout and show
    plt.tight_layout()
    plt.show()

def plot_dim_reduction(df_concat, label_column):
    """
    Plots t-SNE, UMAP, and PCA visualizations side by side with transparency in the dots.

    Parameters:
    df_concat (pd.DataFrame): DataFrame containing the features and cluster labels.
    label_column (str): Name of the column containing the cluster labels.
    """
    # Extract features and labels
    features = df_concat.drop(columns=[label_column])
    labels = df_concat[label_column]

    # Perform dimensionality reductions
    tsne_result = TSNE(random_state=42).fit_transform(features)
    umap_result = umap.UMAP(random_state=42).fit_transform(features)
    pca_result = PCA(n_components=2, random_state=42).fit_transform(features)

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)

    # t-SNE Visualization with transparency
    sns.scatterplot(
        x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette='tab10',
        ax=axes[0], legend='full', alpha=0.5
    )
    axes[0].set_title("t-SNE Visualization")
    axes[0].set_xlabel("Component 1")
    axes[0].set_ylabel("Component 2")

    # UMAP Visualization with transparency
    sns.scatterplot(
        x=umap_result[:, 0], y=umap_result[:, 1], hue=labels, palette='tab10',
        ax=axes[1], legend=None, alpha=0.5
    )
    axes[1].set_title("UMAP Visualization")
    axes[1].set_xlabel("Component 1")
    axes[1].set_ylabel("Component 2")

    # PCA Visualization with transparency
    sns.scatterplot(
        x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, palette='tab10',
        ax=axes[2], legend=None, alpha=0.5
    )
    axes[2].set_title("PCA Visualization")
    axes[2].set_xlabel("Principal Component 1")
    axes[2].set_ylabel("Principal Component 2")

    # Adjust layout and show
    plt.tight_layout()
    plt.show()

"""
FUNCTIONS USED IN DM2425_Part2_10_04.ipynb
"""

def cluster_profiles(df, label_columns, figsize, 
                     cmap="tab10",
                     compare_titles=None):
    """
    Pass df with label columns of one or multiple clustering labels.
    Then specify these label columns to perform the cluster profile according to them.
    """
    
    if compare_titles is None:
        compare_titles = [""] * len(label_columns)
        
    fig, axes = plt.subplots(nrows=len(label_columns), 
                             ncols=2, 
                             figsize=figsize, 
                             constrained_layout=True,
                             squeeze=False)
    
    for ax, label, titl in zip(axes, label_columns, compare_titles):
        # Filtering df to keep only relevant columns
        drop_cols = [i for i in label_columns if i != label]
        dfax = df.drop(drop_cols, axis=1)
        
        # Getting the cluster centroids and counts
        centroids = dfax.groupby(by=label, as_index=False).mean()
        counts = dfax.groupby(by=label, as_index=False).count().iloc[:, [0, 1]]
        counts.columns = [label, "counts"]
        
        # Plotting parallel coordinates for cluster centroids
        pd.plotting.parallel_coordinates(centroids, 
                                         label, 
                                         color=sns.color_palette(cmap),
                                         ax=ax[0])

        # Plotting barplot for cluster sizes (counts)
        sns.barplot(x=label, 
                    hue=label,
                    y="counts", 
                    data=counts, 
                    ax=ax[1], 
                    palette=sns.color_palette(cmap),
                    legend=False
                    )

        # Add counts above each bar
        for p in ax[1].patches:
            ax[1].annotate(f'{int(p.get_height())}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='bottom', 
                           fontsize=15, color='black')
        
        # Setting Layout
        handles, _ = ax[0].get_legend_handles_labels()
        cluster_labels = ["Cluster {}".format(i) for i in range(len(handles))]
        ax[0].annotate(text=titl, xy=(0.95, 1.1), xycoords='axes fraction', fontsize=13, fontweight='heavy') 
        ax[0].axhline(color="black", linestyle="--")
        ax[0].set_title("Cluster Means - {} Clusters".format(len(handles)), fontsize=13)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), 
                              rotation=40,
                              ha='right')
        
        ax[0].legend(handles, cluster_labels,
                     loc='center left', bbox_to_anchor=(1, 0.5), title=label
                     ) # Adaptable to number of clusters
        
        ax[1].set_xticks([i for i in range(len(handles))])
        ax[1].set_xticklabels(cluster_labels)
        ax[1].set_xlabel("")
        ax[1].set_ylabel("Absolute Frequency")
        ax[1].set_title("Cluster Sizes - {} Clusters".format(len(handles)), fontsize=13)
    
    plt.suptitle("Cluster Simple Profiling", fontsize=23)
    plt.show()
    
def plot_customer_age_distribution(df, cluster_col='merged_labels'):
    """
    Creates individual distribution plots for `customer_age` for each `merged_label`.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing the data.
    cluster_col (str): Column name representing the cluster labels.

    Returns:
    None
    """
    # Iterate over each unique cluster label
    for cluster in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster]
        plt.figure(figsize=(8, 5))
        sns.histplot(
            data=cluster_data,
            x='customer_age',
            kde=True,  # Add KDE for smoother distribution visualization
            bins=20,   # Adjust number of bins as needed
            color='blue'
        )
        plt.title(f"Customer Age Distribution - Cluster {cluster}")
        plt.xlabel("Customer Age")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()


def plot_demographics_relative_distribution(df, value_col, cluster_col = 'merged_labels',  title=None, colormap='tab10'):
    """
    Plots a stacked bar chart showing the relative distribution of categorical values
    for each cluster.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing the data.
    cluster_col (str): Column name representing the cluster labels.
    value_col (str): Column name representing the categorical variable to be visualized.
    title (str): Title of the plot. Defaults to None.
    colormap (str): Colormap to use for the plot. Defaults to 'tab10'.

    Returns:
    None
    """
    # Group by cluster and the value column, then count occurrences
    grouped_data = df.groupby([cluster_col, value_col])[value_col].size().unstack()

    # Normalize to get relative distributions (percentages)
    normalized_data = grouped_data.div(grouped_data.sum(axis=1), axis=0)

    # Plot the stacked bar chart
    ax = normalized_data.plot.bar(stacked=True, figsize=(10, 6), colormap=colormap)

    # Add labels and title
    plot_title = title if title else f'Relative Distribution of {value_col.title()} by {cluster_col.title()}'
    plt.title(plot_title, fontsize=14)
    plt.xlabel(cluster_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.legend(title=value_col.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Display the plot
    plt.show()
    
    
def plot_demographics_distribution(df, value_col, cluster_col = 'merged_labels', title=None, figsize=(16, 4)):
    """
    Creates a grid of count plots for a categorical variable across all clusters,
    including an overall count plot for the entire dataset.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing the data.
    cluster_col (str): Column name representing the cluster labels.
    value_col (str): Column name representing the categorical variable to plot.
    title (str): Title of the entire figure. Defaults to None.
    figsize (tuple): Size of the overall figure. Defaults to (16, 4).

    Returns:
    None
    """
    # Determine the number of unique clusters
    num_clusters = df[cluster_col].nunique()

    # Create a subplot grid (1 row, number of clusters + 1 columns)
    fig, axes = plt.subplots(1, num_clusters + 1, figsize=figsize, tight_layout=True)

    # Iterate over all subplots
    for i, ax in enumerate(axes.flatten()):
        if i == 0:
            # Plot for all data
            sns.countplot(
                data=df,
                x=value_col,
                order=df[value_col].value_counts().index,
                ax=ax
            )
            ax.set_title("All Data")
        else:
            # Plot for individual clusters
            cluster_data = df[df[cluster_col] == i - 1]
            sns.countplot(
                data=cluster_data,
                x=value_col,
                order=df[value_col].value_counts().index,
                ax=ax
            )
            ax.set_title(f"Cluster {i - 1}")

        # Add counts above each bar
        for p in ax.patches:
            ax.annotate(
                f'{int(p.get_height())}',  # Display the count
                (p.get_x() + p.get_width() / 2., p.get_height()),  # Position the count above the bar
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points'
            )

        # Rotate x-axis labels for readability
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Add a title for the entire figure
    plt.suptitle(title if title else f"{value_col.title()} Distribution Across Clusters", fontsize=16)
    plt.show()

def plot_radar(grouped_means, title="Radar Plot", figsize=(10, 10), rotation=45):
    """
    Plots a radar chart for grouped data, excluding the 'Overall_Avg' column.

    Parameters:
    - grouped_means: DataFrame containing the data to be plotted. Rows should be categories, and columns should be cluster means.
    - title: Title of the plot (default: "Radar Plot").
    - figsize: Tuple specifying the size of the figure (default: (10, 10)).
    - rotation: Angle for category labels rotation (default: 45).
    """
    # Exclude 'Overall_Avg' column if present
    if 'Overall_Avg' in grouped_means.columns:
        grouped_means = grouped_means.drop(columns=['Overall_Avg'])
    
    # Number of categories
    categories = grouped_means.index.tolist()
    num_categories = len(categories)
    
    # Compute angles for each category
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Plot each cluster
    for column in grouped_means.columns:
        values = grouped_means[column].tolist()
        values += values[:1]  # Close the plot
        ax.plot(angles, values, linewidth=2, label=column)
        ax.fill(angles, values, alpha=0.2)
    
    # Add labels for each category with adjusted rotation
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, rotation=rotation, ha='center', va='center')
    
    # Set y-ticks and gridlines
    max_value = grouped_means.max().max()
    step = max_value / 5
    y_ticks = np.arange(step, max_value + step, step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{x:.1f}' for x in y_ticks], color='gray', fontsize=8)
    ax.grid(True)
    
    # Add title and legend
    plt.title(title, size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
def plot_simple_boxplots(df, features, x_col="merged_labels", figsize=(8, 6)):
    """
    Plots boxplots for the specified features against a given x-column.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        features (list of str): List of feature names for the y-axis.
        x_col (str): The column name for the x-axis (default is "merged_labels").
        figsize (tuple): Figure size for each plot.
    """
    for feature in features:
        plt.figure(figsize=figsize)
        sns.boxplot(x=x_col, y=feature, data=df)
        plt.title(f"{feature} vs {x_col}")
        plt.xlabel("Cluster Labels")
        plt.ylabel(feature)
        plt.show()

def plot_cluster_heatmap(df, features, groupby_column='merged_labels', figsize=(15, 15), cmap="BrBG"):
    """
    Generates a heatmap of mean values for specified features grouped by a column.
    
    Args:
        df (pd.DataFrame): Dataframe containing the data.
        features (list): List of feature names to plot in the heatmap.
        groupby_column (str): Column name to group by (default is 'merged_labels').
        figsize (tuple): Size of the plot (default is (15, 15)).
        cmap (str): Color map to use for the heatmap (default is "BrBG").
    """
    # Group by the specified column and calculate the mean for the selected features
    km_profile = df.groupby(groupby_column).mean(numeric_only=True)[features].T

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(km_profile,
                center=0, annot=True, cmap=cmap, fmt=".2f",
                ax=ax)

    ax.set_xlabel("Merged Cluster Labels")
    ax.set_title(f"Customer Segmentations by KMeans Clustering")
    plt.show()
