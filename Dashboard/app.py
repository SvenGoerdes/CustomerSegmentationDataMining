from dash import Dash, dash_table, dcc, callback, Output, Input, html
import pandas as pd
import plotly.express as px
import dash_mantine_components as dmc
import json

# ------------------------------------------------------------------
# 1) Load and prepare data
# ------------------------------------------------------------------
df_clust = pd.read_csv('../Data/df_clustering_non_standardized_mergedlabel.csv')

with open('initial_perspectives.json') as f:
    initial_perspective = json.load(f)

# The JSON file presumably has keys: 'customer_behavior', 'cuisine_preferences', 'demographics'
cust_col = initial_perspective['customer_behavior']     # e.g. behavior columns
cui_col = initial_perspective['cuisine_preferences']    # e.g. cuisine columns
demogr_col = initial_perspective['demographics']        # e.g. region/generation/payment_method, etc.

# For demonstration, let's assume you want to scatter-plot from the same set of behavior columns:
metric_features = cust_col + cust_col  # (Your code doubled it; adjust as needed)

# Identify the columns that start with 'prop_' from the behavior set:
prop_cols = [c for c in cust_col if c.startswith('prop_')]

# Melt DF for box plots (using the cuisine columns)
df_cui_melted = df_clust.melt(
    id_vars=['merged_labels_name'],
    value_vars=cui_col,   # from your JSON
    var_name='cuisine',
    value_name='proportion'
)

# Tab-to-cluster mapping
cluster_map = {
    'tab-1-example-graph': 'The Chain Enthusiasts',
    'tab-2-example-graph': 'The Indian Food Lovers',
    'tab-3-example-graph': 'The Average Consumers',
    'tab-4-example-graph': 'The Dawn Spenders',
    'tab-5-example-graph': 'Frequent High-Spenders',
    'tab-6-example-graph': 'The Morning Snackers'
}

# ------------------------------------------------------------------
# 2) Initialize the Dash app
# ------------------------------------------------------------------
app = Dash()

# ------------------------------------------------------------------
# 3) Define the layout with Mantine
# ------------------------------------------------------------------
app.layout = dmc.Container(
    children=[
        dmc.Title("Data Mining Cluster Dashboard", color='#000000', size="h1"),
        html.Hr(),
        html.Br(),
    
        dmc.Title("Choose the Cluster:", color='#000000', size="h4"),
        dcc.Tabs(
            id="tabs-example-graph",
            value='tab-1-example-graph',
            children=[
                dcc.Tab(label='The Chain Enthusiasts', value='tab-1-example-graph'),
                dcc.Tab(label='The Indian Food Lovers', value='tab-2-example-graph'),
                dcc.Tab(label='The Average Consumers', value='tab-3-example-graph'),
                dcc.Tab(label='The Dawn Spenders', value='tab-4-example-graph'),
                dcc.Tab(label='Frequent High-Spenders', value='tab-5-example-graph'),
                dcc.Tab(label='The Morning Snackers', value='tab-6-example-graph'),
                dcc.Tab(label='All Individuals', value='tab-7-example-graph'),
            ],
            style={'marginBottom': '20px'}
        ),

        html.Br(),

        dmc.Text(
            "Select the two columns for the scatterplot:",
            size="md",
            weight=600,
            id="select-columns"
        ),
        dcc.Dropdown(
            options=metric_features,
            value=metric_features[0],
            id='controls-and-dropdown-1',
            className='dash-dropdown',
            style={'marginBottom': '10px', 'width': '250px'}
        ),
        dcc.Dropdown(
            options=metric_features,
            value=metric_features[1],
            id='controls-and-dropdown-2',
            className='dash-dropdown',
            style={'marginBottom': '20px', 'width': '250px'}
        ),

        # Cuisine Section
        dmc.Title("Cuisine preferences", color="blue", size="h2"),
        html.Hr(),
        dmc.Grid([
            dmc.Col([
                dcc.Graph(
                    figure={},
                    id='controls-and-graph',
                    className='dark-graph'
                )
            ], span=6),
            dmc.Col([
                dcc.Graph(
                    figure={},
                    id='cuisine-box-graph'
                )
            ], span=6),
        ], gutter="md"),

        # Demographics Section
        dmc.Title("Demographics", color="blue", size="h2"),
        html.Hr(),
        dmc.Grid([
            dmc.Col([
                dcc.Graph(figure={}, id='barplot-region')
            ], span=4),
            dmc.Col([
                dcc.Graph(figure={}, id='barplot-generation')
            ], span=4),
            dmc.Col([
                dcc.Graph(figure={}, id='barplot-payment-method')
            ], span=4),
        ], gutter="md"),

        # Behavior Section
        dmc.Title("Behavior", color="blue", size="h2"),
        html.Hr(),

        dmc.Text("Choose a behavior column to see its distribution:", size="md", weight=600),
        dcc.Dropdown(
            options=cust_col,   # all behavior columns from your JSON
            value=cust_col[0],  # default selection
            id='behavior-dropdown',
            style={'marginBottom': '20px', 'width': '350px'}
        ),

        dmc.Grid([
            dmc.Col([
                dcc.Graph(id='behavior-hist')        # Distribution histogram
            ], span=6),
            dmc.Col([
                dcc.Graph(id='behavior-mean-prop')  # Mean proportion bar
            ], span=6),

            # dmc.Col([
            #     dcc.Graph(id='behavior-heatmap')     # Correlation heatmap
            # ], span=12),

        ], gutter="md"),

    ],
    fluid=True,
    style={"padding": "10px"}
)

# ------------------------------------------------------------------
# 4) Callback: Return all 8 figures
# ------------------------------------------------------------------
@callback(
    Output('controls-and-graph', 'figure'),
    Output('cuisine-box-graph', 'figure'),
    Output('barplot-region', 'figure'),
    Output('barplot-generation', 'figure'),
    Output('barplot-payment-method', 'figure'),
    Output('behavior-mean-prop', 'figure'),
    # Output('behavior-heatmap', 'figure'),
    Output('behavior-hist', 'figure'),
    Input('controls-and-dropdown-1', 'value'),
    Input('controls-and-dropdown-2', 'value'),
    Input('tabs-example-graph', 'value'),
    Input('behavior-dropdown', 'value')
)
def update_graph(col_chosen_1, col_chosen_2, active_tab, behavior_col):
    """
    Returns 8 figures:

      1) Scatter Plot
      2) Box Plot of cuisine columns
      3) Bar Plot of 'customer_region' (relative %)
      4) Bar Plot of 'generation' (relative %)
      5) Bar Plot of 'payment_method' (relative %)
      6) Bar Plot: Mean of prop_ columns
      7) Correlation Heatmap among all behavior columns
      8) Histogram for the chosen behavior column
    """

    # Decide which cluster data to filter
    if active_tab == 'tab-7-example-graph':
        df_filtered = df_clust.copy()
        cluster_title = "All Individuals"
    else:
        cluster_name = cluster_map[active_tab]
        df_filtered = df_clust[df_clust['merged_labels_name'] == cluster_name].copy()
        cluster_title = cluster_name

    # 1) Scatter Plot
    fig_scatter = px.scatter(
        df_filtered,
        x=col_chosen_1,
        y=col_chosen_2,
        height=700,
        title=f'Scatter: {col_chosen_1} vs. {col_chosen_2} — {cluster_title}'
    )

    # 2) Box Plot of cuisine columns
    if active_tab == 'tab-7-example-graph':
        df_cui_filtered = df_cui_melted.copy()
    else:
        df_cui_filtered = df_cui_melted[df_cui_melted['merged_labels_name'] == cluster_title]

    fig_box = px.box(
        df_cui_filtered,
        x='cuisine',
        y='proportion',
        height=700,
        title=f'Cuisine Proportion Distribution — {cluster_title}'
    )
    fig_box.update_layout(showlegend=False)
    fig_box.update_xaxes(tickangle=45)

    # Helper for relative percentages
    def to_percentages(local_df, col):
        temp = (
            local_df
            .groupby(local_df[col].astype(str))
            .size()
            .reset_index(name='count')
            .rename(columns={col: f'{col}_str'})
        )
        total_count = temp['count'].sum()
        temp['percentage'] = temp['count'] / total_count * 100 if total_count > 0 else 0
        return temp

    # 3) Bar Plot: customer_region
    df_region_count = to_percentages(df_filtered, 'customer_region')
    fig_region = px.bar(
        df_region_count,
        x='customer_region_str',
        y='percentage',
        text='percentage',
        labels={'percentage': 'Percentage (%)'},
        height=700,
        title=f'Customer Region (% of rows) — {cluster_title}'
    )
    fig_region.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside'
    )

    # 4) Bar Plot: generation
    df_gen_count = to_percentages(df_filtered, 'generation')
    fig_generation = px.bar(
        df_gen_count,
        x='generation_str',
        y='percentage',
        text='percentage',
        labels={'percentage': 'Percentage (%)'},
        height=700,
        title=f'Generation (% of rows) — {cluster_title}'
    )
    fig_generation.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside'
    )

    # 5) Bar Plot: payment_method
    df_payment_count = to_percentages(df_filtered, 'payment_method')
    fig_payment = px.bar(
        df_payment_count,
        x='payment_method_str',
        y='percentage',
        text='percentage',
        labels={'percentage': 'Percentage (%)'},
        height=700,
        title=f'Payment Method (% of rows) — {cluster_title}'
    )
    fig_payment.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside'
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 6) Bar Chart: Mean of columns that start with "prop_"
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if len(df_filtered) > 0:
        # Only consider prop_ columns that actually exist in df_filtered
        existing_prop_cols = [c for c in prop_cols if c in df_filtered.columns]
        if existing_prop_cols:
            mean_values = df_filtered[existing_prop_cols].mean().reset_index()
            mean_values.columns = ["prop_col", "mean_value"]
        else:
            mean_values = pd.DataFrame({"prop_col": [], "mean_value": []})
    else:
        mean_values = pd.DataFrame({"prop_col": [], "mean_value": []})

    fig_prop_mean = px.bar(
        mean_values,
        x="prop_col",
        y="mean_value",
        title=f"Average of prop_ Columns — {cluster_title}",
        labels={"prop_col": "prop_ Column", "mean_value": "Mean Value"},
        height=500
    )
    fig_prop_mean.update_xaxes(tickangle=45)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 7) Correlation Heatmap among all behavior columns
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We'll consider all columns in cust_col that exist in df_filtered
    existing_behavior_cols = [c for c in cust_col if c in df_filtered.columns]
    if len(existing_behavior_cols) > 1:
        corr_data = df_filtered[existing_behavior_cols].corr()
        fig_heatmap = px.imshow(
            corr_data,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title=f"Correlation Heatmap — {cluster_title}",
            aspect="auto",
            height=500
        )
    else:
        # Not enough columns to compute correlation
        fig_heatmap = px.imshow([[0]], text_auto=True)
        fig_heatmap.update_layout(title="Correlation Heatmap (Not enough data)")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 8) Distribution Histogram for chosen behavior column
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if behavior_col in df_filtered.columns:
        fig_hist = px.histogram(
            df_filtered,
            x=behavior_col,
            nbins=30,
            height=500,
            title=f"Distribution of {behavior_col} — {cluster_title}"
        )
    else:
        fig_hist = px.histogram(
            pd.DataFrame({"NoData": []}),
            x="NoData",
            title=f"No data for {behavior_col} in {cluster_title}"
        )

    return (
        fig_scatter,
        fig_box,
        fig_region,
        fig_generation,
        fig_payment,
        fig_prop_mean,
        # fig_heatmap,
        fig_hist
    )

# ------------------------------------------------------------------
# 5) Run the app
# ------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
