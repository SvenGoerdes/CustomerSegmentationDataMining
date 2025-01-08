from dash import Dash, dash_table, dcc, callback, Output, Input, html
import pandas as pd
import plotly.express as px
import dash_mantine_components as dmc

# ------------------------------------------------------------------
# 1) Load and prepare data
# ------------------------------------------------------------------
df = pd.read_csv('../Data/DM2425_ABCDEats_DATASET.csv')
df_clust = pd.read_csv('../Data/df_clustering_non_standardized_mergedlabel.csv')

# Numeric columns, ignoring CUI/DOW/HR columns (example from your code)
metric_features = df.select_dtypes(include=['number']).columns.tolist()
cui_columns = df.filter(like="CUI_").columns.tolist()
dow_columns = df.filter(like="DOW_").columns.tolist()
hr_columns = df.filter(like="HR_").columns.tolist()

metric_features_excluding_cui_dow_and_hr = [
    feat for feat in metric_features
    if feat not in cui_columns + dow_columns + hr_columns
]

# Cuisine columns (relative spending)
prop_cui_cols = [col for col in df_clust.columns if col.startswith('prop_cui')]

# Melt DF for box plots
df_cui_melted = df_clust.melt(
    id_vars=['merged_labels_name'],
    value_vars=prop_cui_cols,
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
        dmc.Title("Data Mining Cluster Dashboard", color= '#000000', size="h1"),
        # dmc.Text("Data Mining Cluster Dashboard", color="blue", size="h2"),
        html.Hr(),
        html.Br(),
    
        dmc.Title("Choose the Cluster:", color= '#000000', size="h4"),
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

        dmc.Text("Select the two columns for the scatterplot:", size="md", weight=600
                 , id = "select-columns"),
        dcc.Dropdown(
            options=metric_features_excluding_cui_dow_and_hr,
            value=metric_features_excluding_cui_dow_and_hr[0],
            id='controls-and-dropdown-1',
            className='dash-dropdown',
            style={'marginBottom': '10px', 'width': '250px'}
        ),

        dcc.Dropdown(
            options=metric_features_excluding_cui_dow_and_hr,
            value=metric_features_excluding_cui_dow_and_hr[1],
            id='controls-and-dropdown-2',
            className='dash-dropdown',
            style={'marginBottom': '20px', 'width': '250px'}
        ),

        dmc.Title("Cuisine preferences", color="blue", size="h2"),
        html.Hr(),
        # First row: Scatter + Box Plot
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

        
        dmc.Title("Demographics", color="blue", size="h2"),
        html.Hr(),
        # Second row: Three bar plots side-by-side
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



    dmc.Title("Behaviour", color="blue", size="h2"),
    html.Hr(),


    ],
    fluid=True,
    style={"padding": "10px"}
)

# ------------------------------------------------------------------
# 4) Callback: Return all 5 figures
# ------------------------------------------------------------------
@callback(
    Output('controls-and-graph', 'figure'),
    Output('cuisine-box-graph', 'figure'),
    Output('barplot-region', 'figure'),
    Output('barplot-generation', 'figure'),
    Output('barplot-payment-method', 'figure'),
    Input('controls-and-dropdown-1', 'value'),
    Input('controls-and-dropdown-2', 'value'),
    Input('tabs-example-graph', 'value')
)
def update_graph(col_chosen_1, col_chosen_2, active_tab):
    """
    Returns 5 figures:

      1) Scatter (controls-and-graph)
      2) Box plot of cuisine proportions (cuisine-box-graph)
      3) Bar plot of 'customer_region' (barplot-region) -> relative percentages
      4) Bar plot of 'generation' (barplot-generation) -> relative percentages
      5) Bar plot of 'Payment method' (barplot-payment-method) -> relative percentages
    """

    # A) Filter the cluster
    if active_tab == 'tab-7-example-graph':
       df_filtered = df_clust
    else:
        cluster_name = cluster_map[active_tab]
        df_filtered = df_clust[df_clust['merged_labels_name'] == cluster_name]

    # 1) Scatter Plot
    fig_scatter = px.scatter(
        df_filtered,
        x=col_chosen_1,
        y=col_chosen_2,
        height =700,
        title=f'Scatter: {col_chosen_1} vs. {col_chosen_2} ({cluster_name})'
    )

    # 2) Box Plot (cuisine proportions)
    df_cui_filtered = df_cui_melted[df_cui_melted['merged_labels_name'] == cluster_name]
    fig_box = px.box(
        df_cui_filtered,
        x='cuisine',
        y='proportion',
        height =700,
        title=f'Cuisine Proportion Distribution ({cluster_name})'
    )
    fig_box.update_layout(showlegend=False)
    fig_box.update_xaxes(tickangle=45)

    # ----------------------------------------------------------------------
    # Helper function: Given a dataframe and a column, return df with count & %
    # ----------------------------------------------------------------------
    def to_percentages(df_local, col):
        """
        1) Interpret col as string.
        2) Count how many rows per category.
        3) Compute share of total as percentage.
        4) Return a DataFrame with columns [col_as_str, count, percentage].
        """
        temp = (
            df_local
            .groupby(df_local[col].astype(str))  # cast to string
            .size()
            .reset_index(name='count')
            .rename(columns={col: f'{col}_str'})  # rename for clarity
        )
        total_count = temp['count'].sum()
        temp['percentage'] = temp['count'] / total_count * 100
        return temp

    # 3) Bar Plot: customer_region
    df_region_count = to_percentages(df_filtered, 'customer_region')
    fig_region = px.bar(
        df_region_count,
        x='customer_region_str',
        y='percentage',
        text='percentage',  # display text on bars
        labels={'percentage': 'Percentage (%)'},
        height =700,
        title=f'Customer Region (% of rows) — {cluster_name}'
    )
    fig_region.update_traces(
        texttemplate='%{text:.2f}%',   # format the text as XX.XX%
        textposition='outside'
    )

    # 4) Bar Plot: generation
    df_gen_count = to_percentages(df_filtered, 'generation')
    fig_generation = px.bar(
        df_gen_count,
        x='generation_str',    # from rename in to_percentages()
        y='percentage',
        text='percentage',
        labels={'percentage': 'Percentage (%)'},
        height =700,
        title=f'Generation (% of rows) — {cluster_name}'
    )
    fig_generation.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside'
    )

    # 5) Bar Plot: Payment method
    df_payment_count = to_percentages(df_filtered, 'payment_method')
    fig_payment = px.bar(
        df_payment_count,
        x='payment_method_str',
        y='percentage',
        text='percentage',
        labels={'percentage': 'Percentage (%)'},
        height =700,
        title=f'Payment_method (% of rows) — {cluster_name}'
    )
    fig_payment.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside'
    )

    return fig_scatter, fig_box, fig_region, fig_generation, fig_payment



# ------------------------------------------------------------------
# 5) Run the app
# ------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
