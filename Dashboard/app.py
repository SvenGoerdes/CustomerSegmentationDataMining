# dash, dash_table, dcc, callback, etc.
from dash import Dash, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_mantine_components as dmc
import json

# ------------------------------------------------------------------
# 1) Load your data
# ------------------------------------------------------------------
df = pd.read_csv('../Data/DM2425_ABCDEats_DATASET.csv')
df_clust = pd.read_csv('../Data/df_clustering_non_standardized_mergedlabel.csv')

# read in the json with the initial perspectives
with open("initial_perspectives.json", "r") as file:
    perspectives = json.load(file)


# 
# Example data structures (adapt to your scenario)
metric_features = df.select_dtypes(include=['number']).columns.tolist()
cui_columns = df.filter(like="CUI_").columns.tolist()
dow_columns = df.filter(like="DOW_").columns.tolist()
hr_columns = df.filter(like="HR_").columns.tolist()

cui_columns = perspectives['cuisine_preferences']
demographics = perspectives['demographics']
customer_behaviour = perspectives['customer_behavior']

metric_features_excluding_cui_dow_and_hr = [
    feat for feat in metric_features 
    if feat not in cui_columns + dow_columns + hr_columns
]

# Identify all columns that start with 'prop_cui'
prop_cui_cols = [col for col in df_clust.columns if col.startswith('prop_cui')]

# Melt DataFrame for box plot
df_cui_melted = df_clust.melt(
    id_vars=['merged_labels_name'],  # keep cluster label
    value_vars=prop_cui_cols,       # melt these cuisine columns
    var_name='cuisine',
    value_name='proportion'
)

# A map for cluster tabs
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
# 3) Define the layout using Dash Mantine Components
# ------------------------------------------------------------------
app.layout = dmc.Container(
    children=[
        # A nice heading/title
        dmc.Title("Data Mining Cluster Dashboard", color="blue", size="h3"),

        # Dropdowns for X and Y scatter plot selection
        dmc.Text("Select the two columns for the scatterplot:", size="md", weight=600),
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

        # Tabs for cluster selection
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
            ],
            style={'marginBottom': '20px'}
        ),

        # Use a Mantine Grid to place two plots side by side
        dmc.Grid([
            dmc.Col([
                dcc.Graph(
                    figure={},
                    id='controls-and-graph',
                    className='dark-graph'
                )
            ], span=6),  # 6 out of 12 columns

            dmc.Col([
                dcc.Graph(
                    figure={},
                    id='cuisine-box-graph'
                )
            ], span=6),   # 6 out of 12 columns

            dmc.Col([
                dcc.Graph(
                    figure={},
                    id='region_barplot'
                )
            ], span=6)   # 6 out of 12 columns
            
            # dmc.Col([
            #     dcc.Graph(
            #         figure={},
            #         id='cuisine-box-graph'
            #     )
            # ], span=6)   # 6 out of 12 columns




        ]),
    ],
    fluid=True
)

# ------------------------------------------------------------------
# 4) Callback: Two outputs, two inputs + tab
# ------------------------------------------------------------------
@callback(
    Output('controls-and-graph', 'figure'),
    Output('cuisine-box-graph', 'figure'),
    Output('region_barplot', 'figure'),
    Input('controls-and-dropdown-1', 'value'),
    Input('controls-and-dropdown-2', 'value'),
    Input('tabs-example-graph', 'value')
)
def update_graph(col_chosen_1, col_chosen_2, active_tab):
    # Get cluster name from the tab
    cluster_name = cluster_map[active_tab]

    # Filter the main clustering df for the chosen cluster
    df_filtered = df_clust[df_clust['merged_labels_name'] == cluster_name]

    # Scatter plot
    fig_scatter = px.scatter(
        df_filtered,
        x=col_chosen_1,
        y=col_chosen_2,
        height=700,

        title=f'Scatter: {col_chosen_1} vs. {col_chosen_2} — Cluster: {cluster_name}'
    )

    # Filter melted df for the chosen cluster
    df_cui_filtered = df_cui_melted[df_cui_melted['merged_labels_name'] == cluster_name]

    # Box plot for only that cluster's cuisine distribution
    fig_box = px.box(
        df_cui_filtered,
        x='cuisine',
        y='proportion',
        color='cuisine',  # optional
        height=700,

        title=f'Cuisine Proportion Distribution — {cluster_name}'
    )

    fig_region = fig = px.bar(df_filtered,
                   x='customer_region',
                   y='city',
              height = 700)


    fig_box.update_layout(showlegend=False)  # remove legend if repetitive
    fig_box.update_xaxes(tickangle=45)

    return fig_scatter, fig_box, fig_region

# ------------------------------------------------------------------
# 5) Run the app
# ------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
