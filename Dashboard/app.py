# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px

# Incorporate data
df = pd.read_csv('../Data/DM2425_ABCDEats_DATASET.csv')
df_clust = pd.read_csv('../Data/df_clustering_non_standardized_mergedlabel.csv')

metric_features = df_clust.select_dtypes(include=['number']).columns.tolist()
cui_columns = df_clust.filter(like="CUI_").columns.tolist()
dow_columns = df_clust.filter(like="DOW_").columns.tolist()
hr_columns = df_clust.filter(like="HR_").columns.tolist()

# define the columns that are in the dataset
metric_features_excluding_cui_dow_and_hr = [
    feat for feat in metric_features 
    if feat not in cui_columns + dow_columns + hr_columns
]

# Dictionary to map each tab's value to the respective cluster name
cluster_map = {
    'tab-1-example-graph': 'The Chain Enthusiasts',
    'tab-2-example-graph': 'The Indian Food Lovers',
    'tab-3-example-graph': 'The Average Consumers',
    'tab-4-example-graph': 'The Dawn Spenders',
    'tab-5-example-graph': 'Frequent High-Spenders',
    'tab-6-example-graph': 'The Morning Snackers'
}

# Initialize the app
app = Dash()
app.layout = html.Div(html.H1('Heading', style={'backgroundColor':'blue'}))

# App layout
app.layout = html.Div(
    children=[
        html.Div(
            html.H1('Data Mining Cluster Dashboard'),
        ),
        
        html.Hr(),
        
        html.Div(
            html.H4('Select your two columns that you want to compare in the scatterplot:')
        ),

        dcc.Dropdown(
            options=metric_features_excluding_cui_dow_and_hr,
            value=metric_features_excluding_cui_dow_and_hr[0],
            id='controls-and-dropdown-1',
            className='dash-dropdown'
        ),

        dcc.Dropdown(
            options=metric_features_excluding_cui_dow_and_hr,
            value=metric_features_excluding_cui_dow_and_hr[1],
            id='controls-and-dropdown-2',
            className='dash-dropdown'
        ),

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
        ),

        dcc.Graph(
            figure={},
            id='controls-and-graph',
            className='dark-graph'
        )
    ]
)

# Add controls to build the interaction
@callback(
    Output('controls-and-graph', 'figure'),
    Input('controls-and-dropdown-1', 'value'),
    Input('controls-and-dropdown-2', 'value'),
    Input('tabs-example-graph', 'value')  # <-- NEW INPUT for the tabs
)
def update_graph(col_chosen_1, col_chosen_2, active_tab):
    # 1) Identify the cluster name based on the active tab
    cluster_name = cluster_map[active_tab]

    # 2) Filter your clustering dataframe based on the chosen cluster
    df_filtered = df_clust[df_clust['merged_labels_name'] == cluster_name]

    # 3) Create your scatter plot on the filtered dataframe
    fig = px.scatter(
        df_filtered, 
        x=col_chosen_1, 
        y=col_chosen_2, 
        title=f'Scatter Plot of {col_chosen_1} vs. {col_chosen_2} — Cluster: {cluster_name}'
    )

    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
