# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
# import dash_mantine_components as dmc

# Incorporate data
df = pd.read_csv('../Data/DM2425_ABCDEats_DATASET.csv')

df_clust = pd.read_csv('../Data/df_clustering_non_standardized_mergedlabel.csv')
metric_features = df.select_dtypes(include=['number']).columns.tolist()
cui_columns = df.filter(like="CUI_").columns.tolist()
dow_columns = df.filter(like="DOW_").columns.tolist()
hr_columns = df.filter(like="HR_").columns.tolist()

# define the columns that are in the dataset
metric_features_excluding_cui_dow_and_hr = [feat for feat in metric_features if feat not in cui_columns + dow_columns + hr_columns]

# Initialize the app
app = Dash()
app.layout = html.Div(html.H1('Heading', style={'backgroundColor':'blue'}))

# App layout
app.layout = html.Div(
    # style={
    #     'backgroundColor': '#484848',  # example light-blue background
    #     'minHeight': '100vh',         # optional: ensures full viewport height
    #     'margin': '0',                # optional: remove default browser margin
    #     'padding': '20px'            # optional: add space around content
    # },
    children=[
        html.Div(
            
            
            html.H1('Data Mining Cluster Dashboard',
                    ),
            ),

        html.Hr(),
        
        html.Div(

                html.H4('Select your two columns that you want to compare in the scatterplot:')
            ),
        # html.Hr(),

        # First Dropdown
        dcc.Dropdown(
            options=metric_features_excluding_cui_dow_and_hr,
            value=metric_features_excluding_cui_dow_and_hr[0],
            id='controls-and-dropdown-1',
            className='dash-dropdown'
        ),

        # Second Dropdown
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
            # className='dccTabs'
        ),

        dcc.Graph(
            figure={}, 
            id='controls-and-graph',
            className = 'dark-graph'
        )
    ]
)
# Add controls to build the interaction
@callback(
    Output(component_id='controls-and-graph', component_property='figure'),
    Input(component_id='controls-and-dropdown-1', component_property='value'),
    Input(component_id='controls-and-dropdown-2', component_property='value')
)
def update_graph(col_chosen_1, col_chosen_2):
    fig = px.scatter(
        df, 
        x=col_chosen_1, 
        y=col_chosen_2, 
        # color='continent',
        title=f'Scatter Plot of {col_chosen_1} vs. {col_chosen_2}'
    )
    return fig # , fig_2

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
