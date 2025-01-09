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

cust_col = initial_perspective['customer_behavior']     # e.g. behavior columns
cui_col = initial_perspective['cuisine_preferences']    # e.g. cuisine columns
demogr_col = initial_perspective['demographics']        # e.g. region/generation/payment_method, etc.

# combine all columns
all_cols = cust_col + cui_col + demogr_col

# filter columns that are numeric
numeric_features = df_clust[all_cols].select_dtypes(include=['number']).columns

# For demonstration, you originally used the same list twice:
metric_features = cust_col + cust_col  + demogr_col

# Filter for all columns that start with 'prop_'
metric_prop_cols = [c for c in df_clust.columns if c.startswith('prop_')]

# Identify columns that start with 'prop_' from the behavior set
prop_cols_behav = [c for c in cust_col if c.startswith('prop_')]

# The four time-of-day columns
time_of_day_cols = [
    "prop_orders_dawn",
    "prop_orders_morning",
    "prop_orders_afternoon",
    "prop_orders_evening"
]
# The weekend/weekday columns
week_cols = [
    "prop_weekend_orders",
    "prop_weekday_orders"
]

# Melt DF for box plots (using the cuisine columns)
df_cui_melted = df_clust.melt(
    id_vars=['merged_labels_name'],
    value_vars=cui_col,
    var_name='cuisine',
    value_name='proportion'
)

# Tab-to-cluster mapping (for everything except the metadata tab & "all individuals" tab)
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
        # Title & Tabs
        dmc.Title("Data Mining Cluster Dashboard", color='#000000', size="h1"),
        # write a small disclamier that we included the outliers into the dataset and used the nearest centroid to assign the cluster label to the outliers
        html.P('Disclaimer: The Dashboard shows plots that include the outliers. We used the nearest centroid to assign the cluster label to the outliers.'),




        html.Hr(),
        html.Br(),
    
        dmc.Title("Choose the Cluster you want to inspect:", color='#000000', size="h4"),
        dcc.Tabs(
            id="tabs-example-graph",
            # value='tab-1-example-graph',
            value='tab-1-example-graph',
            children=[
                dcc.Tab(label='Metadata Overview', value='metadata-overview'),
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

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # A) METADATA OVERVIEW SECTION
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        html.Div(
            id='metadata-overview-content',
            style={'display': 'none'},  # hidden by default
            children=[
                # A nice heading
                dmc.Title("Metadata: Overview", color="blue", size="h2"),
                html.Hr(),
                # Short introduction about the metadata and dataset. The dataset contains about 31236 rows and 44 columns. The original columns have been preprocessed and transformed into 
                # relative proportions, which are used for clustering. The dataset contains information about customer behavior, cuisine preferences, and demographics.
                # 

                html.B(
                    'The dataset contains about 31236 rows and 44 columns. '
                    'Some of the original columns have been preprocessed and transformed into relative proportions, '
                    'which are used for clustering and for the dashboard. The dataset information/columns can be splitted into customer behavior, '
                    'cuisine preferences, and demographics. '
                    'Additionally, we have added the outliers back into the dataset for the dashboard. We used the nearest centroid to assign the cluster label to the outliers.'


                ),

                html.Br(),
                html.Br(),
                html.Br(),
                html.B('The following sections provide an overview of the columns in the dataset:'),
                html.Br(),


                html.Br(),
                html.Br(),
                # make a empty paragaphr


                # Customer Behavior columns
                html.H3("Customer Behavior Columns"),
                html.Hr(),
                # add a short text about the metadata of the customer behaviour columns
                # html.P('This is some text about the metadata of the behaviour columns'),
                # html.Ul([html.Li(col) for col in cust_col]),

                html.P('The behaviour columns are defined as following.'),
                

                html.Ul([
                html.Li("prop_chain_orders — The share of total orders placed at chain restaurants."),
                html.Li("prop_weekend_orders — The proportion of orders on weekends (Sat/Sun)."),
                html.Li("prop_weekday_orders — The proportion of orders on weekdays (Mon–Fri)."),
                html.Li("prop_orders_dawn — Fraction of orders during dawn hours."),
                html.Li("prop_orders_morning — Fraction of orders in the morning."),
                html.Li("prop_orders_afternoon — Fraction of orders in the afternoon."),
                html.Li("prop_orders_evening — Fraction of orders in the evening."),
                html.Li("first_order — A number indicating when the first order was. A 0 indicates at the start of the observation period."),
                html.Li("last_order — A number indicating when the last order was. A 90 indicates at the end of the observation period"),
                html.Li("order_recency — Time elapsed since the last order in the A 90 days timeframe of the dataset."),
                html.Li("product_count — How many distinct products the customer has purchased."),
                html.Li("vendor_count — How many different vendors the customer has used."),
                html.Li("total_cui_spending — The total amount spent (across all cuisine orders)."),
                html.Li("total_orders — The total number of orders the customer has made."),
                html.Li("avg_daily_orders — Average orders per day, showing overall ordering frequency."),
                html.Li("avg_order_value — Average monetary value per order."),
                html.Li("products_per_vendor — Average distinct products purchased per vendor."),
                ]),
                # Cuisine columns
                html.H3("Cuisine Preferences Columns"),
                html.Hr(),
                html.P('''The following columns specify how often a customer orders a certain cuisine. The values are in proportion to the total value of orders.
                        As an example: if a customer orders 100 Euros worth of food total and 30 Euros in the section Indian food, the value in the column "prop_orders_indian" would be 0.3.'''),
                # The columns can be overlapping. For example 
                # html.H5('As an example: if a customer orders 10 times in total and 3 times Indian food, the value in the column "prop_orders_indian" would be 0.3.'),

                html.Ul([html.Li(col) for col in cui_col]),

                html.Br(),

                # Demographics columns
                html.H3("Demographics Columns"),
                html.Hr(),
                html.Ul([
                    html.Li("customer_region — The geographic region where the customer resides."),
                    html.Li("city — The city where the customer resides. Represented as an int."),
                    html.Li("generation — The customer’s generational group (e.g., Gen-Z, Millennial)."),
                    html.Li("customer_age — The numerical age of the customer."),
                    html.Li("last_promo — Details of the most recent promo the customer used."),
                    html.Li("payment_method — The method used by the customer to pay for orders."),
                    html.Li("promo_used — A binary indicating whether the customer used a promotional offer."),
                ]),

            ]
        ),

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # B) REMAINDER OF LAYOUT (PLOTS)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Cuisine Section
        dmc.Title("Cuisine Preferences", color="#000066", size="h2"),
        html.Hr(),
        # Scatter Plot Controls
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
        dmc.Title("Demographics", color="#660000", size="h2"),
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
        dmc.Title("Behavior", color="#006600", size="h2"),
        html.Hr(),
        dmc.Text("Choose a behavior column to see its distribution:", size="md", weight=600),
        dcc.Dropdown(
            options=cust_col,   # all behavior columns from your JSON
            value=cust_col[0],  # default selection
            id='behavior-dropdown',
            style={'marginBottom': '20px', 'width': '350px'}
        ),

        # Behavior Plots: 2 rows, 2 columns each
        # Row 1: Mean prop_... + histogram
        dmc.Grid([
            dmc.Col([
                dcc.Graph(id='behavior-mean-prop')  # 6) Mean proportion bar
            ], span=6),
            dmc.Col([
                dcc.Graph(id='behavior-hist')       # 7) Distribution histogram
            ], span=6),
        ], gutter="md"),

        # Row 2: Time-of-day + Weekend/Weekday
        dmc.Grid([
            dmc.Col([
                dcc.Graph(id='behavior-time-of-day')  # 8) Time-of-day bar
            ], span=6),
            dmc.Col([
                dcc.Graph(id='behavior-weekend')      # 9) Weekend vs. Weekday
            ], span=6),
        ], gutter="md"),



        # Heatmap
        dmc.Title("Average Metric Values by Cluster", color="#000000", size="h2"),
        html.Hr(),
        html.P('The heatmap shows the average metric values by cluster without the outliers. The values are between 0 and 1. For more information about the clusters, please refer to the "Metadata Overview" tab.'),
        dcc.Graph(
            figure={},
            id='cluster-heatmap',

        ),
    ],
    fluid=True,
    style={"padding": "10px"}
)

# ------------------------------------------------------------------
# 4) A) Callback to SHOW/HIDE Metadata tab
# ------------------------------------------------------------------
@callback(
    Output('metadata-overview-content', 'style'),
    Input('tabs-example-graph', 'value')
)
def toggle_metadata_overview(selected_tab):
    """
    If 'metadata-overview' is selected, we show the metadata.
    Otherwise, we hide it.
    """
    if selected_tab == 'metadata-overview':
        return {'display': 'block', 'marginBottom': '20px'}
    else:
        return {'display': 'none'}

# ------------------------------------------------------------------
# 4) B) Callback: Return 9 figures for the other tabs
# ------------------------------------------------------------------
@callback(
    Output('controls-and-graph', 'figure'),
    Output('cuisine-box-graph', 'figure'),
    Output('barplot-region', 'figure'),
    Output('barplot-generation', 'figure'),
    Output('barplot-payment-method', 'figure'),
    Output('behavior-hist', 'figure'),
    Output('behavior-mean-prop', 'figure'),
    Output('behavior-time-of-day', 'figure'),
    Output('behavior-weekend', 'figure'),
    Output('cluster-heatmap', 'figure'),
    Input('controls-and-dropdown-1', 'value'),
    Input('controls-and-dropdown-2', 'value'),
    Input('tabs-example-graph', 'value'),
    Input('behavior-dropdown', 'value')
)
def update_graph(col_chosen_1, col_chosen_2, active_tab, behavior_col):
    """
    Returns 9 figures:

      1) Scatter Plot
      2) Cuisine Box Plot
      3) Bar Plot of 'customer_region' (relative %)
      4) Bar Plot of 'generation' (relative %)
      5) Bar Plot of 'payment_method' (relative %)
      6) Bar Plot: Mean of all prop_ columns
      7) Histogram for chosen behavior column
      8) Time-of-Day Bar Plot (average of 4 columns)
      9) Weekend/Weekday Bar Plot (average of 2 columns)
    """
    # If the user is on the "Metadata Overview" tab, just return 9 empty figures:
    if active_tab == 'metadata-overview':
        empty_fig = px.scatter(title="(No Plot Shown - Metadata Tab)")
        return (empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, empty_fig, empty_fig)

    # ~~~~~~~~~~~ Decide which cluster data to filter ~~~~~~~~~~~
    if active_tab == 'tab-7-example-graph':
        df_filtered = df_clust.copy()
        cluster_title = "All Individuals"
    else:
        cluster_name = cluster_map[active_tab]
        df_filtered = df_clust[df_clust['merged_labels_name'] == cluster_name].copy()
        cluster_title = cluster_name

    # ~~~~~~~~~~~ 1) Scatter Plot ~~~~~~~~~~~
    if active_tab == 'tab-7-example-graph':
        # color by cluster if "All Individuals"
        fig_scatter = px.scatter(
            df_filtered,
            x=col_chosen_1,
            y=col_chosen_2,
            height=700,
            color='merged_labels_name',
            title=f'Scatter: {col_chosen_1} vs. {col_chosen_2} — {cluster_title}'
        )
    else:
        fig_scatter = px.scatter(
            df_filtered,
            x=col_chosen_1,
            y=col_chosen_2,
            height=700,
            title=f'Scatter: {col_chosen_1} vs. {col_chosen_2} — {cluster_title}'
        )
        fig_scatter.update_traces(marker_color="rgba(0, 0, 102, 0.5)")

    # ~~~~~~~~~~~ 2) Box Plot: Cuisine ~~~~~~~~~~~
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
    fig_box.update_traces(marker_color="rgba(0, 0, 102, 0.5)")

    # Helper for relative percentages in demographics
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

    # ~~~~~~~~~~~ 3) Bar Plot: customer_region ~~~~~~~~~~~
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
    fig_region.update_traces(marker_color="rgba(102, 0, 0, 0.5)")
    # ~~~~~~~~~~~ 4) Bar Plot: generation ~~~~~~~~~~~
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

    # fig_generation.update_traces(marker_color="rgba(102, 0, 0, 0.5)")
    fig_generation.update_traces(marker_color="rgba(102, 0, 0, 0.5)")

    # ~~~~~~~~~~~ 5) Bar Plot: payment_method ~~~~~~~~~~~
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

    fig_payment.update_traces(marker_color="rgba(102, 0, 0, 0.5)")


    # ~~~~~~~~~~~ 6) Mean of all "prop_" columns  of the behavior columns ~~~~~~~~~~~
    existing_prop_cols = [c for c in prop_cols_behav if c in df_filtered.columns]
    if len(existing_prop_cols) > 0 and len(df_filtered) > 0:
        mean_values = df_filtered[existing_prop_cols].mean().reset_index()
        mean_values.columns = ["prop_col", "mean_value"]
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
    fig_prop_mean.update_traces(marker_color="rgba(0, 102, 0, 0.5)")

    # ~~~~~~~~~~~ 7) Distribution Histogram for chosen column ~~~~~~~~~~~
    if behavior_col in df_filtered.columns:
        fig_hist = px.histogram(
            df_filtered,
            x=behavior_col,
            nbins=30,
            height=500,
            title=f"Distribution of {behavior_col} — {cluster_title}"
        )
        fig_hist.update_traces(marker_color="rgba(0, 102, 0, 0.5)")
    else:
        fig_hist = px.histogram(
            pd.DataFrame({"NoData": []}),
            x="NoData",
            title=f"No data for {behavior_col} in {cluster_title}"
        )

    # ~~~~~~~~~~~ 8) Time-of-Day Bar Plot ~~~~~~~~~~~
    existing_tod_cols = [c for c in time_of_day_cols if c in df_filtered.columns]
    if existing_tod_cols and len(df_filtered) > 0:
        mean_tod = df_filtered[existing_tod_cols].mean().reset_index()
        mean_tod.columns = ["time_of_day", "avg_value"]
    else:
        mean_tod = pd.DataFrame({"time_of_day": [], "avg_value": []})

    fig_time_of_day = px.bar(
        mean_tod,
        x="time_of_day",
        y="avg_value",
        title=f"Average Time-of-Day Orders — {cluster_title}",
        labels={"time_of_day": "Time of Day", "avg_value": "Avg Proportion"},
        height=500
    )
    fig_time_of_day.update_xaxes(tickangle=45)
    fig_time_of_day.update_traces(marker_color="rgba(0, 102, 0, 0.5)")

    # ~~~~~~~~~~~ 9) Weekend/Weekday Bar Plot ~~~~~~~~~~~
    existing_week_cols = [c for c in week_cols if c in df_filtered.columns]
    if existing_week_cols and len(df_filtered) > 0:
        mean_weeks = df_filtered[existing_week_cols].mean().reset_index()
        mean_weeks.columns = ["week_col", "avg_value"]
    else:
        mean_weeks = pd.DataFrame({"week_col": [], "avg_value": []})

    fig_weekend = px.bar(
        mean_weeks,
        x="week_col",
        y="avg_value",
        title=f"Average Weekend/Weekday Orders — {cluster_title}",
        labels={"week_col": "Column", "avg_value": "Avg Proportion"},
        height=500
    )
    fig_weekend.update_xaxes(tickangle=45)
    fig_weekend.update_traces(marker_color="rgba(0, 102, 0, 0.5)")



    #  calculate the average metric values by cluster withouth the outliers
    # and create a heatmap
    # 
# 1) Group data and remove outliers
    grouped_data = (
        df_clust[~df_clust['is_outlier']]
        .groupby('merged_labels_name')[metric_prop_cols]
        .mean()
        .reset_index()
)

# 2) Transpose the DataFrame so clusters become columns
#    and metric features become row indices
    grouped_data_t = grouped_data.set_index('merged_labels_name').T

# 3) Create the heatmap with the transposed data
    fig_heatmap = px.imshow(
        grouped_data_t,                 # transposed data
        labels=dict(y="Metric Features", x="Clusters", color="Average Value"),
        x=grouped_data_t.columns,       # cluster names
        y=grouped_data_t.index,         # metric feature names
        color_continuous_scale="BrBG",
        text_auto=True,                 # automatically add text (cell values)
        title="Average Metric Values by Cluster (without outliers)",
        aspect="auto",
        height=1000
    )



    # Return all 10 figures
    return (
        fig_scatter,      # 1
        fig_box,          # 2
        fig_region,       # 3
        fig_generation,   # 4
        fig_payment,      # 5
        fig_prop_mean,    # 6
        fig_hist,         # 7
        fig_time_of_day,  # 8
        fig_weekend,      # 9
        fig_heatmap       #10
    )


# ------------------------------------------------------------------
# 5) Run the app
# ------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
