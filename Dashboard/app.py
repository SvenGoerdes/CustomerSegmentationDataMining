from dash import Dash, dash_table, dcc, callback, Output, Input, html
import pandas as pd
import plotly.express as px
import dash_mantine_components as dmc
import json
from sklearn.preprocessing import StandardScaler

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

metric_features = cust_col + cust_col + demogr_col  # old usage in your code

# Filter for all columns that start with 'prop_'
metric_prop_cols = [c for c in df_clust.columns if c.startswith('prop_')]

# append the fowlloing columns to the list 'first_order', 'order_recency', 'total_cui_spending', 'avg_order_value', 'products_per_vendor'
metric_prop_cols.extend(['first_order', 'order_recency', 'total_cui_spending', 'avg_order_value', 'products_per_vendor'])

# standardscale all the metric_prop_cols
scaler = StandardScaler()

# only where is_outliers is False
df_clust_scaled = scaler.fit_transform(df_clust[metric_prop_cols][~df_clust['is_outlier']])

# convert to dataframe
df_clust_scaled = pd.DataFrame(df_clust_scaled, columns=metric_prop_cols)

# add generation column 
df_clust_scaled['generation'] = df_clust['generation']
df_clust_scaled['is_outlier'] = df_clust['is_outlier']
df_clust_scaled['merged_labels_name'] = df_clust['merged_labels_name']


# Identify columns that start with 'prop_' from the behavior set
prop_cols_behav = [c for c in cust_col if c.startswith('prop_')]

time_of_day_cols = [
    "prop_orders_dawn",
    "prop_orders_morning",
    "prop_orders_afternoon",
    "prop_orders_evening"
]
week_cols = [
    "prop_weekend_orders",
    "prop_weekday_orders"
]

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
        html.P('Note: The Dashboard includes plots that display outliers as part of the analysis.'),
        html.Hr(),
        html.Br(),
    
        dmc.Title("Choose the Cluster you want to inspect:", color='#000000', size="h4"),
        dcc.Tabs(
            id="tabs-example-graph",
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

        dmc.Text("Filter by Generation:", size="md", weight=600),
        dcc.Dropdown(
            options=[
                {'label': gen, 'value': gen}
                for gen in sorted(df_clust['generation'].dropna().unique())
            ],
            value=sorted(df_clust['generation'].dropna().unique()),
            placeholder='Select generation(s)...',
            multi=True,
            id='gen-filter',
            style={'marginBottom': '20px', 'width': '700px'}
        ),

        html.Br(),

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # METADATA OVERVIEW SECTION
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                html.Hr(),
                html.Br(),
                html.Br(),
                # make a empty paragaphr
                # Customer Behavior columns
                html.H3("Customer Behavior Columns"),
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
                html.P('''The following columns specify how often a customer orders a certain cuisine. The values are in proportion to the total number of orders.
                        As an example: if a customer orders 10 times in total and 3 times Indian food, the value in the column "prop_orders_indian" would be 0.3.'''),
                # The columns can be overlapping. For example 
                # html.H5('As an example: if a customer orders 10 times in total and 3 times Indian food, the value in the column "prop_orders_indian" would be 0.3.'),
                html.Ul([html.Li(col) for col in cui_col]),
                html.Br(),
                # Demographics columns
                html.H3("Demographics Columns"),
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
        # PLOTS
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        dmc.Title("Cuisine Preferences", color="#000066", size="h2"),
        html.Hr(),

        dmc.Text("Select the two columns for the scatterplot:", size="md", weight=600),
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
                dcc.Graph(figure={}, id='controls-and-graph', className='dark-graph')
            ], span=6),
            dmc.Col([
                dcc.Graph(figure={}, id='cuisine-box-graph')
            ], span=6),
        ], gutter="md"),

        dmc.Title("Demographics", color="#660000", size="h2"),
        html.Hr(),
        dmc.Grid([
            dmc.Col([dcc.Graph(figure={}, id='barplot-region')], span=4),
            dmc.Col([dcc.Graph(figure={}, id='barplot-generation')], span=4),
            dmc.Col([dcc.Graph(figure={}, id='barplot-payment-method')], span=4),
        ], gutter="md"),

        dmc.Title("Behavior", color="#006600", size="h2"),
        html.Hr(),
        dmc.Text("Choose a behavior column to see its distribution:", size="md", weight=600),
        dcc.Dropdown(
            options=cust_col,
            value=cust_col[0],
            id='behavior-dropdown',
            style={'marginBottom': '20px', 'width': '350px'}
        ),

        dmc.Grid([
            dmc.Col([dcc.Graph(id='behavior-mean-prop')], span=6),
            dmc.Col([dcc.Graph(id='behavior-hist')], span=6),
        ], gutter="md"),

        dmc.Grid([
            dmc.Col([dcc.Graph(id='behavior-time-of-day')], span=6),
            dmc.Col([dcc.Graph(id='behavior-weekend')], span=6),
        ], gutter="md"),

        # Heatmap
        dmc.Title("Average Metric Values by Cluster", color="#000000", size="h2"),
        html.Hr(),
        html.P(
            'The heatmap shows the average values by cluster without the outliers. The values have been standardized before. Therefore, a value above 0 means that the cluster has a higher value than the overall average.'
        ),
        dcc.Graph(figure={}, id='cluster-heatmap'),
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
# 4) B) Callback: Return 10 figures for the other tabs
# ------------------------------------------------------------------
@callback(
    Output('controls-and-graph', 'figure'),
    Output('cuisine-box-graph', 'figure'),
    Output('barplot-region', 'figure'),
    Output('barplot-generation', 'figure'),
    Output('barplot-payment-method', 'figure'),
    Output('behavior-mean-prop', 'figure'),
    Output('behavior-hist', 'figure'),
    Output('behavior-time-of-day', 'figure'),
    Output('behavior-weekend', 'figure'),
    Output('cluster-heatmap', 'figure'),
    Input('controls-and-dropdown-1', 'value'),
    Input('controls-and-dropdown-2', 'value'),
    Input('tabs-example-graph', 'value'),
    Input('behavior-dropdown', 'value'),
    Input('gen-filter', 'value')  # new generation filter
)
def update_graph(col_chosen_1, col_chosen_2, active_tab, behavior_col, gen_filter_list):
    """
    Returns 10 figures:
      1) Scatter Plot
      2) Cuisine Box Plot
      3) Bar Plot of 'customer_region' (relative %)
      4) Bar Plot of 'generation' (relative %)
      5) Bar Plot of 'payment_method' (relative %)
      6) Bar Plot: Mean of all prop_ columns
      7) Histogram for chosen behavior column
      8) Time-of-Day Bar Plot
      9) Weekend/Weekday Bar Plot
      10) Heatmap (average metric values by cluster)
    """
    # 1) If on Metadata, return empty figs
    if active_tab == 'metadata-overview':
        empty_fig = px.scatter(title="(No Plot Shown - Metadata Tab)")
        return tuple([empty_fig]*10)

    # 2) Filter by generation first
    if not gen_filter_list:
        # If user unselects everything, show no data or revert to all. 
        # We'll show no data for this example.
        df_filtered = df_clust.copy()
        df_filtered_scaled = df_clust_scaled.copy()
    else:
        df_filtered = df_clust[df_clust['generation'].isin(gen_filter_list)].copy()
        df_filtered_scaled = df_clust_scaled[df_clust_scaled['generation'].isin(gen_filter_list)].copy()

    # 3) Filter by cluster
    if active_tab == 'tab-7-example-graph':
        cluster_title = "All Individuals"
    else:
        cluster_name = cluster_map[active_tab]
        df_filtered = df_filtered[df_filtered['merged_labels_name'] == cluster_name]
        cluster_title = cluster_name

    # ~~~~~~~~~~~ 1) Scatter Plot ~~~~~~~~~~~
    if active_tab == 'tab-7-example-graph':
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

    # ~~~~~~~~~~~ 2) Cuisine Box Plot ~~~~~~~~~~~
    # Melt the filtered DF for the cuisine columns
    # So it reflects only the chosen cluster & generation(s)
    if df_filtered.empty:
        # If there's no data after filtering, create an empty DF
        df_cui_filtered = pd.DataFrame(columns=['merged_labels_name','cuisine','proportion'])
    else:
        # Combine 'merged_labels_name' with the relevant columns, then melt
        df_cui_filtered = df_filtered[['merged_labels_name'] + cui_col].melt(
            id_vars=['merged_labels_name'],
            value_vars=cui_col,
            var_name='cuisine',
            value_name='proportion'
        )

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
        temp['percentage'] = (temp['count'] / total_count * 100) if total_count > 0 else 0
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
        textposition='outside',
        marker_color="rgba(102, 0, 0, 0.5)"
    )

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
        textposition='outside',
        marker_color="rgba(102, 0, 0, 0.5)"
    )

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
        textposition='outside',
        marker_color="rgba(102, 0, 0, 0.5)"
    )

    # ~~~~~~~~~~~ 6) Mean of all "prop_" columns from behavior ~~~~~~~~~~~
    existing_prop_cols = [c for c in prop_cols_behav if c in df_filtered.columns]
    if len(existing_prop_cols) > 0 and not df_filtered.empty:
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
    if (behavior_col in df_filtered.columns) and (not df_filtered.empty):
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
    if existing_tod_cols and not df_filtered.empty:
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
    if existing_week_cols and not df_filtered.empty:
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

    # scale the data for the metric_prop_cols features



    # ~~~~~~~~~~~ 10) Heatmap ~~~~~~~~~~~
    grouped_data = (
        df_filtered_scaled[~df_filtered_scaled['is_outlier']]
        .groupby('merged_labels_name')[metric_prop_cols]
        .mean()
        .reset_index()
    )
    grouped_data_t = grouped_data.set_index('merged_labels_name').T

    # please order the clusters as follows:
    # The Chain Enthusiasts  # The Indian Food Lovers    # The Average Consumers    # The Dawn Spenders    # Frequent High-Spenders    # The Morning Snackers

    grouped_data_t = grouped_data_t[['The Chain Enthusiasts', 'The Indian Food Lovers', 'The Average Consumers', 'The Dawn Spenders', 'Frequent High-Spenders', 'The Morning Snackers']]
    # round the values of grouped_data_t to 2 decimal places
    grouped_data_t = grouped_data_t.round(2)


    fig_heatmap = px.imshow(
        grouped_data_t,
        labels=dict(y="Metric Features", x="Clusters", color="Average Value"),
        x=grouped_data_t.columns,
        y=grouped_data_t.index,
        color_continuous_scale="BrBG",
        text_auto=True,
        title="Average Metric Values by Cluster (without outliers)",
        aspect="auto",
        height=1000
    )

    return (
        fig_scatter,
        fig_box,
        fig_region,
        fig_generation,
        fig_payment,
        fig_prop_mean,
        fig_hist,
        fig_time_of_day,
        fig_weekend,
        fig_heatmap
    )


# ------------------------------------------------------------------
# 5) Run the app
# ------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
