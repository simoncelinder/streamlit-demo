import pandas as pd
from plotly import graph_objects as go

from streamlit_demo.helpers import resample_data


def prepare_first_graph(
    df: pd.DataFrame,
    selected_frequency: str,
    agg_dict: dict,
    selected_products: list
) -> go.Figure:
    resampled_df = resample_data(df, selected_frequency, agg_dict)
    fig = go.Figure()
    for product in selected_products:
        product_df = resampled_df[resampled_df['product'] == product]
        fig.add_trace(go.Scatter(x=product_df['order_date'], y=product_df['quantity_ordered'], mode='lines+markers', name=product))

    fig.update_layout(title=f'Sales Over Time (freq = {selected_frequency})', xaxis_title='Date', yaxis_title='Quantity Sold')
    return fig


def prepare_second_graph(
    df: pd.DataFrame,
    selected_slicer: str,
    selected_products: list
) -> go.Figure:
    if selected_slicer == 'weekday':
        df[selected_slicer] = df['order_date'].dt.weekday
    elif selected_slicer == 'hour_of_day':
        df[selected_slicer] = df['order_date'].dt.hour
    slicer_df = df.groupby([selected_slicer, 'product']).agg({'quantity_ordered': 'sum'}).reset_index()

    fig = go.Figure()
    for product in selected_products:
        product_slicer_df = slicer_df[slicer_df['product'] == product]
        fig.add_trace(go.Scatter(x=product_slicer_df[selected_slicer], y=product_slicer_df['quantity_ordered'],
                                        mode='lines+markers', name=product))
    fig.update_layout(title=f'Sales Sliced by {selected_slicer}', xaxis_title=selected_slicer,
                             yaxis_title='Quantity Sold')

    return fig
