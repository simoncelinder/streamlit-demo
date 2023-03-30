import os
import pandas as pd

import streamlit as st

from streamlit_demo.graph_formatting import prepare_second_graph, prepare_first_graph
from streamlit_demo.helpers import initial_formatting


# Load data
abs_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(abs_path, 'data/Sales_September_2019.csv')
df = pd.read_csv(data_path)

# Format initial data for later slicing
df = initial_formatting(df)

# Set up title and sidebar
st.title('Sales Dashboard')
st.sidebar.header('Filter Options')

# Freq slicer for first graph
frequencies = ['hourly', 'daily', 'weekly', 'monthly']
default_ix = frequencies.index('daily')
selected_frequency = st.sidebar.selectbox('Select Frequency (First Graph)', frequencies, index=default_ix)

# Other time slicer for second graph
slicer = ['weekday', 'hour_of_day']
default_ix = slicer.index('weekday')
selected_slicer = st.sidebar.selectbox('Select Slicer (Second Graph)', slicer, index=default_ix)

# Product slicer for both graphs
product_volume = df.groupby('product')['quantity_ordered'].sum().sort_values(ascending=False)
top_3_products = product_volume.head(3).index.tolist()
unique_products = sorted(df['product'].unique())
selected_products = st.sidebar.multiselect('Select Product (Both Graphs)', unique_products, default=top_3_products)

# Filter data and resample
filtered_df = df[df['product'].isin(selected_products)]
agg_dict = {'quantity_ordered': 'sum', 'price_each': 'mean'}
per_product_data = filtered_df.groupby(['order_date', 'product']).agg(agg_dict).reset_index()

# Create first line plot
first_fig = prepare_first_graph(per_product_data, selected_frequency, agg_dict, selected_products)
st.plotly_chart(first_fig)

# Create second line plot
second_fig = prepare_second_graph(per_product_data, selected_slicer, selected_products)
st.plotly_chart(second_fig)
