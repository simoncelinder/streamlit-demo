import os
import pandas as pd

import streamlit as st

from graph_formatting import prepare_second_graph, prepare_first_graph
from helpers import initial_formatting


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
frequencies = ['hourly', 'daily', 'weekly']
default_ix = frequencies.index('daily')
selected_frequency = st.sidebar.selectbox('Select Frequency (First Graph)', frequencies, index=default_ix)

# Other time slicer for second graph
slicer = ['weekday', 'hour_of_day']
default_ix = slicer.index('weekday')
selected_slicer = st.sidebar.selectbox(
    'Select Slicer (Second Graph)',
    # TODO fill gap here with arguments to the streamlit "selectbox" function (for example google an example / documentation)
    #  or use hints/definition in your IDE/Python environment
)

# Product slicer for both graphs
# TODO - You might first do this in a notebook for below - but make the initial_formatting of the df the same as we do here
#  1) create a list "top_3_products" of the 3 products with the highest volume in df
#  hint: use df.groupby() and df.sort_values()
#  2) Also create a list "unique_products" of all unique products in df,
#  this will also be an argument to st.sidebar.multiselect

# selected_products = st.sidebar.multiselect('Select Product (Both Graphs)', unique_products, default=top_3_products)

# Filter data and resample
# filtered_df = df[df['product'].isin(selected_products)]

# TODO complete this dict for how to aggregate data (for ex using 'mean', 'sum' or something like this.
#  We want to be able to see total quantity ordered together with price per product
#  (Price isnt actually plotted in the current dashbaord but it would be a reasonable extension.)
# agg_dict = {'quantity_ordered': <HIDDEN>, 'price_each': <HIDDEN>}
# per_product_data = filtered_df.groupby(['order_date', 'product']).agg(agg_dict).reset_index()

# Create first line plot
# TODO uncomment below once the data preparation above and also logic inside the helper is completed
#first_fig = prepare_first_graph(per_product_data, selected_frequency, agg_dict, selected_products)
#st.plotly_chart(first_fig)

# Create second line plot
# TODO: uncomment below once the data preparation above is completed (nothing to fix inside the helper)
#second_fig = prepare_second_graph(per_product_data, selected_slicer, selected_products)
#st.plotly_chart(second_fig)
