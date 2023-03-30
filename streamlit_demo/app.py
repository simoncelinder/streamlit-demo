import os

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# TODO: Refactor everything below to be grouped / extracted into cleaner parts

# Get the absolute path of the current file
abs_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(abs_path, 'data/Sales_September_2019.csv')
df = pd.read_csv(data_path)
df.columns = [c.lower().replace(' ', '_') for c in df.columns]
df = df.dropna()


def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%m/%d/%y %H:%M')
    except ValueError:
        return None


def resample_data(df, frequency, agg_dict):
    resampled_data_list = []
    unique_products = df['product'].unique()

    for product in unique_products:
        product_data = df[df['product'] == product]

        if frequency == 'hourly':
            resampled_product_data = product_data.resample('H', on='order_date').agg(agg_dict)
        elif frequency == 'daily':
            resampled_product_data = product_data.resample('D', on='order_date').agg(agg_dict)
        elif frequency == 'weekly':
            resampled_product_data = product_data.resample('W', on='order_date').agg(agg_dict)
        else:
            Exception('Invalid frequency')

        resampled_product_data['product'] = product
        resampled_data_list.append(resampled_product_data)

    resampled_data = pd.concat(resampled_data_list)
    return resampled_data.reset_index()


# Formatting
df['order_date'] = df['order_date'].apply(parse_date)

# Convert columns to numeric and handle non-numeric values
df['quantity_ordered'] = pd.to_numeric(df['quantity_ordered'], errors='coerce')
df['price_each'] = pd.to_numeric(df['price_each'], errors='coerce')

df = df.dropna()

# Set title and sidebar header
st.title('Sales Dashboard')
st.sidebar.header('Filter Options')

# Calculate total sales volume for each product
product_volume = df.groupby('product')['quantity_ordered'].sum().sort_values(ascending=False)

# Get the top 3 products
top_3_products = product_volume.head(3).index.tolist()

# Create product filter with default selection as top 3 products
unique_products = sorted(df['product'].unique())

# Frequency filter
frequencies = ['hourly', 'daily', 'weekly', 'monthly']
default_ix = frequencies.index('daily')
selected_frequency = st.sidebar.selectbox('Select Frequency (First Graph)', frequencies, index=default_ix)

# Other slicer 2nd plot
slicer = ['weekday', 'hour_of_day']
default_ix = slicer.index('weekday')
selected_slicer = st.sidebar.selectbox('Select Slicer (Second Graph)', slicer, index=default_ix)

# Create product filter
selected_product = st.sidebar.multiselect('Select Product (Both Graphs)', unique_products, default=top_3_products)

# Filter df based on selected products
filtered_df = df[df['product'].isin(selected_product)]

# Aggregation dictionary for resampling
agg_dict = {'quantity_ordered': 'sum', 'price_each': 'mean'}


# Group data by date and selected product(s)
grouped_data = (
    filtered_df
    .groupby(['order_date', 'product'])
    .agg(
        {
            'quantity_ordered': 'sum',
            'price_each': 'mean',
        })
    .reset_index()
)

# Resample data based on selected frequency and product(s)
resampled_data = resample_data(grouped_data, selected_frequency, agg_dict)

# Drop last since might not be full week, month etc and looks skewed
resampled_data = resampled_data.iloc[0:-1]

# Create line plot with Plotly
fig = go.Figure()

for product in selected_product:
    product_df = resampled_data[resampled_data['product'] == product]
    fig.add_trace(go.Scatter(x=product_df['order_date'], y=product_df['quantity_ordered'], mode='lines+markers', name=product))

fig.update_layout(title=f'Sales Over Time (freq = {selected_frequency})', xaxis_title='Date', yaxis_title='Quantity Sold')
st.plotly_chart(fig)


# Group df by year, week, and selected product(s)

if selected_slicer == 'weekday':
    filtered_df[selected_slicer] = filtered_df['order_date'].dt.weekday
elif selected_slicer == 'hour_of_day':
    filtered_df[selected_slicer] = filtered_df['order_date'].dt.hour

slicer_df = filtered_df.groupby([selected_slicer, 'product']).agg({'quantity_ordered': 'sum'}).reset_index()

# Filter slicer_df based on selected products
slicer_df = slicer_df[slicer_df['product'].isin(selected_product)]

# Create line plot with Plotly for weekly sales
weekly_fig = go.Figure()

for product in selected_product:
    product_slicer_df = slicer_df[slicer_df['product'] == product]
    weekly_fig.add_trace(go.Scatter(x=product_slicer_df[selected_slicer], y=product_slicer_df['quantity_ordered'], mode='lines+markers', name=product))

weekly_fig.update_layout(title=f'Sales Sliced by {selected_slicer}', xaxis_title=selected_slicer, yaxis_title='Quantity Sold')
st.plotly_chart(weekly_fig)
