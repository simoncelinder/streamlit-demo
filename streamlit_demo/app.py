import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# Helper functions

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


# Load and process data

abs_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(abs_path, 'data/Sales_September_2019.csv')
df = pd.read_csv(data_path)
df.columns = [c.lower().replace(' ', '_') for c in df.columns]
df = df.dropna()

df['order_date'] = df['order_date'].apply(parse_date)
df['quantity_ordered'] = pd.to_numeric(df['quantity_ordered'], errors='coerce')
df['price_each'] = pd.to_numeric(df['price_each'], errors='coerce')
df = df.dropna()


# Set up Streamlit

st.title('Sales Dashboard')
st.sidebar.header('Filter Options')

product_volume = df.groupby('product')['quantity_ordered'].sum().sort_values(ascending=False)
top_3_products = product_volume.head(3).index.tolist()
unique_products = sorted(df['product'].unique())

frequencies = ['hourly', 'daily', 'weekly', 'monthly']
default_ix = frequencies.index('daily')
selected_frequency = st.sidebar.selectbox('Select Frequency (First Graph)', frequencies, index=default_ix)

slicer = ['weekday', 'hour_of_day']
default_ix = slicer.index('weekday')
selected_slicer = st.sidebar.selectbox('Select Slicer (Second Graph)', slicer, index=default_ix)

selected_product = st.sidebar.multiselect('Select Product (Both Graphs)', unique_products, default=top_3_products)

# Filter data and resample

filtered_df = df[df['product'].isin(selected_product)]
agg_dict = {'quantity_ordered': 'sum', 'price_each': 'mean'}
grouped_data = filtered_df.groupby(['order_date', 'product']).agg(agg_dict).reset_index()
resampled_data = resample_data(grouped_data, selected_frequency, agg_dict)
resampled_data = resampled_data.iloc[0:-1]

# Create first line plot

fig = go.Figure()

for product in selected_product:
    product_df = resampled_data[resampled_data['product'] == product]
    fig.add_trace(go.Scatter(x=product_df['order_date'], y=product_df['quantity_ordered'], mode='lines+markers', name=product))

fig.update_layout(title=f'Sales Over Time (freq = {selected_frequency})', xaxis_title='Date', yaxis_title='Quantity Sold')
st.plotly_chart(fig)


# Create second line plot

if selected_slicer == 'weekday':
    filtered_df[selected_slicer] = filtered_df['order_date'].dt.weekday
elif selected_slicer == 'hour_of_day':
    filtered_df[selected_slicer] = filtered_df['order_date'].dt.hour

slicer_df = filtered_df.groupby([selected_slicer, 'product']).agg({'quantity_ordered': 'sum'}).reset_index()
slicer_df = slicer_df[slicer_df['product'].isin(selected_product)]

weekly_fig = go.Figure()

for product in selected_product:
    product_slicer_df = slicer_df[slicer_df['product'] == product]
    weekly_fig.add_trace(go.Scatter(x=product_slicer_df[selected_slicer], y=product_slicer_df['quantity_ordered'], mode='lines+markers', name=product))

weekly_fig.update_layout(title=f'Sales Sliced by {selected_slicer}', xaxis_title=selected_slicer, yaxis_title='Quantity Sold')
st.plotly_chart(weekly_fig)