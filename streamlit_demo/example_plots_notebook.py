# +
# %load_ext autoreload
# %autoreload 2

import pandas as pd
import cufflinks as cf

cf.go_offline()
# -


from helpers import initial_formatting

df = pd.read_csv('data/Sales_September_2019.csv')

df = initial_formatting(df)

# # Price per product (not in dahsboard - could be bonus task!)

(
    df
    .groupby('product')
    .mean()
    ['price_each']
    .sort_values(ascending=False)
    .iplot(kind='bar')
)



# # Filter out some main products

# +
top_3 = (
    df
    [['product', 'quantity_ordered']]
    .groupby('product')
    .sum()
    .sort_values('quantity_ordered', ascending=False)
    .head(3)
    .index
)

df = df[df['product'].isin(top_3)]
# -



# # Resampling example 

# ## The function used in the dashboard (explicit loop over products)

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


resample_data(df, 'hourly', {'quantity_ordered': 'sum', 'price_each': 'mean'})

# ## Without for loops

# +
ignore_product = (
    df
    .groupby('order_date')
    .sum()
    [['quantity_ordered']]
)

(
    ignore_product
    .resample('1H')
    .sum()
    #.iloc[0:-1]
    #.rolling(5, center=True)
    #.mean()
    .iplot()
)
# -



# # Group/Slice over some time dimension (hour of day etc)

df['dom'] = df['order_date'].dt.day
df['dow'] = df['order_date'].dt.weekday
df['hod'] = df['order_date'].dt.hour

slice_by = 'hod'
(
    df
    [['quantity_ordered', 'product', slice_by]]
    .groupby(['product', slice_by])
    .sum()
    .unstack(level='product')  # This was a bit new to me, is not in the dashboard solution :-)
).iplot()





# # Other - Likely skip

# +
# If want to avoid explicit for-loop, but can quickly become more advanced

def agg_on_group(df: pd.DataFrame, freq: str, agg_dict: dict) -> pd.DataFrame:
    return df.resample(freq).agg(agg_dict)

(
    df
    .set_index('order_date')
    .groupby('product')
    .apply(agg_on_group, freq='1D', agg_dict={'quantity_ordered': 'sum', 'price_each': 'mean'})
    .unstack(level='product')
    .iplot()
)
