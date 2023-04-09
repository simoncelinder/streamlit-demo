import pandas as pd


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


def initial_formatting(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TODO lowercase col names and replace spaces with underscores
    #df.columns = [...]

    df['order_date'] = df['order_date'].apply(parse_date)
    df['quantity_ordered'] = pd.to_numeric(df['quantity_ordered'], errors='coerce')
    df['price_each'] = pd.to_numeric(df['price_each'], errors='coerce')

    # TODO drop all nulls
    # df =

    return df
