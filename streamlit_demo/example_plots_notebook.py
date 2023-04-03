# +
import numpy as np
import pandas as pd
import seaborn as sns  # pip install / optional
import matplotlib.pyplot as plt
import plotly.express as px
import cufflinks as cf

cf.go_offline()


def generate_synthetic_data(
    hour_offset: bool = True,
    weekday_offset: bool = True,
    month_offset: bool = True,
    add_noise: bool = True,
    multiply_with_sin: bool = True,
    freq: str = 'h',
    start: str = '2017-12-01 00:00',
    stop: str = '2023-01-07 23:00',
    trend_start_value: float = 100.0,
    trend_stop_value: float = 200.0,
    col_name: str = 'target'
) -> pd.DataFrame:
    start = pd.to_datetime(start)
    stop = pd.to_datetime(stop)

    # Create placeholder df
    df = pd.DataFrame(index=[start, stop])
    df = df.asfreq(freq)

    # Create linear trend
    if trend_stop_value is not None:
        df[col_name] = np.linspace(trend_start_value, trend_stop_value, len(df))
    else:
        df[col_name] = trend_start_value

    # Offset hour of day
    if hour_offset:
        assert freq == 'h'
        hour_offset_dict = {
            0: 0.2,
            1: 0.1,
            2: 0.1,
            3: 0.05,
            4: 0.05,
            5: 0.1,
            6: 0.5,
            7: 0.7,
            8: 1.1,
            9: 1.2,
            10: 1.1,
            11: 1,
            12: 1,
            13: 0.9,
            14: 1,
            15: 1.1,
            16: 1.15,
            17: 1.1,
            18: 1.1,
            19: 0.9,
            20: 0.85,
            21: 0.75,
            22: 0.5,
            23: 0.3
        }
        for hour in hour_offset_dict.keys():
            df.loc[df.index.hour == hour, col_name] = df.loc[df.index.hour == hour, col_name] * hour_offset_dict[hour]

    # Offset day of week
    if weekday_offset:
        weekday_offset_dict = {
            0: 1.2,  # Monday high
            1: 1.1,
            2: 1.05,
            3: 1,
            4: 1,
            5: 0.90,
            6: 0.90
        }
        for weekday in weekday_offset_dict.keys():
            df.loc[df.index.weekday == weekday, col_name] = (
                    df.loc[df.index.weekday == weekday, col_name] * weekday_offset_dict[weekday])

    # Offset by month
    if month_offset:
        month_offset_dict = {
            7: 0.80
        }
        for month in month_offset_dict.keys():
            df.loc[df.index.month == month, col_name] = (
                    df.loc[df.index.month == month, col_name] * month_offset_dict[month])
    assert df.index.freq in ['h', 'd']

    # add noise
    if add_noise:
        df[col_name] = df[col_name] + np.random.normal(0, trend_stop_value/40, len(df))

    # multiply by a sin wave with some smoothing
    if multiply_with_sin:
        df[col_name] = df[col_name] * (4 + np.sin(np.linspace(0, 2*np.pi, len(df))))/4
        
    df[col_name] = df[col_name].clip(lower=0)

    return df

df = generate_synthetic_data()
# +
# Prio order:
# plotly > seaborn > matplotlib? :-)
# -

df[['target']].iplot()

df[['target']].resample('1W').sum().iloc[1:-1].iplot()

# barplot using cufflinks
df['month'] = df.index.month
df.groupby('month').mean()[['target']].iplot()

df[['target']].iplot(kind='hist', title='Histogram with plotly')

# histogram using seaborn
sns.histplot(data=df, x='target', bins=10, kde=True, color='green')
plt.title('Histogram using Seaborn')
plt.xlabel('Data Value')
plt.ylabel('Frequency')
plt.show()

# distplot using seaborn
sns.kdeplot(data=df, x='target', fill=True, color='purple')
plt.title('Distplot using Seaborn')
plt.xlabel('Data Value')
plt.ylabel('Density')
plt.show()

# +
# boxplot that separates groups using seaborn and hue
df['weekday'] = df.index.weekday
df_groups = pd.concat([
    df.assign(group='group_1'),
    (df.assign(target=lambda x: x.target*1.5).assign(group='group_2'))
])

sns.boxplot(data=df_groups.reset_index(), x='weekday', y='target', hue='group', palette='Set2')
plt.title('Boxplot using Seaborn with Group Separation')
plt.xlabel('Month')
plt.ylabel('Data Value')
plt.legend(title='Quarter', loc='upper right')
plt.show()
# -


