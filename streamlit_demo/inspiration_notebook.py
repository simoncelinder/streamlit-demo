# # !pip install scikit-learn
import pandas as pd
import numpy as np


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


# ## 1) Use plotly and cufflinks for plots!
# - Interactive (more fun and saves a lot of time)
# - Looks more professional
# - You likely need to install these 3: plotly, cufflinks, chart-studio
# - Cufflinks is for being able to call df.iplot() (else need to explicitly import plotly and have more syntax)

# +
import cufflinks as cf

cf.go_offline()
df = generate_synthetic_data()
df.iplot()
# -

# Good to make it a habit to split lines at dot syntax for readibility
# Dont need to bload state of notebook with lots of intermediary variables
# Always use pandas time formatting features if time series
# (Below ignores the effect of linear trend in the data, just an example, should account for that first)
(
    df
    .groupby(df.index.month)
    .sum()
    .iplot(kind='bar')
)

# # 2) Make sure to preproccess the data correctly for choice of model
# - Else not capturing relationships
# - You might embarass yourself
# - You can discuss preprocessing decisions with ChatGPT!

# ## 2.1) Missing to preprocess
# - Imagine clustering store locations on a geographical map where one axis is in km and the other axis is in mm for example => super biased clusters

# +
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import cufflinks as cf

# Enable the cufflinks offline mode to run in Jupyter Notebook
cf.go_offline()

# Set random seed for reproducibility
np.random.seed(42)

# Create 2D dataset with random points centered around 3 centroids
n_samples = 300
centroids = np.array([[0, 0], [3, 3], [-3, 3]])
points_per_cluster = n_samples // centroids.shape[0]

# Generate data points around the centroids
data = np.vstack([np.random.normal(centroid, 1, size=(points_per_cluster, 2)) for centroid in centroids])

# Scale one of the dimensions temporarily
scaling_factor = 10
data[:, 1] = data[:, 1] / scaling_factor

# Fit KMeans to the dataset
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)

# Scale the dimension back to the original scale
data[:, 1] = data[:, 1] * scaling_factor

# Create a DataFrame with the data points and cluster labels
df = pd.DataFrame(data, columns=['X', 'Y'])
df['Cluster'] = kmeans.labels_

# Create an interactive scatter plot
fig = px.scatter(df, x='X', y='Y', color='Cluster', title=f'KMeans Clustering - Biased to split vertically (factor = {scaling_factor})', hover_data=['Cluster'])
fig.add_scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1]*scaling_factor, mode='markers', marker=dict(size=10, color='red', symbol='x'), name='Centroids')
fig.update_layout(width=700, height=700, autosize=False)
fig.show()
# -

# ## 2.2) Preprocess when not needed
# - Tree based models dont care about monotonic transformations, they only care about rank!

# +
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)

# Create a dummy dataset
X = np.random.rand(100, 2)
y = X[:, 0] + 2 * X[:, 1]

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the RandomForest regressor with the scaled data
regressor_scaled = RandomForestRegressor(random_state=42)
regressor_scaled.fit(X_train_scaled, y_train)

# Fit the RandomForest regressor with the original data
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Predict and calculate the mean squared error for the scaled data
y_pred_scaled = regressor_scaled.predict(X_test_scaled)
mse_scaled = mean_squared_error(y_test, y_pred_scaled)

# Predict and calculate the mean squared error for the original data
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error (with standard scaling):", mse_scaled)
print("Mean Squared Error (without standard scaling):", mse)
# -

# # 3) Make sure to choose a reasonable model for the data you are modelling
# - Else might not capture relationships or extrapolate
# - You might embarass yourself

# +
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import cufflinks as cf

cf.go_offline()


# Create a dummy dataset
y = pd.DataFrame(np.arange(100), columns=['y_true'])

# Split into train and test set
n_train = 50
y_train = y[0:n_train]
y_test = y[n_train::]

# Fit RandomForest regressor
regressor = RandomForestRegressor()
regressor.fit(y_train, y_train.values.ravel())

# Make predictions
y_pred = pd.DataFrame(
    index=y_test.index,
    data=regressor.predict(y_test),
    columns=['y_pred_random_forest']
)

# Plot the data and predictions
y_train.columns = ['y_true_train']
y_test.columns = ['y_true_test']
(
    pd.concat([y_train, y_test], axis=0)
    .join(y_pred)
    .iplot(title='Ooops I failed to extrapolate because Im a tree based model!')
)

# +
# # !pip install torch

# +
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
import cufflinks as cf

# Enable the cufflinks offline mode to run in Jupyter Notebook
cf.go_offline()

# Create a new dataset
X = np.linspace(0, 2, 200).reshape(-1, 1)
y = X.copy()

# Split into train and test set
n_train = 100
X_train = X[:n_train].astype(np.float32)
X_test = X[n_train:].astype(np.float32)
y_train = y[:n_train].astype(np.float32)
y_test = y[n_train:].astype(np.float32)

# Convert the data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train)
X_test_tensor = torch.from_numpy(X_test)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)

# Create a neural network with sigmoid activation
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        #x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        return x

model = SimpleNN()

# Set loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()

# Make predictions
y_pred_tensor = model(X_test_tensor)
y_pred = y_pred_tensor.detach().numpy()

# Combine data into a single DataFrame
data = pd.DataFrame({
    'X': np.concatenate([X_train, X_test]).ravel(),
    'y_train': np.concatenate([y_train, np.full_like(y_test, np.nan)]).ravel(),
    'y_test': np.concatenate([np.full_like(y_train, np.nan), y_test]).ravel(),
    'y_pred': np.concatenate([np.full_like(y_train, np.nan), y_pred]).ravel()
})

# Plot the data and predictions using Cufflinks
fig = go.Figure()
fig.add_scatter(x=data['X'], y=data['y_train'], mode='markers', name='Training set', marker=dict(color='blue'))
fig.add_scatter(x=data['X'], y=data['y_test'], mode='markers', name='Test set', marker=dict(color='green'))
fig.add_scatter(x=data['X'], y=data['y_pred'], mode='markers', name='Neural network', marker=dict(color='red'))

fig.update_layout(title='Neural network fails to extrapolate(sigmoid activation)', xaxis_title='X', yaxis_title='y', width=800, height=600)
fig.show()
# -


