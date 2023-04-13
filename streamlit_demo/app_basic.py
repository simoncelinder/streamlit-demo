import pandas as pd
import streamlit as st
from plotly import graph_objects as go

st.title('My awesome dashboard')

dummy_y = [1, 2, 8, 3, 15, 14, 5, 6, 2, 5, 1, 8, 7, 4]
df = pd.DataFrame({
    'x': range(len(dummy_y)),
    'y': dummy_y}
)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines+markers'))

st.plotly_chart(fig)