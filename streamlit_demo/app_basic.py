import pandas as pd
# TODO: Import streamlit here and alias it as st (just as you alias pandas as pd)
from plotly import graph_objects as go

# TODO: Set title of dashboard using st.title(...)

# TODO: Create a dummy list of values for the y-axis (for example 15 values)
# dummy_y = [1, 2, 8, 3, ...]

df = pd.DataFrame(
    {
        'x': range(len(dummy_y)),
        'y': dummy_y
    }
)

fig = go.Figure()
# TODO fill out the missing col names for creating the plotly figure below
#  Hint: Just look at how the dataframe is created above, its nothing fancy!)
fig.add_trace(
    go.Scatter(
        x=df['<MISSING>'],
        y=df['<MISSING>'],
        mode='lines+markers'
    )
)

# TODO: Pass the figure you created above to st.plotly_chart(...) to plot it in the dashboard
