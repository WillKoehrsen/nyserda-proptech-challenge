import plotly.express as px
from plotly.offline import plot


def plot_time_series(time_series):
    """
    Plot time series as a line plot.
    """
    row_count = time_series["meter"].nunique()

    line_fig = (
        px.line(
            time_series.reset_index(),
            x="date_time",
            y="consumption",
            facet_row="meter",
            title="Consumption by Meter over Time",
            template="presentation",
            height=row_count * 600,
        )
        .update_yaxes(matches=None)
        .update_xaxes(showticklabels=True, matches=None)
    )

    plot(line_fig, show_link=True, filename="plots/consumption_time_series.html")
