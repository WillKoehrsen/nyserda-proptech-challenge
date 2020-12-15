import pandas as pd


def get_datetime_info(df, date_col, timezone=None, drop=False):
    """
    Extract date and time information from a column in dataframe
    and add as new columns. Time zones are converted to local time if specified.

    :param df: pandas dataframe
    :param date_col: string representing the column containing datetimes. Can also be 'index' to use the index
    :param timezone: string for the time zone. If passed, times are converted to local
    :param drop: boolean indicating whether the original column should be dropped from the df

    :return df: dataframe with added date and time columns
    """
    df = df.copy()

    # Extract the field
    if date_col == "index":
        fld = df.index.to_series()
        prefix = df.index.name if df.index.name is not None else "datetime"
    else:
        fld = df[date_col]
        prefix = date_col

    # Make sure the field type is a datetime
    if timezone is not None:
        fld = pd.to_datetime(fld, utc=True)
    else:
        fld = pd.to_datetime(fld)

    # Convert to local time and then remove time zone information
    if timezone:
        fld = fld.dt.tz_convert(timezone).dt.tz_localize(None)
        df["local"] = fld

    # Used for naming the columns
    prefix += "_"

    # Basic attributes
    attr = ["Year", "Month", "Day", "Dayofweek", "Dayofyear"]

    # Additional attributes to extract
    attr = attr + [
        "Is_month_end",
        "Is_month_start",
        "Is_quarter_end",
        "Is_quarter_start",
        "Is_year_end",
        "Is_year_start",
    ]

    # Time attributes
    attr = attr + ["Hour", "Minute", "Second"]

    # Iterate through each attribute and add it to the dataframe
    for n in attr:
        if n == "week":
            df[prefix + n] = getattr(fld.dt.isocalendar(), n.lower())
        else:
            df[prefix + n] = getattr(fld.dt, n.lower())

    # Add fractional time of day
    df[prefix + "FracDay"] = (
        (df[prefix + "Hour"] / 24)
        + (df[prefix + "Minute"] / 60 / 24)
        + (df[prefix + "Second"] / 60 / 60 / 24)
    )

    # Add fractional time of week
    df[prefix + "FracWeek"] = (
        (df[prefix + "Dayofweek"] * 24) + (df[prefix + "FracDay"] * 24)
    ) / (7 * 24)

    # Drop the column if specified
    if drop:
        if date_col == "index":
            df = df.reset_index().iloc[:, 1:].copy()
        else:
            df = df.drop(date_col, axis=1)

    return df


from src.exploration import prepare_data

data_dict = prepare_data()

meter_data = data_dict["meter_data"]
meter_data.head()

tenant_usage_data = data_dict["tenant_data"]
tenant_usage_data.head()

tenant_sums = tenant_usage_data.groupby(tenant_usage_data.index)[
    "consumption", "max_demand", "min_demand", "avg_demand"
].sum()

data_dict["meter_lookup"]
occupancy_data = data_dict["occupancy_data"]

daily_meter_sums = meter_data.groupby(meter_data.index.date)["consumption"].sum()
daily_totals = occupancy_data.merge(daily_meter_sums, left_index=True, right_index=True)

daily_tenant_sums = (
    tenant_usage_data.groupby(
        [tenant_usage_data.index.date, tenant_usage_data["meter"]]
    )["consumption"]
    .sum()
    .reset_index()
    .rename(columns=dict(level_0="date"))
)

daily_tenant_sums_pivoted = daily_tenant_sums.pivot_table(columns='meter', index='date')
daily_tenant_sums_pivoted.merge(occupancy_data, left_index=True, right_index=True).corr()