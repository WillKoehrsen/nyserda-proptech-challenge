import os

import numpy as np
import pandas as pd

from src.constants import (
    COLUMNS_TO_KEEP,
    DATASET_CSV_NAME,
    DATE_READ_SETTINGS,
    DEFAULT_FEATURES,
    ELECTRIC_INTERVAL_DATA_CSV_NAME,
    FEATURES_TARGET_CSV_NAME,
    OCCUPANCY_DATA_CSV_NAME,
    TENANT_ELECTRIC_INTERVAL_DATA_CSV_NAME,
    TEST_DATASET_CSV_NAME,
    TESTING_DATE,
)


def create_dataset_from_csvs(save=False):
    """
    Construct a dataset with features and target (consumption) from the
    raw csv data.
    """
    (electric_interval_data, tenant_electric_interval_data, occupancy_data,) = (
        pd.read_csv(
            file_name,
            parse_dates=True,
            index_col="date_time" if file_name != OCCUPANCY_DATA_CSV_NAME else "date",
        )
        for file_name in [
            ELECTRIC_INTERVAL_DATA_CSV_NAME,
            TENANT_ELECTRIC_INTERVAL_DATA_CSV_NAME,
            OCCUPANCY_DATA_CSV_NAME,
        ]
    )

    occupancy_data = calculate_baseline_change(occupancy_data)

    # Linear interpolation of temperature and humidity values
    electric_interval_data = interpolate_weather_features(electric_interval_data)

    # Add time and date features
    electric_interval_data = get_datetime_info(
        electric_interval_data, date_col="index"
    ).rename(columns=dict(date_time_date="date"))

    occupancy_data["date"] = occupancy_data.index.date

    electric_interval_data = (
        electric_interval_data.reset_index(drop=False)
        .merge(occupancy_data.reset_index(drop=True), on="date", how="outer")
        .set_index("date_time")
    ).assign(meter="Building")

    # Get the weather, datetime, and occupancy features for each individual meter
    tenant_electric_interval_data = (
        tenant_electric_interval_data.merge(
            electric_interval_data.drop(
                columns=[
                    "consumption",
                    "max_demand",
                    "min_demand",
                    "avg_demand",
                    "meter",
                ]
            ),
            on="date_time",
        )
        .reset_index()
        .sort_values(by=["meter", "date_time"])
    ).set_index("date_time")

    tenant_electric_interval_data = pd.concat(
        [
            tenant_electric_interval_data,
            electric_interval_data,
        ],
        axis="rows",
    ).drop(columns=["name"])

    if not os.path.exists(TENANT_ELECTRIC_INTERVAL_DATA_CSV_NAME) or save:
        tenant_electric_interval_data.to_csv(DATASET_CSV_NAME)

    testing_data = electric_interval_data[
        electric_interval_data["date"] == TESTING_DATE
    ]

    assert testing_data.shape[0] == 96
    testing_data.to_csv(TEST_DATASET_CSV_NAME)

    return tenant_electric_interval_data


def calculate_baseline_change(occupancy_data):
    """
    Calculate the occupancy change from the baseline, which is the first measurement.
    Adds the occupancy as a percentage of the baseline occupancy.
    """
    occupancy_data = occupancy_data.copy()
    baseline_entries = occupancy_data["entries"][0]
    occupancy_data["baseline_change"] = (
        100 * (occupancy_data["entries"] - baseline_entries) / baseline_entries
    )
    return occupancy_data


def interpolate_weather_features(dataset):
    """
    Linear time-based interpolate of temperature and humidity features.
    """
    dataset = dataset.copy()
    for feature in ["temp", "humidity"]:
        dataset[feature] = dataset[feature].interpolate(
            method="time", limit_direction="both"
        )
    return dataset


def get_datetime_info(
    df, date_col, utc=False, timezone=None, drop=False, additional_attributes=False
):
    """
    Extract date and time information from a column in dataframe
    and add as new columns. If time is originally is UTC but local timezone is provided,
    then features will be in local timezone.

    :param df: pandas dataframe
    :param date_col: string representing the column containing datetimes. Can also be 'index' to use the index
    :param utc: boolean for whether the timestamps in UTC
    :param timezone: string for the time zone. If passed, times are converted to local.
        A local timezone should only be passed if timestamps are originally in utc.
    :param drop: boolean indicating whether the original column should be dropped from
        the dataframe
    :param additional_attributes: boolean for whether additional date attributes should
        be extracted.

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
    fld = pd.to_datetime(fld, utc=utc)

    # If timezone originally in UTC but want in local time
    if timezone:
        # Timestamps must originally be in UTC timezone
        if not utc:
            raise ValueError("Timestamps were not declared as UTC")
        fld = fld.dt.tz_convert(timezone).dt.tz_localize(None)
        df["local"] = fld

    # Used for naming the columns
    prefix += "_"

    # Basic attributes
    attr = ["year", "month", "day", "dayofweek", "dayofyear", "date"]

    if additional_attributes:
        attr = attr + [
            "is_month_end",
            "is_month_start",
            "is_quarter_end",
            "is_quarter_start",
            "is_year_end",
            "is_year_start",
        ]

    # Time attributes
    attr = attr + ["hour", "minute", "second"]

    # Iterate through each attribute and add it to the dataframe
    for n in attr:
        # The week or weekofyear attributes are deprecated, so must use
        # isocalendar().week
        if n == "week":
            df[prefix + n] = getattr(fld.dt.isocalendar(), n.lower())
        else:
            df[prefix + n] = getattr(fld.dt, n.lower())

    # Add fractional time of day by converting to hours
    df[prefix + "fracday"] = (
        (df[prefix + "hour"] / 24)
        + (df[prefix + "minute"] / 60 / 24)
        + (df[prefix + "second"] / 60 / 60 / 24)
    )

    # Add fractional time of week by converting to hours
    df[prefix + "fracweek"] = (
        (df[prefix + "dayofweek"] * 24) + (df[prefix + "fracday"] * 24)
    ) / (7 * 24)

    # Add fractional time of month by converting to hours
    df[prefix + "fracmonth"] = (
        # First day of month is 1
        ((df[prefix + "day"] - 1) * 24)
        + (df[prefix + "fracday"] * 24)
    ) / (fld.dt.daysinmonth * 24)

    # Add fractional time of year by converting to hours
    df[prefix + "fracyear"] = (
        # First day of year is 1
        ((df[prefix + "dayofyear"] - 1) * 24)
        + (df[prefix + "fracday"] * 24)
    ) / (np.where(fld.dt.is_leap_year, 366, 365) * 24)

    # Add a trend which measures days since first measurement
    df[prefix + "trend"] = (fld - fld.min()) / pd.Timedelta(days=1)

    # Drop the column if specified
    if drop:
        if date_col == "index":
            df = df.reset_index().iloc[:, 1:].copy()
        else:
            df = df.drop(date_col, axis=1)

    return df


def remove_extreme_values_from_interval_data(
    intervals,
    meter_name,
    mean_multiplier=4,
    extreme_upper_limit=1_000,
    # Set the lower limit as -1 to deal with '1-010' meter
    lower_limit=-1,
):
    """

    Remove extreme consumption measurements resulting from time gaps
    in the data.

    1. Remove points immediately after gaps of more than 15 minutes
    2. Remove days with total consumption = 0
    3. Remove consumption measurements = 0 only if the intervals has fewer
        than 1% zero values.
    4. Remove measurements more than mean_multiplier greater than the mean
        or less than -1.
    5. Return the dataframe with extreme measurements removed.
    """
    series = intervals.copy().sort_index()

    # Find the difference between successive measurements
    series["time_diff"] = series.index.to_series().diff()

    # Identify gaps in measurements
    gaps = series[series["time_diff"] != pd.Timedelta(minutes=15)]
    new_series = series.drop(gaps.index)

    assert (new_series["time_diff"] == pd.Timedelta(minutes=15)).all()

    # Remove any days with all zero values
    daily_sums = new_series.groupby("date", as_index=False)["consumption"].sum()
    days_to_drop = daily_sums.loc[daily_sums["consumption"] == 0, "date"].unique()
    new_series = new_series[~new_series["date"].isin(days_to_drop)]

    # Find percentage of values that = 0
    percent_equals_zero = 100 * (new_series["consumption"] == 0).sum() / len(new_series)

    if percent_equals_zero < 1:
        new_series = new_series[new_series["consumption"] > 0]

    upper_limit = min(
        mean_multiplier
        * new_series.loc[new_series["consumption"] > 0, "consumption"].abs().mean(),
        extreme_upper_limit,
    )

    outside_limits = new_series[
        ~new_series["consumption"].between(
            # In the case where there are no measurements > 0, set the upper limit to 1
            lower_limit,
            upper_limit if not pd.isna(upper_limit) else 1,
            inclusive=False,
        )
    ]

    new_series = new_series.drop(outside_limits.index)

    print(f"{meter_name} Total rows removed: {len(intervals) - len(new_series)}")

    return new_series


def engineer_features(columns_to_keep=COLUMNS_TO_KEEP):
    """
    Build the new features, remove outliers, and save dataset
    """
    dataset = create_dataset_from_csvs()
    non_extreme_values_dataset = pd.concat(
        [
            remove_extreme_values_from_interval_data(meter_data, meter_name)
            for meter_name, meter_data in dataset.groupby("meter")
        ]
    )
    non_extreme_values_dataset.to_csv(FEATURES_TARGET_CSV_NAME)


def read_features_and_targets():
    return pd.read_csv(
        FEATURES_TARGET_CSV_NAME,
        parse_dates=["date_time", "date"],
        index_col="date_time",
    )
