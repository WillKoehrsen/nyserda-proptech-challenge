import os
import pandas as pd
import numpy as np


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
    attr = ["Year", "Month", "Day", "Dayofweek", "Dayofyear", "Date"]

    if additional_attributes:
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
        # The week or weekofyear attributes are deprecated, so must use
        # isocalendar().week
        if n == "week":
            df[prefix + n] = getattr(fld.dt.isocalendar(), n.lower())
        else:
            df[prefix + n] = getattr(fld.dt, n.lower())

    # Add fractional time of day by converting to hours
    df[prefix + "FracDay"] = (
        (df[prefix + "Hour"] / 24)
        + (df[prefix + "Minute"] / 60 / 24)
        + (df[prefix + "Second"] / 60 / 60 / 24)
    )

    # Add fractional time of week by converting to hours
    df[prefix + "FracWeek"] = (
        (df[prefix + "Dayofweek"] * 24) + (df[prefix + "FracDay"] * 24)
    ) / (7 * 24)

    # Add fractional time of month by converting to hours
    df[prefix + "FracMonth"] = (
        # First day of month is 1
        ((df[prefix + "Day"] - 1) * 24) + (df[prefix + "FracDay"] * 24)
    ) / (fld.dt.daysinmonth * 24)

    # Add fractional time of year by converting to hours
    df[prefix + "FracYear"] = (
        # First day of year is 1
        ((df[prefix + "Dayofyear"] - 1) * 24) + (df[prefix + "FracDay"] * 24)
    ) / (np.where(fld.dt.is_leap_year, 366, 365) * 24)

    # Add a trend which measures days since first measurement
    df[prefix + "Trend"] = (fld - fld.min()) / pd.Timedelta(days=1)

    # Drop the column if specified
    if drop:
        if date_col == "index":
            df = df.reset_index().iloc[:, 1:].copy()
        else:
            df = df.drop(date_col, axis=1)

    return df


## CONVERSION FROM EXCEL TO CSV ##
def read_raw_data():
    """
    Read in raw data, either from Excel or csv.
    """
    date_read_settings = dict(parse_dates=["date"], index_col="date")
    date_time_read_settings = dict(parse_dates=["date_time"], index_col="date_time")

    METER_DATA_CSV_NAME = "data/csvs/ConEd_Electric.csv"

    if not os.path.exists(METER_DATA_CSV_NAME):
        meter_data = pd.read_excel(
            "data/raw/ConEd_Electric.xlsx", **date_time_read_settings
        )
        meter_data.to_csv(METER_DATA_CSV_NAME)
    else:
        print("Meter data found")
        meter_data = pd.read_csv(METER_DATA_CSV_NAME, **date_time_read_settings)

    TENANT_DATA_CSV_NAME = "data/csvs/Tenant_Usage.csv"
    METERS_CSV_NAME = "data/csvs/Meters.csv"
    if not os.path.exists(TENANT_DATA_CSV_NAME):

        # Create an ExcelFile from the tenant usage.
        xls = pd.ExcelFile("data/raw/Tenant_Usage.xlsx")

        sheet_names = xls.sheet_names
        meter_lookup = pd.read_excel(xls, sheet_names[0])
        meter_lookup.to_csv(METERS_CSV_NAME, index=False)

        # Read in all the individual tenant consumption data.
        tenant_data = pd.concat(
            [
                pd.read_excel(xls, sheet_name, **date_time_read_settings).assign(
                    name=sheet_name
                )
                for sheet_name in sheet_names[1:]
            ]
        )
        tenant_data.to_csv(TENANT_DATA_CSV_NAME)
    else:
        print("Tenant and meter lookup data found")
        tenant_data = pd.read_csv(TENANT_DATA_CSV_NAME, **date_time_read_settings)
        meter_lookup = pd.read_csv(METERS_CSV_NAME)

    OCCUPANCY_DATA_CSV_NAME = "data/csvs/Occupancy.csv"
    if not os.path.exists(OCCUPANCY_DATA_CSV_NAME):
        occupancy_data = pd.read_excel("data/raw/Occupancy.xlsx", **date_read_settings)
        occupancy_data.to_csv(OCCUPANCY_DATA_CSV_NAME)
    else:
        print("Occupancy data found")
        occupancy_data = pd.read_csv(OCCUPANCY_DATA_CSV_NAME, **date_read_settings)

    STEAM_DATA_CSV_NAME = "data/csvs/ConEd_Steam.csv"
    if not os.path.exists(STEAM_DATA_CSV_NAME):
        steam_data = pd.read_excel(
            "data/raw/ConEd_Steam.xlsx", **date_time_read_settings
        )
        steam_data.to_csv(STEAM_DATA_CSV_NAME)
    else:
        print("Steam data found")
        steam_data = pd.read_csv(STEAM_DATA_CSV_NAME, **date_time_read_settings)

    return dict(
        meter_data=meter_data,
        tenant_data=tenant_data,
        occupancy_data=occupancy_data,
        steam_data=steam_data,
        meter_lookup=meter_lookup,
    )


def prepare_data():
    """
    Read in and create features from data.
    """
    data_dict = read_raw_data()
    meter_data, tenant_data, occupancy_data, steam_data, meter_lookup = (
        data_dict["meter_data"],
        data_dict["tenant_data"],
        data_dict["occupancy_data"],
        data_dict["steam_data"],
        data_dict["meter_lookup"],
    )

    occupancy_data = calculate_baseline_change(occupancy_data)

    # Linear interpolation of temperature and humidity values
    meter_data = interpolate_weather_features(meter_data)

    # Add time and date features
    meter_data = get_datetime_info(meter_data, date_col="index").rename(
        columns=dict(date_time_Date="date")
    )
    occupancy_data["date"] = occupancy_data.index.date

    meter_data = (
        meter_data.reset_index(drop=False)
        .merge(occupancy_data.reset_index(drop=True), on="date", how="outer")
        .set_index("date_time")
    )

    tenant_data = (
        tenant_data.merge(
            meter_data.drop(
                columns=["consumption", "max_demand", "min_demand", "avg_demand"]
            ),
            on="date_time",
        )
        .reset_index()
        .sort_values(by=["meter", "date_time"])
    ).set_index("date_time")

    tenant_data = pd.concat(
        [tenant_data, meter_data.assign(meter="building")], axis="rows"
    )

    return dict(
        meter_data=meter_data,
        tenant_data=tenant_data,
        occupancy_data=occupancy_data,
        steam_data=steam_data,
        meter_lookup=meter_lookup,
    )


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