import os
import pandas as pd

from src.utils import get_datetime_info

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

    # Linear interpolation of temperature and humidity values
    meter_data = interpolate_weather_features(meter_data)

    # Add time and date features
    meter_data = get_datetime_info(meter_data, date_col="index")

    tenant_data = tenant_data.merge(
        meter_data.drop(
            columns=["consumption", "max_demand", "min_demand", "avg_demand"]
        ),
        on="date_time",
    )
    return dict(
        meter_data=meter_data,
        tenant_data=tenant_data,
        occupancy_data=occupancy_data,
        steam_data=steam_data,
        meter_lookup=meter_lookup,
    )


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