import pandas as pd

from src.constants import (
    DATE_READ_SETTINGS,
    DATE_TIME_READ_SETTINGS,
    ELECTRIC_INTERVAL_DATA_CSV_NAME,
    METER_TO_LOCATION_CSV_NAME,
    OCCUPANCY_DATA_CSV_NAME,
    STEAM_DATA_CSV_NAME,
    TENANT_ELECTRIC_INTERVAL_DATA_CSV_NAME,
)


def read_raw_excel_data_to_csv():
    """
    Read in raw Excel data, parse, and save as csvs.
    """
    electric_interval_data = pd.read_excel(
        "data/raw/ConEd_Electric.xlsx", **DATE_TIME_READ_SETTINGS
    )
    electric_interval_data.to_csv(ELECTRIC_INTERVAL_DATA_CSV_NAME)

    # Create an ExcelFile from the tenant usage.
    xls = pd.ExcelFile("data/raw/Tenant_Usage.xlsx")

    sheet_names = xls.sheet_names
    meter_to_location_lookup = pd.read_excel(xls, sheet_names[0])
    meter_to_location_lookup.to_csv(METER_TO_LOCATION_CSV_NAME, index=False)

    # Read in all the individual tenant consumption data.
    tenant_electric_interval_data = pd.concat(
        [
            pd.read_excel(xls, sheet_name, **DATE_TIME_READ_SETTINGS).assign(
                name=sheet_name
            )
            for sheet_name in sheet_names[1:]
        ]
    )
    tenant_electric_interval_data.to_csv(TENANT_ELECTRIC_INTERVAL_DATA_CSV_NAME)

    occupancy_data = pd.read_excel("data/raw/Occupancy.xlsx", **DATE_READ_SETTINGS)
    occupancy_data.to_csv(OCCUPANCY_DATA_CSV_NAME)

    steam_data = pd.read_excel("data/raw/ConEd_Steam.xlsx", **DATE_TIME_READ_SETTINGS)
    steam_data.to_csv(STEAM_DATA_CSV_NAME)
