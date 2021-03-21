import os

import pandas as pd

from src.utils import prepare_data


def preliminary():
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
    daily_totals = occupancy_data.merge(
        daily_meter_sums, left_index=True, right_index=True
    )

    daily_tenant_sums = (
        tenant_usage_data.groupby(
            [tenant_usage_data.index.date, tenant_usage_data["meter"]]
        )["consumption"]
        .sum()
        .reset_index()
        .rename(columns=dict(level_0="date"))
    )

    daily_tenant_sums_pivoted = daily_tenant_sums.pivot_table(
        columns="meter", index="date"
    )
    daily_tenant_sums_pivoted.merge(
        occupancy_data, left_index=True, right_index=True
    ).corr()
