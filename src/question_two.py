from src.feature_engineering import read_features_and_targets
from src.modeling import (
    occupancy_consumption_correlation,
    plot_consumption_before_and_with_covid,
)

if __name__ == "__main__":

    features_and_targets = read_features_and_targets()

    occupancy_consumption_correlation(features_and_targets)
    plot_consumption_before_and_with_covid(features_and_targets)
