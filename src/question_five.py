from src.feature_engineering import read_features_and_targets
from src.modeling import compute_efficiency_for_occupancy

if __name__ == "__main__":
    features_and_targets = read_features_and_targets()

    compute_efficiency_for_occupancy(features_and_targets)
