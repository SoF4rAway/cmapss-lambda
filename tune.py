from src.training.tuner import run_tuning_pipeline, finalize_model
from src.core.config import set_seed

if __name__ == "__main__":
    set_seed(42)
    result = run_tuning_pipeline()
    if result is not None:
        best_params, train_data, val_data, preprocessor, feature_cols, study, best_trial = result
        finalize_model(best_params, train_data, val_data, preprocessor, feature_cols, study, best_trial)
