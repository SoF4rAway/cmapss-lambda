from src.training.tuner import run_tuning_pipeline, finalize_model
from src.core.config import set_seed
if __name__ == "__main__":
    set_seed(42)
    best_params, train_data, val_data = run_tuning_pipeline()
    if best_params:
        finalize_model(best_params, train_data, val_data)
