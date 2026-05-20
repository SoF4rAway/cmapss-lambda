import argparse
from src.evaluation.evaluator import run_evaluation_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate C-MAPSS RUL prediction models.")
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Specific model version timestamp (e.g., 20260517_145257) to evaluate. Defaults to latest."
    )
    parser.add_argument(
        "--no-prod-latency",
        action="store_true",
        help="Skip querying Elasticsearch for production latency and fallback to synthetic latency."
    )
    args = parser.parse_args()
    
    run_evaluation_pipeline(
        model_timestamp=args.model_version,
        fetch_prod_latency=not args.no_prod_latency
    )
