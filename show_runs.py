import mlflow
import os
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=["0"], order_by=["metrics.accuracy DESC"])
print(f"{'Run Name':<30} {'n_estimators':<14} {'max_depth':<11} {'min_split':<11} {'Accuracy':<10} {'F1'}")
print("-" * 88)
for r in runs:
    p = r.data.params
    m = r.data.metrics
    print(f"{r.info.run_name:<30} {p.get('n_estimators','?'):<14} {p.get('max_depth','?'):<11} {p.get('min_samples_split','?'):<11} {m.get('accuracy',0):<10.4f} {m.get('f1_score',0):.4f}")
