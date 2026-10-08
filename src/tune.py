from sklearn.model_selection import RandomizedSearchCV

from src.models import get_models
from src.train import build_pipeline

from src.data_preparation import load_data, split_data

from src.evaluate import calculate_metrics

import mlflow

mlflow.set_experiment("crop-recommendation-tuning")

X, y = load_data()

X_train, X_test, y_train, y_test = split_data(X, y)

models = get_models()

random_forest = models["random_forest"]
pipeline = build_pipeline(random_forest)

param_distributions = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"],
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="f1_macro",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

with mlflow.start_run(run_name="random_forest_tuning"):

    # 1. Tune
    search.fit(X_train, y_train)

    # 2. Log best parameters
    for param_name, param_value in search.best_params_.items():
        mlflow.log_param(param_name, param_value)

    # 3. Log CV score
    mlflow.log_metric("best_cv_macro_f1", search.best_score_)

    # 4. Get best model
    best_model = search.best_estimator_

    # 5. Final test evaluation
    y_pred = best_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred)

    mlflow.sklearn.log_model(
    best_model,
    name="tuned_crop_recommendation_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    # 7. Log test metrics
    for metric_name, value in test_metrics.items():
        mlflow.log_metric(f"test_{metric_name}", value)

    # 8. Print results
    print("\nFinal Test Results:")

    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
'''print("\nBest model:")
print(best_model)

print("Best parameters:")
print(search.best_params_)

print("\nBest CV Macro F1:")
print(search.best_score_)'''