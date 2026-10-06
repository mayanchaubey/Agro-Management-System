from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def get_models():
    return {
        "svm_linear": SVC(kernel="linear"),
        "svm_rbf": SVC(kernel="rbf"),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

if __name__ == "__main__":
    models = get_models()

    for name, model in models.items():
        print(name, "->", model)