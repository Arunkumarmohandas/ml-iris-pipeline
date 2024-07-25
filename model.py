from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


def train_model():
    data = load_iris()
    X, y = data.data, data.target
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model


def evaluate_model(model):
    data = load_iris()
    X, y = data.data, data.target
    scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean()}")


if __name__ == "__main__":
    model = train_model()
    print("Model trained successfully!")
    evaluate_model
