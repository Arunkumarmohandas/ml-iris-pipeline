from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

def train_model():
    data = load_iris()
    X, y = data.data, data.target
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

if __name__ == "__main__":
    model = train_model()
    print("Model trained successfully!")
