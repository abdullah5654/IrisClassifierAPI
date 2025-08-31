import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

MODEL_PATH = "iris_model.pkl"


def train_and_save_model():
    """Train a simple Logistic Regression model and save it."""
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, iris.target_names), f)


def load_model():
    """Load trained model from pickle file."""
    with open(MODEL_PATH, "rb") as f:
        model, target_names = pickle.load(f)
    return model, target_names


def predict_species(features):
    """Make a prediction for a single input."""
    model, target_names = load_model()
    prediction = model.predict([features])[0]
    return target_names[prediction]
