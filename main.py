import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Iris_Classification")

with mlflow.start_run():
    n_estimators = 100
    max_depth = 5
    
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)

    signature = infer_signature(X_train, model.predict(X_train[:5]))

    mlflow.sklearn.log_model(
        model,
        "random_forest_model",
        signature = signature,
        input_example = X_train[:1],
        registered_model_name = "IrisClassifier" 
    )
    
    print(f"Model accuracy: {accuracy:.2f}")