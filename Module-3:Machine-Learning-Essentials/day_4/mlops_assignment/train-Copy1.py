import fire
import mlflow
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split

def get_data():
    """Return a tuple containing:
    * training features
    * test features
    * training targets
    * test targets
    * data preprocessing function (already applied to the provided data)
    When serving the model in production you might need to preprocess the raw input data. You can
    use the last element of the returned tuple for this.
    """
    wine = datasets.load_wine()
    wine_x = wine.data
    wine_y = wine.target

    x_train, x_test, y_train, y_test = train_test_split(
        wine_x, wine_y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (x_train, x_test, y_train, y_test)


# we use scikit-learn pipeline to package standarization into single object with model
def setup_knn_pipeline(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    pipe = make_pipeline(StandardScaler(), knn)
    return pipe


 
def track_with_mlflow(model, y_pred, Y_test, mlflow, model_metadata):
    mlflow.log_params(model_metadata)
    mlflow.log_metric("accuracy", model.score(X_test, Y_test))
    mlflow.log_metric("f1", f1_score(Y_test, y_pred, average="micro"))
    mlflow.sklearn.log_model(model, "knn", registered_model_name="sklearn_knn")


    

def main(max_k: int):

    X_train, X_test, Y_train, Y_test = get_data()
    # let's check some other k
    k_list = range(1, max_k)

    for k in k_list:
        with mlflow.start_run():
            knn_pipe = setup_knn_pipeline(k)
            knn_pipe.fit(X_train, Y_train)
            model_metadata = {"k": k}
            y_pred = knn_pipe.predict(X_test)
            track_with_mlflow(knn_pipe, y_pred, Y_test, mlflow, model_metadata)


if __name__ == "__main__":
    fire.Fire(main)
