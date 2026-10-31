from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm.auto import tqdm

class MLPReadmissionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dims=(64, 32), max_iter=1000, alpha=0.0001,
                 random_state=42, verbose=True):
        self.hidden_dims = hidden_dims
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self.model_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_dims,
            activation="relu",
            alpha=self.alpha,
            max_iter=self.max_iter,
            warm_start=True,
            random_state=self.random_state
        )

        prev_loss = None
        iterator = tqdm(range(self.max_iter), disable=not self.verbose, desc="MLP training")
        for _ in iterator:
            self.model_.fit(X, y)
            iterator.set_postfix(loss=f"{self.model_.loss_:.4f}")
            if prev_loss is not None and abs(prev_loss - self.model_.loss_) < self.model_.tol:
                break
            prev_loss = self.model_.loss_

        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)