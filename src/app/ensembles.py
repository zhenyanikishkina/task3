import numpy as np
import time
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        if 'random_state' in trees_parameters:
            self.random_state = trees_parameters['random_state']
        else:
            self.random_state = 137
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.models = None

    def fit(self, X, y, X_val=None, y_val=None, trace=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """

        self.models = []
        if trace:
            acc_train = []
            time_md = []
            y_train = np.zeros(y.shape[0])
        if X_val is not None:
            y_val_pred = np.zeros(y_val.shape[0])
            acc_val = []

        self.feature_subsample_size = X.shape[1] // 3 if\
            self.feature_subsample_size is None else\
            self.feature_subsample_size

        self.ff_arr = []
        start = time.time()
        r = np.random.RandomState(self.random_state)
        for i in range(self.n_estimators):
            ind_row = r.choice(y.shape[0], y.shape[0])
            ind_col = r.choice(X.shape[1], self.feature_subsample_size, replace=False)
            self.ff_arr.append(ind_col)
            model = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            model.fit(X[ind_row][:, ind_col], y[ind_row])
            self.models.append(model)
            if trace:
                y_train += model.predict(X[:, ind_col])
                acc_train.append(mean_squared_error(y, y_train / (i + 1), squared=False))
            if X_val is not None:
                y_val_pred += model.predict(X_val[:, ind_col])
                acc_val.append(mean_squared_error(y_val, y_val_pred / (i + 1), squared=False))
            time_md.append(time.time() - start)

        if X_val is not None:
            return (acc_train, acc_val, time_md) if trace else acc_val
        return (acc_train, time_md) if trace else None

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        return np.mean([self.models[i].predict(X[:, self.ff_arr[i]]) for i in range(len(self.models))], axis=0)


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        if 'random_state' in trees_parameters:
            self.random_state = trees_parameters['random_state']
        else:
            self.random_state = 137
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.mean = None
        self.weights = None
        self.models = None

    def fit(self, X, y, X_val=None, y_val=None, trace=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        self.mean = np.mean(y)
        self.models = []
        self.weights = np.zeros(self.n_estimators)

        if trace:
            acc_train = []
            time_md = []
        if X_val is not None:
            acc_val = []
            pred_val = np.zeros(y_val.shape[0]) + np.mean(y_val)
            ans_val = pred_val.copy()

        self.feature_subsample_size = X.shape[1] // 3 if\
            self.feature_subsample_size is None else\
            self.feature_subsample_size

        pred_train = np.zeros(y.shape[0]) + self.mean
        ans_train = pred_train.copy()

        self.ff_arr = []
        start = time.time()
        r = np.random.RandomState(self.random_state)
        for i in range(self.n_estimators):
            ind_row = r.choice(y.shape[0], y.shape[0])
            ind_col = r.choice(X.shape[1], self.feature_subsample_size, replace=False)
            self.ff_arr.append(ind_col)
            model = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            model.fit(X[ind_row][:, ind_col], (y - ans_train)[ind_row])
            self.models.append(model)
            pred_train = model.predict(X[:, ind_col])
            self.weights[i] = minimize_scalar(lambda alpha: mean_squared_error(y,
                                        ans_train + alpha * pred_train, squared=False)).x
            ans_train = ans_train + self.weights[i] * pred_train * self.learning_rate
            if trace:
                acc_train.append(mean_squared_error(y, ans_train, squared=False))
            if X_val is not None:
                pred_val = model.predict(X_val[:, ind_col])
                ans_val = ans_val + self.weights[i] * pred_val * self.learning_rate
                acc_val.append(mean_squared_error(y_val, ans_val, squared=False))
            time_md.append(time.time() - start)

        if X_val is not None:
            return (acc_train, acc_val, time_md) if trace else acc_val
        return (acc_train, time_md) if trace else None

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pred = self.mean + np.zeros(X.shape[0])
        return pred + (np.array([self.models[i].predict(X[:, self.ff_arr[i]]) for i in range(len(self.models))])\
                                * self.weights.reshape(-1, 1)).sum(axis=0) * self.learning_rate
