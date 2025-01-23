from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression, f_regression

class ModelLinear:

    def __init__(self, n_features = 6, n_poly = 6, alpha = 30, test_size = 0.3):
        self.N_FEATURES = n_features
        self.N_POLY = n_poly
        self.ALPHA = alpha
        self.TEST_SIZE = test_size

    def train_model(self, feature_df=None, target=None):
        """Train a linear regression model with input feature_df and target."""
        steps = [
            ('feature_selection', SelectKBest(r_regression, k=self.N_FEATURES)),
            ('scalar', StandardScaler()),
            ('poly', PolynomialFeatures(degree=self.N_POLY)),
            ('model', Ridge(alpha=self.ALPHA, fit_intercept=True))
        ]
        pipeline = Pipeline(steps)

        # split data
        X_train, X_test, y_train, y_test = train_test_split(feature_df, target, test_size=self.TEST_SIZE)
        pipeline.fit(X_train, y_train)

        print('Training score: {}'.format(pipeline.score(X_train, y_train)))
        print('Test score: {}'.format(pipeline.score(X_test, y_test)))
        return pipeline



