import pathlib
import pandas as pd
import typing
import numpy as np
import statsmodels.api as sm
from scipy.stats import t


# LINEAR MODEL
class LinearRegressionSM:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self._model = None

    def fit(self):
        X = sm.add_constant(self.right_hand_side)
        y = self.left_hand_side
        model = sm.OLS(y, X).fit()
        self._model = model

    def get_params(self):
        if self._model is not None:
            params = self._model.params
            params = params.rename('Beta coefficients')
            return params
        else:
            raise ValueError("Fit the model first to get parameters.")

    def get_pvalues(self):
        if self._model is not None:
            params = self._model.pvalues
            params = params.rename('P-values for the corresponding coefficients')
            return params
        else:
            raise ValueError("Fit the model first to get parameters.")

    def get_wald_test_result(self, restriction_matrix):
        if self._model is not None:
            wald_test_result = self._model.wald_test(restriction_matrix)
            fvalue = wald_test_result.statistic[0][0]
            pvalue = wald_test_result.pvalue
            return f"F-value: {fvalue:.3f}, p-value: {pvalue:.3f}"
        else:
            raise ValueError("Fit the model first to get goodness-of-fit values.")

    def get_model_goodness_values(self):
        if self._model is not None:
            r_squared = self._model.rsquared_adj
            aic = self._model.aic
            bic = self._model.bic
            return f'Adjusted R-squared: {r_squared:.3f}, Akaike IC: {aic:.3f}, Bayes IC: {bic:.3f}'
        else:
            raise ValueError("Fit the model first to get goodness-of-fit values.")

class LinearRegressionNP:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.beta = None
        self.p_values = None
        self.centered_r_squared = None
        self.adjusted_r_squared = None
        self.wald_statistic = None

    def fit(self):
        X = sm.add_constant(self.right_hand_side)
        model = sm.OLS(self.left_hand_side, X).fit()
        self.beta = model.params
        self._model = model

    def get_params(self):
        return pd.Series(self.beta, name="Beta coefficients")

    def get_pvalues(self):
        if self._model is not None:
            params = self._model.pvalues
            params = params.rename('P-values for the corresponding coefficients')
            return params
        else:
            raise ValueError("Fit the model first to get parameters.")

    def get_wald_test_result(self, restriction_matrix):
        if self._model is not None:
            wald_test_result = self._model.wald_test(restriction_matrix)
            fvalue = wald_test_result.statistic[0][0]
            pvalue = wald_test_result.pvalue
            return f"Wald: {fvalue:.3f}, p-value: {pvalue:.3f}"
        else:
            raise ValueError("Fit the model first to get goodness-of-fit values.")

    def get_model_goodness_values(self, include_constant=True):
        n = len(self.left_hand_side)
        p = len(self.beta) - (1 if include_constant else 0)
        y_mean = np.mean(self.left_hand_side)
        tss = np.sum((self.left_hand_side - y_mean) ** 2)
        rss = np.sum((self.left_hand_side - np.dot(sm.add_constant(self.right_hand_side), self.beta)) ** 2)
        adj_r_squared = 1 - (rss / (n - p)) / (tss / (n))
        cen_r_squared = 1 - (rss / (n - p)) / (tss / (n - p))
        return f"Centered R-squared: {cen_r_squared:.3f}, Adjusted R-squared: {adj_r_squared:.3f}"