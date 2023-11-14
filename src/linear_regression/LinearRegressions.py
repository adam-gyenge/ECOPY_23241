import pathlib
import pandas as pd
import typing
import numpy as np
import statsmodels.api as sm
from scipy.stats import t, f
from scipy import stats


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
        self.beta = None  # Placeholder for estimated coefficients

    def fit(self):
        # Add a constant to the right-hand side for intercept term
        self.X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))

        # Perform OLS regression
        self.beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.left_hand_side

    def get_params(self):
        if self.beta is not None:
            return pd.Series(self.beta, name='Beta coefficients')
        else:
            raise ValueError("Model has not been fitted. Call fit() first.")

    def get_pvalues(self):
        if self.beta is not None:
            n = len(self.left_hand_side)
            k = len(self.beta)

            # Calculate t-statistics and corresponding p-values
            t_stats = self.beta / np.sqrt(np.diag(np.linalg.inv(self.X.T @ self.X)) * (np.sum((self.left_hand_side - self.X @ self.beta)**2) / (n - k)))

            # Calculate two-tailed p-values
            p_values = 2 * (1 - t.cdf(np.abs(t_stats), df=n - k))

            return pd.Series(p_values, name='P-values for the corresponding coefficients')
        else:
            raise ValueError("Model has not been fitted. Call fit() first.")

    def get_wald_test_result(self, R):
        if self.beta is not None:
            one = (np.array(R) @ self.beta).T
            mid = np.array(R) @ np.linalg.inv(self.X.T @ self.X) @ np.array(R).T
            end = np.array(R) @ self.beta

            n = len(self.left_hand_side)
            m, k = np.array(R).shape

            # sigma squared
            residuals = self.left_hand_side - self.X @ self.beta
            sigma_squared = np.sum(residuals ** 2) / (n - k)

            wald_value = (one @ np.linalg.inv(mid) @ end) / (m * sigma_squared)
            p_value = 1 - f.cdf(wald_value, dfn=m, dfd=n - k)

            return f"Wald: {wald_value:.3f}, p-value: {p_value:.3f}"
        else:
            raise ValueError("Model has not been fitted. Call fit() first.")

    def get_model_goodness_values(self):
        if self.beta is not None:
            n = len(self.left_hand_side)
            k = len(self.beta)

            # Calculate centered R-squared
            y_hat = self.X @ self.beta
            y_bar = np.mean(self.left_hand_side)
            centered_r_squared = 1 - np.sum((self.left_hand_side - y_hat)**2) / np.sum((self.left_hand_side - y_bar)**2)

            # Calculate adjusted R-squared
            adjusted_r_squared = 1 - (1 - centered_r_squared) * ((n - 1) / (n - k))

            return f"Centered R-squared: {centered_r_squared:.3f}, Adjusted R-squared: {adjusted_r_squared:.3f}"
        else:
            raise ValueError("Model has not been fitted. Call fit() first.")