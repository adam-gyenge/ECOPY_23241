import pathlib
import pandas as pd
import typing
import numpy as np
import statsmodels.api as sm
from scipy.stats import t, f


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


class LinearRegressionGLS:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side.values
        self.right_hand_side = right_hand_side.values
        self.beta = None

    def fit(self):
        #self.X = self.right_hand_side.values
        self.X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        self.y = self.left_hand_side
        beta_ols = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        residuals = self.y - self.X @ beta_ols

        X_new = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side)) #np.column_stack((np.log(squared_residuals), self.X))
        y_new = np.log(residuals**2) #np.log(self.y ** 2)
        beta_gls = np.linalg.inv(X_new.T @ X_new) @ X_new.T @ y_new

        # V inverz mátrix előállítása
        self.V_inv = np.diag(1 / np.sqrt(np.exp(X_new @ beta_gls)))
        beta_gls = np.linalg.inv(X_new.T @ self.V_inv @ X_new) @ X_new.T @ self.V_inv @ self.left_hand_side
        self.coefficients = beta_gls
        return beta_gls

    def get_params(self):
        beta_values = self.fit()
        params_series = pd.Series(beta_values, name='Beta coefficients')
        return params_series

    def get_pvalues(self):
        n_obs = len(self.y)
        n_params = len(self.coefficients)

        residuals = self.left_hand_side - np.column_stack((np.ones(n_obs), self.right_hand_side)) @ self.coefficients
        sigma_squared = (residuals @ residuals) / (n_obs - n_params)
        var_cov_matrix = sigma_squared * np.linalg.inv(self.X.T @ self.V_inv @ self.X)

        se = np.sqrt(np.diagonal(var_cov_matrix))
        t_stat = self.coefficients / se

        #p_values = (1 - t.cdf(np.abs(t_stat), n_obs - n_params)) * 2
        p_values = [min(value, 1 - value) * 2 for value in t.cdf(np.abs(t_stat), df=n_obs - n_params)]

        p_values = pd.Series(p_values, name='P-values for the corresponding coefficients')
        return p_values

    def get_wald_test_result(self, R):
        one = (np.array(R) @ self.coefficients).T
        mid = np.array(R) @ np.linalg.inv(self.X.T @ self.V_inv @ self.X) @ np.array(R).T
        end = np.array(R) @ self.coefficients

        R = np.array(R)
        n = len(self.left_hand_side)
        m, k = np.array(R).shape

        # sigma squared
        residuals = self.left_hand_side - self.X @ self.coefficients
        sigma_squared = (residuals @ residuals) / (n - k)

        wald_value = (one @ np.linalg.inv(mid) @ end) / (m * sigma_squared)
        p_value = 1 - f.cdf(wald_value, dfn=m, dfd=n - k)

        return f"Wald: {wald_value:.3f}, p-value: {p_value:.3f}"

    def get_model_goodness_values(self):
        total_sum_of_squares = self.y.T @ self.V_inv @ self.y
        residual_sum_of_squares = self.y.T @ self.V_inv @ self.X @ np.linalg.inv(
            self.X.T @ self.V_inv @ self.X) @ self.X.T @ self.V_inv @ self.y
        centered_r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
        adjusted_r_squared = 1 - (residual_sum_of_squares / (len(self.y) - self.X.shape[1])) * (
                len(self.y) - 1) / total_sum_of_squares
        return f"Centered R-squared: {centered_r_squared:.3f}, Adjusted R-squared: {adjusted_r_squared:.3f}"