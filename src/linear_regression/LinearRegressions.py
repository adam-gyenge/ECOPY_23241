import pathlib
import pandas as pd
import typing
import statsmodels.api as sm


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