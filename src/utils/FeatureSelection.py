from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import shap
import pandas as pd
import numpy as np




class FeatureSelection():
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def CorrFeatureSelection(self, dependent_col):
        """Calulates the Correlation Cofficient with respect to Dependent Column"""
        ## Correlation Metric of all Columns with Respect to Dependent Column (Out put Column)
        self.feature_corr_value = self.dataframe.corr()[dependent_col].reset_index().rename(columns={'index': 'feature', dependent_col: 'corr_coeff'})
        self.feature_corr_value = self.feature_corr_value[self.feature_corr_value['feature'] != dependent_col].reset_index(drop=True)
        return self.feature_corr_value.sort_values('corr_coeff', ascending=False)
        

    def DataLabel(self,dependent_col):
        self.X_label = self.dataframe.drop(dependent_col, axis=1)
        self.y_label = self.dataframe[dependent_col] 
        return self.X_label, self.y_label



    def RFRFeatureSelection(self, dependent_col, n_estimators=100):
        """Calculates feature importance using Random Forest Regressor with respect to the specified dependent column."""
        X_label, y_label = self.DataLabel(dependent_col) # Get the features and label
        
        # Train a Random Forest regressor on label encoded data
        rf_label = RandomForestRegressor(n_estimators=n_estimators, random_state=42) ## Class instantiation (RandomForestRegressor)
        rf_label.fit(X_label, y_label)

        # Extract feature importance scores for label encoded data
        self.rfr_value = pd.DataFrame({'feature': X_label.columns, 'rf_importance': rf_label.feature_importances_})
        return  self.rfr_value.sort_values(by='rf_importance', ascending=False)
    


    def GBRFeatureSelection(self, dependent_col):
        """Calculates feature importance using Grandient Boosting regressor with respect to the specified dependent column."""
        ## Train and Test Split
        X_label, y_label = self.DataLabel(dependent_col) # Get the features and label

        # Train a Grandient Boosting regressor on label encoded data
        gb_label = GradientBoostingRegressor()
        gb_label.fit(X_label, y_label)

        # Extract feature importance scores for label encoded data
        self.gdr_value = pd.DataFrame({'feature': X_label.columns,'gb_importance': gb_label.feature_importances_})
        return self.gdr_value.sort_values(by='gb_importance', ascending=False)
    


    def PermutationFeatureSelection(self,dependent_col, n_estimators=100,n_repeats=30):
        """Calculates feature importance using Permutation Techniques with respect to the specified dependent column."""
        X_label, y_label = self.DataLabel(dependent_col) # Get the features and label
        X_train_label, X_test_label, y_train_label, y_test_label = train_test_split(X_label, y_label, test_size=0.2, random_state=42)

        # Train a Random Forest regressor on label encoded data
        rf_label = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf_label.fit(X_train_label, y_train_label)

        # Calculate Permutation Importance
        perm_importance = permutation_importance(rf_label, X_test_label, y_test_label, n_repeats=n_repeats, random_state=42)

        # Organize results into a DataFrame
        pif_value = pd.DataFrame({'feature': X_label.columns,'permutation_importance': perm_importance.importances_mean})
        return pif_value.sort_values(by='permutation_importance', ascending=False)
    


    def RecursiveFeatureElimination(self, dependent_col):
        """Calculates feature importance using Recursive Feature Elimination Techniques with respect to the specified dependent column."""
        X_label, y_label = self.DataLabel(dependent_col) # Get the features and label
        estimator = RandomForestRegressor() # Initialize the base estimator
        
        # Apply RFE on the label-encoded and standardized training data
        selector_label = RFE(estimator, n_features_to_select=X_label.shape[1], step=1)
        selector_label = selector_label.fit(X_label, y_label)

        # Get the selected features based on RFE
        selected_features = X_label.columns[selector_label.support_]
        # Extract the coefficients for the selected features from the underlying linear regression model
        selected_coefficients = selector_label.estimator_.feature_importances_

        # Organize the results into a DataFrame
        rfe_value = pd.DataFrame({'feature': selected_features,'rfe_score': selected_coefficients})
        return rfe_value.sort_values(by='rfe_score', ascending=False)



    def LassoFeatureSelection(self,dependent_col):
        """Calculates feature importance using Lasso Regression Techniques with respect to the specified dependent column."""
        X_label, y_label = self.DataLabel(dependent_col) # Get the features and label
        scaler = StandardScaler() # Standardize the features
        X_scaled = scaler.fit_transform(X_label)

        lasso = Lasso(alpha=0.01, random_state=42) # Train a LASSO regression model
        lasso.fit(X_scaled, y_label)
        # Extract coefficients
        lasso_value = pd.DataFrame({'feature': X_label.columns,'lasso_coeff': lasso.coef_})
        return lasso_value.sort_values(by='lasso_coeff', ascending=False)



    def LinerRegressionFeature(self, dependent_col):
        """Calculates feature importance using Linear Regression Techniques with respect to the specified dependent column."""
        X_label, y_label = self.DataLabel(dependent_col) # Get the features and label
        scaler = StandardScaler() # Standardize the features
        X_scaled = scaler.fit_transform(X_label)
        lin_reg = LinearRegression()
        lin_reg.fit(X_scaled, y_label)

        # Extract coefficients
        lin_value = pd.DataFrame({'feature': X_label.columns,'reg_coeffs': lin_reg.coef_})
        return lin_value.sort_values(by='reg_coeffs', ascending=False)



    def ShapFeature(self, dependent_col):
        """Calculates feature importance using Shap Techniques with respect to the specified dependent column."""
        X_label, y_label = self.DataLabel(dependent_col) # Get the features and label
        # Compute SHAP values using the trained Random Forest model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_label, y_label)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_label)
        shao_val = pd.DataFrame({'feature': X_label.columns,'SHAP_score': np.abs(shap_values).mean(axis=0)})
        return shao_val.sort_values(by='SHAP_score', ascending=False)

