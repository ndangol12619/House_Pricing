
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy


class MyCustomException(Exception):
  def __init__(self,message):
    """Custome Exception"""
    super().__init__(message)
    print(message)


### TableAnalysis Class
class TableAnalysis:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.numeric_df = None
        self.duplicate_record = dataframe[dataframe.duplicated()].reset_index(drop=True)
        # print(f"TableAnalysis()")
        # if self.duplicate_record.shape[0]>0:
        #     print(f"Duplicates: {dataframe.duplicated().sum()}\nFor Duplicate Records: ShowDuplicates()")
       
    def __str__(self):
            df_shape = f"Rows: {self.dataframe.shape[0]}\nColumns: {self.dataframe.shape[1]}"
            if self.duplicate_record.shape[0]>0:
                duplicate = f"Duplicates: {self.dataframe.duplicated().sum()}\nFor Duplicate Records: ShowDuplicates()"
                message_print = f"{df_shape}\n{duplicate}"
            else:
                message_print = df_shape
            return message_print
    

    def ShowDuplicates(self):
        """Fetch All Duplicate Records."""
        try:
            if self.duplicate_record.empty:
                raise MyCustomException ('No Duplicate Detected')
            else:
                return self.duplicate_record
        except MyCustomException as e:
            return pd.DataFrame()
       


    def CategoricalFeatureSummary(self):
        """High-Level Table Summary with categorical stats where applicable"""
        categorical_df= self.dataframe.select_dtypes(include=object)
        summary_data = []
        for column in categorical_df.columns:
            value_count = categorical_df[column].value_counts(dropna=False)
            null_count = categorical_df[column].isnull().sum()
            total_count = categorical_df.shape[0]
            unique_count = categorical_df[column].nunique(dropna=False)
            top_value = value_count.index[0]
            top_count = value_count.iloc[0]
            top_pct = round((top_count / total_count) * 100, 2)
            top_3 = value_count.head(3)
            top_3_names = list(top_3.index)
            top_3_pct = list((top_3 / total_count * 100).round(2))
            rare_count = (value_count / total_count < 0.01).sum()

        ## Table Summary into Data Frame 
            summary_data.append({
                'field': column,
                'non_null_count': total_count - null_count,
                'null_count': null_count,
                'null%': round((null_count / total_count) * 100, 4),
                'dtype': categorical_df[column].dtype,
                'unique_count': unique_count,
                'top_value': top_value,
                'top_count': top_count,
                'top_pct': top_pct,
                'top_3': top_3_names,
                'top_3_pct': top_3_pct,
                'rare(<1%)_count': rare_count,
                'singleton_count': (value_count == 1).sum(),
                'entropy': round(entropy(value_count / total_count), 4),
                'dominance_ratio': round(top_count / total_count, 4),
                'is_binary': unique_count == 2,
                'high_cardinality': unique_count > 50,
            })
        return pd.DataFrame(summary_data)

        


    def NumericalFeatureSummary(self):
        """Statiscal Summary"""
        self.num_df = self.dataframe.select_dtypes(include=(int, float))
        try:
            if self.num_df.empty:
                raise MyCustomException('No Numerical Column in df')
            # Quantile Calculation Logic
            Q1 = self.num_df.quantile(0.25)
            Q2 = self.num_df.quantile(0.50)
            Q3 = self.num_df.quantile(0.75)
            IQR = Q3 - Q1

            # Outlier thresholds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Calculate outliers count and percentage
            outliers_mask = (self.num_df < lower_bound) | (self.num_df > upper_bound)
            outliers_count = outliers_mask.sum()
            outlier_percentage = (outliers_mask.sum() / self.num_df.count() * 100).round(2)
            # # Calculate outlier percentage
            # outlier_percentage = ((self.num_df < lower_bound) | (self.num_df > upper_bound)).sum() / self.num_df.count() * 100
            # outlier_percentage = outlier_percentage.round(2)

            # Skewness and Kurtosis
            skewness = self.num_df.skew().round(2)
            kurtosis = self.num_df.kurt().round(2)
             # Distribution summary logic
            distribution_summary = []
            for s, k in zip(skewness, kurtosis):
                # Skewness interpretation
                if s == 0:
                    skew_text = "Symmetric"
                elif abs(s) < 0.5:
                    skew_text = "Fairly Symmetric"
                elif abs(s) < 1:
                    skew_text = "Moderately Skewed"
                else:
                    skew_text = "Highly Skewed"

                if s > 0:
                    skew_text += " (Positive: tail on the right)"
                elif s < 0:
                    skew_text += " (Negative: tail on the left)"

                # Kurtosis interpretation
                if k < 0:
                    kurt_text = "Platykurtic (Light tails, Negative)"
                elif k == 0:
                    kurt_text = "Mesokurtic (Normal)"
                elif k <= 3:
                    kurt_text = "Mildly Leptokurtic (Positive)"
                elif k <= 10:
                    kurt_text = "Leptokurtic (Heavy tails, Positive)"
                else:
                    kurt_text = "Very Leptokurtic (Extreme outliers, Positive)"

                # Combine messages
                dist_text = f"{skew_text}, {kurt_text}"
                distribution_summary.append(dist_text)

            # Statistical Summary Into DataFrame 
            summary = pd.DataFrame({
                'field': self.num_df.columns,
                'non_null_count': self.num_df.count(),
                'null_count': self.num_df.isnull().sum(),
                'null%': round((self.num_df.isnull().sum() / self.num_df.shape[0]) *100,4),
                'dtype': self.num_df.dtypes.astype(str),
                'min': round(self.num_df.min(), 2),
                'max': round(self.num_df.max(), 2),
                'mean': round(self.num_df.mean(), 2),
                'median': round(self.num_df.median(), 2),
                'std': round(self.num_df.std(), 2),
                'var (M)': round(self.num_df.var() / 1_000_000, 2),
                '1 %' : round(self.num_df.quantile(0.01),2),
                '5 %' : round(self.num_df.quantile(0.05),2),
                '25 %': round(Q1, 2),
                '50 %': round(Q2, 2),
                '75 %': round(Q3, 2),
                '95 %': round(self.num_df.quantile(0.95),2),
                '99 %': round(self.num_df.quantile(0.99),2),
                'IQR': round(IQR, 2),
                'lower_bound': round(lower_bound, 4),
                'upper_bound': round(upper_bound, 4),
                'outliers_count' : outliers_count.values,
                'outlier_percentage': outlier_percentage.values,
                'skewness': skewness.values,
                'kurtosis': kurtosis.values,
                'distribution_summary': distribution_summary
            }).reset_index(drop=True)
            return summary
        except :
            # print(f"Warning: {e}")
            return pd.DataFrame()  # Optionally return an empty DataFrame
        

    def get_outliers_for_column(self, column):
        """Return All the Outliers Records"""
        # if column not in self.numeical_cols:
        #     raise MyCustomException(f"'{column}' is not a numerical column")

        Q1 = self.dataframe[column].quantile(0.25)
        Q3 = self.dataframe[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        mask_outliers = (self.dataframe[column] < lower_bound) | (self.dataframe[column] > upper_bound)
        return self.dataframe[mask_outliers]

            
  