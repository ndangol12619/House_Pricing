

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os
import pathlib

from utils.DataAnalysis import TableAnalysis,MyCustomException

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)


### Univariate Analysis Class
class UnivariateAnalysis(TableAnalysis): # Inherits functionality from the TableAnalysis class for univariate exploration
    def __init__(self,dataframe):
        super().__init__(dataframe)  # Initialize the parent DataAnalysis class with the dataframe
        self.dataframe = dataframe
        self.numeical_cols = list(self.dataframe.select_dtypes(include=(int,float)).columns) ## Numerical Column 
        self.categorical_cols = list(self.dataframe.select_dtypes(include=(object)).columns) ## Categorical Column 
        

    def __str__(self):
        return f"categorical_columns= {self.categorical_cols}\nnumerical_columns= {self.numeical_cols}"
    
    def ColumnVariable(self):
        column_type =  {'categorical_columns':self.categorical_cols,'numerical_columns':self.numeical_cols}
        print(column_type.keys())
        return column_type


    def CategoricalFeaturesAnalysis(self,column, figsize=(8,5)):
        """Categorical Featiure Analysis And Visualization"""
        summary = self.CategoricalFeatureSummary()     ## Method Call from Data Analysis Paresnt Class
        summary = summary.loc[summary['field'].isin([column])].transpose().reset_index()
        summary.columns = summary.iloc[0]
        summary = summary.iloc[1:].reset_index(drop=True)
         
        self.value_count = self.dataframe[column].value_counts() ## Value Count for Each Column 
        if len(self.value_count) < 20: ## If column value count is Less than 20 this block will get excecuted
            fig, ax = plt.subplots(1,2, figsize=figsize)
            sns.countplot(data=self.dataframe, x=column,ax=ax[0]) ## Count Plot
            ## Add gridlines
            ax[0].set_axisbelow(True)
            ax[0].yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
            for p in ax[0].patches:   ## Print Column Value in chat
                height =p.get_height()
                width = p.get_width()
                x_cordinate = p.get_x()
                position = 0.99
                ## ax.annotate(text, xy, **kwargs)
                ax[0].annotate(f'{int(height)}', (x_cordinate + width / 2., height * position), ha='center', va='bottom', fontsize=8, fontweight='bold')
                ax[0].set_title(f'[{column}] Value Count')
                ax[0].tick_params(axis='x', rotation=90)
            ax[1].pie(self.value_count, labels=self.value_count.index,autopct='%.1f%%')
            ax[1].set_title(f'[{column}] Percentage(%)')
            ax[1].axis('equal')
            plt.tight_layout()
            plt.show()
        else:
            ## If column value count is Greater than 20 this block will get excecuted
            print(f"[{column}] Has Higher Unique Values, Picked Top Higher Frequency Value")
            self.top_freqcy = self.dataframe[column].value_counts().head(20).reset_index()
            self.top_freqcy.columns = [column, 'count'] ## Column Assign to the Newly created Dataframe [self.top_freqcy]
            plt.figure(figsize=figsize)
            ax = sns.barplot(data=self.top_freqcy,x=column, y='count')
            ## Add gridlines
            ax.set_axisbelow(True) 
            ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
            for p in ax.patches:
                height =p.get_height()
                width = p.get_width()
                x_cordinate = p.get_x()
                position = 0.99
                ## ax.annotate(text, xy, **kwargs)
                ax.annotate(f'{int(height)}', (x_cordinate + width / 2., height * position), ha='center', va='bottom', fontsize=8, fontweight='bold')
            plt.xticks(rotation=90)
            plt.title(f'[{column}] Top 10 High Frequency Value Barchart')
            plt.tight_layout()
            plt.show()
        return summary




    def NumericalFeatureAnalysis(self,column,figsize=(10,6)):
        """Numerical Feature Analysis And Visualization"""
        stat_summary = self.NumericalFeatureSummary() ## Method Call from Data Analysis Paresnt Class
        stat_summary = stat_summary.loc[stat_summary['field'].isin([column])].transpose().reset_index()
        stat_summary.columns = stat_summary.iloc[0]
        stat_summary = stat_summary.iloc[1:].reset_index(drop=True)

        outlier_df = self.get_outliers_for_column(column) ## Return OUtlier DF 

        self.value_count = self.dataframe[column].value_counts() ## Value Count for Each Column     
        ## Considering Descrete, If value Count of Numerical 
        if len(self.value_count) < 20: 
            print(f"Considered Descrete - Low Cardinality: {column}({self.dataframe[column].dtypes})")
            fig, ax = plt.subplots(2,2, figsize=figsize)
            sns.countplot(data=self.dataframe, x=column,ax=ax[0,0]) ## Count Plot
            ## Add gridlines
            for axis in ax.flatten():
                axis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')  # Grid lines visible behind the plots
            for p in ax[0,0].patches:   ## Print Column Value in chat
                height =p.get_height()
                width = p.get_width()
                x_cordinate = p.get_x()
                position = 0.99
                ## ax.annotate(text, xy, **kwargs)
                ax[0,0].annotate(f'{int(height)}', (x_cordinate + width / 2., height * position), ha='center', va='bottom', fontsize=8, fontweight='bold')
                ax[0,0].set_title(f'[{column}] Value Count')
                ax[0,0].tick_params(axis='x', rotation=90)
            
            sns.histplot(self.dataframe[column], bins=len(self.dataframe[column].unique()), discrete=True, ax=ax[0,1]) ## Histogram Plot
            ax[0,1].set_title(f'Frequency Distribution [{column}]')

            sns.boxplot(x=self.dataframe[column], ax=ax[1,0]) ## Box Plot
            ax[1,0].set_title(f'Outliers [{column}]')

            ax[1,1].pie(self.value_count, labels=self.value_count.index,autopct='%.1f%%')
            ax[1,1].set_title(f'[{column}] Percentage(%)')
            ax[1,1].axis('equal')
            plt.tight_layout()
            plt.show()
        else:
            fig,ax=plt.subplots(2,2,figsize=figsize) ## Subplots
            for axis in ax.flatten():
                axis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')  # Grid lines visible behind the plots

            sns.histplot(self.dataframe[column],kde=True, ax=ax[0,0]) ## Histogram Plot
            ax[0,0].set_title(f'Frequency Distribution [{column}]')
            
            sns.kdeplot(x=self.dataframe[column], ax=ax[0,1]) ## KDE Plot
            ax[0,1].set_title(f'Density Distribution [{column}]')

            sns.boxplot(x=self.dataframe[column], ax=ax[1,0]) ## Box Plot
            ax[1,0].set_title(f'Outliers [{column}]')

            sns.histplot(x=self.dataframe[column], cumulative=True, stat='density', element='step',ax=ax[1,1])
            ax[1,1].set_title(f'Cumulative Distribution [{column}]')

            plt.tight_layout()
            plt.show()
        return stat_summary, outlier_df
    




    
### Bivariate Analysis Class
class BivariateAnalysis(TableAnalysis): # Inherits functionality from the TableAnalysis class for univariate exploration
    def __init__(self,dataframe):
        super().__init__(dataframe)  # Initialize the parent DataAnalysis class with the dataframe
        self.dataframe = dataframe
        self.columns = list[self.dataframe.columns]
        self.numeical_cols = list(self.dataframe.select_dtypes(include=(int,float)).columns) ## Numerical Column 
        self.categorical_cols = list(self.dataframe.select_dtypes(include=(object)).columns) ## Categorical Column 

    def __str__(self):
        return f"categorical_columns= {self.categorical_cols}\nnumerical_columns= {self.numeical_cols}"

    def CategoricalNumericalBivar(self, cat_col, num_col, hue=None, estimator='median', figsize=(18,4)):
        """Categorical and Numerical Bivariate Analysis"""
        try:
            if cat_col not in self.categorical_cols:
                raise MyCustomException (f"[{cat_col}] Not a Categorical Column")
            elif num_col not in self.numeical_cols:
                raise MyCustomException(f"[{num_col}] Not a Numerical Column")
            else:
                self.value_count = self.dataframe[num_col].value_counts() ## Value Count for Each Column     
                ## Considering Descrete, If value Count of Numerical 
                if len(self.value_count) < 20: 
                    print(f"Considered Descrete - Low Cardinality: {num_col} ({self.dataframe[num_col].dtypes})")
                    fig,ax = plt.subplots(1,2, figsize= figsize)
                    sns.countplot(data=self.dataframe, x=cat_col, hue=num_col,ax=ax[0]) ## Count Plot
                    ax[0].set_title(f"Count Distribution ['{cat_col}' Across '{num_col}']")

                    crosstab = pd.crosstab(self.dataframe[cat_col], self.dataframe[num_col])
                    sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=ax[1]) ## Heat Map
                    ax[1].set_title(f"Heatmap: ['{cat_col}' vs '{num_col}']")
                    plt.show()

                else:
                    estimator_func = {
                                    'mean': np.mean,'median': np.median,'sum': np.sum,'std': np.std,'min': np.min,'max': np.max,
                                    }.get(estimator) 
                    fig,ax = plt.subplots(1,3, figsize= figsize) ## Bar Plot
                    sns.barplot(x=cat_col,y=num_col, data=self.dataframe, hue=hue, estimator=estimator_func, ax=ax[0]) ## Bar Plot
                    ax[0].set_title(f"Bar Plot [{estimator} of '{num_col}' grouped by '{cat_col}']")

                    sns.boxplot(x=self.dataframe[cat_col], y=self.dataframe[num_col], ax=ax[1]) ## Box Plot
                    ax[1].set_title(f"Box plot ['{num_col}' across '{cat_col}']")

                    # KDE Plot (one KDE per category)
                    for category in self.dataframe[cat_col].dropna().unique():
                        subset = self.dataframe[self.dataframe[cat_col] == category]
                        sns.kdeplot(subset[num_col], label=str(category), ax=ax[2])

                    ax[2].legend(title=cat_col)
                    ax[2].set_title(f"KDE Plot ['{num_col}' by '{cat_col}']")

                    plt.tight_layout()
                    plt.show()
        except MyCustomException as e:
            pass

    def CategoricalBivar(self, cat_col1, cat_col2, hue=None, estimator='median', figsize=(18,4)):
        """Categorical Bivariate Analysis"""
        try:
            if cat_col1 not in self.categorical_cols:
                raise MyCustomException (f"[{cat_col1}] Not a Categorical Colum")
            elif cat_col2 not in self.categorical_cols:
                raise MyCustomException(f"[{cat_col2}] Not a Categorical Column")
            else:
                fig,ax = plt.subplots(1,2, figsize= figsize)
                sns.countplot(data=self.dataframe, x=cat_col1, hue=cat_col2,ax=ax[0]) ## Count Plot
                ax[0].set_title(f"Count Distribution ['{cat_col1}' Across '{cat_col2}']")

                crosstab = pd.crosstab(self.dataframe[cat_col1], self.dataframe[cat_col2])
                sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=ax[1]) ## Heat Map
                ax[1].set_title(f"Heatmap: ['{cat_col1}' vs '{cat_col2}']")
                plt.show()
                return crosstab.reset_index()
        except MyCustomException as e:
            pass



    def NumericBivariate(self, num_col1, num_col2,hue=None, estimator='median',figsize=(12,8)):
        """Bivariate Analysis between Numerical columns"""
        try:
            if num_col1 not in self.numeical_cols:
                raise MyCustomException (f"[{num_col1}] Not a Numerical Column")
            elif num_col2 not in self.numeical_cols:
                raise MyCustomException(f"[{num_col2}] Not a Numerical Column")
            else:
                self.value_count = self.dataframe[num_col1].value_counts() ## Value Count for Each Column
                estimator_func = {
                                    'mean': np.mean,'median': np.median,'sum': np.sum,'std': np.std,'min': np.min,'max': np.max,
                                    }.get(estimator)  
                if len(self.value_count) < 20:
                    plt.figure(figsize=figsize)
                    print(f"Considered Descrete - Low Cardinality: {num_col1} ({self.dataframe[num_col1].dtypes})")
                    sns.barplot(x=num_col1,y=num_col2, data=self.dataframe, hue=hue, estimator=estimator_func) ## Bar Plot
                    plt.title(f"Bar Plot [{estimator} of '{num_col1}' grouped by '{num_col2}']")
                    plt.xticks(rotation=90)
                else:
                    plt.figure(figsize=figsize)
                    sns.scatterplot(data=self.dataframe,x=num_col1,y=num_col2, hue=hue, palette='dark') ## Scatter Plot
                    plt.title(f"Scatter Plot [{num_col1} vs {num_col2} by {hue}]")
                    plt.tight_layout()
                    plt.show()
        except MyCustomException as e:
            pass
