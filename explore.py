# IMPORTS
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr

# FUNCTIONS
def plot_variable_pairs(train):
    """
    This function is specific to zillow. It will
    - accept the train dataframe (target is 'tax_value')
    - set the columns to visualize to be the continuous variables with the target at the end
    - only look at a sample of 1000 to keep the run time reasonable
    - do sns.lmplot for the target variable vs each feature
    """
    cols = ['bedrooms', 'bathrooms', 'square_feet', 'year_built', 'tax_amount', 'tax_value']

    sample = train.sample(1000)

    for i, col in enumerate(cols[:-1]):
        sns.lmplot(data=sample, x=col, y='tax_value')
        plt.title(f'tax_value vs {col}')
        plt.show()

def plot_categorical_and_continuous_vars(train, cols_contin, cols_cat):
    """
    This function will
    - plot 3 plots (boxen, violin, and box) for each categorical variable vs each continuous variable
    - accepts a dataframe (train), a list of continuous column names (cols_contin),
      and a list of categorical column names (cols_cat)
    - prints all the plots
    - returns nothing
    """
    # set sample to something that will run in a reasonable amount of time
    sample = train.sample(10000, random_state=42)

    for cat in cols_cat:

        for col in cols_contin:

            sns.boxenplot(data=sample, x=cat, y=col)
            plt.title(f'{cat} vs. {col}, boxen')
            plt.show()

            sns.violinplot(data=sample, x=cat, y=col)
            plt.title(f'{cat} vs. {col}, violin')
            plt.show()

            sns.boxplot(data=sample, x=cat, y=col)
            plt.title(f'{cat} vs. {col}, boxplot')
            plt.show()

