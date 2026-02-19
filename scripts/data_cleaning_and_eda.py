#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from math import exp
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Reading in Data")
    names = pd.read_csv('data/census_data/census-bureau.columns') #Load the header names into a dataframe
    df = pd.read_csv('data/census_data/census-bureau.data', names = names['age']) #Load the data in with the names being the first column of the header dataframe
    df = df.reset_index(drop = False, names = 'age') #Since age was loaded in as the index we will reset the index and name its column "age"

    print("Cleaning Data")
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns] #List comprehension to strip whitespace, convert to lowercase and add _ between words


    df['label'] = np.where(df['label']=='- 50000.', '<50', '>50') #Replace all instances of greater or less than 50000 with either <50 or >50


    df['age_bucket'] = pd.cut(df['age'], bins=9)



    money_cols = [
        'wage_per_hour',
        'capital_gains',
        'capital_losses',
        'dividends_from_stocks'
    ]


    len(df[(df['age']<20) & (df['label']=='<50')])


    # In[19]:


    df = df[df['age']>=20]


    lst = []

    for col in df.columns:
        if df[col].dtype == "object":  # ensure it's string-like
            count_1 = df[col].str.contains("?", na=False, regex = False).sum() # 
            count_2 = df[col].str.contains("Not in universe", case = False, na=False).sum()
            count = count_1 + count_2
            lst.append((col, count))


    cols_to_drop = []
    for column, val in lst:
        if val/len(df)>0.40:
            cols_to_drop.append(column)


    df = df.drop(columns = cols_to_drop)

    df = df.drop(columns = [
        'country_of_birth_self',
        'country_of_birth_mother',
        'country_of_birth_father',
        'detailed_occupation_recode',
        'detailed_industry_recode',
        'detailed_household_and_family_stat',
        'hispanic_origin',
        'weight',
        'own_business_or_self_employed',
        'detailed_household_summary_in_household',
        'tax_filer_stat',
        'year',
        'age_bucket'])


    print("Completing Feature Engineering")
    def group_education(edu):
        edu = str(edu).lower()

        # Advanced Degrees
        if any(x in edu for x in ['masters', 'doctorate', 'prof school']):
            return 'advanced_degree'

        # College (Bachelors)
        elif 'bachelors' in edu:
            return 'college'

        # Some College / Associates
        elif 'some college' in edu or 'associates' in edu:
            return 'some_college'

        # High School and everything below
        else:
            return 'high_school_or_less'

    # Apply the grouping
    df['education_grouped'] = df['education'].apply(group_education)

    df = df.drop(columns = ['education'])



    def group_citizenship(edu):
        edu = str(edu).lower()

        # Advanced Degrees
        if 'native' in edu:
            return 'us citizen'

        # College (Bachelors)
        elif 'naturalization' in edu:
            return 'us citizen'

        # High School and everything below
        else:
            return 'not a citizen'

    # Apply the grouping
    df['citizenship_grouped'] = df['citizenship'].apply(group_citizenship)

    df = df.drop(columns = ['citizenship'])


    def group_job_status(edu):
        edu = str(edu).lower()

        # Advanced Degrees
        if any(x in edu for x in ['unemployed', 'children', 'labor force']):
            return 'unemployed'

        # College (Bachelors)
        elif 'schedules' in edu:
            return 'fulltime'

        # Some College / Associates
        elif 'pt' in edu:
            return 'partime'


    # Apply the grouping
    df['ft_or_pt_grouped'] = df['full_or_part_time_employment_stat'].apply(group_job_status)

    df = df.drop(columns = ['full_or_part_time_employment_stat'])


    def group_class(edu):
        edu = str(edu).lower()

        # Advanced Degrees
        if any(x in edu for x in ['without pay', 'never worked']):
            return 'unemployed'

        # College (Bachelors)
        elif 'self-employed' in edu:
            return 'self employed'

        # Some College / Associates
        else:
            return edu


    # Apply the grouping
    df['class_of_worker_grouped'] = df['class_of_worker'].apply(group_class)

    df = df.drop(columns = ['class_of_worker'])


    def group_marital(edu):
        edu = str(edu).lower()

        # Advanced Degrees
        if 'never' in edu:
            return ' never married'

        # College (Bachelors)
        elif any(x in edu for x in ['widowed', 'divorced', 'separated']):
            return 'previously married'

        # Some College / Associates
        else:
            return 'married'


    # Apply the grouping
    df['marital_stat_grouped'] = df['marital_stat'].apply(group_marital)

    df = df.drop(columns = ['marital_stat'])


    def threshold_grouping(df, column, threshold_pct=0.05):
        # Calculate frequencies
        counts = df[column].value_counts(normalize=True)

        rare_categories = counts[counts < threshold_pct].index.tolist()

        # Replace rare ones with 'Other'
        df[column] = df[column].replace(rare_categories, 'other')

        return df


    df = threshold_grouping(df, 'major_industry_code', 0.05)
    df = threshold_grouping(df, 'major_occupation_code', 0.05)



    df['has_financial_activity'] = (df[money_cols] > 0).any(axis=1).astype(int)
    df = df.drop(columns = money_cols)
    df = df.reset_index(drop=True)
    print("Saving Final CSV")
    df.to_csv('data/processed/cleaned_dataset.csv')
    print("Final CSV saved to cleaned_dataset.csv")

if __name__ == '__main__':
    main()