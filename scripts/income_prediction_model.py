#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.pipeline import Pipeline
from math import exp
import pickle


def main():
    df = pd.read_csv('data/processed/cleaned_dataset.csv')
    df = df.drop(columns = ['Unnamed: 0'], errors = 'ignore')


    df["label"] = df["label"].astype(str).str.strip()

    X = df.drop(columns=["label"])
    y = (df["label"] == ">50").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify = y, random_state = 42) #Set the random state for reproducible results


    categorical = [ #Create the list of categorical variables
        'class_of_worker_grouped',
        'education_grouped',
        'marital_stat_grouped',
        'major_industry_code',
        'major_occupation_code',
        'race',
        'sex',
        'ft_or_pt_grouped',
        'citizenship_grouped',
        'veterans_benefits',
        'has_financial_activity',
        'num_persons_worked_for_employer'
    ]

    numerical = [ #Create the list of numerical variables
        'age',
        'weeks_worked_in_year'
    ]

    ca = ColumnTransformer(transformers = [
        ("cat", OneHotEncoder(handle_unknown = 'ignore', sparse_output = False), categorical), 
        ("num", StandardScaler(), numerical)
        ],
        remainder = 'drop'
    )
    print("Creating Model")
    pipe_final = Pipeline(steps = [
        ('preprocess', ca),
        ('lr', LogisticRegression(random_state = 42,
                                C = 0.01,
                                max_iter = 1000,
                                penalty = 'elasticnet',
                                solver = 'saga',
                                l1_ratio = 1))
        ])
    print("Training Model")
    pipe_final.fit(X_train, y_train)

    print("Compiling Predictions and Evaluation Metrics")
    threshold = 0.55
    y_proba = pipe_final.predict_proba(X_test)[:, 1]
    predictions = (y_proba >= threshold).astype(int)
    print('Printing Classification Report')
    print(classification_report(y_test, predictions))



    auc_score = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {auc_score:.4f}")


    feature_names = pipe_final.named_steps["preprocess"].get_feature_names_out()

    coefficients = pipe_final.named_steps["lr"].coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients
    })

    coef_df_sorted = coef_df.sort_values(by="coefficient", ascending=False)


    coef_df_sorted['coefficient'] = coef_df_sorted['coefficient'].apply(exp)

    print('Top features increasing odds of >50K (highest odds ratios):')
    print(coef_df_sorted.head(3))
    print('Top features decreasing odds of >50K (lowest odds ratios):')
    print(coef_df_sorted.tail(3))
    print("Saving All Coefficients to outputs/income_prediction_outputs/model_coefficients.csv")
    coef_df_sorted.to_csv('outputs/income_prediction_outputs/model_coefficients.csv', index = False)
    with open("models/final_income_model.pkl", "wb") as f:
        pickle.dump(pipe_final, f)

    print("Saved model: final_income_model.pkl")


if __name__ == '__main__':
    main()