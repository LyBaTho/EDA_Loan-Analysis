"""
| Field          | Description                                                                           |
|----------------|---------------------------------------------------------------------------------------|
| Loan_status    | Whether a loan is paid off on in collection                                           |
| Principal      | Basic principal loan amount at the                                                    |
| Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
| Effective_date | When the loan got originated and took effects                                         |
| Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
| Age            | Age of applicant                                                                      |
| Education      | Education of applicant                                                                |
| Gender         | The gender of applicant                                                               |

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import re
import scipy
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2

# Section 1: Data loading ---------------------------------------------------------------------------------------

loan = pd.read_csv('loan.csv')

  # Identify variables' data type and convert them if necessary

loan.shape
loan.info()
loan.describe()

"""
Input:
    * Principal: Categorical variable (Nominal)
    * Terms: Categorical variable (Ordinal)
    * Effective_date: Time Series
    * Due_date: Time Series
    * Age: Numerical variable (Discrete)
    * Education: Categorical variable (Ordinal)
    * Gender: Categorical variable (Nominal)
  
Output:
    * Loan_status: Categorical variable (Nominal)
"""

# Section 2: Data Preprocessing ---------------------------------------------------------------------------------------

  # Remove unnecessary columns
loan.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], axis = 1, inplace = True)
loan.columns = ['loan_status', 'principal', 'terms', 'effective_date', 'due_date', 'age', 'education', 'gender']

  # Identify null and duplicated variables and erase if necessary

loan.isnull().sum()

loan.duplicated().any()
loan.duplicated().sum()
loan[loan.duplicated()]
loan = loan.drop_duplicates().reset_index(drop = True)

"""
Comment:

  * No null values in these variables
  * There are total 40 duplicated rows
"""

  # Convert data type
cols_1 = ['age']
loan[cols_1] = loan[cols_1].apply(pd.to_numeric, errors = 'coerce')

cols_2 = ['loan_status', 'principal', 'terms' ,'effective_date', 'due_date', 'education', 'gender']
loan[cols_2].astype('object')

loan['principal'] = loan['principal'].astype('object')
loan['terms'] = loan['terms'].astype('object')

loan['due_date'] = pd.to_datetime(loan['due_date'])
loan['effective_date'] = pd.to_datetime(loan['effective_date'])

# Apply Feature Engineer to convert the categorical data types into numbers

  # Create new column 'dayofweek_encoder' from 'effective_date'
loan['dayofweek_encoder'] = loan['effective_date'].dt.dayofweek

  # gender (male: 1, female: 0)
loan['gender_encoder'] = loan['gender'].apply(lambda x: 1 if x == 'male' else 0)

  # terms (7: 0, 15: 1, 30: 2)

terms_type_mapping = {7: 0, 15: 1, 30: 2}
label_encoder_s1 = preprocessing.LabelEncoder()
loan['terms_encoder'] = loan['terms'].map(terms_type_mapping)
loan['terms_encoder'] = label_encoder_s1.fit_transform(loan['terms_encoder'])

  # Scaling terms_encoder by Normalization

scaler = MinMaxScaler()
terms_values = loan['terms_encoder'].values.reshape(-1, 1)
loan['scaled_std_terms'] = scaler.fit_transform(terms_values)

  # education ('High School or Below':0, 'college':1,'Bechalor':2, 'Master or Above':3)
education_type_mapping = {'High School or Below':0, 'college':1,'Bechalor':2, 'Master or Above':3}
label_encoder = preprocessing.LabelEncoder()
loan['education_encoder'] = loan['education'].map(education_type_mapping)
loan['education_encoder'] = label_encoder.fit_transform(loan['education_encoder'])

  # principal (300, 500, 800, 900, 1000)
principal_type_mapping = {300:0, 500:1, 800:2, 900:3, 1000:4}
label_encoder = preprocessing.LabelEncoder()
loan['principal_encoder'] = loan['principal'].map(principal_type_mapping)
loan['principal_encoder'] = label_encoder.fit_transform(loan['principal_encoder'])

  # Scaling principal_encoder by Normalization

scaler = MinMaxScaler()
principal_values = loan['principal_encoder'].values.reshape(-1, 1)
loan['scaled_std_principal'] = scaler.fit_transform(principal_values)

  # loan_status (collection: 1, paidoff: 0)
loan['loan_status_encoder'] = loan['loan_status'].apply(lambda x: 1 if x == 'COLLECTION' else 0)

  # Reconstructure data

loan_train = loan[['scaled_std_principal', 'scaled_std_terms', 'age', 'education_encoder', 'gender_encoder', 'dayofweek_encoder', 'loan_status_encoder']]

  # Convert data types

loan_train['scaled_std_principal'] = loan['scaled_std_principal'].apply(pd.to_numeric, errors = 'coerce')
loan['scaled_std_terms'] = loan['scaled_std_terms'].apply(pd.to_numeric, errors = 'coerce')

loan_train['gender_encoder'] = loan_train['gender_encoder'].astype('category')
loan_train['education_encoder'] = loan_train['education_encoder'].astype('category')
loan_train['loan_status_encoder'] = loan_train['loan_status_encoder'].astype('category')
loan_train['dayofweek_encoder'] = loan_train['dayofweek_encoder'].astype('category')

# Section 3: Univariate Analysis ---------------------------------------------------------------------------------------
  # Categorical Variables

cat_cols = loan_train.select_dtypes('category').columns
cat_cols = cat_cols.tolist()

for column in cat_cols:
  print('\n* Column:', column)
  print(len(loan_train[column].unique()), 'unique values')

def univariate_analysis_categorical_variable(df, group_by_col): 
    print(df[group_by_col].value_counts())
    df[group_by_col].value_counts().plot.bar(figsize=(5, 6),rot=0)
    plt.show()

for cat in cat_cols:
  print('Variable: ', cat)
  univariate_analysis_categorical_variable(loan_train, cat)
  print()

"""
Comment:
education ('High School or Below':0, 'college':1,'Bechalor':2, 'Master or Above':3)
gender (male: 1, female: 0)
dayofweek (Monday: 0, Sunday: 6)
loan_status (collection: 1, paidoff: 0)

  * education_encoder: Most borrowers are College graduates and High School or below
  * gender_encoder: Male gender dominates the porportion
  * dayofweek_encoder: Borrowers start their loan mostly on Sunday and Monday, Thursday witness the lowest number
  * loan_status_encoder: Significant number of borrowers paid off their debt
"""

  # Numerical Variables

num_cols = loan_train.select_dtypes('number').columns
num_cols = num_cols.tolist()

for column in num_cols:
  print('\n* Column:', column)
  print(len(loan_train[column].unique()), 'unique values')

def univariate_analysis_continuous_variable(df, feature):
    print("Describe:")
    print(feature.describe(include='all'))
    print("Mode:", feature.mode())
    print("Range:", feature.values.ptp())
    print("IQR:", scipy.stats.iqr(feature))
    print("Var:", feature.var())
    print("Std:", feature.std())
    print("Skew:", feature.skew())
    print("Kurtosis:", feature.kurtosis())

def check_outlier(df, feature):
    plt.boxplot(feature)
    plt.show()
    Q1 = np.percentile(feature, 25)
    Q3 = np.percentile(feature, 75)
    n_O_upper = df[feature > (Q3 + 1.5*scipy.stats.iqr(feature))].shape[0]
    print("Number of upper outliers:", n_O_upper)
    n_O_lower = df[feature < (Q1 - 1.5*scipy.stats.iqr(feature))].shape[0]
    print("Number of lower outliers:", n_O_lower)
    # Percentage of ouliers
    outliers_per = (n_O_lower + n_O_upper)/df.shape[0]
    print("Percentage of ouliers:", outliers_per)
    return Q1, Q3, n_O_upper, n_O_lower, outliers_per

def univariate_visualization_analysis_continuous_variable_new(feature):
    # Histogram
    feature.plot.kde()
    plt.show()      
    feature.plot.hist()
    plt.show() 

for con in num_cols:
  print('Variable: ', con)
  univariate_analysis_continuous_variable(loan_train, loan_train[con])
  check_outlier(loan_train, loan_train[con])
  univariate_visualization_analysis_continuous_variable_new(loan_train[con])
  print()

"""
Comment:

  * scaled_std_principal: no outliers, most common value is 0.58, positive Kurtosis with 2 tops and left-sided skewness
  * scaled_std_terms: no outliers, most common value is 0.93, negative Kurtosis with 2 tops and left-sided skewness
  * age: few outliers (4), most common value is 26, positive Kurtosis and right-sided skewness (a bit like Normal Distribution)
"""

# Section 4: Bivariate Analysis: Input - Output ---------------------------------------------------------------------------------------
  # Numerical - Categorical

output = 'loan_status_encoder'
cat_cols.remove('loan_status_encoder')

# ANOVA

import statsmodels.api as sm
from statsmodels.formula.api import ols

def variables_cont_cat(df, col1, col2):
    df_sub = df[[col1, col2]]
    plt.figure(figsize=(5,6))
    sns.boxplot(x=col1, y=col2, data=df_sub, palette="Set3")
    plt.show()
    chuoi = str(col2)+' ~ '+str(col1)
    model = ols(chuoi, data=df_sub).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print('ANOVA table: ', anova_table)

col1 = 'loan_status_encoder'
for i in range(0, len(num_cols)):
    col2 = num_cols[i]
    print('2 variables:', col1, 'and', col2)
    variables_cont_cat(loan_train, col1, col2)
    print()

"""
Comment:

  * loan_status_encoder and scaled_std_principal: suggests that there's an influence because the P-value (0.19) < 0.05
  * loan_status_encoder and scaled_std_terms: suggests that there's an influence because the P-value (0.02) < 0.05
  * loan_status_encoder and age: suggests that there's no influence because the P-value (0.6) > 0.05

There's a weak relationship between input(age) and output(loan_status) 
    => We may consider to drop the 'age' feature, compared to principal feature, it has a bit stronger link so we may keep it
"""

# z-test

'''
Check the Numerical variables in 2 group (Collection: 1, Paidoff: 0) of loan_status_encoder

*  **H0:** There is no mean difference in turn of Numerical variables between Collection: 1, Paidoff: 0
*  **H1:** There is mean difference in turn of Numerical variables between Collection: 1, Paidoff: 0
'''

from statsmodels.stats.weightstats import ztest

def z_test_loop(data, group_column, value_columns, alpha):

    results = {}
    for column in value_columns:
        group1_data = data[data[group_column] == 0][column]
        group2_data = data[data[group_column] == 1][column]
        z_score, p_value = ztest(group1_data, group2_data, value=group1_data.mean())
        if p_value > alpha:
            result = "Accept the null hypothesis that the means are equal."
        else:
            result = "Reject the null hypothesis that the means are equal."
        results[column] = result
    return results

group_column = 'loan_status_encoder'
alpha = 0.05

for i in range(len(num_cols)):
    value_columns = [num_cols[i]]
    results = z_test_loop(loan_train, group_column, value_columns, alpha)
    for column, result in results.items():
        print("Column: {}".format(column))
        print(result)
        print()

'''
Comment:

Since p-value of 'age' is less than 0.05, we have enough evidence to reject hypothesis H0.

However, p-values of 'scaled_std_principal' and 'scaled_std_terms' is higher than 0.05, we don't have enough evidence to reject hypothesis H0
'''

  # Categorical - Categorical

col2 = 'loan_status_encoder'
lst = []

def categorical_categorical(feature1, feature2):
    # Contingency table
    table_FB = pd.crosstab(feature1, feature2)
    print(table_FB)
    table_FB.plot(kind='bar', stacked=True, figsize=(5, 6),rot=0)
    plt.show()
    table_FB.plot.bar(figsize=(5, 6))
    plt.show()
    
    # Chi-Square Test
    stat, p, dof, expected = chi2_contingency(table_FB)
    print('dof=%d' % dof)
    print('p=', p)
    # interpret test-statistic
    prob = 0.95
    critical = chi2.ppf(prob, dof)
    print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
    # interpret p-value
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
        print('Dependent (reject H0)')
        x1 = feature1.name
        x2 = feature2.name
        chuoi = x1 + ' and ' + x2
        return chuoi
    else:
        print('Independent (fail to reject H0)')

for i in range(0, len(cat_cols)):
    col1 = cat_cols[i]
    print('2 variables:', col1, 'and', col2)
    chuoi = categorical_categorical(loan_train[col1], loan_train[col2])
    lst.append(chuoi)
    print()

"""
Comment:

  * education_encoder and loan_status_encoder: p-value (0.88) > alpha (0.05) => fail to reject the null hypothesis => 2 independent variables
  * gender_encoder and loan_status_encoder: p-value (0.02) < alpha (0.05) => reject the null hypothesis => 2 dependent variables
  * dayofweek_encoder and loan_status_encoder: p-value (4.1e-12) < alpha (0.05) => reject the null hypothesis => 2 dependent variables
"""

# Section 4: Bivariate Analysis: Input - Input ---------------------------------------------------------------------------------------
  # Numerical - Numerical

for i in range(0, len(num_cols)):
    col1 = num_cols[i]
    for j in range(i+1, len(num_cols)):
        col2 = num_cols[j]
        print('Correlation between 2 variables:', col1, 'and', col2)
        print(loan_train[[col1, col2]].corr())
        print('Pearson Correlation between 2 variables:', col1, 'and', col2)
        print(stats.pearsonr(loan_train[col1], loan_train[col2]))
        print('Spearman Correlation between 2 variables:', col1, 'and', col2)
        print(stats.spearmanr(loan_train[col1], loan_train[col2]))
        sns.pairplot(loan_train[[col1, col2]])
        plt.show()        
        print()

"""
Comment:

  * scaled_std_principal and scaled_std_terms: moderate positive linear relationship and moderate positive monotonic relationship
  * scaled_std_principal and age: strongly negative linear relationship and strongly negative monotonic relationship
  * scaled_std_terms and age: almost no linear relationship and very weak monotonic relationship

There's a weak/not too significant relationship between age column and others
"""

# Section 4: Bivariate Analysis: Input - Input
  # Categorical - Categorical

lst = []
for i in range (0, len(cat_cols)):
  col1 = cat_cols[i]
  for j in range (i+1, len(cat_cols)):
    col2 = cat_cols[j]
    print('2 variables:', col1, 'and', col2)
    chuoi = categorical_categorical(loan_train[col1], loan_train[col2])
    lst.append(chuoi)
    print()

"""
Comment:

  * education_encoder and gender_encoder: p-value (0.59) > alpha (0.05) => fail to reject the null hypothesis => 2 independent variables
  * education_encoder and dayofweek_encoder: p-value (0.84) > alpha (0.05) => fail to reject the null hypothesis => 2 independent variables
  * gender_encoder and dayofweek_encoder: p-value (0.19) > alpha (0.05) => fail to reject the null hypothesis => 2 independent variables
"""

# Section 4: Bivariate Analysis: Input - Input
  # Numerical - Categorical

for i in range(0, len(cat_cols)):
    col1 = cat_cols[i]
    for j in range(0, len(num_cols)):
        col2 = num_cols[j]
        print('2 variables:', col1, 'and', col2)
        variables_cont_cat(loan_train, col1, col2)
        print()

"""
Comment:

  * education_encoder and scaled_std_principal: suggests that there's no influence because the P-value (0.89) > 0.05
  * education_encoder and scaled_std_terms: suggests that there's no influence because the P-value (0.5) > 0.05
  * education_encoder and age: suggests that there's an influence because the P-value (0.001) < 0.05 
---
  * gender_encoder and scaled_std_principal: suggests that there's no influence because the P-value (0.9) > 0.05
  * gender_encoder and scaled_std_terms: suggests that there's no influence because the P-value (0.72) > 0.05
  * gender_encoder and age: suggests that there's no influence because the P-value (0.63) > 0.05
---
  * dayofweek_encoder and scaled_std_principal: suggests that there's no influence because the P-value (0.47) > 0.05
  * dayofweek_encoder and scaled_std_terms: suggests that there's an influence because the P-value (0.03) < 0.05
  * dayofweek_encoder and age: suggests that there's no influence because the P-value (0.31) > 0.05
"""
'''
Recommend removing the attribute age.

Because:

  * ANOVA: The age column does not show a significant influence on the output variable (loan_status_encoder), as the p-value is higher than the significance level of 0.05. 
      Therefore, removing the age column would not have a major impact on the analysis based on the ANOVA results.
  * Chi-squared: The age column is not directly involved in the chi-squared analysis, as it focuses on the relationship between categorical variables. 
      Removing the age column would not affect the chi-squared analysis or the determination of independence between variables.
  * Pearson and Spearman Coefficients: Removing the age column would result in the loss of the relationship between scaled_std_terms and age. 
      However, based on the analysis results, the relationship between scaled_std_terms and age is considered to have almost 
      no linear relationship and a very weak monotonic relationship. 
      Therefore, removing the age column would not significantly impact the overall analysis based on the correlation coefficients.
'''

# Section 5: Data Prediction ---------------------------------------------------------------------------------------

X = loan_train[['scaled_std_principal','scaled_std_terms', 'education_encoder', 'gender_encoder', 'dayofweek_encoder', 'loan_status_encoder']] 
y = loan_train['loan_status_encoder']

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

'''
Use KNN for this project prediction 

Removing age column => KNN can capture local patterns and adapt to 
  complex decision boundaries that may not be well approximated by linear models

Normalization scaling is applied to the terms and principal features, which is beneficial for KNN as it maintains equal feature contribution, 
  prevents dominance by features with larger ranges, and allows for similarity-based decision making.
'''

def knn_with_varying_k(X, y):
    k_values = [3, 5, 7, 9, 11, 13, 15]
    results = {}

    for k in k_values:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[k] = accuracy
    return results

accuracy_results = knn_with_varying_k(X, y)

for k, accuracy in accuracy_results.items():
    print("Accuracy for k =", k, ":", accuracy)
