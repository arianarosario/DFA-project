#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/arianarosario/DFA-project/blob/main/deliverable.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # 1. Data imports & data cleaning

# ### Importing the data

# In[ ]:


from ucimlrepo import fetch_ucirepo

# fetch dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

# data (as pandas dataframes)
feature_data = cdc_diabetes_health_indicators.data.features
target_data = cdc_diabetes_health_indicators.data.targets


# In[ ]:


# metadata
cdc_diabetes_health_indicators.metadata


# In[ ]:


# variable information
variable_info = cdc_diabetes_health_indicators.variables
variable_info


# # 2. Summary statistics

# ### Exploring the data

# In[ ]:


# view the first few rows of the data
feature_data.head()


# In[ ]:


feature_data.describe(include='all')


# In[ ]:


feature_data.dtypes


# In[ ]:


sensitive_attributes = ['Sex', 'Age', 'Education', 'Income']

variable_info[variable_info['name'].isin(sensitive_attributes)]


# In[ ]:


# Variable Detailed Description from Codebook
# source: https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf

Sex_desc = {
    0: 'Female',
    1: 'Male'
}

Age_desc = {
    1: 'Age 18 to 24',
    2: 'Age 25 to 29',
    3: 'Age 30 to 34',
    4: 'Age 35 to 39',
    5: 'Age 40 to 44',
    6: 'Age 45 to 49',
    7: 'Age 50 to 54',
    8: 'Age 55 to 59',
    9: 'Age 60 to 64',
    10: 'Age 65 to 69',
    11: 'Age 70 to 74',
    12: 'Age 75 to 79',
    13: 'Age 80 or older'
    }

Education_desc = {
    1: 'Never attended school or only kindergarten',
    2: 'Grades 1 through 8 (Elementary)',
    3: 'Grades 9 through 11 (Some high school)',
    4: 'Grade 12 or GED (High school graduate)',
    5: 'College 1 year to 3 years (Some college or technical school)',
    6: 'College 4 years or more (College graduate)'
}

Income_desc = {
    1: 'Less than $10,000',
    2: 'Less than $15,000 ($10,000 to less than $15,000)',
    3: 'Less than $20,000 ($15,000 to less than $20,000)',
    4: 'Less than $25,000 ($20,000 to less than $25,000)',
    5: 'Less than $35,000 ($25,000 to less than $35,000)',
    6: 'Less than $50,000 ($35,000 to less than $50,000)',
    7: 'Less than $75,000 ($50,000 to less than $75,000)',
    8: '$75,000 or more',
}


# ### a. How many rows & columns of data do you have, both overall, and per sensitive attribute subgroup?

# In[ ]:


# How many rows & columns of data do you have, both overall, and per sensitive attribute subgroup?

rows, columns = feature_data.shape
print(f'There are {rows} rows and {columns} columns in the dataset')

for att in sensitive_attributes:
    print(f'\n{att} Variable:')

    n_subgroups = feature_data[att].nunique()
    desc_dict_name = att + '_desc'
    if att=='Sex':
        first_val = 0
    else:
        first_val = 1

    for subgroup in range(first_val, n_subgroups + first_val):

        subgroup_name = eval(desc_dict_name)[subgroup]
        print(f'  {subgroup} ({subgroup_name}): {len(feature_data[feature_data[att] == subgroup])} rows')


# ### b. What are your outcome variables of interest?
# > Our outcome variable is called *"Diabetes_binary"* and is stored in target_data
# 

# In[ ]:


# What mean / median / standard deviation values do the outcome variables have overall?

target_data.describe()


# In[ ]:


# What about these statistics within each relevant sensitive attribute subgroup?

# merging target and feature data
merged_data = feature_data.copy()
merged_data['Diabetes_binary'] = target_data

for att in sensitive_attributes:
    print(f'\n{att} Variable:')

    n_subgroups = merged_data[att].nunique()
    desc_dict_name = att + '_desc'
    if att=='Sex':
        first_val = 0
    else:
        first_val = 1

    for subgroup in range(first_val, n_subgroups + first_val):

        subgroup_name = eval(desc_dict_name)[subgroup]
        subgroup_diabetes_percent = merged_data[merged_data[att] == subgroup]['Diabetes_binary'].mean()
        print(f'\t{subgroup} ({subgroup_name}): \t{subgroup_diabetes_percent:.2%} have diabetes')


# ### c. Plot at least as many figures as your # group members, and explain concisely but meaningfully what the plot shows in markdown text

# In[ ]:


# helper function to plot a varible distribution

import matplotlib.pyplot as plt
import seaborn as sns

def plot_dist(data, col, labels):
    sns.countplot(x=col, data=data,
                  stat="percent", palette=['lightcoral','skyblue'])
    if labels:
        plt.xticks(ticks=[0,1], labels=labels.values())
        plt.xlabel('')


# #### Relative Distribution of Binary Variables

# In[ ]:


# hiding warnings
import warnings
warnings.filterwarnings('ignore')

# setting up multi plot
fig = plt.figure(figsize=(10,20))
plt.suptitle('Relative Distribution of Binary Variables',
             va='bottom', y=0.9, fontsize=20)
fig_count = 1

# plotting target variable
col = 'Diabetes_binary'
x_labels = {0:'No',1:'Yes'}

plt.subplot(5, 3, fig_count)
plt.title("#"+str(fig_count)+": "+col)
plot_dist(target_data, col, x_labels)

# plotting binary features (14 out of 21)
binary_features = feature_data.columns[feature_data.nunique() == 2]

for col in binary_features:
    fig_count += 1

    if col == 'Sex':
        x_labels = {0:'Female',1:'Male'}

    plt.subplot(5, 3, fig_count)
    plt.title("#"+str(fig_count)+": "+col)
    plot_dist(feature_data, col, x_labels)

plt.show()


# #### Relative Distribution of Other Variables

# In[ ]:


other_features = [feat for feat in feature_data.columns if feat not in binary_features]

# setting up multi plot
fig = plt.figure(figsize=(10,20))
plt.suptitle('Relative Distribution of Other Variables',
             va='bottom', y=0.9, fontsize=20)
fig_count = 1

for att in other_features:
    plt.subplot(4, 2, fig_count)
    sns.histplot(x=att , data=merged_data, stat="percent", shrink=5)
    plt.xlabel('')

    plt.title("#"+str(fig_count+15)+": "+att)

    fig_count += 1
plt.show()


# #### Correlation Heatmap

# In[ ]:


# creating a heatmap to show correlation between the features
plt.figure(figsize=(12,10))
correlation = feature_data.corr()

# selecting the highly correlated features (abs corr > 0.25)
hi_corr = correlation.copy()
threshold = 0.25
hi_corr[(abs(hi_corr) < threshold) | (abs(hi_corr) > 0.99)] = 0
sns.heatmap(hi_corr, annot=False, cmap='coolwarm', center=0)


# # 3. Research Question, Hypotheses, and Analysis Plan

# ### a. Concretely, what is (are) your research question(s)? Be specific: what are the inputs, outputs, and evaluation metrics you are interested in, and why?
# 
# > * Does diabetes prediction accuracy change across income groups?
# > * Does diabetes prediction accuracy change across sex groups?
# > * If there do exist differences, what are the driving factors?
# 
# #### Inputs
# > * Our inputs include: HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, and DiffWalk.
# > * The demographic input data are Sex, Age, Education, and Income.
# 
# #### Outputs
# > Our output is the binary prediction such that 1 is the patient is diagnosed with diabetes or prediabetes and 0 represents that the patient is not diabetic.
# 
# #### Evaluation metrics
# > The evaluation metrics we want to use for our algorithm include assessing if the accuracy, precision, recall, and FNR is different for various income and sex groups.
# 
# ### b. What are your hypotheses? e.g., do you notice any potential biases from your summary statistics? What are they, and why might these exist?
# 
# > * From the summary statistics, we observe that there are about **27% more females** than males identified in the data.
# > * Additionally, almost all participants had indicated that they had their cholesterol check in the past 5 years. There could be a bias in the dataset towards individuals that are more health conscious, have had family history of high cholesterol, and previous concerns regarding their cholesterol levels and are predisposed to a diagnosis related to their cholesterol levels. Additionally, this indicator could be associated with most of the participants having an income level of around $75,000 or more. Many of the participants in this dataset could have the financial ability to upkeep with annual check ups and take preventative cautions in regards to cholesterol levels and a potential diabetes diagnosis.
# 
# ### c. What analyses are you going to run (in section 4) to test your hypotheses presented above?
# > * We will be testing our hypotheses with the following models: Linear Regression, SVM, LightGBM, Naive Bayes, and Random Forest.
# > * To observe potential biases in prediction, we will utilize upsampling methods for minority groups and compare with its reciprocal model without upsampling methods employed.
# 
# 
# 
# 

# # 4. Modeling

# ## Model 1: Naive Bayes
# ### a. Model

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, auc, roc_curve, roc_auc_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score


# In[ ]:


X = feature_data
y = target_data.Diabetes_binary


# In[ ]:


X_train_test, X_test, y_train_test, y_test = train_test_split(X, y, test_size=0.2)

X_train, X_val, y_train, y_val = train_test_split(X_train_test, y_train_test, test_size=0.25)


# In[ ]:


# Train the model
model = GaussianNB()
model.fit(X_train, y_train)


# ### b. Evaluation Metrics

# In[ ]:


# Predict on the validation set
y_val_pred = model.predict(X_val)


# In[ ]:


# Evaluate the model
f1 = f1_score(y_val, y_val_pred, average='weighted')
print(f"F1-Score on Validation Set: {f1}")


# In[ ]:


# Predict on the test set
y_test_pred = model.predict(X_test)

# Evaluate the model
f1 = f1_score(y_test, y_test_pred, average='weighted')
print(f"F1-Score on Test Set: {f1}")


# Since the f1-score for the validation set and test set are comparable, we can state that there are no signs of overfitting.

# In[ ]:


print(metrics.classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_test_pred))


# 3087 participants were identified as non-diabetic when they were diabetic or prediabetic in actuality. The FNR from utilizing Naive Bayes is 0.432.

# In[ ]:


female_idx = X_test['Sex']==0
male_idx = X_test['Sex']==1

female_test = y_test.loc[female_idx]
male_test = y_test.loc[male_idx]

female_pred  = y_test_pred[female_idx]
male_pred = y_test_pred[male_idx]

y_pred_prob = model.predict_proba(X_test)[:, 1]
female_score = y_pred_prob[female_idx]
male_score = y_pred_prob[male_idx]

true_y = {'all': y_test, 'female':female_test, 'male':male_test}
pred_y = {'female':female_pred, 'male':male_pred}
prob_y = {'all':y_pred_prob, 'female':female_score, 'male':male_score}

for group in ['all', 'female', 'male']:
    AUC = metrics.roc_auc_score(true_y[group], prob_y[group])
    print(f'AUC ({group} participants): {AUC:.4f}')


tn, fp, fn, tp = {}, {}, {}, {}
for group in ['female', 'male']:
    tn[group], fp[group], fn[group], tp[group] = metrics.confusion_matrix(true_y[group], pred_y[group]).ravel()


# In[ ]:


for group in ['female', 'male']:
    FNR = fn[group]/(tp[group]+fn[group])
    print(f'FNR ({group} participants): {FNR:.4f}')


# In[ ]:


for group in ['female', 'male']:
    FPR = fp[group]/(tn[group]+fp[group])
    print(f'FPR ({group} participants): {FPR:.4f}')


# In[ ]:


for group in ['female', 'male']:
    PR = pred_y[group].mean()
    print(f'Fraction positive ({group} participants): {PR:.4f}')


# In[ ]:


X_test.Income.unique()


# In[ ]:


one_idx = X_test['Income']==1
two_idx = X_test['Income']==2
three_idx = X_test['Income']==3
four_idx = X_test['Income']==4
five_idx = X_test['Income']==5
six_idx = X_test['Income']==6
seven_idx = X_test['Income']==7
eight_idx = X_test['Income']==8

one_test = y_test.loc[one_idx]
two_test = y_test.loc[two_idx]
three_test = y_test.loc[three_idx]
four_test = y_test.loc[four_idx]
five_test = y_test.loc[five_idx]
six_test = y_test.loc[six_idx]
seven_test = y_test.loc[seven_idx]
eight_test = y_test.loc[eight_idx]

one_pred  = y_test_pred[one_idx]
two_pred = y_test_pred[two_idx]
three_pred = y_test_pred[three_idx]
four_pred = y_test_pred[four_idx]
five_pred = y_test_pred[five_idx]
six_pred = y_test_pred[six_idx]
seven_pred = y_test_pred[seven_idx]
eight_pred = y_test_pred[eight_idx]

y_pred_prob = model.predict_proba(X_test)[:, 1]
one_score = y_pred_prob[one_idx]
two_score = y_pred_prob[two_idx]
three_score = y_pred_prob[three_idx]
four_score = y_pred_prob[four_idx]
five_score = y_pred_prob[five_idx]
six_score = y_pred_prob[six_idx]
seven_score = y_pred_prob[seven_idx]
eight_score = y_pred_prob[eight_idx]


true_y = {'all': y_test, 'Income 1':one_test, 'Income 2':two_test, 'Income 3':three_test,
          'Income 4':four_test,'Income 5':five_test,'Income 6':six_test,'Income 7':seven_test,'Income 8':eight_test}
pred_y = {'Income 1':one_pred, 'Income 2':two_pred, 'Income 3':three_pred,
          'Income 4':four_pred,'Income 5':five_pred,'Income 6':six_pred,'Income 7':seven_pred,'Income 8':eight_pred}
prob_y = {'all': y_pred_prob, 'Income 1':one_score, 'Income 2':two_score, 'Income 3':three_score,
          'Income 4':four_score,'Income 5':five_score,'Income 6':six_score,'Income 7':seven_score,'Income 8':eight_score}


for group in ['all', 'Income 1', 'Income 2', 'Income 3', 'Income 4', 'Income 5', 'Income 6', 'Income 7', 'Income 8']:
    AUC = metrics.roc_auc_score(true_y[group], prob_y[group])
    print(f'AUC ({group} participants): {AUC:.4f}')


tn, fp, fn, tp = {}, {}, {}, {}
for group in ['Income 1', 'Income 2', 'Income 3', 'Income 4', 'Income 5', 'Income 6', 'Income 7', 'Income 8']:
    tn[group], fp[group], fn[group], tp[group] = metrics.confusion_matrix(true_y[group], pred_y[group]).ravel()


# In[ ]:


for group in ['Income 1', 'Income 2', 'Income 3', 'Income 4', 'Income 5', 'Income 6', 'Income 7', 'Income 8']:
    FNR = fn[group]/(tp[group]+fn[group])
    print(f'FNR ({group} participants): {FNR:.4f}')


# In[ ]:


for group in ['Income 1', 'Income 2', 'Income 3', 'Income 4', 'Income 5', 'Income 6', 'Income 7', 'Income 8']:
    FPR = fp[group]/(tn[group]+fp[group])
    print(f'FPR ({group} participants): {FPR:.4f}')


# In[ ]:


for group in 'Income 1', 'Income 2', 'Income 3', 'Income 4', 'Income 5', 'Income 6', 'Income 7', 'Income 8':
    PR = pred_y[group].mean()
    print(f'Fraction positive ({group} participants): {PR:.4f}')


# ### c. Informative Plots

# In[ ]:


train_sizes = np.linspace(0.1, 1.0, 10)
train_scores = []

for size in train_sizes:
    subset_size = int(size * X_train.shape[0])
    X_subset, y_subset = X_train[:subset_size], y_train[:subset_size]
    model.fit(X_subset, y_subset)
    train_scores.append(model.score(X_val, y_val))

# Plotting the learning curve over accuracy in validation set
plt.plot(train_sizes, train_scores, label='Accuracy')
plt.title('Learning Curve')
plt.xlabel('Fraction of Training Data Used')
plt.ylabel('Accuracy on Validation Set')
plt.legend()
plt.show()


# In[ ]:


cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = metrics.confusion_matrix(y_test, y_test_pred), display_labels = [False, True])

cm_display.plot()
plt.show()


# ## Model 2: Logistic Regression
# ### a. Model

# In[ ]:


from sklearn.linear_model import LogisticRegression
X = feature_data
y = target_data.Diabetes_binary

# splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# training the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# predicting on the test set
y_pred_logreg = logistic_model.predict(X_test)
y_pred_proba_logreg = logistic_model.predict_proba(X_test)[:, 1]


# ### b. Evaluation Metrics

# In[ ]:


from sklearn.metrics import roc_auc_score
# calculating the AUC and F1 score
AUC_logreg = roc_auc_score(y_test, y_pred_proba_logreg)
f1_logreg = f1_score(y_test, y_pred_logreg)

print(f'AUC: {AUC_logreg:.4f}')
print(f'F1 Score: {f1_logreg:.4f}')


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_logreg))


# In[ ]:


print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred_logreg))


# In[ ]:


# calculating AUC and FNR for each sensitive attribute subgroup

AUC_subgroups = {}
FNR_subgroups = {}
FPR_subgroups = {}

for att in sensitive_attributes:
    AUC_subgroups[att] = {}
    FNR_subgroups[att] = {}
    FPR_subgroups[att] = {}

    subgroups = feature_data[att].unique()
    for subgroup in subgroups:
        curr_subgroup_index = X_test[att] == subgroup
        curr_X_test = X_test[curr_subgroup_index]
        curr_y_test = y_test[curr_subgroup_index]
        curr_y_pred = logistic_model.predict(curr_X_test)
        curr_y_pred_proba = logistic_model.predict_proba(curr_X_test)[:, 1]
        curr_AUC = roc_auc_score(curr_y_test, curr_y_pred_proba)
        tn, fp, fn, tp = confusion_matrix(curr_y_test, curr_y_pred).ravel()
        curr_FNR = fn/(tp+fn)
        curr_FPR = fp/(tn+fp)

        AUC_subgroups[att][subgroup] = curr_AUC
        FNR_subgroups[att][subgroup] = curr_FNR
        FPR_subgroups[att][subgroup] = curr_FPR


# In[ ]:


# formatting the results into a df

data = [(att, subgroup) for att, subgroups in AUC_subgroups.items() for subgroup in subgroups]
results_df = pd.DataFrame(data, columns=['Attribute', 'Subgroup'])

results_df['AUC'] = [AUC_subgroups[att][subgroup] for att, subgroups in AUC_subgroups.items() for subgroup in subgroups]
results_df['FNR'] = [FNR_subgroups[att][subgroup] for att, subgroups in FNR_subgroups.items() for subgroup in subgroups]
results_df['FPR'] = [FPR_subgroups[att][subgroup] for att, subgroups in FPR_subgroups.items() for subgroup in subgroups]

results_df


# ### c. Informative Plots

# In[ ]:





# ## Model 3: SVC
# 
# ### a. Model
# 

# In[ ]:


from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
X = feature_data
y = target_data['Diabetes_binary']

X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
from sklearn.utils import resample
from sklearn.svm import LinearSVC

# Selecting a random subset of the data
X_train_subset, y_train_subset = resample(X_train_smote, y_train_smote, n_samples=int(len(X_train) * 0.1), random_state=42, replace=False)

# Training the LinearSVC model on the subset
linear_svc_subset_model = LinearSVC(random_state=42, max_iter=1000, C=0.1)
linear_svc_subset_model.fit(X_train_subset, y_train_subset)

# Predicting on the test set
y_pred_subset = linear_svc_subset_model.predict(X_test)


# In[ ]:


y_pred_subset


# ### b. Evaluation Metrics

# In[ ]:


from sklearn.metrics import roc_auc_score

# Calculating AUC
auc_score = roc_auc_score(y_test, y_pred_subset)
auc_score


# In[ ]:


# Calculating accuracy for the LinearSVC model trained on the subset
accuracy_subset = accuracy_score(y_test, y_pred_subset)
accuracy_subset


# In[ ]:


from sklearn.metrics import confusion_matrix

# Generating the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_subset)

# Extracting false negatives (FN) and true positives (TP)
FN = conf_matrix[1][0]
TP = conf_matrix[1][1]

# Calculating the False Negative Rate (FNR)
FNR = FN / (FN + TP)
FNR


# In[ ]:


from sklearn.metrics import classification_report

# Classification report
class_report = classification_report(y_test, y_pred_subset)
print(class_report)


# In[ ]:


y_pred_subset_df = pd.DataFrame(y_pred_subset, columns=['Predicted_Diabetes_binary'])

if len(y_pred_subset_df) == len(X_test):
    # Add the predictions to the test dataset
    X_test_df = pd.DataFrame(X_test, columns=feature_data.columns)
    test_with_pred_df = X_test_df.reset_index(drop=True)
    test_with_pred_df['Predicted_Diabetes_binary'] = y_pred_subset_df.reset_index(drop=True)
    test_with_pred_df['True_Diabetes_binary'] = y_test.reset_index(drop=True)
else:
    result = "The length of the prediction set does not match the test set, can't proceed with merging."

# Checking the first few rows
test_with_pred_df


# In[ ]:


from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np

def calculate_fnr_fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    return fnr, fpr

# Function to calculate AUC
def calculate_auc(y_true, y_pred_proba):
    # Check if both classes are present
    if len(np.unique(y_true)) == 1:
        return np.nan
    return roc_auc_score(y_true, y_pred_proba)

results = {}

# Demographic categories
categories = ['Sex', 'Income', 'Education', 'Age']

# Loop through each category and calculate metrics
for cat in categories:
    results[cat] = {}
    for subgroup in test_with_pred_df[cat].unique():
        subgroup_mask = test_with_pred_df[cat] == subgroup
        y_true_subgroup = test_with_pred_df[subgroup_mask]['True_Diabetes_binary']
        y_pred_subgroup = test_with_pred_df[subgroup_mask]['Predicted_Diabetes_binary']

        # Calculate FNR
        results[cat][subgroup] = {
            'FNR': calculate_fnr_fpr(y_true_subgroup, y_pred_subgroup)
        }

        results[cat][subgroup]['AUC'] = calculate_auc(y_true_subgroup, y_pred_subgroup)


# In[ ]:


results_df = pd.DataFrame(columns=['Demographic', 'Subgroup', 'FNR', 'FPR', 'AUC'])

for cat in results:
    for subgroup in results[cat]:
        fnr, fpr = calculate_fnr_fpr(test_with_pred_df[test_with_pred_df[cat] == subgroup]['True_Diabetes_binary'],
                                     test_with_pred_df[test_with_pred_df[cat] == subgroup]['Predicted_Diabetes_binary'])
        auc = results[cat][subgroup]['AUC']

        results_df = results_df.append({
            'Demographic': cat,
            'Subgroup': subgroup,
            'FNR': fnr,
            'FPR': fpr,
            'AUC': auc
        }, ignore_index=True)

# Adjusting data types for better formatting
results_df['Subgroup'] = results_df['Subgroup'].astype(int)
results_df['FNR'] = results_df['FNR'].astype(float)
results_df['FPR'] = results_df['FPR'].astype(float)
results_df['AUC'] = results_df['AUC'].astype(float)

results_df


# ### c. Informative Plots

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

conf_matrix = confusion_matrix(y_test, y_pred_subset)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.xticks(ticks=[0.5, 1.5], labels=['Negative (0)', 'Positive (1)'])
plt.yticks(ticks=[0.5, 1.5], labels=['Negative (0)', 'Positive (1)'], rotation=0)
plt.show()


# ## Model 4: LightGBM
# ### a. Model

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, f1_score, ConfusionMatrixDisplay


# In[ ]:


from ucimlrepo import fetch_ucirepo

# fetch dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

# data (as pandas dataframes)
feature_data = cdc_diabetes_health_indicators.data.features
target_data = cdc_diabetes_health_indicators.data.targets


# In[ ]:


X = feature_data
y = target_data.Diabetes_binary


# In[ ]:


X_train_test, X_test, y_train_test, y_test = train_test_split(X, y, test_size=0.2)

X_train, X_val, y_train, y_val = train_test_split(X_train_test, y_train_test, test_size=0.25)


# In[ ]:


from lightgbm import LGBMClassifier
import lightgbm as lgb


# In[ ]:


lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=False)


# In[ ]:


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',

    'learning_rate': 0.0001,
    'num_leaves': 15, # more leaves increases accuracy, but may lead to overfitting.

    'max_depth': 4, # the maximum tree depth. Shallower trees reduce overfitting.
    'min_split_gain': 0., # minimal loss gain to perform a split
    'min_child_samples': 20, # or min_data_in_leaf: specifies the minimum samples per leaf node.
    # 'min_child_weight': 5, # minimal sum hessian in one leaf. Controls overfitting.

    # 'lambda_l1': 0.1, # L1 regularization
    'lambda_l2': 0.9, # L2 regularization

    'feature_fraction': 0.7, # randomly select a fraction of the features before building each tree.
    # Speeds up training and controls overfitting.
    # 'bagging_fraction': 0.5, # allows for bagging or subsampling of data to speed up training.
    # 'bagging_freq': 0, # perform bagging on every Kth iteration, disabled if 0.

    'scale_pos_weight': 6, # add a weight to the positive class examples (compensates for imbalance).

    # 'subsample_for_bin': 2000, # amount of data to sample to determine histogram bins
    # 'max_bin': 10, # the maximum number of bins to bucket feature values in.
    # LightGBM autocompresses memory based on this value. Larger bins improves accuracy.
    # 'nthread': 4, # number of threads to use for LightGBM, best set to number of actual cores.
}


# In[ ]:


evals_result = {}
gbm = lgb.train(params, # parameter dict to use
                lgb_train,
                num_boost_round=10000, # the boosting rounds or number of iterations.
                early_stopping_rounds=50, # early stopping iterations.
                # stop training if *no* metric improves on *any* validation data.
                valid_sets=[lgb_train, lgb_val],
                evals_result=evals_result, # dict to store evaluation results in.
                verbose_eval=100) # print evaluations during training.


# ### b. Evaluation Metrics

# In[ ]:


y_val_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
threshold = sorted(y_val_pred)[43625] # Determine the threshold based on positive/negative ratio
y_val_pred = np.where(y_val_pred > threshold, 1, 0)
f1 = f1_score(y_val, y_val_pred, average='weighted')
print(f"F1-Score on Validation Set: {f1}")


# In[ ]:


# Predict on the test set
y_test_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_test_pred = np.where(y_test_pred > threshold, 1, 0)
# Evaluate the model
f1 = f1_score(y_test, y_test_pred, average='weighted')
print(f"F1-Score on Test Set: {f1}")


# In[ ]:


print(metrics.classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_test_pred))


# In[ ]:


female_idx = X_test['Sex']==0
male_idx = X_test['Sex']==1

female_test = y_test.loc[female_idx]
male_test = y_test.loc[male_idx]

female_pred  = y_test_pred[female_idx]
male_pred = y_test_pred[male_idx]

y_pred_prob = gbm.predict(X_test, num_iteration=gbm.best_iteration)
female_score = y_pred_prob[female_idx]
male_score = y_pred_prob[male_idx]

true_y = {'all': y_test, 'female':female_test, 'male':male_test}
pred_y = {'female':female_pred, 'male':male_pred}
prob_y = {'all':y_pred_prob, 'female':female_score, 'male':male_score}

for group in ['all', 'female', 'male']:
    AUC = metrics.roc_auc_score(true_y[group], prob_y[group])
    print(f'AUC ({group} participants): {AUC:.4f}')


tn, fp, fn, tp = {}, {}, {}, {}
for group in ['female', 'male']:
    tn[group], fp[group], fn[group], tp[group] = metrics.confusion_matrix(true_y[group], pred_y[group]).ravel()


# In[ ]:


for group in ['female', 'male']:
    FNR = fn[group]/(tp[group]+fn[group])
    print(f'FNR ({group} participants): {FNR:.4f}')


# In[ ]:


for group in ['female', 'male']:
    FPR = fp[group]/(tn[group]+fp[group])
    print(f'FPR ({group} participants): {FPR:.4f}')


# In[ ]:


for group in ['female', 'male']:
    PR = pred_y[group].mean()
    print(f'Fraction positive ({group} participants): {PR:.4f}')


# In[ ]:


one_idx = X_test['Income']==1
two_idx = X_test['Income']==2
three_idx = X_test['Income']==3
four_idx = X_test['Income']==4
five_idx = X_test['Income']==5
six_idx = X_test['Income']==6
seven_idx = X_test['Income']==7
eight_idx = X_test['Income']==8

one_test = y_test.loc[one_idx]
two_test = y_test.loc[two_idx]
three_test = y_test.loc[three_idx]
four_test = y_test.loc[four_idx]
five_test = y_test.loc[five_idx]
six_test = y_test.loc[six_idx]
seven_test = y_test.loc[seven_idx]
eight_test = y_test.loc[eight_idx]

one_pred  = y_test_pred[one_idx]
two_pred = y_test_pred[two_idx]
three_pred = y_test_pred[three_idx]
four_pred = y_test_pred[four_idx]
five_pred = y_test_pred[five_idx]
six_pred = y_test_pred[six_idx]
seven_pred = y_test_pred[seven_idx]
eight_pred = y_test_pred[eight_idx]

y_pred_prob = gbm.predict(X_test, num_iteration=gbm.best_iteration)
one_score = y_pred_prob[one_idx]
two_score = y_pred_prob[two_idx]
three_score = y_pred_prob[three_idx]
four_score = y_pred_prob[four_idx]
five_score = y_pred_prob[five_idx]
six_score = y_pred_prob[six_idx]
seven_score = y_pred_prob[seven_idx]
eight_score = y_pred_prob[eight_idx]


true_y = {'all': y_test, 'Income 1':one_test, 'Income 2':two_test, 'Income 3':three_test,
          'Income 4':four_test,'Income 5':five_test,'Income 6':six_test,'Income 7':seven_test,'Income 8':eight_test}
pred_y = {'Income 1':one_pred, 'Income 2':two_pred, 'Income 3':three_pred,
          'Income 4':four_pred,'Income 5':five_pred,'Income 6':six_pred,'Income 7':seven_pred,'Income 8':eight_pred}
prob_y = {'all': y_pred_prob, 'Income 1':one_score, 'Income 2':two_score, 'Income 3':three_score,
          'Income 4':four_score,'Income 5':five_score,'Income 6':six_score,'Income 7':seven_score,'Income 8':eight_score}


for group in ['all', 'Income 1', 'Income 2', 'Income 3', 'Income 4', 'Income 5', 'Income 6', 'Income 7', 'Income 8']:
    AUC = metrics.roc_auc_score(true_y[group], prob_y[group])
    print(f'AUC ({group} participants): {AUC:.4f}')


tn, fp, fn, tp = {}, {}, {}, {}
for group in ['Income 1', 'Income 2', 'Income 3', 'Income 4', 'Income 5', 'Income 6', 'Income 7', 'Income 8']:
    tn[group], fp[group], fn[group], tp[group] = metrics.confusion_matrix(true_y[group], pred_y[group]).ravel()


# In[ ]:


for group in ['Income 1', 'Income 2', 'Income 3', 'Income 4', 'Income 5', 'Income 6', 'Income 7', 'Income 8']:
    FNR = fn[group]/(tp[group]+fn[group])
    print(f'FNR ({group} participants): {FNR:.4f}')


# In[ ]:


for group in ['Income 1', 'Income 2', 'Income 3', 'Income 4', 'Income 5', 'Income 6', 'Income 7', 'Income 8']:
    FPR = fp[group]/(tn[group]+fp[group])
    print(f'FPR ({group} participants): {FPR:.4f}')


# In[ ]:


for group in 'Income 1', 'Income 2', 'Income 3', 'Income 4', 'Income 5', 'Income 6', 'Income 7', 'Income 8':
    PR = pred_y[group].mean()
    print(f'Fraction positive ({group} participants): {PR:.4f}')


# ### c. Informative Plots

# In[ ]:


cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'positive'])
disp = disp.plot(cmap=plt.cm.Blues,values_format='g')
# disp.plot()
plt.title(f'LightGBM (threshold={round(threshold, 4)})')
plt.show()
# calculate score
score = f1_score(y_test, y_test_pred, average='micro')
print('F1 Score: %.3f' % score)


# # 5. Results

# ### a. Interpret the results of each model

# ## Model 1: Naive Bayes
# In Model 1, the f1-score was 0.795, with an overall AUC of 0.7828, and FNR of 0.432. For most participants, the model predicted their diabetes outcome correctly. When we looked at these metrics for the different subgroups within sex and income, we saw that overall the metrics did not change for each subgroup. Looking at the sex subgroup, female particpants had a FNR of 0.3943 while male participants had a FNR of 0.4735. The slightly higher FNR for male participants could be attributed to the higher number of female participants and thus more data for this subgroup as we highlighted in the summary statistics above. As for different income groups, the AUC score and FNR had a positive relationship with the income group, with lower income groups having a lower AUC and FNR score while higher income groups having higher AUC and FNR scores. The highest FNR recorded was for Income group 8, which are participants who recorded having an income of \$ 75,000 or more. This result does not follow our initial hypothesis for lower income groups potentially having more unfair results from the model, as the higher income groups having higher FNR and FPR. However, this could also be attributed to a higher representation of high income groups in the dataset. Additionally, the fraction positive proportion had a negative relationship with income subgroups, with lower subgroups having the highest fraction positive such as 0.6299 and 0.6330 for Income 1 and Income 2 participants who have an income level around or less than 10,000. We can hypothesize that lower income groups may be less represented in the dataset, but the ones that did participate in the survey have had history of health issues related to diabetes or high cholestrol.
# 
# 
# ## Model 3: SVC
# After examining the diabetes prediction model based on Support Vector Classifier (SVC), it became clear that its performance varied across different groups of people. For men and women, the model had a slightly higher tendency to miss diabetes cases in men (with a False Negative Rate, or FNR, of 64.3%) compared to women (FNR of 62.3%). Despite this, the chance of incorrectly identifying someone as diabetic (False Positive Rate, or FPR) hovered around 35% for both genders, and the model's overall accuracy (measured by AUC) was just over 50%, barely better than a random guess.
# 
# This indicates that the model doesn't particularly favor one gender over the other; however, its effectiveness in accurately predicting diabetes is lacking. AUC values, indicating the model's ability to distinguish between those with and without diabetes, are disappointingly low across various groups, pointing to a struggle in making precise predictions.
# 
# There is many reasons why SVC model didn't perform well. First, the data used to train the model might lack diversity or fail to include crucial information that significantly impacts diabetes predictions, like dietary habits or physical activity levels. Second, the selected SVC model parameters might not be optimal, potentially making the model too simplistic or overly complex, thus hindering its learning capability.
# 
# 
# 
# 
# 
# 

# ### b. Compare the performance of your models from part 4 on the evaluation metrics you noted in section 3a

# # 6. Contribution Notes
# 

# [ here ]

# # 7. Sources Cited

# * Behavioral Risk Factor Surveillance System (BRFSS) 2015 Codebook Report: https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf
# * INFO 4390 Assignment 1 (answered)
# * [1] UC Irvine Machine Learning Repository. 2023. CDC Diabetes Health Indicators. https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators. Accessed February 19th, 2024
