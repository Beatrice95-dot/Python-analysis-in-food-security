#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 18:34:16 2025

@author: beatricenyarko
"""




import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
    

#load the dataset to read with pandas

df = pd.read_csv('dec22pub.csv')

print(df)

print(df.head())

print(df.isnull().sum())

#I use this code to select the variables that are revelant for my analysis

cols = ["HRFS12M1", "HEFAMINC", "PEMLR", "GESTFIPS"]

#I save the selected columns into a new dataset
data = df[cols].copy()

#Renaming the selected columns
data = data.rename(columns= {"HRFS12M1":"food_security",
                             "HEFAMINC": "household_income",
                             "PEMLR": "employment_status",
                             "GESTFIPS": "location"
                             })

print(data.columns)

#Find missing values in the selected columns
print(data.isna())
#I checked the sum of na just to confirm there are no missing values
print(data.isna().sum())

#From the dataset, -9 and -1 showed no responses. I will treat these as missing values and will remove them#
#This code identify  the no reponse code and treats them as missing
data["food_security"].replace([-9,-1], np.nan, inplace = True )
data["household_income"].replace([-9,-1], np.nan, inplace = True )
data["employment_status"].replace([-9,-1], np.nan, inplace = True )


#Remove missing values 
data.dropna(subset = ["food_security", "household_income", "employment_status"], inplace = True)

data.describe()

# Replacing employment status codes with real names
data["household_income"] = data["household_income"].replace({
    1: "LESS THAN $5,000",
    2: "5,000 TO 7,499",
    3: "7,500 TO 9,999",
    4: "10,000 TO 12,499",
    5: "12,500 TO 14,999",
    6: "15,000 TO 19,999",
    7: "20,000 TO 24,999",
    8: "25,000 TO 29,999",
    9: "30,000 TO 34,999",
    10: "35,000 TO 39,999",
    11: "40,000 TO 49,999",
    12: "50,000 TO 59,999",
    13: "60,000 TO 74,999",
    14: "75,000 TO 99,999",
    15: "100,000 TO 149,999",
    16: "150,000 OR MORE"   
})


# Replacing employment status codes with real names
data["employment_status"] = data["employment_status"].replace({
    1: "Employed",
    2: "Employed",       
    3: "Unemployed",     
    4: "Unemployed",     
    5: "Retired",        
    6: "Disabled",       
    7: "Not in labour force"           
})

# Replacing employment status codes with real names
data["location"] = data["location"].replace({
    1:  "Alabama",
    2:  "Alaska",
    4:  "Arizona",
    5:  "Arkansas",
    6:  "California",
    8:  "Colorado",
    9:  "Connecticut",
    10: "Delaware",
    11: "District of Columbia",
    12: "Florida",
    13: "Georgia",
    15: "Hawaii",
    16: "Idaho",
    17: "Illinois",
    18: "Indiana",
    19: "Iowa",
    20: "Kansas",
    21: "Kentucky",
    22: "Louisiana",
    23: "Maine",
    24: "Maryland",
    25: "Massachusetts",
    26: "Michigan",
    27: "Minnesota",
    28: "Mississippi",
    29: "Missouri",
    30: "Montana",
    31: "Nebraska",
    32: "Nevada",
    33: "New Hampshire",
    34: "New Jersey",
    35: "New Mexico",
    36: "New York",
    37: "North Carolina",
    38: "North Dakota",
    39: "Ohio",
    40: "Oklahoma",
    41: "Oregon",
    42: "Pennsylvania",
    44: "Rhode Island",
    45: "South Carolina",
    46: "South Dakota",
    47: "Tennessee",
    48: "Texas",
    49: "Utah",
    50: "Vermont",
    51: "Virginia",
    53: "Washington",
    54: "West Virginia",
    55: "Wisconsin",
    56: "Wyoming"
        
})



print(data["household_income"])
# Checking the measure of central tendancy
#print('Mean income:')
#print(statistics.mean(data["household_income"]))
#print('Median Income:')
#print(statistics.median(data['household_income']))
#mode_income = data["household_income"].mode()[0]
#print('Mode for Income:')
#print(mode_income)

#Checking the mode for employment status and location

mode_employment = data["employment_status"].mode()[0]
print('Mode for employment status:')
print(mode_employment)

mode_location = data["location"].mode()[0]
print('Mode for location:')
print(mode_location)

print(data['food_security'].value_counts())
#Because I am doing a binary logistics regression analysis,
#I will collapse the three responses to two, to have a situation 
#where households are food secure and where households are not secure#
#I will recode the responses to have 1: food secure, 0: food insecure

data["food_security"] = np.where(data["food_security"] == 1,1,0)
print(data["food_security"].value_counts())

#This code helps me find the percentage of food security responses
percentage_secure = (data["food_security"].mean()*100)
print('percentage of food security:')
print(percentage_secure)


#Starting with the regression analysis#

#Defining my dependent and independent variables

X = data[['household_income', 'employment_status', 'location']]
Y = data['food_security']

#I create dummy variales for the categorical predictors
X = pd.get_dummies(X, columns=["household_income", "employment_status", "location"], drop_first=True)
X = X.astype(float)
#Add a constant to the model
X_const = sm.add_constant(X)

#Fit the logistic regression model
logit_model = sm.Logit(Y,X_const)
result = logit_model.fit(disp = False)
print(result.summary())
print("Food security",data['food_security'])
print("household income",data['household_income'])
print("Employment Status",data['employment_status'])
print("Location",data['location'])



# This code helps me to find the Odds ratios with p-values
or_table = pd.DataFrame({
    "Variable": result.params.index,
    "Coef": result.params.values,
    "Odds Ratio": np.exp(result.params.values),
    "P-value": result.pvalues.values
})
print(or_table)

# Predict, classify, and evaluate
probs = result.predict(X_const)
pred = (probs >= 0.5).astype(int)

#This code  helps me to find the confusion matrix and classification report
print("Confusion Matrix",confusion_matrix(Y, pred))
print("Classification report",classification_report(Y, pred))



##VISUALIZATION##

#Distribution plots of food security
#This code sets the size of the plot, with the width set to 8 inches and the height set to 6 inches.
plt.figure(figsize=(8,6))
ax = sns.countplot(x=data["food_security"], palette="Blues")

#Giving s title to the plot.
plt.title("Distribution of Food Security", fontsize=14)
#setting the x axis label
plt.xlabel("Food Security (1 = Secure, 0 = Insecure)", fontsize=12)
#Setting the y axis label
plt.ylabel("Count", fontsize=12)

# Add percentage labels
#Counting the total number of observations in the dataset
total = len(data)
#Checking the count in each category by calculating the count in each height
for p in ax.patches:
    height = p.get_height()
    #Finding the percentage of each category
    percentage = f'{(height/total)*100:.1f}%'
    ax.annotate(percentage,
                (p.get_x() + p.get_width() / 2, height),
                ha='center', va='bottom', fontsize=12)

# This code creates legend in our graph
legend_labels = {0: "Food Insecure", 1: "Food Secure"}
handles = [plt.Rectangle((0,0),1,1, color=ax.patches[i].get_facecolor()) for i in range(2)]
labels = [legend_labels[i] for i in sorted(legend_labels)]
plt.legend(handles, labels, title="Category")

# Tick labels
plt.xticks([0, 1], ["Insecure (0)", "Secure (1)"], fontsize=12)

plt.show()



### Distribution of people whoo are employed.
plt.figure(figsize=(10,6))
ax = sns.countplot(x=data["employment_status"], palette="Set2")

plt.title("Distribution of Employment Status", fontsize=14)
plt.xlabel("Employment Status", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)

#Add peercentage labels  on each bar
total = len(data)

for p in ax.patches:
    count = p.get_height()
    # This code format as % with 1 decimal
    percentage = f'{(count/total)*100:.1f}%'  
    x = p.get_x() + p.get_width() / 2        
    y = p.get_height()                       
    ax.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=11)

plt.show()



#Plot for Food Security by Employment Status

plt.figure(figsize=(10,6))
ax = sns.barplot(
    x="employment_status",
    y="food_security",
    data=data,
    palette="viridis"
)

plt.title("Food Security Rate by Employment Status", fontsize=14)
plt.ylabel("Food Secure (%)", fontsize=12)
plt.xlabel("Employment Status", fontsize=12)

# Add percentage labels on bars
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height*100:.1f}%',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=12)

# Improve tick labels
plt.xticks(rotation=45, fontsize=12)

# Add legend explaining colors
# (one color per bar, so just label them generically)
#handles = [plt.Rectangle((0,0),1,1,color=p.get_facecolor()) for p in ax.patches]
#labels  = [f"Status {i}" for i in range(len(ax.patches))]
#plt.legend(handles, labels, title="Color Guide")

#plt.show()
employment_categories = ["Retired", "Employed", "Other", "Disabled", "Unemployed"]
handles = [plt.Rectangle((0,0),1,1, color=p.get_facecolor()) for p in ax.patches]

plt.legend(handles, employment_categories, title="Employment Status", loc="lower left")

plt.show()

#Distribution for household income

income_order = [
    "LESS THAN $5,000",
    "5,000 TO 7,499",
    "7,500 TO 9,999",
    "10,000 TO 12,499",
    "12,500 TO 14,999",
    "15,000 TO 19,999",
    "20,000 TO 24,999",
    "25,000 TO 29,999",
    "30,000 TO 34,999",
    "35,000 TO 39,999",
    "40,000 TO 49,999",
    "50,000 TO 59,999",
    "60,000 TO 74,999",
    "75,000 TO 99,999",
    "100,000 TO 149,999",
    "150,000 OR MORE"
]


plt.figure(figsize=(14,6))
ax = sns.countplot(y=data["household_income"], order=income_order, palette="tab20")

total = len(data)

for p in ax.patches:
    count = p.get_width()
    percentage = f'{(count/total)*100:.1f}%'
    ax.annotate(percentage,
                (count + 100, p.get_y() + p.get_height()/2),
                ha='left', va='center')

plt.title("Distribution of Household Income Categories")
plt.xlabel("Count")
plt.ylabel("Income Bracket")
plt.show()


#Plot for distribution by state

plt.figure(figsize=(16,8))
ax = sns.countplot(
    y=data["location"], 
    order=data["location"].value_counts().index,
    palette="tab20"
)

plt.title("Distribution of Respondents by State", fontsize=16)
plt.xlabel("Count", fontsize=14)
plt.ylabel("State", fontsize=14)

# Add percentages
total = len(data)
for p in ax.patches:
    count = p.get_width()
    percentage = f'{(count/total)*100:.1f}%'
    ax.annotate(percentage,
                (count + 100, p.get_y() + p.get_height()/2),
                ha='left', va='center', fontsize=10)

plt.show()

#Finding food security rate by state

# Compute food security rate by state
state_fs = data.groupby("location")["food_security"].mean().sort_values()

plt.figure(figsize=(16,8))
ax = sns.barplot(
    y=state_fs.index,
    x=state_fs.values,
    palette="viridis"
)

plt.title("Food Security Rate by State", fontsize=16)
plt.xlabel("Food Security (%)", fontsize=14)
plt.ylabel("State", fontsize=14)

# Add percentage labels
for i, v in enumerate(state_fs.values):
    ax.text(v + 0.005, i, f"{v*100:.1f}%", va='center', fontsize=10)

plt.show()



#Finding the top 10 most secure state

# Calculate food security rate by state
state_fs = data.groupby("location")["food_security"].mean() * 100  # convert to %
state_fs = state_fs.sort_values()


plt.figure(figsize=(12,6))
top10 = state_fs.tail(10)

ax = sns.barplot(
    x=top10.values,
    y=top10.index,
    palette="Greens"  # green = high security
)

plt.title("Top 10 Food Secure States", fontsize=16)
plt.xlabel("Food Security (%)", fontsize=14)
plt.ylabel("State", fontsize=14)

# Add percentage labels
for i, v in enumerate(top10.values):
    ax.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=10)

plt.show()


# Finding the bottom 10 Food Secure States

plt.figure(figsize=(12,6))
bottom10 = state_fs.head(10)

ax = sns.barplot(
    x=bottom10.values,
    y=bottom10.index,
    palette="Reds_r"   # red = low security
)

plt.title("Bottom 10 Food Secure States", fontsize=16)
plt.xlabel("Food Security (%)", fontsize=14)
plt.ylabel("State", fontsize=14)

# Add percentage labels
for i, v in enumerate(bottom10.values):
    ax.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=10)

plt.show()
