import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind

np.random.seed(42)

samples = 1000

age = np.random.normal(35, 10, samples)
height = np.random.normal(170, 50, samples)
weight = np.random.normal(70, 10, samples)
gender = np.random.choice(['Male', 'Female'], samples, p=[0.5, 0.5])
income = np.random.normal(50000, 15000, samples)


data = np.core.records.fromarrays([age, height, weight, gender, income], names = "Age, Height, Weight, Gender, Income")

print(data[:5])

mean_age = np.mean(data['Age'])
median_age = np.median(data['Age'])
std_age = np.std(data['Age'])
var_age = np.var(data['Age'])


mean_height = np.mean(data['Height'])
median_height = np.median(data['Height'])
std_height = np.std(data['Height'])
var_height = np.var(data['Height'])


mean_weight = np.mean(data['Weight'])
median_weight = np.median(data['Weight'])
std_weight = np.std(data['Weight'])
var_weight = np.var(data['Weight'])


mean_income = np.mean(data['Income'])
median_income = np.median(data['Income'])
std_income = np.std(data['Income'])
var_income = np.var(data['Income'])

print(f"Age: Mean = {mean_age}, Median = {median_age}, Std Dev = {std_age}, Variance = {var_age}")
print(f"Height: Mean = {mean_height}, Median = {median_height}, Std Dev = {std_height}, Variance = {var_height}")
print(f"Weight: Mean = {mean_weight}, Median = {median_weight}, Std Dev = {std_weight}, Variance = {var_weight}")
print(f"Income: Mean = {mean_income}, Median = {median_income}, Std Dev = {std_income}, Variance = {var_income}")

# Set up the figure for the histograms and KDE plots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Plot histograms
sns.histplot(data['Age'], bins=30, kde=False, ax=axes[0, 0])
axes[0, 0].set_title('Age Histogram')

sns.histplot(data['Height'], bins=30, kde=False, ax=axes[0, 1])
axes[0, 1].set_title('Height Histogram')

sns.histplot(data['Weight'], bins=30, kde=False, ax=axes[0, 2])
axes[0, 2].set_title('Weight Histogram')

sns.histplot(data['Income'], bins=30, kde=False, ax=axes[0, 3])
axes[0, 3].set_title('Income Histogram')

# Plot KDE plots
sns.kdeplot(data['Age'], ax=axes[1, 0], shade=True)
axes[1, 0].set_title('Age KDE')

sns.kdeplot(data['Height'], ax=axes[1, 1], shade=True)
axes[1, 1].set_title('Height KDE')

sns.kdeplot(data['Weight'], ax=axes[1, 2], shade=True)
axes[1, 2].set_title('Weight KDE')

sns.kdeplot(data['Income'], ax=axes[1, 3], shade=True)
axes[1, 3].set_title('Income KDE')

# Adjust layout
plt.tight_layout()
plt.show()

# Combine into a pandas DataFrame
data = pd.DataFrame({
    'Age': age,
    'Height': height,
    'Weight': weight,
    'Gender': gender,
    'Income': income
})

# Task 3: Boxplots to identify outliers
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

sns.boxplot(data=data, y='Age', ax=axes[0])
axes[0].set_title('Age Boxplot')

sns.boxplot(data=data, y='Height', ax=axes[1])
axes[1].set_title('Height Boxplot')

sns.boxplot(data=data, y='Weight', ax=axes[2])
axes[2].set_title('Weight Boxplot')

sns.boxplot(data=data, y='Income', ax=axes[3])
axes[3].set_title('Income Boxplot')

plt.tight_layout()
plt.show()

# Task 4: Correlation Analysis
# Calculate the Pearson correlation coefficient
correlation_matrix = data[['Age', 'Height', 'Weight', 'Income']].corr()

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Task 5: Inferential Statistics
# Perform a t-test to see if there is a significant difference in Income between Male and Female
male_income = data[data['Gender'] == 'Male']['Income']
female_income = data[data['Gender'] == 'Female']['Income']

t_stat, p_value = ttest_ind(male_income, female_income)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
