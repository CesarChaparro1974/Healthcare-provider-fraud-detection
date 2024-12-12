#!/usr/bin/env python
# coding: utf-8

# ### Project Objectives

# Provider Fraud is one of the biggest problems facing Medicare. According to the government, the total Medicare spending increased exponentially due to frauds in Medicare claims. Healthcare fraud is an organized crime which involves peers of providers, physicians, beneficiaries acting together to make fraud claims.

# Rigorous analysis of Medicare data has yielded many physicians who indulge in fraud. They adopt ways in which an ambiguous diagnosis code is used to adopt costliest procedures and drugs. Insurance companies are the most vulnerable institutions impacted due to these bad practices. Due to this reason, insurance companies increased their insurance premiums and as result healthcare is becoming costly matter day by day.

# Healthcare fraud and abuse take many forms. Some of the most common types of frauds by providers are:<br>
# * Billing for services that were not provided.<br>
# * Duplicate submission of a claim for the same service.<br>
# * Misrepresenting the service provided.<br>
# *  Charging for a more complex or expensive service than was actually provided.<br>
# * Billing for a covered service when the service actually provided was not covered.

# ### Problem Statement

# The goal of this project is to " predict the potentially fraudulent providers " based on the claims filed by them. Along with this, we will also discover important variables helpful in detecting the behaviour of potentially fraud providers. further, we will study fraudulent patterns in the provider's claims to understand the future behaviour of providers.

# ### Introduction to the Dataset

# For the purpose of this project, we are considering Inpatient claims, Outpatient claims and Beneficiary details of each provider.<br>
# Lets see their details :<br>
# * Inpatient Data:<br>
# This data provides insights about the claims filed for those patients who are admitted in the hospitals. It also provides additional details like their admission and discharge dates and admit d diagnosis code.<br>
# * Outpatient Data:<br>
# This data provides details about the claims filed for those patients who visit hospitals and not admitted in it.<br>
# * Beneficiary Details Data:<br>
# This data contains beneficiary KYC details like health conditions,regioregion they belong to etc.

# ### Defintions:<br>
# * DOB: Date of birth
# * DOD: Date of death
# * Part A coverage: Medicare Part A, often referred to as hospital insurance, primarily covers inpatient care in hospitals, skilled nursing facilities, hospice care, and some home health services.<br>
# * Part B coverage: Medicare Part B is a component of Original Medicare that covers medically necessary services and preventive care, such as doctor visits, outpatient care, and certain medical supplies.<br>
# * IPAnnualReimbursementAmt: This refers to the total amount reimbursed by the insurance provider for inpatient (IP) services over a year. It represents the financial compensation received for covered inpatient medical expenses.<br>
# * OPAnnualReimbursementAmt: Similar to the inpatient counterpart, this term denotes the total amount reimbursed by the insurance provider for outpatient (OP) services over a year. It reflects the financial compensation for covered outpatient medical expenses.<br>
# * IPAnnualDeductibleAmt: This is the total amount that an individual must pay out-of-pocket for inpatient services before the insurance coverage begins to reimburse any costs. It is a yearly threshold that must be met.<br>
# * OPAnnualDeductibleAmt: This indicates the total out-of-pocket expense that an individual must cover for outpatient services before their insurance starts to pay for those costs. Like the inpatient deductible, it is calculated annually. 

# In[457]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 


# In[365]:


# open datasets
test_beneficiary = pd.read_csv("data_base/Test_Beneficiarydata-1542969243754.csv")
test_inpatient = pd.read_csv("data_base/Test_Inpatientdata-1542969243754.csv")
test_outpatient = pd.read_csv("data_base/Test_Outpatientdata-1542969243754.csv")
test = pd.read_csv("data_base/Test-1542969243754.csv")
train_beneficiary = pd.read_csv("data_base/Train_Beneficiarydata-1542865627584.csv")
train_inpatient = pd.read_csv("data_base/Train_Inpatientdata-1542865627584.csv")
train_outpatient = pd.read_csv("data_base/Train_Outpatientdata-1542865627584.csv")
train = pd.read_csv("data_base/Train-1542865627584.csv")


# #### Beneficiaries Data Set.

# In[366]:


test_beneficiary.head()


# In[367]:


test_beneficiary.info()


# In[368]:


test_beneficiary.shape


# In[369]:


# checking for duplicates on beneficiaries column.
# display duplicated beneficiaries in the column.
print(f"There are {test_beneficiary['BeneID'].duplicated().sum()} duplicate beneficiaries.")


# In[370]:


# display unique number of beneficiaries.
print(f"There are {len(test_beneficiary.BeneID.unique())} unique beneficiaries.")


# #### Beneficiaries by Gender.

# In[371]:


# distribution of beneficiaries by gender
gender_distribution = test_beneficiary['Gender'].value_counts()
print("Distribution of Beneficiaries by Gender:")
print(gender_distribution)


# In[372]:


# calculate percentage distribution
percentage_distribution = (test_beneficiary['Gender'].value_counts(normalize=True) * 100).round(2)
print("\nPercentage Distribution:")
print(percentage_distribution)


# In[373]:


# create the plot
plt.figure(figsize=(8, 6))

ax = sns.countplot(data=test_beneficiary, x='Gender', 
                  palette=['#2E86C1', '#F39C12'])

# Calculate percentages
total = len(test_beneficiary)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()/2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')

plt.title('Distribution of Beneficiaries by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Beneficiaries')
plt.show()


# #### Beneficiaries by Year of Birth.

# ##### DOB column is object, we have to convert it into datetime type in order to manipulate it.

# In[374]:


# Simple conversion (since it's in standard ISO format)
test_beneficiary['DOB'] = pd.to_datetime(test_beneficiary['DOB'])

# Verify the conversion
print("New data type:", test_beneficiary['DOB'].dtype)

# Display a few examples to confirm the conversion
print("\nFirst few dates after conversion:")
print(test_beneficiary['DOB'].head())


# In[375]:


test_beneficiary['Year'] = test_beneficiary['DOB'].dt.year
# group by year and count the number of beneficiaries
beneficiaries_by_year = test_beneficiary.groupby('Year').size().reset_index(name='Count')
beneficiaries_by_year.head()


# In[376]:


# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Year', y='Count', data=beneficiaries_by_year, color='skyblue')

# Adjust x-axis to show every 5th year
xticks = beneficiaries_by_year['Year'][::5]
plt.xticks(ticks=range(0, len(beneficiaries_by_year), 5), labels=xticks, rotation=45, ha='right')

plt.xlabel('Year')
plt.ylabel('Number of Beneficiaries')
plt.title('Beneficiaries by Year')
plt.tight_layout()  # Adjust layout to fit labels
plt.show()


# ##### Most of beneficiaries are born between 1919-1943.<br>
# ##### We can categorize beneficiaries by year in 6 groups:<br>
# * 1. 1909-1918.<br>
# * 2. 1919-1943.<br>
# * 3. 1944-1953.<br>
# * 4. 1954-1963.<br>
# * 5. 1964-1973.<br>
# * 6. 1974-1983.

# In[377]:


# Define the year ranges and corresponding group labels
bins = [1908, 1918, 1943, 1953, 1963, 1973, 1983]  # Bin edges
labels = ['1909-1918', '1919-1943', '1944-1953', '1954-1963', '1964-1973', '1974-1983']  # Bin labels

# Create a new column for the year groups
beneficiaries_by_year['Year_Category'] = pd.cut(beneficiaries_by_year['Year'], bins=bins, labels=labels, right=True)

# Group by the year groups and count the number of beneficiaries
beneficiaries_by_category = beneficiaries_by_year.groupby('Year_Category')['Count'].sum().reset_index()
# Calculate the total number of beneficiaries
total_beneficiaries = beneficiaries_by_category['Count'].sum()
# Display the result
beneficiaries_by_category


# In[378]:


# calculate the percentage for each category
beneficiaries_by_category['Percentage'] = (beneficiaries_by_category['Count'] / total_beneficiaries) * 100

# plotting the distribution by categories
plt.figure(figsize=(10, 6))
sns.barplot(x='Year_Category', y='Count', data=beneficiaries_by_category, palette='Blues_d')

# adding percentages on top of each bar
for index, row in beneficiaries_by_category.iterrows():
    plt.text(index, row['Count'] + 50, f'{row["Percentage"]:.1f}%', ha='center', va='bottom')

# adding labels and title
plt.xlabel('Year Category')
plt.ylabel('Number of Beneficiaries')
plt.title('Distribution of Beneficiaries by Year Category with Percentages')

# display the plot
plt.tight_layout()
plt.show()


# #### Deceased beneficiaries.

# ##### Let's check the column of DOD.

# In[379]:


# Simple conversion (since it's in standard ISO format)
test_beneficiary['DOD'] = pd.to_datetime(test_beneficiary['DOD'])

# Verify the conversion
print("New data type:", test_beneficiary['DOD'].dtype)

# Display a few examples to confirm the conversion
print("\nFirst few dates after conversion:")
print(test_beneficiary['DOD'].head())


# In[380]:


test_beneficiary['DOD'].unique()


# ##### There are only dead dates for year 2009.

# ##### Let's investigate the rows of patients who are dead.

# In[381]:


# Calculate the age by subtracting DOB from DOD and getting the difference in years
test_beneficiary['Age_at_death'] = (test_beneficiary['DOD'] - test_beneficiary['DOB']).dt.days / 365.25

# Round the age to the nearest integer
test_beneficiary['Age_at_death'] = test_beneficiary['Age_at_death'].round()

# Check the distribution of ages, you can use:
print(test_beneficiary['Age_at_death'].describe())


# In[382]:


# Create the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=test_beneficiary.index, y=test_beneficiary['Age_at_death'], color='blue')

# Highlight the patient who died at the age of 37
patient_37 = test_beneficiary[test_beneficiary['Age_at_death'] == 37]
plt.scatter(patient_37.index, patient_37['Age_at_death'], color='red', label='Patient aged 37')

# Title and labels
plt.title('Scatter Plot of Patient Ages at Death', fontsize=14)
plt.xlabel('Patient Index', fontsize=12)
plt.ylabel('Age at Death', fontsize=12)

# Show the legend
plt.legend()

# Display the plot
plt.show()


# ##### We can conclude that the minimum age of death is not an outlier, also there is no specific pattern of ages through the patients, the ages are distributed uniformly from the minimum to the maximum.

# ##### Average death age is 74 years old amongst 574 patients. Date of death only for 2009. Missing data for any other years.

# #### Beneficiaries by Race.

# In[383]:


# group by race and count the number of beneficiaries
beneficiaries_by_race = test_beneficiary.groupby('Race').size().reset_index(name='Count')
beneficiaries_by_race.head()


# In[384]:


# create the bar plot
plt.figure(figsize=(12, 8))
ax = sns.barplot(data=beneficiaries_by_race, x='Race', y='Count', 
                 palette=['#2E86C1', '#5DADE2', '#85C1E9', '#AED6F1'])

# calculate the percentage for each category
beneficiaries_by_race['Percentage'] = (beneficiaries_by_race['Count'] / total_beneficiaries) * 100

# adding percentages on top of each bar
for index, row in beneficiaries_by_race.iterrows():
    plt.text(index, row['Count'] + 50, f'{row["Percentage"]:.1f}%', ha='center', va='bottom')

# add title and labels
plt.title('Distribution of Beneficiaries by Race', fontsize=14)
plt.xlabel('Race', fontsize=12)
plt.ylabel('Number of Beneficiaries', fontsize=12)

# show the plot
plt.show()



# #### Beneficiearies by State.

# In[385]:


# group by state and count the number of beneficiaries
beneficiaries_by_state = test_beneficiary.groupby('State').size().reset_index(name='Count')
beneficiaries_by_state.head()


# ##### There are 54 States in this Dataset.

# In[386]:


plt.figure(figsize=(12, 8))

# sort the data in descending order by 'Count'
beneficiaries_by_state_sorted = beneficiaries_by_state.sort_values(by='Count', ascending=False)

# normalize the counts to create a gradient effect
normalized_counts = beneficiaries_by_state_sorted['Count'] / beneficiaries_by_state_sorted['Count'].max()

# use a colormap to generate colors based on normalized counts
colormap = cm.get_cmap('Blues')  # Choose a gradient colormap
colors = colormap(normalized_counts)

# create the bar plot with explicit order for the x-axis
ax = sns.barplot(
    data=beneficiaries_by_state_sorted,
    x='State',
    y='Count',
    palette=colors,
    order=beneficiaries_by_state_sorted['State']
)

# calculate total number of beneficiaries and mean
total_beneficiaries = beneficiaries_by_state_sorted['Count'].sum()
mean_beneficiaries = beneficiaries_by_state_sorted['Count'].mean()

# add an average line
plt.axhline(mean_beneficiaries, color='grey', linestyle='--', linewidth=2, label=f'Mean: {mean_beneficiaries:.0f}')

# annotate the value of the mean on the line
plt.text(
    x=len(beneficiaries_by_state_sorted) - 1,  # Position near the end of the x-axis
    y=mean_beneficiaries + 50,  # Slightly above the line for visibility
    s=f'Mean: {mean_beneficiaries:.0f}', 
    color='grey',
    fontsize=12,
    ha='right'
)

# set plot title and labels
plt.title('Number of Beneficiaries by State', fontsize=16)
plt.xlabel('State', fontsize=14)
plt.ylabel('Number of Beneficiaries', fontsize=14)

# rotate x-axis labels for better readability
plt.xticks(rotation=45)

# add legend for the average line
plt.legend(loc='upper right', fontsize=12)

# show the plot
plt.tight_layout()
plt.show()


# In[387]:


beneficiaries_by_state.describe()


# #### Beneficiaries by County.

# In[388]:


# group by county and count the number of beneficiaries
beneficiaries_by_county = test_beneficiary.groupby('County').size().reset_index(name='Count')
beneficiaries_by_county.head()


# In[389]:


beneficiaries_by_county.describe()


# ##### There are 999 Counties in this Dataset.

# ##### I'am interested in analyzing more populous counties, to focus on larger data that can provide more meaningful trends.
# ##### I am going to filter counties with ≥323 beneficiaries, as they are part of the top 25% of counties by population of beneficiaries.

# In[390]:


# filter counties where the count of beneficiaries is >= 323
filtered_beneficiaries_by_county = beneficiaries_by_county[beneficiaries_by_county['Count'] >= 323]

filtered_beneficiaries_by_county.head()


# In[391]:


plt.figure(figsize=(12, 8))

# sort the data in descending order by 'Count'
beneficiaries_by_county_sorted = filtered_beneficiaries_by_county.sort_values(by='Count', ascending=False)

# normalize the counts to create a gradient effect
normalized_counts = beneficiaries_by_county_sorted['Count'] / beneficiaries_by_county_sorted['Count'].max()

# use a colormap to generate colors based on normalized counts
colormap = cm.get_cmap('Blues')  # Choose a gradient colormap
colors = colormap(normalized_counts)

# create the bar plot with explicit order for the x-axis
ax = sns.barplot(
    data=beneficiaries_by_county_sorted,
    x='County',
    y='Count',
    palette=colors,
    order=beneficiaries_by_county_sorted['County']
)

# calculate total number of beneficiaries and mean
total_beneficiaries = beneficiaries_by_county_sorted['Count'].sum()
mean_beneficiaries = beneficiaries_by_county_sorted['Count'].mean()

# add an average line
plt.axhline(mean_beneficiaries, color='grey', linestyle='--', linewidth=2, label=f'Mean: {mean_beneficiaries:.0f}')

# annotate the value of the mean on the line
plt.text(
    x=len(beneficiaries_by_county_sorted) - 1,  # Position near the end of the x-axis
    y=mean_beneficiaries + 50,  # Slightly above the line for visibility
    s=f'Mean: {mean_beneficiaries:.0f}', 
    color='grey',
    fontsize=12,
    ha='right'
)

# set plot title and labels
plt.title('Number of Beneficiaries by County', fontsize=16)
plt.xlabel('County', fontsize=14)
plt.ylabel('Number of Beneficiaries', fontsize=14)

# rotate x-axis labels for better readability
plt.xticks(rotation=45)

# add legend for the average line
plt.legend(loc='upper right', fontsize=12)

# show the plot
plt.tight_layout()
plt.show()


# ##### Beneficiaries by Chronic Condition.

# In[392]:


# Aggregate by summing each chronic condition
aggregated_data = test_beneficiary[['ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
                                    'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
                                    'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
                                    'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke']].sum()

# Sort the aggregated data in descending order
aggregated_data_sorted = aggregated_data.sort_values(ascending=False)

# Create a bar plot with a gradient of blues (from dark to light)
plt.figure(figsize=(10, 6))

# Generate a colormap of blues with a gradient from dark to light
cmap = plt.cm.Blues
norm = plt.Normalize(vmin=min(aggregated_data_sorted.values), vmax=max(aggregated_data_sorted.values))
colors = [cmap(norm(value)) for value in aggregated_data_sorted.values]

bars = plt.bar(aggregated_data_sorted.index, aggregated_data_sorted.values, color=colors)

# Renaming x-axis labels by removing 'ChronicCond_' prefix
new_labels = [label.replace('ChronicCond_', '') for label in aggregated_data_sorted.index]
plt.xticks(ticks=range(len(new_labels)), labels=new_labels, rotation=45)

# Adding labels and title
plt.xlabel('Chronic Conditions')
plt.ylabel('Number of Beneficiaries')
plt.title('Chronic Conditions Aggregated by Beneficiaries')

plt.tight_layout()

# Show the plot
plt.show()


# ***

# ##### Annual reinbursement for IP and OP beneficiaries analysis.

# In[393]:


# check statistics to investigate the existence of outliers for IP Patients.
test_beneficiary['IPAnnualReimbursementAmt'].describe()


# In[394]:


# check statistics to investigate the existence of outliers for OP Patients.
test_beneficiary['OPAnnualReimbursementAmt'].describe()


# In[395]:


# create the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=test_beneficiary.index, y=test_beneficiary['IPAnnualReimbursementAmt'], color='blue')

# highlight the patient who died at the age of 37
patient_155600 = test_beneficiary[test_beneficiary['IPAnnualReimbursementAmt'] == 155600]
plt.scatter(patient_155600.index, patient_155600['IPAnnualReimbursementAmt'], color='red', label='Patient max. Reimbursement')

# title and labels
plt.title('Scatter Plot of IP Patient Reimbursement', fontsize=14)
plt.xlabel('Patient Index', fontsize=12)
plt.ylabel('Annual Reimbursement', fontsize=12)

# show the legend
plt.legend()

# display the plot
plt.show()


# In[396]:


# create the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=test_beneficiary.index, y=test_beneficiary['OPAnnualReimbursementAmt'], color='blue')

# highlight the patient who died at the age of 37
patient_155600 = test_beneficiary[test_beneficiary['OPAnnualReimbursementAmt'] == 155600]
plt.scatter(patient_155600.index, patient_155600['OPAnnualReimbursementAmt'], color='red', label='Patient max. Reimbursement')

# title and labels
plt.title('Scatter Plot of OP Patient Reimbursement', fontsize=14)
plt.xlabel('Patient Index', fontsize=12)
plt.ylabel('Annual Reimbursement', fontsize=12)

# show the legend
plt.legend()

# display the plot
plt.show()


# ##### 🚩 Conclusion: We can see some pattern outside the average behavior of patients on reimbursements received which may indicate fraud movements.

# ##### Both OP and IP Patients have negative amounts. For the purpose of the analysis we will clean the dataset not taking into consideration the negative values with a clean dataset.

# In[397]:


# filter IP beneficiaries with negative reimbursement amounts
IP_negative_reimbursements = test_beneficiary[test_beneficiary['IPAnnualReimbursementAmt'] < 0]

# count the number of IP beneficiaries with negative reimbursements
IP_count_negative_reimbursements = IP_negative_reimbursements['BeneID'].nunique()

print(f"Number of IP beneficiaries with negative reimbursements: {IP_count_negative_reimbursements}")


# In[398]:


IP_negative_reimbursements


# In[399]:


# filter OP beneficiaries with negative reimbursement amounts
OP_negative_reimbursements = test_beneficiary[test_beneficiary['OPAnnualReimbursementAmt'] < 0]

# count the number of OP beneficiaries with negative reimbursements
OP_count_negative_reimbursements = OP_negative_reimbursements['BeneID'].nunique()

print(f"Number of OP beneficiaries with negative reimbursements: {OP_count_negative_reimbursements}")


# In[400]:


OP_negative_reimbursements


# In[401]:


# check if there are beneficiaries on both datasets.
common_beneficiaries = set(OP_negative_reimbursements['BeneID']) & set(IP_negative_reimbursements['BeneID'])
print(f"Therer are {len(common_beneficiaries)} beneficiaries in both datasets{', '.join(map(str, common_beneficiaries))}.")


# ##### Clean the test_beneficiary dataset removing the negative amounts from the reimbursements and saving the clean dataset.

# In[402]:


# filter out rows with negative values in the specified columns
cleaned_test_beneficiary = test_beneficiary[
    (test_beneficiary['IPAnnualReimbursementAmt'] >= 0) & 
    (test_beneficiary['OPAnnualReimbursementAmt'] >= 0)
]

# save the cleaned DataFrame to a new CSV file
cleaned_test_beneficiary.to_csv("cleaned_test_beneficiary.csv", index=False)


# In[403]:


# check to see if it is clean of the negative values.
cleaned_test_beneficiary['IPAnnualReimbursementAmt'].describe()


# ##### Next are same steps but for the Out Patient dataset.

# In[404]:


# check if the data set is clean of negative values.
cleaned_test_beneficiary['OPAnnualReimbursementAmt'].describe()


# ##### 🚩 Conclusion: The beneficiaries who may seem outliers are actually not outliers as they may identify fraudulent movement.

# ***

# #### Patients who were admitted in the hospital.

# In[405]:


test_inpatient.head()


# In[406]:


test_inpatient.info()


# In[407]:


test_inpatient['InscClaimAmtReimbursed'].describe()


# In[408]:


# checking for duplicates on claims column.
# display duplicated claims in the column.
print(f"There are {test_inpatient['ClaimID'].duplicated().sum()} duplicate Claims.")


# In[409]:


# check unique Providers.
print(test_inpatient['Provider'].nunique())


# In[410]:


# join both tables on BeneID
merged_df = pd.merge(test_beneficiary, test_inpatient, on='BeneID', how='inner')

# ensure the DOD and ClaimsStartDt columns are in datetime format
merged_df['DOD'] = pd.to_datetime(merged_df['DOD'], errors='coerce')
merged_df['ClaimStartDt'] = pd.to_datetime(merged_df['ClaimStartDt'], errors='coerce')
merged_df['ClaimEndDt'] = pd.to_datetime(merged_df['ClaimEndDt'], errors='coerce')
merged_df['AdmissionDt'] = pd.to_datetime(merged_df['AdmissionDt'], errors='coerce')
merged_df['DischargeDt'] = pd.to_datetime(merged_df['DischargeDt'], errors='coerce')

# filter beneficiaries where ClaimsStartDt is later than DOD
claims_after_dod = merged_df[merged_df['ClaimStartDt'] > merged_df['DOD']]

# count the number of unique beneficiaries and total claims
num_beneficiaries = claims_after_dod['BeneID'].nunique()  # Count unique beneficiaries
num_claims = claims_after_dod.shape[0]  # Total number of claims

# print the result
print(f"There are {num_beneficiaries} beneficiaries who filed {num_claims} claims after they were deceased.")


# #### There are no Beneficiaries who filed claims after they died.

# In[411]:


ended_claims_after_dod = merged_df[merged_df['ClaimEndDt'] > merged_df['DOD']]

num_beneficiaries = ended_claims_after_dod['BeneID'].nunique()

print(f"There are {num_beneficiaries} beneficiaries who ended their claims after they were deceased.")


# #### There are no Beneficiaries who ended their claims after they died.

# In[412]:


admissions_after_dod = merged_df[merged_df['AdmissionDt'] > merged_df['DOD']]

num_beneficiaries = admissions_after_dod['BeneID'].nunique()

print(f"There are {num_beneficiaries} beneficiaries who were admitted after they were deceased.")


# #### There are no Beneficiaries who were admitted after they died.

# In[413]:


discharged_after_dod = merged_df[merged_df['DischargeDt'] > merged_df['DOD']]

num_beneficiaries = discharged_after_dod['BeneID'].nunique()

print(f"There are {num_beneficiaries} beneficiaries who were discharged after they were deceased.")


# #### There are no Beneficiaries who were discharged after they died.

# In[414]:


# Filter beneficiaries who have a non-null DischargeDt and InsClaimAmtReimbursed equal to 0
filtered_beneficiaries = test_inpatient[(test_inpatient['DischargeDt'].notnull()) & (test_inpatient['InscClaimAmtReimbursed'] == 0)]

# Count the number of unique BeneID values that meet the condition
filtered_beneficiaries_count = filtered_beneficiaries['BeneID'].nunique()

# Print the result
print(f"Number of beneficiaries who received a service but had 0 reimbursement: {filtered_beneficiaries_count}")


# In[415]:


# Filter beneficiaries who have a non-null DischargeDt and InsClaimAmtReimbursed equal to 0
filtered_beneficiaries = test_inpatient[(test_inpatient['DischargeDt'].notnull()) & (test_inpatient['DeductibleAmtPaid'] == 0)]

# Count the number of unique BeneID values that meet the condition
filtered_beneficiaries_count = filtered_beneficiaries['BeneID'].nunique()

# Print the result
print(f"Number of beneficiaries who received a service but paid 0 deductible amount for the service: {filtered_beneficiaries_count}")


# In[416]:


# Filter beneficiaries who have a non-null DischargeDt and InsClaimAmtReimbursed equal to 0
filtered_beneficiaries = test_inpatient[(test_inpatient['DischargeDt'].notnull()) & (test_inpatient['InscClaimAmtReimbursed'] == 0) & (test_inpatient['DeductibleAmtPaid'] == 0)]

# Count the number of unique BeneID values that meet the condition
filtered_beneficiaries_count = filtered_beneficiaries['BeneID'].nunique()

# Print the result
print(f"Number of beneficiaries who received a service but had 0 reimbursement and paid the deductable amount for the service: {filtered_beneficiaries_count}")


# In[417]:


# Filter beneficiaries who have a non-null DischargeDt and InsClaimAmtReimbursed equal to 0
filtered_beneficiaries = test_inpatient[(test_inpatient['DischargeDt'].notnull()) & (test_inpatient['InscClaimAmtReimbursed'] == 0) & (test_inpatient['DeductibleAmtPaid'] > 0)]

# Count the number of unique BeneID values that meet the condition
filtered_beneficiaries_count = filtered_beneficiaries['BeneID'].nunique()

# Print the result
print(f"Number of beneficiaries who received a service but had 0 reimbursement and paid the deductable amount for the service: {filtered_beneficiaries_count}")


# In[418]:


# Group by 'ClmAdmitDiagnosisCode' and count the number of unique 'BeneID' for each diagnosis code
in_grouped_by_diagnosis = test_inpatient.groupby('DiagnosisGroupCode')['BeneID'].nunique().reset_index(name='BeneficiaryCount')

# Print the result
in_grouped_by_diagnosis.describe()


# In[419]:


# Plot a histogram of the BeneficiaryCount
plt.figure(figsize=(10, 6))
plt.hist(in_grouped_by_diagnosis['BeneficiaryCount'], bins=30, edgecolor='black', color='skyblue')
plt.title('Distribution of In Patients Count by Diagnosis Code')
plt.xlabel('Number of Beneficiaries')
plt.ylabel('Frequency')
plt.show()


# In[420]:


# Extract the numeric columns from the dataset
numeric_columns = test_inpatient.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
correlation_matrix = numeric_columns.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, 
            linewidths=0.5, annot_kws={"fontsize": 10})

# Add labels and title
plt.title("Correlation Matrix Heatmap for IP", fontsize=14)
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)
plt.tight_layout()

# Display the heatmap
plt.show()


# In[421]:


clean_test_inpatient = test_inpatient.drop(['ClmProcedureCode_5'], axis=1)
clean_test_inpatient.head()


# ***

# #### Patients who visited the hospital but were not admited in the hospital.

# In[422]:


test_outpatient.head()


# In[423]:


test_outpatient.info()


# In[424]:


# ensure the Claims, Admission and Discharge columns are in datetime format
merged_df['ClaimStartDt'] = pd.to_datetime(merged_df['ClaimStartDt'], errors='coerce')
merged_df['ClaimEndDt'] = pd.to_datetime(merged_df['ClaimEndDt'], errors='coerce')
merged_df['AdmissionDt'] = pd.to_datetime(merged_df['AdmissionDt'], errors='coerce')
merged_df['DischargeDt'] = pd.to_datetime(merged_df['DischargeDt'], errors='coerce')


# In[425]:


# checking for duplicates on claims column.
# display duplicated claims in the column.
print(f"There are {test_outpatient['ClaimID'].duplicated().sum()} duplicate Claims.")


# ##### I will check if there are Beneficiaries who were deceased and reported to visit hospital after DOD.

# In[426]:


# filter beneficiaries where ClaimsStartDt is later than DOD
claims_after_dod = merged_df[merged_df['ClaimStartDt'] > merged_df['DOD']]

# count the number of unique beneficiaries and total claims
num_beneficiaries = claims_after_dod['BeneID'].nunique()  # Count unique beneficiaries
num_claims = claims_after_dod.shape[0]  # Total number of claims

# print the result
print(f"There are {num_beneficiaries} beneficiaries who filed {num_claims} claims after they were deceased.")


# In[427]:


# filter beneficiaries where ClaimEndDt is later than DOD
ended_claims_after_dod = merged_df[merged_df['ClaimEndDt'] > merged_df['DOD']]

# count the number of unique beneficiaries and total claims
num_beneficiaries = ended_claims_after_dod['BeneID'].nunique()  # Count unique beneficiaries
num_claims = ended_claims_after_dod.shape[0]  # Total number of claims

# print the result
print(f"There are {num_beneficiaries} beneficiaries who ended {num_claims} claims after they were deceased.")


# ##### There are no beneficiaries who filed and ended a claim after they were deceased.

# In[428]:


# filter for claims with dates later than the date of death
claims_after_death = merged_df[
    (merged_df['ClaimStartDt'] > merged_df['DOD']) | 
    (merged_df['ClaimEndDt'] > merged_df['DOD'])
]

# count the unique providers
providers_with_late_claims = claims_after_death['Provider'].nunique()

# print the result
print(f"There are {providers_with_late_claims} providers who had Beneficiaries who sent a claim after they were deceased.")


# In[429]:


# count the unique attending physicians
attending_physicians_with_late_claims = claims_after_death['AttendingPhysician'].nunique()

# print the result
print(f"There are {attending_physicians_with_late_claims} attending physicians who had Beneficiaries who sent a claim after they were deceased.")


# In[430]:


# count the unique providers
operating_physicians_with_late_claims = claims_after_death['OperatingPhysician'].nunique()

# print the result
print(f"There are {operating_physicians_with_late_claims} operating physicians who had Beneficiaries who sent a claim after they were deceased.")


# In[431]:


# count the unique providers
other_physicians_with_late_claims = claims_after_death['OtherPhysician'].nunique()

# print the result
print(f"There are {other_physicians_with_late_claims} other physicians who had Beneficiaries who sent a claim after they were deceased.")


# In[432]:


# Extract the numeric columns from the dataset
numeric_columns = test_outpatient.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
correlation_matrix = numeric_columns.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, 
            linewidths=0.5, annot_kws={"fontsize": 10})

# Add labels and title
plt.title("Correlation Matrix Heatmap for OP", fontsize=14)
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)
plt.tight_layout()

# Display the heatmap
plt.show()


# In[433]:


clean_test_outpatient = test_outpatient.drop(['ClmProcedureCode_3', 'ClmProcedureCode_4'], axis=1)
clean_test_outpatient.head()


# ##### IP Deductible amounts vs Reimbursed amounts for IP & OP.

# In[434]:


# Function to format the y-axis labels
def format_y_axis(value, _):
    if value >= 1e6:
        return f'{value / 1e6:.0f}M'
    elif value >= 1e3:
        return f'{value / 1e3:.0f}K'
    else:
        return f'{value:.0f}'

# Aggregate the data for IP
ip_total_deductible = test_beneficiary['IPAnnualDeductibleAmt'].sum()
ip_total_reimbursement = test_beneficiary['IPAnnualReimbursementAmt'].sum()

# Aggregate the data for OP
op_total_deductible = test_beneficiary['OPAnnualDeductibleAmt'].sum()
op_total_reimbursement = test_beneficiary['OPAnnualReimbursementAmt'].sum()

# Create a DataFrame for plotting
plot_data = pd.DataFrame({
    'Category': ['Inpatient (IP)', 'Outpatient (OP)'],
    'Deductible Amount': [ip_total_deductible, op_total_deductible],
    'Reimbursement Amount': [ip_total_reimbursement, op_total_reimbursement]
})

# Plotting
plt.figure(figsize=(10, 6))

# Stacked bar chart
plt.bar(plot_data['Category'], plot_data['Deductible Amount'], label='Deductible Amount', color='#2E86C1')
plt.bar(plot_data['Category'], plot_data['Reimbursement Amount'], 
        bottom=plot_data['Deductible Amount'], label='Reimbursement Amount', color='#F39C12')

# Add labels and title
plt.xlabel('Type of Patient', fontsize=12)
plt.ylabel('Total Amount ($)', fontsize=12)
plt.title('Deductible and Reimbursement Amounts for IP & OP', fontsize=14)

# Format y-axis labels
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_axis))
plt.ylabel('Total Amount ($)', fontsize=12)

# Add a legend
plt.legend()

# Display values on the bars
for idx, row in plot_data.iterrows():
    total = row['Deductible Amount'] + row['Reimbursement Amount']
    plt.text(idx, row['Deductible Amount'] / 2, f'{row["Deductible Amount"] / 1e6:.1f}M', ha='center', color='white', fontsize=10)
    plt.text(idx, row['Deductible Amount'] + (row['Reimbursement Amount'] / 2), f'{row["Reimbursement Amount"] / 1e6:.1f}M', ha='center', color='black', fontsize=10)

# Improve layout
plt.tight_layout()

# Show plot
plt.show()


# ***

# ##### Split the source data into training set, validations set and test set.<br>
# ##### 60% training dataset.<br>
# ##### 20% validating dataset.<br>
# ##### 20% test dataset.

# In[443]:


# First, split the dataset into 80% (training + validation) and 20% (testing)
train_val_beneficiary, cleaned_test_beneficiary = train_test_split(train_beneficiary, test_size=0.2, random_state=42)
train_val_inpatient, clean_test_inpatient = train_test_split(train_inpatient, test_size=0.2, random_state=42)
train_val_outpatient, clean_test_outpatient = train_test_split(train_outpatient, test_size=0.2, random_state=42)

# Second, split the 80% (train + validation) into 60% (training) and 20% (validation)
train_beneficiary, val_beneficiary = train_test_split(train_val_beneficiary, test_size=0.25, random_state=42)
train_inpatient, val_inpatient = train_test_split(train_val_inpatient, test_size=0.25, random_state=42)
train_outpatient, val_outpatient = train_test_split(train_val_outpatient, test_size=0.25, random_state=42)

# - 60% training (train_beneficiary, train_inpatient, train_outpatient)
# - 20% validation (val_beneficiary, val_inpatient, val_outpatient)
# - 20% testing (test_beneficiary, test_inpatient, test_outpatient)


# In[452]:


clean_test_inpatient.info()


# In[ ]:


# Set a threshold for the 'IPAnnualReimbursementAmt' column
high_reimbursement_threshold = 10000

# Create a FraudulentClaim column based on the conditions
def create_fraudulent_flag(row):
    # Check for high reimbursement amount
    if row['InscClaimAmtReimbursed'] > high_reimbursement_threshold:
        return 1  # Fraudulent
    # You can add more conditions here, e.g., suspicious procedure codes or missing data
    # If procedure codes are missing, it might indicate fraud
    if any(pd.isnull(row[f'ClmProcedureCode_{i}']) for i in range(1, 7)):
        return 1  # Fraudulent due to missing procedure codes
    return 0  # Not fraudulent

# Apply the function to create the 'FraudulentClaim' column
clean_test_inpatient['FraudulentClaim'] = clean_test_inpatient.apply(create_fraudulent_flag, axis=1)


# In[458]:


# Here, for example, we can assume claims with very high reimbursement amounts are potentially fraudulent.
clean_test_inpatient['FraudulentClaim'] = (clean_test_inpatient['InscClaimAmtReimbursed'] > 10000).astype(int)
clean_test_outpatient['FraudulentClaim'] = (clean_test_outpatient['InscClaimAmtReimbursed'] > 10000).astype(int)

# 2. Feature selection for inpatient and outpatient claims
inpatient_features = clean_test_inpatient[['Provider', 'InscClaimAmtReimbursed', 'ClmAdmitDiagnosisCode', 
                                     'DeductibleAmtPaid', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2']]

outpatient_features = clean_test_outpatient[['Provider', 'InscClaimAmtReimbursed', 'ClmAdmitDiagnosisCode', 
                                      'DeductibleAmtPaid', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2']]

# 3. Merging beneficiary data with the claims data for additional features
# Merge beneficiary data on BeneID for both inpatient and outpatient claims
inpatient_data = pd.merge(inpatient_features, cleaned_test_beneficiary[['BeneID', 'Gender', 'Race']], 
                          left_on='BeneID', right_on='BeneID', how='left')

outpatient_data = pd.merge(outpatient_features, cleaned_test_beneficiary[['BeneID', 'Gender', 'Race']], 
                           left_on='BeneID', right_on='BeneID', how='left')

# 4. Separate target variable (FraudulentClaim) from features for both datasets
X_inpatient = inpatient_data.drop(columns=['FraudulentClaim', 'BeneID'])
y_inpatient = inpatient_data['FraudulentClaim']

X_outpatient = outpatient_data.drop(columns=['FraudulentClaim', 'BeneID'])
y_outpatient = outpatient_data['FraudulentClaim']

# 5. Splitting data into training, validation, and test sets
X_inpatient_train, X_inpatient_test, y_inpatient_train, y_inpatient_test = train_test_split(X_inpatient, y_inpatient, test_size=0.2, random_state=42)
X_outpatient_train, X_outpatient_test, y_outpatient_train, y_outpatient_test = train_test_split(X_outpatient, y_outpatient, test_size=0.2, random_state=42)

# 6. Example of training a RandomForestClassifier on inpatient data
rf_inpatient = RandomForestClassifier(random_state=42)
rf_inpatient.fit(X_inpatient_train, y_inpatient_train)

# Predicting on test data
inpatient_predictions = rf_inpatient.predict(X_inpatient_test)

# Example of training a RandomForestClassifier on outpatient data
rf_outpatient = RandomForestClassifier(random_state=42)
rf_outpatient.fit(X_outpatient_train, y_outpatient_train)

# Predicting on test data
outpatient_predictions = rf_outpatient.predict(X_outpatient_test)


# ##### Creating the Model.

# In[444]:


# creating the model.
dtc_model = DecisionTreeClassifier(random_state = 42)


# ##### Training the Model.

# In[ ]:


# training on the training model.
dtc_model = dtc_model.fit(features_train,target_train)


# ***

# In[436]:


test.head()


# In[437]:


# check unique Providers.
print(test['Provider'].nunique())


# In[438]:


test.info()


# In[439]:


train_beneficiary.head()


# In[440]:


train_inpatient.head()


# In[441]:


train_outpatient.head()


# In[442]:


train.head()

