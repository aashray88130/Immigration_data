#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd

# Load the dataset
df1 = pd.read_csv('MIG_28112022024056641.csv')

# Check the first few rows to see the structure
print(df1.head())

# Print unique country names
print("Unique countries in the dataset:")
print(df1['Country of birth/nationality'].unique())  # Adjust this line based on your column name

# Set the index based on 'Country of birth/nationality' column
df1.set_index('Country of birth/nationality', inplace=True)

# Try to access the United Kingdom data safely
try:
    united_kingdom_data = df1.loc['Italy']
    print(united_kingdom_data)
except KeyError as e:
    print(f"KeyError: {e} - The country may not be in the dataset. Please check the country name.")


# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
df1 = pd.read_csv('MIG_28112022024056641.csv')

# Set the index based on 'Country of birth/nationality' column
df1.set_index('Country of birth/nationality', inplace=True)

# Access data for the United Kingdom
united_kingdom_data = df1.loc['United Kingdom']

# Reset index for easier manipulation
united_kingdom_data.reset_index(inplace=True)

# Filter to only include inflows
uk_inflows = united_kingdom_data[united_kingdom_data['Variable'].str.contains('Inflows', na=False)]

# Select relevant columns for prediction
uk_inflows = uk_inflows[['Year', 'Value']]

# Convert Year to numeric and reshape for the model
uk_inflows['Year'] = pd.to_numeric(uk_inflows['Year'])
X = uk_inflows['Year'].values.reshape(-1, 1)  # Features (years)
y = uk_inflows['Value'].values  # Target (inflows)

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions for future years
future_years = np.array(range(2021, 2026)).reshape(-1, 1)  # Predict for the next 5 years
predictions = model.predict(future_years)

# Create a DataFrame for the predictions
pred_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted Inflows': predictions})

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(uk_inflows['Year'], uk_inflows['Value'], label='Historical Inflows', marker='o')
plt.plot(pred_df['Year'], pred_df['Predicted Inflows'], label='Predicted Inflows', linestyle='--', marker='x')
plt.title('Immigrant Inflows to the United Kingdom')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.legend()
plt.grid()
plt.show()

# Print predicted inflows for the next 5 years
print("Predicted inflows for the next 5 years:")
print(pred_df)


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df1 = pd.read_csv('MIG_28112022024056641.csv')

# Check the first few rows to verify structure
print(df1.head())

# Filter data for Italy and Greece
countries_to_analyze = ['Italy', 'Greece']

# Loop through each country and visualize inflows and outflows
for country in countries_to_analyze:
    try:
        # Filter data for the country
        country_data = df1[df1['Country of birth/nationality'] == country]
        
        # Separate inflows and outflows data
        inflows = country_data[country_data['Variable'].str.contains('Inflows', case=False)]
        outflows = country_data[country_data['Variable'].str.contains('Outflows', case=False)]
        
        # Plot inflows over the years
        plt.figure(figsize=(10, 6))
        plt.plot(inflows['Year'], inflows['Value'], marker='o', label=f'{country} Inflows')
        plt.plot(outflows['Year'], outflows['Value'], marker='o', label=f'{country} Outflows')
        plt.title(f'Inflows and Outflows for {country} Over Time')
        plt.xlabel('Year')
        plt.ylabel('Population')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.show()

    except KeyError as e:
        print(f"KeyError: {e} - {country} may not be in the dataset. Please check the country name.")


# In[4]:


import pandas as pd

# Load the dataset
df1 = pd.read_csv('MIG_28112022024056641.csv')

# Check unique values in the 'Country' column to confirm how Italy is listed
print("Unique destination countries in the dataset:")
print(df1['Country'].unique())

# Filter the dataset for immigration to Italy (adjust based on how Italy is listed)
italy_immigration = df1[df1['Country'].str.contains('Italy', case=False, na=False)]

# Check if there's any data for Italy
if italy_immigration.empty:
    print("No data found for immigration to Italy.")
else:
    # Group by country of birth/nationality and sum the immigration values
    top_immigration_to_italy = italy_immigration.groupby('Country of birth/nationality')['Value'].sum().reset_index()

    # Sort the values in descending order to find the top 5 countries
    top_immigration_to_italy_sorted = top_immigration_to_italy.sort_values(by='Value', ascending=False)

    # Get the top 5 countries
    top_5_countries = top_immigration_to_italy_sorted.head(5)

    # Display the result
    print("Top 5 populations that immigrated to Italy:")
    print(top_5_countries)


# In[6]:


import seaborn as sns
italy_immigration = df1[df1['Country'].str.contains('Italy', case=False, na=False)]

# Check if there's any data for Italy
if italy_immigration.empty:
    print("No data found for immigration to Italy.")
else:
    # Group by country of birth/nationality and sum the immigration values
    top_immigration_to_italy = italy_immigration.groupby('Country of birth/nationality')['Value'].sum().reset_index()

    # Sort the values in descending order to find the top 5 countries
    top_immigration_to_italy_sorted = top_immigration_to_italy.sort_values(by='Value', ascending=False)

    # Get the top 5 countries
    top_5_countries = top_immigration_to_italy_sorted.head(5)

    # Plotting the graph
    plt.figure(figsize=(10,6))
    sns.barplot(x='Country of birth/nationality', y='Value', data=top_5_countries, palette='viridis')
    plt.title('Top 5 Populations that Immigrated to Italy', fontsize=16)
    plt.xlabel('Country of Origin', fontsize=12)
    plt.ylabel('Total Immigrants', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[9]:


us_immigration = df1[df1['Country'].str.contains('United States', case=False, na=False)]

# Check if there's any data for the United States
if us_immigration.empty:
    print("No data found for immigration to the United States.")
else:
    # Group by country of birth/nationality and sum the immigration values
    top_immigration_to_us = us_immigration.groupby('Country of birth/nationality')['Value'].sum().reset_index()

    # Sort the values in descending order to find the top 5 countries
    top_immigration_to_us_sorted = top_immigration_to_us.sort_values(by='Value', ascending=False)

    # Get the top 5 countries
    top_5_countries_us = top_immigration_to_us_sorted.head(5)

    # Plotting a pie chart
    plt.figure(figsize=(8,8))
    plt.pie(top_5_countries_us['Value'], labels=top_5_countries_us['Country of birth/nationality'], autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
    plt.title('Top 5 Populations that Immigrated to the United States', fontsize=16)
    plt.tight_layout()
    plt.show()


# In[31]:


df1 = pd.read_csv('MIG_28112022024056641.csv')

# Filter for immigration to the United States
us_immigration = df1[df1['Country'].str.contains('United States', case=False, na=False)]

# Check if there is data for the United States
if us_immigration.empty:
    print("No data found for immigration to the United States.")
else:
    # Group by gender and country of birth/nationality, summing the immigration values
    gender_distribution = us_immigration.groupby(['Gender', 'Country of birth/nationality'])['Value'].sum().unstack(fill_value=0)

    # Get the total immigration for each country and sort to find the top 5
    top_immigration_to_us_sorted = gender_distribution.sum(axis=1).sort_values(ascending=False)
    top_5_countries_us = top_immigration_to_us_sorted.head(5).index

    # Get the gender distribution for the top 5 countries
    top_gender_distribution = gender_distribution.loc[top_5_countries_us]

    # Reset the index for plotting
    top_gender_distribution = top_gender_distribution.reset_index()

    # Melt the DataFrame for Seaborn
    melted_data = top_gender_distribution.melt(id_vars='Gender', var_name='Country of Birth/Nationality', value_name='Number of Immigrants')

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Country of Birth/Nationality', y='Number of Immigrants', hue='Gender', data=melted_data, palette=['blue', 'pink'])
    
    plt.title('Top 5 Countries of Immigration to the United States by Gender', fontsize=16)
    plt.xlabel('Country of Birth/Nationality', fontsize=12)
    plt.ylabel('Number of Immigrants', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Gender')
    plt.tight_layout()
    plt.show()


# In[ ]:




