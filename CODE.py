# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 20:18:05 2023

@author: kurva
"""
# importing the Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Defining The Function
def readFile_Pop(DataG):
    data = pd.read_csv(DataG)
    return data


# Loading the Dataset
# Read the CSV file into a DataFrame
path_to_csv = 'GlobalLandTemperaturesByCountry.csv'

# Pass the DataFrame into the function
df = readFile_Pop(path_to_csv)

# Removing Null values from DataTypes
print("Null values before removal:")
print(df.isnull().sum())

df_cleaned = df.dropna()
# Display information about null values after removal and general dataset information
print("\nNull values after removal:")
print(df_cleaned.isnull().sum())
print("\nInformation about the cleaned dataset:")
print(df_cleaned.info())

df.head()

df.tail()

df.describe

# Summary Statistics
summary_statistics = df_cleaned.describe()
print("Summary Statistics for Numerical Columns:")
print(summary_statistics)

# grouping the countries by mean
df = df.groupby('Country').agg('mean')
print(df)

# Printing the Structure of Data Type
data_types_structure = df_cleaned.dtypes
print("Data Types Structure:")
print(data_types_structure)

# Histogram Plot for AverageTemperature
def Histo_plot(df_cleaned):
    plt.figure(figsize = (12, 10))
    plt.hist(df_cleaned['AverageTemperature'], 
             bins = 10, color = 'skyblue', label = 'AverageTemperature')
    plt.hist(df_cleaned['AverageTemperatureUncertainty'],
             bins = 10, color = 'salmon', label = 'AverageTemperatureUncertainty')
    plt.title('Distribution of Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Frequency')
    plt.legend()
    # Adds Grid lines to graph
    plt.grid(True)

    #Display the plot
    plt.show()
Histo_plot(df_cleaned)



# Time Series Analysis
df_cleaned['dt'] = pd.to_datetime(
    df_cleaned['dt'], format = '%Y-%m-%d', errors = 'coerce')

# Drop rows with NaT values in the 'dt' column
df_cleaned = df_cleaned.dropna(subset = ['dt'])

# Set 'dt' column as the index
df_cleaned.set_index('dt', inplace = True)
plt.figure(figsize = (12, 8))

# Resample the data annually and compute mean temperature
plt.plot(df_cleaned['AverageTemperature'].resample('A').mean(), color = 'skyblue',
                                 label = 'AverageTemperature')
plt.plot(df_cleaned['AverageTemperatureUncertainty'].resample('A').mean(), color = 'salmon',
                                  label = 'AverageTemperatureUncertainty')

plt.title('Annual Mean Temperature Over Time')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.legend()

# Adds Grid lines to graph
plt.grid(True)

#Display the plot
plt.show()

# Shows the Pair plot for Two type of Temperature
numerical_cols = ['AverageTemperature', 'AverageTemperatureUncertainty']
sns.pairplot(df_cleaned[numerical_cols])
plt.show()

# Boxplot for Average Temperature by Country
plt.figure(figsize = (12, 8))
# Sampled data for better visualization
sns.boxplot(x = 'Country', y = 'AverageTemperature', 
            data = df_cleaned.sample(100))
plt.xticks(rotation = 90)
plt.title('Average Temperature by Country')
plt.xlabel('Country')
plt.ylabel('Average Temperature')

# Adds Grid lines to graph
plt.grid(True)

#Display the Plot
plt.show()

# Boxplot for Average Temperature Uncertainty by Country
plt.figure(figsize = (12, 8))
# Sampled data for better visualization
sns.boxplot(x = 'Country', y = 'AverageTemperatureUncertainty',
            data = df_cleaned.sample(100))
plt.xticks(rotation = 90)
plt.title('Average Temperature Uncertainty by Country')
plt.xlabel('Country')
plt.ylabel('Average Temperature Uncertainty')

# Adds Grid lines to graph
plt.grid(True)

#Display the Plot
plt.show()

# Correlation HeatMap
numerical_cols = df_cleaned.select_dtypes(
    include=['float64', 'int64']).columns.tolist()

correlation_matrix = df_cleaned[numerical_cols].corr()

plt.figure(figsize = (8, 6))
sns.heatmap(correlation_matrix, cmap = 'coolwarm', annot = True, fmt = '.2f',
            linewidths = .5)
plt.title('Correlation Heatmap')
plt.show()
