# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataset
data = pd.read_csv("Airbnb.csv")
print(data)


#Dataset view
data.head()

# Rows and Columns
print("Rows:",data.shape[0])
print("Columns:",data.shape[1])

# Info of dataset
data.info()

# Duplicat values
print("Duplicate values:",data.duplicated().sum())

# NUll/ Missing Values
print("Null Values:\n",data.isna().sum())

# Visualizing the missing values
plt.title("Airbnb dataset Null Values")
sns.heatmap(data.isna(),cbar=True)
plt.show()

# Dataset Columns
col = list(data.columns)
print(col)

# Dataset Describe
print("Statsic of Dataset:\n",data.describe())

# Check Unique Values for each variable.
for i in col:
  print(f"Unique values in {i} is {data[i].nunique()}")

#check for chatagorical columns
cat_col = data.select_dtypes(include="object").columns
cat_col

#check for numeric columns
num_col = data.select_dtypes(exclude='object').columns
num_col

# dropping unnecessary columns
data.drop(['latitude','longitude','last_review','reviews_per_month'],axis=1,inplace=True)

print("After droping columns \n",data.head(5))

print("Null Values:",data.isna().sum())

# Drop null values
data.dropna(inplace=True)

host_areas =data.groupby(['host_name','neighbourhood_group'])['calculated_host_listings_count'].max().reset_index()
host_areas.sort_values(by='calculated_host_listings_count',ascending=False)
print(host_areas)

#Distribution of the target variable (price)
plt.figure(figsize=(10, 6))
sns.histplot(data['price'], bins=50, kde=True)
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

#Countplot of room types
plt.figure(figsize=(10, 6))
sns.countplot(x='room_type', data=data)
plt.title('Count of Room Types')
plt.xlabel('Room Type')
plt.ylabel('Count')
plt.show()

#Average price by room type
plt.figure(figsize=(10, 6))
sns.barplot(x='room_type', y='price', data=data)
plt.title('Average Price by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Average Price')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['number_of_reviews'], bins=50, kde=True)
plt.title('Distribution of Number of Reviews')
plt.xlabel('Number of Reviews')
plt.ylabel('Frequency')
plt.show()

#Relationship between price and number of reviews
plt.figure(figsize=(10, 6))
sns.scatterplot(x='number_of_reviews', y='price', data=data)
plt.title('Price vs. Number of Reviews')
plt.xlabel('Number of Reviews')
plt.ylabel('Price')
plt.show()

#Relationship between price and availability
plt.figure(figsize=(10, 6))
sns.scatterplot(x='availability_365', y='price', data=data)
plt.title('Price vs. Availability')
plt.xlabel('Availability (365 days)')
plt.ylabel('Price')
plt.show()

#Boxplot of prices by room type
plt.figure(figsize=(10, 6))
sns.boxplot(x='room_type', y='price', data=data)
plt.title('Boxplot of Prices by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Price')
plt.show()