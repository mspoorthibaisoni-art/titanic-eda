import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset
df = pd.read_csv("titanic.csv")

# 1. Basic Information
print("===== BASIC INFORMATION =====")
print(df.head())
print("\nShape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nStatistics:")
print(df.describe())

# 2. Survival Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Survived")
plt.title("Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

# 3. Survival by Gender
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.show()

# 4. Survival by Passenger Class
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()

# 5. Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.show()

# 6. Fare Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Fare", bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.show()

# 7. Correlation Heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 8. Survival by Gender and Class
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex")
plt.title("Survival Rate by Gender and Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()

# 9. Survival Rates
print("\n===== SURVIVAL RATES =====")
print("\nOverall:")
print(df["Survived"].mean())

print("\nBy Gender:")
print(df.groupby("Sex")["Survived"].mean())

print("\nBy Passenger Class:")
print(df.groupby("Pclass")["Survived"].mean())

print("\nBy Gender and Class:")
print(df.groupby(["Sex", "Pclass"])["Survived"].mean())

# 10. Key Insights
print("""
===== KEY INSIGHTS =====

1. Survival was not evenly distributed among passengers.

2. Gender is associated with survival and can be an important
   feature for predictive modeling.

3. Passenger class is associated with survival rate.

4. Passenger ages vary considerably, making Age a useful variable
   for analysis and modeling.

5. Fare varies among passengers and provides additional information
   about passenger characteristics.

6. Gender and passenger class together provide more detailed
   information about survival.

7. Correlation analysis helps identify relationships between
   numerical variables for future modeling.
""")

print("EDA completed successfully!")
