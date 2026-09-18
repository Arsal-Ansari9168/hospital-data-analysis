import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


df = pd.read_csv("hospital_data.csv")
df.head()

df.shape

df.head(10)

df.tail(10)

df.columns

df.dtypes

df.info()

df.describe()

df.isnull().sum()

df.duplicated().sum()

df = df.drop_duplicates()

df["Department"].unique()

df["Disease"].unique()

df["Admission_Date"] = pd.to_datetime(df["Admission_Date"])
df["Discharge_Date"] = pd.to_datetime(df["Discharge_Date"])

df["Year"] = df["Admission_Date"].dt.year
df["Month"] = df["Admission_Date"].dt.month
df["Month_Name"] = df["Admission_Date"].dt.month_name()

df["Day_Name"] = df["Admission_Date"].dt.day_name()

bins = [0, 18, 35, 50, 65, 100]
labels = ["Child", "Young Adult", "Adult", "Senior", "Elderly"]
df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels)

average_bill = df["Bill_Amount"].mean()
df["High_Bill"] = np.where(df["Bill_Amount"] > average_bill, "Yes", "No")

total_patients = df["Patient_ID"].nunique()
print(total_patients)

total_revenue = df["Bill_Amount"].sum()
print(round(total_revenue, 2))

average_bill = df["Bill_Amount"].mean()
print(round(average_bill, 2))

df["Bill_Amount"].min()

df["Bill_Amount"].max()

df["Length_of_Stay"].mean()

df["Satisfaction"].mean()

df["Gender"].value_counts()

df["City"].value_counts()

df["Age_Group"].value_counts()

df["Age_Group"].value_counts().idxmax()

department_patients = df["Department"].value_counts()
department_patients

department_patients.idxmax()

department_revenue = (
    df.groupby("Department")["Bill_Amount"].sum().sort_values(ascending=False)
)
department_revenue

department_revenue.idxmax()

df.groupby("Department")["Bill_Amount"].mean().sort_values(ascending=False)

df.groupby("Department")["Length_of_Stay"].mean().sort_values(ascending=False)

df.groupby("Department")["Satisfaction"].mean().sort_values(ascending=False)

disease_count = df["Disease"].value_counts()
disease_count

disease_count.head(5)

# Q38. Revenue by disease.
disease_revenue = (
    df.groupby("Disease")["Bill_Amount"].sum().sort_values(ascending=False)
)
disease_revenue

df.groupby("Disease")["Bill_Amount"].mean().sort_values(ascending=False)

disease_revenue.idxmax()

doctor_count = df["Doctor"].value_counts()
doctor_count

doctor_count.head(5)

df.groupby("Doctor")["Satisfaction"].mean().sort_values(ascending=False)

doctor_revenue = (
    df.groupby("Doctor")["Bill_Amount"].sum().sort_values(ascending=False)
)
doctor_revenue

monthly_admissions = df.groupby("Month")["Patient_ID"].count()
monthly_admissions

monthly_revenue = df.groupby("Month")["Bill_Amount"].sum()
monthly_revenue

monthly_admissions.idxmax()

monthly_revenue.idxmax()

df["Length_of_Stay"].max()

long_stay = df[df["Length_of_Stay"] > 10]
long_stay

df[["Length_of_Stay", "Bill_Amount"]].corr()

df["Insurance_Type"].value_counts()

df.groupby("Insurance_Type")["Bill_Amount"].mean()

insurance_revenue = (
    df.groupby("Insurance_Type")["Bill_Amount"]
    .sum()
    .sort_values(ascending=False)
)
insurance_revenue

insurance_revenue.idxmax()

df["Payment_Mode"].value_counts()

df.groupby("Payment_Mode")["Bill_Amount"].sum().sort_values(ascending=False)

df["Readmitted"].value_counts()

readmission_rate = df["Readmitted"].eq("Yes").mean() * 100
print(round(readmission_rate, 2), "%")

pd.crosstab(df["Department"], df["Readmitted"], normalize="index") * 100

readmission_table = (
    pd.crosstab(df["Department"], df["Readmitted"], normalize="index") * 100
)
readmission_table["Yes"].idxmax()

satisfaction_department = (
    df.groupby("Department")["Satisfaction"]
    .mean()
    .sort_values(ascending=False)
)
satisfaction_department

satisfaction_department[satisfaction_department < 4]

df.groupby("Gender")["Satisfaction"].mean()

department_kpi = (
    df.groupby("Department")
    .agg(
        Patients=("Patient_ID", "count"),
        Revenue=("Bill_Amount", "sum"),
        Average_Bill=("Bill_Amount", "mean"),
        Average_Stay=("Length_of_Stay", "mean"),
        Satisfaction=("Satisfaction", "mean"),
    )
    .sort_values("Revenue", ascending=False)
)
department_kpi

pd.crosstab(df["Department"], df["Gender"])

pd.crosstab(df["Disease"], df["Gender"])

pd.crosstab(df["Department"], df["Insurance_Type"])

# 25. Pivot Table
# Q69. Department vs insurance revenue.
pivot = pd.pivot_table(
    df,
    values="Bill_Amount",
    index="Department",
    columns="Insurance_Type",
    aggfunc="sum",
)
pivot

top10 = df.nlargest(10, "Bill_Amount")
top10[
    [
        "Patient_Code",
        "Department",
        "Disease",
        "Length_of_Stay",
        "Bill_Amount",
    ]
]

df[df["Bill_Amount"] > df["Bill_Amount"].mean()]

df[df["Satisfaction"] < 3]

df[(df["Length_of_Stay"] > 7) & (df["Bill_Amount"] > 75000)]

df[(df["Age"] >= 65) & (df["Readmitted"] == "Yes")]


department_count = df["Department"].value_counts().sort_values(ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(department_count.index, department_count.values) #barh=ka matlab hai Horizontal Bar Chart.
plt.title("Number of Patients by Department")
plt.xlabel("Number of Patients")
plt.ylabel("Department")
plt.tight_layout() 
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(
    monthly_admissions.index, monthly_admissions.values, marker="o"
)
plt.title("Monthly Patient Admissions")
plt.xlabel("Month")
plt.ylabel("Number of Patients")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(monthly_revenue.index, monthly_revenue.values, marker="o")
plt.title("Monthly Hospital Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()

gender_count = df["Gender"].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(
    gender_count.values,
    labels=gender_count.index,
    autopct="%1.1f%%",
    startangle=90,
)
plt.title("Patient Gender Distribution")
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(df["Age"], bins=15, edgecolor="black")
plt.title("Patient Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.show()

department_revenue = (
    df.groupby("Department")["Bill_Amount"].sum().sort_values(ascending=True)
)
plt.figure(figsize=(10, 6))
plt.barh(department_revenue.index, department_revenue.values)
plt.title("Revenue by Department")
plt.xlabel("Revenue")
plt.ylabel("Department")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
order = df["Disease"].value_counts().index
sns.countplot(data=df, y="Disease", order=order)
plt.title("Disease Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="Bill_Amount", bins=30, kde=True)
plt.title("Hospital Bill Distribution")
plt.xlabel("Bill Amount")
plt.ylabel("Number of Patients")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="Department", y="Bill_Amount")
plt.title("Hospital Bill Distribution by Department")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, x="Length_of_Stay", y="Bill_Amount", hue="Department"
)
plt.title("Length of Stay vs Hospital Bill")
plt.xlabel("Length of Stay")
plt.ylabel("Bill Amount")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="Department", y="Satisfaction")
plt.title("Patient Satisfaction by Department")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

numeric_columns = ["Age", "Length_of_Stay", "Bill_Amount", "Satisfaction"]
corr = df[numeric_columns].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Hospital Data Correlation Heatmap")
plt.tight_layout()
plt.show()


df.groupby("Department").size().idxmax()
df.groupby("Department")["Bill_Amount"].sum().idxmax()
df.groupby("Department")["Bill_Amount"].mean().idxmax()
df.groupby("Department")["Length_of_Stay"].mean().idxmax()
df.groupby("Department")["Satisfaction"].mean().idxmin()

df["Disease"].value_counts().head(5)
df.groupby("Disease")["Bill_Amount"].sum().sort_values(
    ascending=False
).head(5)
df.groupby("Disease")["Bill_Amount"].mean().sort_values(
    ascending=False
).head(5)

df["Doctor"].value_counts().head(5)
df.groupby("Doctor")["Bill_Amount"].sum().sort_values(
    ascending=False
).head(5)
df.groupby("Doctor")["Satisfaction"].mean().sort_values(
    ascending=False
).head(5)

readmission_rate = pd.crosstab(
    df["Department"],
    df["Readmitted"],
    normalize="index"
) * 100

print(readmission_rate)
readmission_rate["Yes"].sort_values(ascending=False)

df.nlargest(10, "Bill_Amount")

correlation = df["Length_of_Stay"].corr(df["Bill_Amount"])
print(correlation)

plt.scatter(df["Length_of_Stay"], df["Bill_Amount"])
plt.xlabel("Length of Stay")
plt.ylabel("Bill Amount")
plt.title("Length of Stay vs Bill Amount")
plt.show()

df["City"].value_counts()

pd.crosstab(df["Disease"], df["Gender"])

df.groupby("Department")["Satisfaction"].mean().idxmax()

df.groupby("Month")["Bill_Amount"].sum().idxmax()


total_patients = df["Patient_ID"].nunique()
total_revenue = df["Bill_Amount"].sum()
average_bill = df["Bill_Amount"].mean()
average_stay = df["Length_of_Stay"].mean()
average_satisfaction = df["Satisfaction"].mean()
readmission_rate = df["Readmitted"].eq("Yes").mean() * 100
top_department = df["Department"].value_counts().idxmax()
top_disease = df["Disease"].value_counts().idxmax()
top_revenue_department = df.groupby("Department")["Bill_Amount"].sum().idxmax()

print("=" * 50)
print("HOSPITAL MANAGEMENT KPI REPORT")
print("=" * 50)
print("Total Patients:", total_patients)
print("Total Revenue:", round(total_revenue, 2))
print("Average Bill:", round(average_bill, 2))
print("Average Stay:", round(average_stay, 2))
print("Average Satisfaction:", round(average_satisfaction, 2))
print("Readmission Rate:", round(readmission_rate, 2), "%")
print("Highest Patient Volume:", top_department)
print("Most Common Disease:", top_disease)
print("Highest Revenue Department:", top_revenue_department)
print("=" * 50)