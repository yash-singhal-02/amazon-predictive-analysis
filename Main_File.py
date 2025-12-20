import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

amazon_gold = "orange"
amazon_dark = "black"
background_color = "lightgray"

sns.set_theme(style="whitegrid")
plt.rcParams["figure.facecolor"] = background_color

df = pd.read_csv("Amazon.csv")
print(df.head())
print(df.info())
print(df.describe())

df["Profit"] = df["TotalAmount"] - ((df["UnitPrice"] * df["Quantity"]) + df["Tax"] + df["ShippingCost"] - df["Discount"])
print(df[["TotalAmount", "Profit"]].head())

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Distribution of Numeric Columns", fontsize=16)

axes[0,0].hist(df["Quantity"], bins=20, color=amazon_gold)
axes[0,0].set_title("Quantity")

axes[0,1].hist(df["UnitPrice"], bins=20, color=amazon_gold)
axes[0,1].set_title("UnitPrice")

axes[0,2].hist(df["Discount"], bins=20, color=amazon_gold)
axes[0,2].set_title("Discount")

axes[0,3].hist(df["Tax"], bins=20, color=amazon_gold)
axes[0,3].set_title("Tax")

axes[1,0].hist(df["ShippingCost"], bins=20, color=amazon_gold)
axes[1,0].set_title("ShippingCost")

axes[1,1].hist(df["TotalAmount"], bins=20, color=amazon_gold)
axes[1,1].set_title("TotalAmount")

axes[1,2].hist(df["Profit"], bins=20, color=amazon_gold)
axes[1,2].set_title("Profit")

fig.delaxes(axes[1,3])
plt.tight_layout()
plt.show()

numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10,7))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(12,5))
df[numeric_df.columns].boxplot()
plt.title("Outlier Detection using Boxplot")
plt.xticks(rotation=45)
plt.show()

Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1

mask_outliers = ((numeric_df < (Q1 - 1.5*IQR)) | (numeric_df > (Q3 + 1.5*IQR))).any(axis=1)
df_clean = df[~mask_outliers].copy()

print("Original:", df.shape)
print("After Cleaning:", df_clean.shape)

df_clean = df_clean.sort_values("Profit")
total = len(df_clean)

cut1 = int(total/3)
cut2 = int(2*total/3)

df_clean.loc[df_clean.index[:cut1], "Profit_Category"] = "Low"
df_clean.loc[df_clean.index[cut1:cut2], "Profit_Category"] = "Medium"
df_clean.loc[df_clean.index[cut2:], "Profit_Category"] = "High"

print(df_clean["Profit_Category"].value_counts())

df_clean = df_clean.drop("OrderID", axis=1)
df_clean = df_clean.drop("CustomerID", axis=1)
df_clean = df_clean.drop("ProductID", axis=1)
df_clean = df_clean.drop("SellerID", axis=1)
df_clean = df_clean.drop("OrderDate", axis=1)

le = LabelEncoder()

df_clean["CustomerName"] = le.fit_transform(df_clean["CustomerName"])
df_clean["ProductName"] = le.fit_transform(df_clean["ProductName"])
df_clean["Category"] = le.fit_transform(df_clean["Category"])
df_clean["Brand"] = le.fit_transform(df_clean["Brand"])
df_clean["PaymentMethod"] = le.fit_transform(df_clean["PaymentMethod"])
df_clean["OrderStatus"] = le.fit_transform(df_clean["OrderStatus"])
df_clean["City"] = le.fit_transform(df_clean["City"])
df_clean["State"] = le.fit_transform(df_clean["State"])
df_clean["Country"] = le.fit_transform(df_clean["Country"])

X = df_clean.drop("Profit_Category", axis=1)
y = df_clean["Profit_Category"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

X_reg = df_clean[["TotalAmount"]]
y_reg = df_clean["Profit"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)
pred_reg = lr.predict(X_test_reg)

logistic = LogisticRegression(max_iter=500)
logistic.fit(X_train, y_train)
pred_logistic = logistic.predict(X_test)

sns.heatmap(confusion_matrix(y_test, pred_logistic), fmt="d", annot=True, cmap="YlOrBr")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

print("\nRegression Results")
print("Mean Absolute Error: ", mean_absolute_error(y_test_reg, pred_reg))
print("Mean Squared Error: ", mean_squared_error(y_test_reg, pred_reg))
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test_reg, pred_reg)))
print("R2 Score:", r2_score(y_test_reg, pred_reg))

knn = KNeighborsClassifier().fit(X_train, y_train)
pred_knn = knn.predict(X_test)

nb = GaussianNB().fit(X_train, y_train)
pred_nb = nb.predict(X_test)

dt = DecisionTreeClassifier().fit(X_train, y_train)
pred_dt = dt.predict(X_test)

rf = RandomForestClassifier(n_estimators=120, random_state=42).fit(X_train, y_train)
pred_rf = rf.predict(X_test)

sns.heatmap(confusion_matrix(y_test,pred_knn), fmt = "d",annot=True, cmap="YlOrBr")
plt.title("Confusion Matrix - KNN")
plt.show()

sns.heatmap(confusion_matrix(y_test,pred_nb), fmt = "d",annot=True, cmap="YlOrBr")
plt.title("Confusion Matrix - Naive Bayes")
plt.show()

sns.heatmap(confusion_matrix(y_test,pred_dt), fmt = "d",annot=True, cmap="YlOrBr")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

sns.heatmap(confusion_matrix(y_test,pred_rf), fmt = "d",annot=True, cmap="YlOrBr")
plt.title("Confusion Matrix - Random Forest")
plt.show()


accuracies = [accuracy_score(y_test,pred_knn),
              accuracy_score(y_test,pred_nb),
              accuracy_score(y_test,pred_dt),
              accuracy_score(y_test,pred_rf)]

models = ["KNN","Naive Bayes","Decision Tree","Random Forest"]

plt.figure(figsize=(8,5))
sns.barplot(x=models, y=accuracies, color="orange")
plt.title("Accuracy Comparison of Models")
plt.ylabel("Accuracy Score")
plt.show()

sample = df_clean.sample(2000)
sample_scaled = scaler.fit_transform(sample.drop("Profit_Category", axis=1))

kmeans = KMeans(n_clusters=3, random_state=42)
cluster = kmeans.fit_predict(sample_scaled)

pc = PCA(n_components=2).fit_transform(sample_scaled)
plt.scatter(pc[:,0], pc[:,1], c=cluster, cmap="YlOrBr")
plt.title("Clustering performed after applying Principal Component Analysis")
plt.show()
