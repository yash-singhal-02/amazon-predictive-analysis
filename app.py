import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

sns.set_theme(style="whitegrid")

# ===========================
# 1. Sidebar Navigation
# ===========================
st.sidebar.title("Amazon Predictive Analytics")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA", "Regression", "Classification", "Clustering"]
)

# ===========================
# 2. Load and preprocess data
# ===========================
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("Amazon.csv")

    # Profit
    df["Profit"] = df["TotalAmount"] - (
        (df["UnitPrice"] * df["Quantity"]) + df["Tax"] + df["ShippingCost"] - df["Discount"]
    )

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    # Outliers
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    mask_outliers = ((numeric_df < (Q1 - 1.5*IQR)) | (numeric_df > (Q3 + 1.5*IQR))).any(axis=1)
    df_clean = df[~mask_outliers].copy()

    # Profit Category
    df_clean = df_clean.sort_values("Profit")
    total = len(df_clean)
    cut1 = int(total/3)
    cut2 = int(2*total/3)

    df_clean.loc[df_clean.index[:cut1], "Profit_Category"] = "Low"
    df_clean.loc[df_clean.index[cut1:cut2], "Profit_Category"] = "Medium"
    df_clean.loc[df_clean.index[cut2:], "Profit_Category"] = "High"

    # Drop ID-like columns
    df_clean = df_clean.drop(["OrderID","CustomerID","ProductID","SellerID","OrderDate"], axis=1)

    # Label Encoding
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

    # Features / labels
    X = df_clean.drop("Profit_Category", axis=1)
    y = df_clean["Profit_Category"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42
    )

    # Regression data
    X_reg = df_clean[["TotalAmount"]]
    y_reg = df_clean["Profit"]
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.25, random_state=42
    )

    return df, df_clean, X, y, X_scaled, X_train, X_test, y_train, y_test, X_train_reg, X_test_reg, y_train_reg, y_test_reg

df, df_clean, X, y, X_scaled, X_train, X_test, y_train, y_test, X_train_reg, X_test_reg, y_train_reg, y_test_reg = load_and_prepare_data()



if page == "Overview":
    st.title("Amazon Predictive Analytics Dashboard")
    st.write("This app performs EDA, Regression, Classification and Clustering on Amazon sales data.")
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    st.write("Rows:", df.shape[0], " | Columns:", df.shape[1])


if page == "EDA":
    st.title("Exploratory Data Analysis")

    st.write("### Histograms Of Key Numeric Columns")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    axes[0,0].hist(df["Quantity"], bins=20, color="orange")
    axes[0,0].set_title("Quantity")

    axes[0,1].hist(df["UnitPrice"], bins=20, color="orange")
    axes[0,1].set_title("UnitPrice")

    axes[0,2].hist(df["Discount"], bins=20, color="orange")
    axes[0,2].set_title("Discount")

    axes[0,3].hist(df["Tax"], bins=20, color="orange")
    axes[0,3].set_title("Tax")

    axes[1,0].hist(df["ShippingCost"], bins=20, color="orange")
    axes[1,0].set_title("ShippingCost")

    axes[1,1].hist(df["TotalAmount"], bins=20, color="orange")
    axes[1,1].set_title("TotalAmount")

    axes[1,2].hist(df["Profit"], bins=20, color="orange")
    axes[1,2].set_title("Profit")

    fig.delaxes(axes[1,3])
    plt.tight_layout()
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include=["int64","float64"])
    fig2, ax2 = plt.subplots(figsize=(10,7))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.write("### Boxplot for Outliers")
    fig3, ax3 = plt.subplots(figsize=(12,5))
    df[numeric_df.columns].boxplot(ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)



if page == "Regression":
    st.title("Regression - Predict Profit from TotalAmount")

    lr = LinearRegression()
    lr.fit(X_train_reg, y_train_reg)
    pred_reg = lr.predict(X_test_reg)

    mae = mean_absolute_error(y_test_reg, pred_reg)
    mse = mean_squared_error(y_test_reg, pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_reg, pred_reg)

    st.write("### Regression Metrics")
    st.write("Mean Absolute Error:", mae)
    st.write("Mean Squared Error:", mse)
    st.write("Root Mean Squared Error:", rmse)
    st.write("R² Score:", r2)

    st.write("### Actual vs Predicted")
    sample = pd.DataFrame({
        "Actual Profit": y_test_reg.values[:20],
        "Predicted Profit": pred_reg[:20]
    })
    st.dataframe(sample)




if page == "Classification":
    st.title("Classification - Profit Category Prediction")

    # Train models
    knn = KNeighborsClassifier().fit(X_train, y_train)
    nb = GaussianNB().fit(X_train, y_train)
    dt = DecisionTreeClassifier().fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=120, random_state=42).fit(X_train, y_train)
    logistic = LogisticRegression(max_iter=500).fit(X_train, y_train)

    pred_knn = knn.predict(X_test)
    pred_nb = nb.predict(X_test)
    pred_dt = dt.predict(X_test)
    pred_rf = rf.predict(X_test)
    pred_logistic = logistic.predict(X_test)

    st.write("### Confusion Matrix - Select Model")
    model_choice = st.selectbox(
        "Choose Model",
        ["KNN","Naive Bayes","Decision Tree","Random Forest","Logistic Regression"]
    )

    if model_choice == "KNN":
        cm = confusion_matrix(y_test, pred_knn)
    elif model_choice == "Naive Bayes":
        cm = confusion_matrix(y_test, pred_nb)
    elif model_choice == "Decision Tree":
        cm = confusion_matrix(y_test, pred_dt)
    elif model_choice == "Random Forest":
        cm = confusion_matrix(y_test, pred_rf)
    else:
        cm = confusion_matrix(y_test, pred_logistic)

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", ax=ax_cm)
    st.pyplot(fig_cm)

    accuracies = [
        accuracy_score(y_test,pred_knn),
        accuracy_score(y_test,pred_nb),
        accuracy_score(y_test,pred_dt),
        accuracy_score(y_test,pred_rf),
        accuracy_score(y_test,pred_logistic),
    ]
    models = ["KNN","Naive Bayes","Decision Tree","Random Forest","Logistic"]

    st.write("### Accuracy Comparison")
    fig_acc, ax_acc = plt.subplots(figsize=(8,5))
    sns.barplot(x=models, y=accuracies, ax=ax_acc)
    ax_acc.set_ylabel("Accuracy")
    st.pyplot(fig_acc)





if page == "Clustering":
    st.title("Clustering using Principal Component Analysis (PCA) and K-Means")

    sample = df_clean.sample(2000, random_state=42)
    scaler = StandardScaler()
    sample_scaled = scaler.fit_transform(sample.drop("Profit_Category", axis=1))

    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster = kmeans.fit_predict(sample_scaled)

    pc = PCA(n_components=2).fit_transform(sample_scaled)

    fig_clust, ax_clust = plt.subplots(figsize=(8,5))
    scatter = ax_clust.scatter(pc[:,0], pc[:,1], c=cluster, cmap="YlOrBr")
    ax_clust.set_title("PCA Clustering Visualization")
    st.pyplot(fig_clust)

