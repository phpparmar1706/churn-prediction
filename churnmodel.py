# %%
import pandas as pd
import joblib

# %%
data = pd.read_csv("Churn_Modelling.csv")

# %%
data.info()

# %%
data.columns

# %%
data["Exited"].value_counts()

# %%
data = data[['CreditScore','Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Exited']]

# %%
data = pd.get_dummies(
    data,
    columns=["Gender"]
)

# %%
data.info()

# %%
data.corr()

# %%
X = data[
    [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
]
Y = data["Exited"]

# %%
import pandas as pd  # Data preprocessing step

from sklearn.model_selection import (
    train_test_split,
)  # 80 : 20 for train : test purpose -- This will be useful for spliting data
from sklearn.linear_model import (
    LogisticRegression,
)  # This is the base model from scikitlearn for classification
from sklearn.metrics import (
    accuracy_score,
)  # After the model training we will evaluate on test data and find accuracy
from sklearn.preprocessing import (
    StandardScaler,
)  # We will need in preprocssing to scale the data to some extent

# %%
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42
)

# %%
print("X_train Shape", X_train.shape)
print("X_test Shape", X_test.shape)
print("y_train Shape", y_train.shape)
print("y_test Shape", y_test.shape)

# %%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
model = LogisticRegression()

# %%
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

# %%
y_pred

# %%
y_test

# %%
evaluate = accuracy_score(y_test, y_pred)

# %%
evaluate

# %%
newApplicant = pd.DataFrame(
    [{"CreditScore": 1,
        "Age" :22 ,
        "Tenure" : 10,
        "Balance" :252524,
        "NumOfProducts" : 1,
        "HasCrCard" : 0 ,
        "IsActiveMember" : 1,
        "EstimatedSalary" : 242323,
}])

# %%
newApplicant = scaler.transform(newApplicant)

# %%
result = model.predict(newApplicant)
probability = model.predict_proba(newApplicant)

# %%
print(f"Result : {result}")
print(f"Probability : {probability}")

# %%
import matplotlib.pyplot as plt

# %%
label = ["Churn","No Churn"]

# %%
counts = data["Exited"].value_counts()

# %%
plt.bar(label,counts)
plt.ylabel("No. of Customers")
plt.title("Churn Distribustion")
plt.show()

# %%
plt.pie(counts,
        labels=["No churn","Churn"],
        startangle=90,
        )
plt.show()


# %%

# Save Model
joblib.dump(model, "backend/model.pkl")

# Save Scaler
joblib.dump(scaler, "backend/scaler.pkl")

print("Model Saved Successfully")
