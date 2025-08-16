#!/usr/bin/env python
# coding: utf-8

# In[13]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Utility for display
pd.set_option("display.max_columns", None)


# ## 1. Load Data

# In[3]:



train_path = r"C:\Users\Sneha\Downloads\train_v9rqX0R.csv"
test_path = r"C:\Users\Sneha\Downloads\test_AbJTz2l.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


# ## 2. Quick EDA

# In[4]:



display(train.dtypes.to_frame("dtype"))
display(train.isna().sum().sort_values(ascending=False).to_frame("missing_train"))
display(test.isna().sum().sort_values(ascending=False).to_frame("missing_test"))

# Numeric summary
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
display(train[numeric_cols].describe().T)


# In[5]:



plt.figure()
train['Item_Outlet_Sales'].hist(bins=50)
plt.title("Distribution: Item_Outlet_Sales")
plt.xlabel("Sales"); plt.ylabel("Frequency")
plt.show()

plt.figure()
train['Item_MRP'].hist(bins=50)
plt.title("Distribution: Item_MRP")
plt.xlabel("MRP"); plt.ylabel("Frequency")
plt.show()

plt.figure()
train['Item_Visibility'].hist(bins=50)
plt.title("Distribution: Item_Visibility")
plt.xlabel("Visibility"); plt.ylabel("Frequency")
plt.show()

plt.figure()
train['Item_Weight'].dropna().hist(bins=50)
plt.title("Distribution: Item_Weight")
plt.xlabel("Weight"); plt.ylabel("Frequency")
plt.show()


# ## 3. Data Cleaning & Feature Engineering

# In[6]:




def standardize_fat_content(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
        'LF': 'Low Fat',
        'low fat': 'Low Fat',
        'reg': 'Regular'
    })
    return df

def add_item_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Item_Category'] = df['Item_Identifier'].str[:2].map({'FD':'Food', 'DR':'Drinks', 'NC':'Non-Consumable'})
    # Non-consumables are not edible
    df.loc[df['Item_Category'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
    return df

def add_outlet_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Outlet_Age'] = 2013 - df['Outlet_Establishment_Year']
    return df

def impute_item_weight_by_item(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    med_by_item = df.groupby('Item_Identifier')['Item_Weight'].median()
    df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Identifier'].map(med_by_item))
    df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].median())
    return df

def fix_visibility_and_add_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    med_vis_by_item = df.groupby('Item_Identifier')['Item_Visibility'].median()
    zeros = df['Item_Visibility'] == 0
    df.loc[zeros, 'Item_Visibility'] = df.loc[zeros, 'Item_Identifier'].map(med_vis_by_item)
    df['Item_Visibility'] = df['Item_Visibility'].replace(0, np.nan)
    df['Item_Visibility'] = df['Item_Visibility'].fillna(df['Item_Visibility'].median())
    # ratio to per-item mean
    mean_vis_by_item = df.groupby('Item_Identifier')['Item_Visibility'].mean()
    df['Item_Visibility_MeanRatio'] = df['Item_Visibility'] / df['Item_Identifier'].map(mean_vis_by_item)
    return df

def impute_outlet_size(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mode_by_type = df.groupby('Outlet_Type')['Outlet_Size'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Type'].map(mode_by_type))
    if df['Outlet_Size'].isna().any():
        gl_mode = df['Outlet_Size'].mode()
        if not gl_mode.empty:
            df['Outlet_Size'] = df['Outlet_Size'].fillna(gl_mode.iloc[0])
    return df

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = standardize_fat_content(df)
    df = add_item_category(df)
    df = add_outlet_age(df)
    df = impute_item_weight_by_item(df)
    df = fix_visibility_and_add_ratio(df)
    df = impute_outlet_size(df)
    return df


train_prep = prepare(train)
test_prep = prepare(test)


display(train_prep.isna().sum().sort_values(ascending=False).to_frame("missing_after_prep_train").head(10))
display(test_prep.isna().sum().sort_values(ascending=False).to_frame("missing_after_prep_test").head(10))


# ## 4. Modeling & Validation

# In[7]:



target_col = "Item_Outlet_Sales"
X = train_prep.drop(columns=[target_col])
y = train_prep[target_col]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()


ordinal_preprocess = ColumnTransformer(
    transformers=[
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), cat_cols),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), num_cols)
    ]
)

ridge_pipe = Pipeline([
    ("preprocess", ordinal_preprocess),
    ("model", Ridge(alpha=1.0, random_state=42))
])

ridge_pipe.fit(X_train, y_train)
ridge_pred = ridge_pipe.predict(X_valid)
ridge_rmse = mean_squared_error(y_valid, ridge_pred, squared=False)
print("Validation RMSE (Ridge + Ordinal):", round(ridge_rmse, 4))

#RandomForest with OneHot encoding
onehot_preprocess = ColumnTransformer(
    transformers=[
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ]), cat_cols),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), num_cols)
    ]
)

rf_pipe = Pipeline([
    ("preprocess", onehot_preprocess),
    ("model", RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1))
])

rf_pipe.fit(X_train, y_train)
rf_pred = rf_pipe.predict(X_valid)
rf_rmse = mean_squared_error(y_valid, rf_pred, squared=False)
print("Validation RMSE (RandomForest + OneHot):", round(rf_rmse, 4))

best_pipe = rf_pipe if rf_rmse < ridge_rmse else ridge_pipe
best_name = "RandomForest" if rf_rmse < ridge_rmse else "Ridge"
print("Chosen model:", best_name)


# ## 5. Feature Importance (Random Forest)

# In[8]:


# Feature importance 
def plot_rf_feature_importance(fitted_pipeline, cat_cols, num_cols, top_n=20):
    # Get onehot feature names
    preprocess = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        print("Model does not support feature importances.")
        return
    # Build feature name list
    ohe = preprocess.named_transformers_["categorical"].named_steps["onehot"]
    ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
    feature_names = ohe_names + num_cols
    importances = model.feature_importances_
    # Top N
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]
    # Plot (single figure)
    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(names))
    plt.barh(y_pos, vals)
    plt.gca().invert_yaxis()
    plt.yticks(y_pos, names)
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.show()

if best_name == "RandomForest":
    plot_rf_feature_importance(best_pipe, cat_cols, num_cols, top_n=25)
else:
    print("Best model is Ridge; feature importance is not applicable.")


# ## 6. Train on Full Data & Predict Test

# In[12]:


# Retrain chosen model on full data and predict test
# Rebuild preprocessors with full X
cat_cols_full = X.select_dtypes(include=['object']).columns.tolist()
num_cols_full = X.select_dtypes(include=[np.number]).columns.tolist()

if best_name == "RandomForest":
    preprocess_full = onehot_preprocess
    model_full = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
else:
    preprocess_full = ordinal_preprocess
    model_full = Ridge(alpha=1.0, random_state=42)

final_pipe = Pipeline([
    ("preprocess", preprocess_full),
    ("model", model_full)
])

final_pipe.fit(X, y)
test_preds = final_pipe.predict(test_prep)

submission = test[['Item_Identifier','Outlet_Identifier']].copy()
submission['Item_Outlet_Sales'] = test_preds

out_path = "C:/Users/Sneha/Desktop/Test_prediction.csv"
submission.to_csv(out_path, index=False)
print("Saved submission to:", out_path)
submission.head()


# In[ ]:




