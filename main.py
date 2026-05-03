import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

print("\n================= PROJECT START =================\n")

# =========================
# LOAD DATA (ROBUST PATH)
# =========================
print("Current Working Directory:", os.getcwd())

# ✅ Always look in the same folder as main.py — no matter where you run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Dataset.csv")

# ✅ List all CSV files found, so you know what's available
csv_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.csv')]
print("CSV files found in folder:", csv_files)

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"\n❌ Dataset.csv not found!\n"
        f"   Looking in: {BASE_DIR}\n"
        f"   CSV files found: {csv_files}\n"
        f"   👉 Please put Dataset.csv in the same folder as main.py"
    )

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

print("✅ Data Loaded Successfully")
print("Shape:", df.shape)

# =========================
# DATA CLEANING
# =========================
df = df.dropna()
df['Average Cost for two'] = pd.to_numeric(df['Average Cost for two'], errors='coerce')
df = df.dropna(subset=['Average Cost for two'])

# =========================
# TASK 1: RATING PREDICTION
# =========================
print("\n--- Task 1: Rating Prediction ---")

le_city = LabelEncoder()
le_cuisine = LabelEncoder()

df['City_encoded'] = le_city.fit_transform(df['City'])
df['Cuisines_encoded'] = le_cuisine.fit_transform(df['Cuisines'])

X = df[['Average Cost for two', 'City_encoded', 'Cuisines_encoded']]
y = df['Aggregate rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Sample Predictions:", y_pred[:5])
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# =========================
# TASK 2: COST ANALYSIS
# =========================
print("\n--- Task 2: Cost Analysis ---")

cost_area = (
    df.groupby('Locality')['Average Cost for two']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Expensive Areas:\n", cost_area)

plt.figure(figsize=(10,5))
cost_area.plot(kind='bar')
plt.title("Top 10 Expensive Areas")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df['Average Cost for two'], df['Aggregate rating'], alpha=0.5)
plt.xlabel("Cost")
plt.ylabel("Rating")
plt.title("Cost vs Rating")
plt.tight_layout()
plt.show()

# =========================
# TASK 3: CUISINE CLASSIFICATION
# =========================
print("\n--- Task 3: Cuisine Classification ---")

X_c = df[['Average Cost for two', 'City_encoded']]
y_c = df['Cuisines_encoded']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(Xc_train, yc_train)
yc_pred = clf.predict(Xc_test)

print("Cuisine Classification Accuracy:", accuracy_score(yc_test, yc_pred))

# =========================
# TASK 4: RESTAURANT INSIGHTS
# =========================
print("\n--- Task 4: Restaurant Insights ---")

top_rated = (
    df[['Restaurant Name', 'Aggregate rating']]
    .sort_values(by='Aggregate rating', ascending=False)
    .head(10)
)

print("\nTop Rated Restaurants:\n", top_rated)

print("\n✅ ALL TASKS COMPLETED SUCCESSFULLY 🎉")