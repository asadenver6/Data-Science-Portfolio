
from src.data_loader import load_data
from pipelines.preprocessing import build_preprocessor
from src.model_factory import create_model
from sklearn.model_selection import train_test_split

print("1. Loading data...")
data = load_data("data/creditscoring.csv")
print("OK:", data.shape)
input("Press Enter to continue...")


print("\n2. Splitting X/y...")
X = data.drop("good_bad", axis=1)
y = data["good_bad"].map({"bad": 0, "good": 1})
print("OK")
input("Press Enter to continue...")

print("\n3. Train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("OK:", X_train.shape)

print("\n4. Building preprocessor...")
preprocessor = build_preprocessor(X_train)
print("OK")

print("\n5. Testing preprocessing only...")
Xt = preprocessor.fit_transform(X_train)
print("OK:", Xt.shape)

print("\n6. Building model pipeline...")
model = create_model(preprocessor)
print("OK")

print("\n7. Fitting model on small sample...")
model.fit(X_train.sample(500, random_state=42),
          y_train.loc[X_train.sample(500, random_state=42).index])

print("DONE: full pipeline works")
