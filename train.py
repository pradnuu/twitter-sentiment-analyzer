import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib

# Load the real Sentiment140 dataset
print("Loading dataset...")
# The dataset has no headers, so we name them manually
cols = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv('dataset.csv', encoding='latin-1', names=cols)

# Keep only text and target columns
df = df[['text', 'target']]

# Map target: 0 = Negative, 4 = Positive (we ignore neutral=2 for simplicity)
df = df[df['target'].isin([0, 4])]  # Remove neutral tweets
df['target'] = df['target'].map({0: 0, 4: 1})  # 0=Negative, 1=Positive

# Optional: Sample 100,000 rows to speed up training (remove if you want full dataset)
df = df.sample(100000, random_state=42).reset_index(drop=True)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['target'], test_size=0.2, random_state=42
)

# Build model pipeline
model = make_pipeline(TfidfVectorizer(max_features=5000), LogisticRegression(max_iter=1000))

# Train
print("Training model...")
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, 'sentiment_model.joblib')
print("✅ Model saved as 'sentiment_model.joblib'")