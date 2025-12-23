from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib

app = FastAPI(title="Twitter Sentiment Analyzer")

# Load model
try:
    model = joblib.load('sentiment_model.joblib')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

class TweetRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(request: TweetRequest):
    prediction = model.predict([request.text])[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = model.predict_proba([request.text])[0][prediction]
    return {
        "text": request.text,
        "sentiment": sentiment,
        "score": int(prediction),
        "confidence": round(float(confidence), 4)
    }

# 👉 This serves index.html when you open http://127.0.0.1:8000
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()