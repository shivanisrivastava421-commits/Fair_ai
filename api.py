import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI
from pydantic import BaseModel
from services.analysis import generate_response

# APP INIT
app = FastAPI(
    title="FairAI Guardian API",
    description="Detect bias in datasets using AI",
    version="1.0"
)

# REQUEST MODEL
class RequestData(BaseModel):
    dataset_url: str
    target_column: str
    sensitive_feature: str

# ROOT ENDPOINT
@app.get("/")
def home():
    return {
        "message": "FairAI Guardian API is running 🚀",
        "endpoint": "/analyze"
    }

# MAIN API ENDPOINT
@app.post("/analyze")
def analyze(data: RequestData):

    result = generate_response(
        data.dataset_url,
        data.target_column,
        data.sensitive_feature
    )

    return result

# LOCAL RUN (OPTIONAL)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
