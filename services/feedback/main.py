import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="feedback-loop")

class SocLabel(BaseModel):
    hash_id: str
    label: str   # "tp" | "fp" | "tn" | "fn"
    notes: str = ""

@app.post("/label")
def label(feedback: SocLabel):
    # TODO: write to SageMaker Feature Store (via boto3)
    # TODO: recompute θ / α / β periodically (via retrain job)
    return {"status":"stored","hash_id":feedback.hash_id}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","8000")))