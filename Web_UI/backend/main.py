from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from model_loader import load_model, predict

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model, device = load_model()

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    result = predict(model, device, file.file)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
