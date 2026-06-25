from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

model = YOLO("best.pt")

# FASTAPI
API_URL = "http://127.0.0.1:8000/predict"
app = FastAPI(
    title="Waste Classification API",
    description="API untuk klasifikasi sampah menggunakan YOLOv8",
    version="1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ROOT
@app.get("/")
def home():
    return {
        "message": "Waste Classification API Running"
    }

# PREDICT
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        results = model.predict(
            source=image,
            imgsz=224,
            verbose=False
        )

        result = results[0]

        probs = result.probs

        top1 = probs.top1

        confidence = float(
            probs.top1conf
        )

        prediction = model.names[top1]

        all_probs = {}

        for idx, score in enumerate(
            probs.data.cpu().numpy()
        ):

            all_probs[
                model.names[idx]
            ] = round(
                float(score) * 100,
                2
            )

        return {
            "success": True,
            "prediction": prediction,
            "confidence": round(
                confidence * 100,
                2
            ),
            "probabilities": all_probs
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    