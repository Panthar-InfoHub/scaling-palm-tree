from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import TFViTForImageClassification, AutoImageProcessor
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Hugging Face model and processor 
model = TFViTForImageClassification.from_pretrained("pantharinfohub/tf_model_face_recognition")
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Labels (based on your config.json id2label)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Track predictions
expression_summary = {}

@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Preprocess using Hugging Face processor
    inputs = processor(images=image, return_tensors="tf")

    # Predict using ViT model
    outputs = model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
    top_3 = probs.argsort()[-3:][::-1]

    # Summary tracking
    top_label = class_names[top_3[0]]
    expression_summary[top_label] = expression_summary.get(top_label, 0) + 1

    results = [
        {"label": class_names[i], "probability": float(probs[i])}
        for i in top_3
    ]

    return JSONResponse(content={"top_3_predictions": results})

@app.get("/summary/")
def get_summary():
    return JSONResponse(content={"summary": expression_summary})