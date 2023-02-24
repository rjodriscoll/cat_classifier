import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from src.model import model, auto_transforms
import uvicorn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI()


def run_pred(img: str):
    img = Image.open(img)
    img = auto_transforms(img).unsqueeze(0)

    with torch.inference_mode():
        y_pred = model(img.to(DEVICE))
        y_pred_class = torch.sigmoid(y_pred)

    prob = y_pred_class[0][0].item()
    prob_class_0, prob_class_1 = round(1 - prob, 2), round(prob, 2)

    return {
        "class_magnus_prob": prob_class_0,
        "class_winnie_prob": prob_class_1,
        "prediction": "winnie" if prob > 0.5 else "magnus",
    }

@app.get("/")
def read_root():
    return "cat classifier running"

@app.on_event("startup")
def load_model():
    global model
    model.load_state_dict(
        torch.load("models/ccfl_efficient_net_0.pth", map_location=torch.device(DEVICE))
    )
    model.eval()


@app.get("/predict/{img}/", status_code=200)
async def predict(img):
    prediction = run_pred(img)
    if not prediction:

        raise HTTPException(
            status_code=404, detail="Image could not be downloaded"
        )
    
    return prediction

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)