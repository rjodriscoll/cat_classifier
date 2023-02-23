from src.model import model, auto_transforms
from PIL import Image
import gradio as gr
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(
        torch.load("models/ccfl_efficient_net_0.pth", map_location=torch.device(DEVICE))
    )
model.eval()
model.to(DEVICE)

model.eval()

def run_pred(img):

    img = auto_transforms(img).unsqueeze(0)

    with torch.inference_mode():
        y_pred = model(img.to(DEVICE))
        y_pred_class = torch.sigmoid(y_pred)

    prob = y_pred_class[0][0].item()
    prob_class_0, prob_class_1 = round(1 - prob, 2), round(prob, 2)

    return {
        "winnie": prob_class_1,
        "magnus": prob_class_0,
    }



interface = gr.Interface(
    fn=run_pred,
    inputs=gr.inputs.Image(type = 'pil'),
    outputs="label",
    title='Cat Classification',
    description='Classify pictures of my cats'
)

interface.launch()