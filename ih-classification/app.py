import gradio as gr
from transformers import AutoModel, AutoProcessor
import torch
import requests
from PIL import Image
from io import BytesIO

fashion_items = ['top', 'trousers']

# Load model and processor
model_name = 'Marqo/marqo-fashionSigLIP'
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Preprocess and normalize text data
with torch.no_grad():
    # Ensure truncation and padding are activated
    processed_texts = processor(
        text=fashion_items,
        return_tensors="pt",
        truncation=True,  # Ensure text is truncated to fit model input size
        padding=True      # Pad shorter sequences so that all are the same length
    )['input_ids']
    
    text_features = model.get_text_features(processed_texts)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Prediction function
def predict_from_url(url):
    # Check if the URL is empty
    if not url:
        return {"Error": "Please input a URL"}
    
    try:
        image = Image.open(BytesIO(requests.get(url).content))
    except Exception as e:
        return {"Error": f"Failed to load image: {str(e)}"}
    
    processed_image = processor(images=image, return_tensors="pt")['pixel_values']
    
    with torch.no_grad():
        image_features = model.get_image_features(processed_image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_probs = (100 * image_features @ text_features.T).softmax(dim=-1)
        
    return {fashion_items[i]: float(text_probs[0, i]) for i in range(len(fashion_items))}

# Gradio interface
demo = gr.Interface(
    fn=predict_from_url, 
    inputs=gr.Textbox(label="Enter Image URL"),
    outputs=gr.Label(label="Classification Results"), 
    title="Fashion Item Classifier",
    allow_flagging="never"
)

# Launch the interface
demo.launch()
