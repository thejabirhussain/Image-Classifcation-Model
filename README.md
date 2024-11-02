# Fashion Item Classifier Using Marqo's Model with Gradio

This project is an image classification tool that identifies fashion items from images using Marqo's `marqo-fashionSigLIP` model, implemented with Gradio for a user-friendly interface. The model classifies fashion items, specifically identifying whether an image contains items like "top" or "trousers."

## Table of Contents
- [Overview](#overview)
- [Model and Processor](#model-and-processor)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [License](#license)

## Overview
This project uses Hugging Face's `transformers` library to load Marqo's `marqo-fashionSigLIP` model, which specializes in identifying fashion-related items from images. The Gradio interface allows users to input an image URL and receive classification results indicating the probability that the item is either a "top" or "trousers."

## Model and Processor
- **Model**: `Marqo/marqo-fashionSigLIP`
- **Processor**: `AutoProcessor` for pre-processing image and text data before feeding them into the model.

## Requirements
- Python 3.x
- `torch`: For handling tensors and running the model.
- `transformers`: For loading Marqo's model and processor.
- `gradio`: To create the web interface.
- `Pillow`: For image processing.
- `requests`: To fetch images from URLs.

Install the dependencies with:
```bash
pip install torch transformers gradio pillow requests
```

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fashion-item-classifier.git
   ```
2. Change directory to the project folder:
   ```bash
   cd fashion-item-classifier
   ```
3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the Gradio app with:
```bash
python app.py
```
The Gradio interface will launch, allowing you to input an image URL for classification.

## How It Works
1. **Text Preprocessing**: Fashion items (like "top" and "trousers") are processed to create normalized text features.
2. **Image Preprocessing**: The image from the URL is fetched, processed, and normalized.
3. **Classification**: The model compares image features with text features to determine the probabilities for each fashion item.
4. **Output**: The Gradio interface displays the probabilities for each fashion item.

## Example
Input an image URL, and the model will classify it with probabilities for each fashion item.
