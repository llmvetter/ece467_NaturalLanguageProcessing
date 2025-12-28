import gradio as gr
from transformers import pipeline


# --- Configuration ---
MODEL_PATH = "llmvetter/bert_product_classifier"
DEVICE = -1

# --- Load Model ---
classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
)


# --- Prediction Function ---
def classify_product(product_name):
    result = classifier(product_name)[0]
    category = result["label"]
    confidence = result["score"] * 100
    return f"Predicted Category: **{category}**\nConfidence: {confidence:.2f}%"


# --- Gradio Interface ---
iface = gr.Interface(
    fn=classify_product,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Enter a product name (e.g., 'Bluetooth noise-cancelling headphones')",
    ),
    outputs="markdown",
    title="Fine-Tuned Product Classifier",
    description="Fine-tuned BERT model for product category classification. Type a product name and get the predicted category.",
)

# --- Launch ---
if __name__ == "__main__":
    iface.launch()
