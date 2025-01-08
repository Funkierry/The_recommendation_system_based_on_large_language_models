import os
import pandas as pd
from transformers import pipeline
from PIL import Image
import torch
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# Paths
input_file = r"C:\Users\User\Desktop\FYP\dataset\val.csv"
images_dir = r"C:\Users\User\Desktop\FYP\dataset\images\val"
output_file = r"C:\Users\User\Desktop\FYP\dataset\val.csv"
batch_size = 100  # Batch size for processing

# Load the dataset in chunks
reader = pd.read_csv(input_file, chunksize=batch_size)

# Load Flan-T5 large model with GPU acceleration
print("Loading the Flan-T5 large model...")
generator = pipeline("text2text-generation", model="google/flan-t5-large", device=0)  # GPU device 0

# Load CLIP model for image embeddings
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Function to extract CLIP image embeddings
def extract_image_features(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = clip_processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            features = clip_model.get_image_features(**image)
        return features.cpu().numpy()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to generate category name based on title and image features
def generate_category(title, image_path):
    try:
        # Extract image features
        image_features = extract_image_features(image_path)
        # Convert image features to a string description (if image exists)
        image_description = "image not available" if image_features is None else "image features extracted"
        # Use Flan-T5 to generate category
        prompt = (
            f"Classify this product title into a suitable category, such as 'Jackets', 'Sweatshirts', "
            f"'Shoes', 'Pants', 'Accessories', or similar categories. "
            f"Title: {title}. Image context: {image_description}."
        )
        result = generator(prompt, max_length=10, num_return_sequences=1)
        return result[0]["generated_text"]
    except Exception as e:
        print(f"Error generating category for title: {title}, image: {image_path}, error: {e}")
        return "Unknown"

# Initialize progress bar
print("Processing dataset in batches...")

# Create an empty DataFrame to store processed data
processed_data = pd.DataFrame()

# Process the dataset in chunks
for chunk in tqdm(reader, desc="Processing batches"):
    # Process each row in the chunk
    chunk["categoryName"] = chunk.apply(
        lambda row: generate_category(
            row["title"],
            os.path.join(images_dir, os.path.basename(row["imgUrl"])) if pd.notnull(row["imgUrl"]) else None
        ),
        axis=1
    )
    # Append processed chunk to the final DataFrame
    processed_data = pd.concat([processed_data, chunk], ignore_index=True)

# Save the updated dataset
processed_data.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"Category names have been updated based on title and image features and saved to {output_file}.")
