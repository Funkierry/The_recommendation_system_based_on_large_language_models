import os
import pandas as pd
from transformers import pipeline
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module="transformers.models.clip.modeling_clip")


input_file = r"C:\Users\User\Desktop\FYP\dataset\train.csv"
images_dir = r"C:\Users\User\Desktop\FYP\dataset\images\train"
output_file = r"C:\Users\User\Desktop\FYP\dataset\train_updated.csv"  
category_counts_output = r"C:\Users\User\Desktop\FYP\dataset\trainCounts.csv"  
batch_size = 100  # Batch size for processing


device = 0 if torch.cuda.is_available() else -1  # GPU device 0 if available, else CPU
print(f"Using device: {'GPU' if device >=0 else 'CPU'}")

# Flan-T5 large
print("Loading the Flan-T5 large model...")
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=device  
)

# CLIP
print("Loading CLIP model...")
clip_device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(clip_device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_image_features(image_path):
    if not image_path:
        return None
    try:
        image = Image.open(image_path).convert("RGB")
        image_inputs = clip_processor(images=image, return_tensors="pt").to(clip_model.device)
        with torch.no_grad():
            features = clip_model.get_image_features(**image_inputs)
        return features.cpu().numpy()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def generate_category(title, image_path):
    try:
        image_features = extract_image_features(image_path)
        image_context = "image available" if image_features is not None else "image not available"
        
        prompt = (
            f"Classify this product title into a detailed and specific category. Examples of detailed categories include 'Men's Running Shoes', "
            f"'Women's Leather Handbags', 'Children's Winter Jackets', etc. "
            f"Title: {title}. Image context: {image_context}."
        )
        
        result = generator(prompt, max_length=50, num_return_sequences=1)
        category = result[0]["generated_text"].strip()
        return category
    except Exception as e:
        print(f"Error generating category for title: {title}, image: {image_path}, error: {e}")
        return "Unknown"

def main():
    print("Counting total number of rows in the dataset...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1  
    print(f"Total rows to process: {total_rows}")

    print("Starting classification of the dataset...")
    reader = pd.read_csv(input_file, chunksize=batch_size, iterator=True)
    
    with tqdm(total=total_rows, desc="Processing dataset", unit="rows") as pbar:
        processed_chunks = []
        for chunk in reader:
            chunk["imagePath"] = chunk["imgUrl"].apply(
                lambda url: os.path.join(images_dir, os.path.basename(url)) if pd.notnull(url) else None
            )
            
            chunk["categoryName"] = chunk.apply(
                lambda row: generate_category(row["title"], row["imagePath"]),
                axis=1
            )
            
            chunk = chunk.drop(columns=["imagePath"])
            
            processed_chunks.append(chunk)
            
            pbar.update(len(chunk))
    
    print("Concatenating all processed chunks...")
    processed_data = pd.concat(processed_chunks, ignore_index=True)
    
    print(f"Saving the updated dataset to {output_file}...")
    processed_data.to_csv(output_file, index=False, encoding="utf-8-sig")
    
    print("Computing all categories and their sample counts based on reclassification...")
    category_counts = processed_data['categoryName'].value_counts().reset_index()
    category_counts.columns = ['Category Name', 'Sample Count']
    
    print(f"Saving category counts to {category_counts_output}...")
    category_counts.to_csv(category_counts_output, index=False, encoding="utf-8-sig")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()

