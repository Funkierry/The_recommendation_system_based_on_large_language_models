import pandas as pd
import requests
import os
import hashlib
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

def download_image(url, save_dir):
    try:
        filename = hashlib.md5(url.encode()).hexdigest() + '.jpg'
        filepath = os.path.join(save_dir, filename)
        
        if os.path.exists(filepath):
            return filename
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:   
                f.write(response.content)
            return filename
        return None
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return None

def process_dataset(input_file, image_dir, output_dir):
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(input_file)
    
    category_counts = df['categoryName'].value_counts()
    
    valid_categories = category_counts[(category_counts >= 9000) & (category_counts <= 25000)].index
    
    df['local_image_path'] = None
    
    print("Downloading images...")
    num_valid_categories = len(valid_categories)
    current_category = 0
    for category in valid_categories:
        category_df = df[df['categoryName'] == category]
        for idx in tqdm(category_df.index, desc=f"Processing category {category} ({current_category+1}/{num_valid_categories})"):
            url = category_df.loc[idx, 'imgUrl']
            filename = download_image(url, image_dir)
            if filename:
                category_df.loc[idx, 'local_image_path'] = filename
        
        category_df['imgUrl'] = category_df['local_image_path']
        category_df = category_df.drop('local_image_path', axis=1)
        df.loc[category_df.index] = category_df
        current_category += 1
    
    print(f"Total rows before filtering: {len(df)}")
    df_success = df.dropna(subset=['imgUrl'])
    print(f"Successfully downloaded images: {len(df_success)}")
    print(f"Failed downloads: {len(df) - len(df_success)}")
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    print("\nCategory distribution before split:")
    print(df_success['categoryName'].value_counts())
    
    for category in df_success['categoryName'].unique():
        category_df = df_success[df_success['categoryName'] == category]
        
        train_df, temp_df = train_test_split(category_df, train_size=0.8, random_state=42)
        val_df, test_df = train_test_split(temp_df, train_size=0.5, random_state=42)
        
        train_dfs.append(train_df)
        val_dfs.append(val_df)
        test_dfs.append(test_df)
    
    final_train_df = pd.concat(train_dfs)
    final_val_df = pd.concat(val_dfs)
    final_test_df = pd.concat(test_dfs)
    
    print("\nFinal dataset statistics:")
    print(f"Train set size: {len(final_train_df)}")
    print(f"Validation set size: {len(final_val_df)}")
    print(f"Test set size: {len(final_test_df)}")
    
    print("\nCategory distribution in train set:")
    print(final_train_df['categoryName'].value_counts())
    print("\nCategory distribution in validation set:")
    print(final_val_df['categoryName'].value_counts())
    print("\nCategory distribution in test set:")
    print(final_test_df['categoryName'].value_counts())
    
    final_train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    final_val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    final_test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

if __name__ == "__main__":
    INPUT_FILE = r"C:\Users\User\Desktop\FYP\dataset\amz_uk_processed_data.csv"
    IMAGE_DIR = "images"
    OUTPUT_DIR = "dataset"
    
    process_dataset(INPUT_FILE, IMAGE_DIR, OUTPUT_DIR)
         
                 