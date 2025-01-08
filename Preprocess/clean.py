import pandas as pd
import os

def preprocess_csv(input_file, output_file, images_dir):
    df = pd.read_csv(input_file)
    initial_length = len(df)
    
    # 保留每个样本的第一个图像路径
    def process_imgurl(imgurl):
        if pd.isnull(imgurl):
            return None
        if ';' in imgurl:
            return imgurl.split(';')[0].strip()
        return imgurl.strip()
    
    df['imgUrl'] = df['imgUrl'].apply(process_imgurl)
    
    # 检查图像是否存在，如果不存在则设为None
    def verify_image(imgurl):
        if pd.isnull(imgurl):
            return False
        return os.path.exists(os.path.join(images_dir, os.path.basename(imgurl)))
    
    df['valid_image'] = df['imgUrl'].apply(verify_image)
    
    # 选择只保留有效图像的样本
    cleaned_df = df[df['valid_image']].drop(columns=['valid_image'])
    cleaned_length = len(cleaned_df)
    print(f"Processed {input_file}: {initial_length} -> {cleaned_length} samples after cleaning.")
    
    # 保存清理后的CSV
    cleaned_df.to_csv(output_file, index=False)
    print(f"Cleaned CSV saved to {output_file}")

if __name__ == '__main__':
    preprocess_csv(
        input_file=r"C:\Users\User\Desktop\FYP\dataset\train.csv",
        output_file=r"C:\Users\User\Desktop\FYP\dataset\train_clean.csv",
        images_dir=r"C:\Users\User\Desktop\FYP\dataset\images\train"
    )
    preprocess_csv(
        input_file=r"C:\Users\User\Desktop\FYP\dataset\val.csv",
        output_file=r"C:\Users\User\Desktop\FYP\dataset\val_clean.csv",
        images_dir=r"C:\Users\User\Desktop\FYP\dataset\images\val"
    )
    preprocess_csv(
        input_file=r"C:\Users\User\Desktop\FYP\dataset\test.csv",
        output_file=r"C:\Users\User\Desktop\FYP\dataset\test_clean.csv",
        images_dir=r"C:\Users\User\Desktop\FYP\dataset\images\test"
    )

