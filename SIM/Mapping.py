import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# 路径设置
TRAIN_CSV = r'C:\Users\User\Desktop\FYP\dataset\train_updated.csv'
TEST_CSV = r'C:\Users\User\Desktop\FYP\dataset\test_updated.csv'
VAL_CSV = r'C:\Users\User\Desktop\FYP\dataset\val_updated.csv'
RESULTS_DIR = r'C:\Users\User\Desktop\FYP\results'

def read_csv_with_encoding(file_path):
    encodings = ['utf-8', 'latin1', 'gbk', 'big5', 'gb18030']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read {file_path} with any of the attempted encodings")

# 读取所有数据集
train_df = read_csv_with_encoding(TRAIN_CSV)
test_df = read_csv_with_encoding(TEST_CSV)
val_df = read_csv_with_encoding(VAL_CSV)

# 合并所有类别
all_categories = pd.concat([
    train_df['categoryName'],
    test_df['categoryName'],
    val_df['categoryName']
]).unique()

# 创建标签编码器
label_encoder = LabelEncoder()
label_encoder.fit(all_categories)

# 保存标签映射信息
label_info = {
    'label_encoder': label_encoder,
    'num_classes': len(label_encoder.classes_)
}

# 保存到文件
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, 'label_mapping.pkl'), 'wb') as f:
    pickle.dump(label_info, f)

print(f"Label mapping saved. Total number of classes: {len(label_encoder.classes_)}")