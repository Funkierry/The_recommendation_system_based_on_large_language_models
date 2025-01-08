import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import json
import warnings
import pickle
from collections import Counter

# 忽略警告
warnings.filterwarnings("ignore")

# 路径设置
MODEL_PATH = r'C:\Users\User\Desktop\FYP\results\best_model.pth'
TRAIN_CSV = r'C:\Users\User\Desktop\FYP\dataset\train_updated.csv'
TEST_CSV = r'C:\Users\User\Desktop\FYP\dataset\test_updated.csv'
BASE_DIR = r'C:\Users\User\Desktop\FYP\dataset'
RESULTS_DIR = r'C:\Users\User\Desktop\FYP\results'
LABEL_MAP_PATH = os.path.join(RESULTS_DIR, 'label_mapping.pkl')

# 确保输出目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)

# 数据预处理转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def read_csv_with_encoding(file_path):
    """尝试使用不同的编码读取CSV文件"""
    encodings = ['utf-8', 'latin1', 'gbk', 'big5', 'gb18030']
    
    for encoding in encodings:
        try:
            print(f"Trying to read {file_path} with {encoding} encoding...")
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully read with {encoding} encoding")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading file with {encoding} encoding: {str(e)}")
            continue
    
    raise ValueError(f"Unable to read {file_path} with any of the attempted encodings")

def get_training_samples(train_csv_path):
    """获取训练集中每个类别的样本数量"""
    train_df = read_csv_with_encoding(train_csv_path)
    return dict(Counter(train_df['categoryName']))

class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super(FusionModel, self).__init__()
        self.resnet = nn.Sequential(
            *list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1]
        )
        self.fc1 = nn.Linear(2048, 512)
        self.dropout = nn.Dropout(0.5)
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc2 = nn.Linear(768, 512)
        
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, image, input_ids, attention_mask):
        img_features = self.resnet(image).view(image.size(0), -1)
        img_features = torch.relu(self.fc1(img_features))
        img_features = self.dropout(img_features)
        
        text_output = self.bert(input_ids, attention_mask=attention_mask)
        text_features = torch.relu(self.fc2(text_output.last_hidden_state[:, 0, :]))
        text_features = self.dropout(text_features)
        
        img_features = img_features.unsqueeze(1)
        text_features = text_features.unsqueeze(1)
        combined_features = torch.cat((img_features, text_features), dim=1)
        
        attn_output, _ = self.attention(combined_features, combined_features, combined_features)
        fused_features = attn_output.mean(dim=1)
        
        output = self.fc3(fused_features)
        return output

class TestDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, tokenizer=None):
        self.data = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['imgUrl'])
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
            
        title = row['title']
        inputs = self.tokenizer(
            title,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(row['label'], dtype=torch.long),
            'category_name': row['categoryName']
        }

def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
    return torch.utils.data.dataloader.default_collate(batch)

def calculate_metrics(class_predictions, test_samples, train_samples, min_test_samples=10):
    """计算每个类别的各项指标"""
    class_metrics = {}
    
    for category, stats in class_predictions.items():
        if stats['total'] >= min_test_samples:  # 只考虑测试样本充足的类别
            test_acc = stats['correct'] / stats['total']
            test_count = stats['total']
            train_count = train_samples.get(category, 0)
            
            # 计算综合得分：考虑准确率和样本数量
            sample_weight = np.log1p(test_count)  # 使用log1p避免log(1)=0的情况
            score = test_acc * sample_weight
            
            class_metrics[category] = {
                'accuracy': test_acc,
                'test_samples': test_count,
                'train_samples': train_count,
                'correct_predictions': stats['correct'],
                'score': score
            }
    
    return class_metrics

def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 加载标签映射
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label mapping file not found at {LABEL_MAP_PATH}")
        
    with open(LABEL_MAP_PATH, 'rb') as f:
        saved_label_info = pickle.load(f)
        label_encoder = saved_label_info['label_encoder']
        num_classes = saved_label_info['num_classes']
    print(f"Loaded label mapping with {num_classes} classes")

    # 2. 获取训练集样本统计
    print("Loading training set statistics...")
    training_samples = get_training_samples(TRAIN_CSV)
    print(f"Found {len(training_samples)} categories in training set")

    # 3. 加载测试数据
    print("Loading test data...")
    test_df = read_csv_with_encoding(TEST_CSV)
    print(f"Test data shape: {test_df.shape}")

    # 4. 检查并处理未知类别
    unknown_categories = set(test_df['categoryName']) - set(label_encoder.classes_)
    if unknown_categories:
        print(f"Warning: Found {len(unknown_categories)} categories in test set that were not in training set.")
        print("These categories will be skipped during evaluation.")
        test_df = test_df[~test_df['categoryName'].isin(unknown_categories)]

    # 5. 标签编码
    test_df['label'] = label_encoder.transform(test_df['categoryName'])
    
    # 6. 初始化模型并加载权重
    model = FusionModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Model loaded successfully")
    
    # 7. 创建数据加载器
    test_dataset = TestDataset(test_df, BASE_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0
    )
    
    # 8. 初始化评估指标
    class_predictions = {name: {'correct': 0, 'total': 0} for name in label_encoder.classes_}
    total_loss = 0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    
    # 9. 评估过程
    print("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if not batch:
                continue
                
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            categories = batch['category_name']
            
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            
            for pred, label, category in zip(preds.cpu(), labels.cpu(), categories):
                class_predictions[category]['total'] += 1
                if pred == label:
                    class_predictions[category]['correct'] += 1
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 10. 计算整体指标
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 11. 设置最小样本阈值并计算类别指标
    min_test_samples = 10
    class_metrics = calculate_metrics(
        class_predictions, 
        test_df['categoryName'].value_counts().to_dict(),
        training_samples,
        min_test_samples
    )
    
    # 12. 获取前200个类别
    top_200_classes = dict(sorted(
        class_metrics.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )[:200])
    
    # 13. 保存结果
    results = {
        'overall_metrics': {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1,
            'total_test_samples': len(test_df),
            'total_classes_in_test': len(class_metrics),
            'min_test_samples_threshold': min_test_samples
        },
        'top_200_classes': top_200_classes
    }
    
    # 14. 保存JSON格式结果
    output_file = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 15. 创建易读的文本报告
    report_file = os.path.join(RESULTS_DIR, 'top_200_classes_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"Top 200 Categories (Minimum {min_test_samples} test samples required)\n")
        f.write("=" * 120 + "\n")
        header = f"{'Category Name':<50} {'Accuracy':<10} {'Train Samples':<15} {'Test Samples':<15} {'Score':<10}\n"
        f.write(header)
        f.write("-" * 120 + "\n")
        
        for category, metrics in top_200_classes.items():
            line = f"{category:<50} {metrics['accuracy']:<10.4f} {metrics['train_samples']:<15} "
            line += f"{metrics['test_samples']:<15} {metrics['score']:<10.4f}\n"
            f.write(line)

    # 16. 打印总结
    print("\nEvaluation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Total test samples: {len(test_df)}")
    print(f"Classes with sufficient test samples (>={min_test_samples}): {len(class_metrics)}")
    print(f"\nDetailed results have been saved to:")
    print(f"1. JSON format: {output_file}")
    print(f"2. Text report: {report_file}")

if __name__ == '__main__':
    evaluate_model()