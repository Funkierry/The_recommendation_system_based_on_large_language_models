import pandas as pd

def print_class_distribution(file_path, dataset_name):
    df = pd.read_csv(file_path)
    
    class_counts = df['categoryName'].value_counts()
    
    print(f"\n{dataset_name} set class distribution:")
    print(class_counts.to_string())

if __name__ == "__main__":
    train_file = "dataset/train.csv"
    val_file = "dataset/val.csv"
    test_file = "dataset/test.csv"

    print_class_distribution(train_file, "Train")
    print_class_distribution(val_file, "Validation")
    print_class_distribution(test_file, "Test")
