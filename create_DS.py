import os
import shutil


new_DS_path = r'C:\Users\lucas\github\Deep-Learning-Binary-Classification\Dataset'
train_dir = os.path.join(new_DS_path, "Train")
validation_dir = os.path.join(new_DS_path, "Validation")
test_dir = os.path.join(new_DS_path, "Test")
test_dir_glasses = os.path.join(test_dir, "Glasses")
test_dir_tables = os.path.join(test_dir, "Tables")
#print(train_dir)
