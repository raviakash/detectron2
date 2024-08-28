from detectron2.data.datasets import register_coco_instances
import sys, os


sys.path.insert(0, os.path.join('/home', 'raviakash', 'codebase'))
#file structure
path = os.path.join('/home', 'raviakash', 'codebase')
# Specify the directory containing your LabelMe annotations
labelme_folder = os.path.join(path, 'TestCase')

# Paths to JSON annotation files and image directories
train_json_file = os.path.join(labelme_folder, 'TestCase', 'train_data', 'train_dataset.json')
val_json_file = os.path.join(labelme_folder, 'TestCase', 'validation_data', 'validation_dataset.json')
image_root = os.path.join(labelme_folder, 'TestCase')

# Register training and validation datasets
register_coco_instances("train_data", {}, train_json_file, f"{image_root}/train_data/images")
register_coco_instances("validation_data", {}, val_json_file, f"{image_root}/validation_data/images")
