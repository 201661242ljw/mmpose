import json

josn_path = r"../data/00_new_dataset/tower_info_val_2.json"

data = json.load(open(josn_path, "r", encoding="utf-8"), strict=False)
a = data['04_4_020_head_no_0_0.JPG']
josn_path = r"../data/00_new_dataset/tower_info_train_2.json"

data = json.load(open(josn_path, "r", encoding="utf-8"), strict=False)
b = data['04_1_056_head_no_0_0.JPG']

print(a)

a = 1
