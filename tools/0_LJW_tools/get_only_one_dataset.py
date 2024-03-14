import json
import os


json_path = r"E:\LJW\Git\mmpose\tools\data\00_Tower_Dataset\1024\anns\tower_keypoints_test_2.json"




data = json.load(open(json_path,"r",encoding="utf-8"), strict=False)
a = 1

new_datat = {}
new_datat['info'] = data['info']
new_datat['licenses'] = data['licenses']
new_datat['categories'] = data['categories']
new_datat['images'] = data['images'][:2]
new_datat['annotations'] = data['annotations'][:2]

b = json.dumps(new_datat, ensure_ascii=False, indent=4)
f2 = open(json_path, 'w', encoding='utf-8')
f2.write(b)
f2.close()
