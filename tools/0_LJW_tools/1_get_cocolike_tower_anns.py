import copy
import json
import os

import numpy as np

src_dir = r"E:\LJW\Git\00_Tower_Dataset"

json_path = r"../data/coco/annotations/person_keypoints_train2017.json"
template_data = json.load(open(json_path, "r", encoding="utf-8"), strict=False)

for input_size in [256, 384, 512, 640, 768, 1024, 1280]:
    img_id = 0
    ann_id = 0
    for dataset in ['train', 'val', 'test']:
        output_data = copy.deepcopy(template_data)
        output_data['categories'] = [{'supercategory': "tower",
                                      'id': 0,
                                      'name': 'tower',
                                      'keypoints': ['edgepoint',
                                                    'skeletonpoints',
                                                    'edgepoint_2',
                                                    'bileipoint',
                                                    'skeletonpoint_2'],
                                      'skeleton': []
                                      }]
        output_data['images'] = []
        output_data['annotations'] = []

        json_path = r"{}/{}/anns/tower_info_{}.json".format(src_dir, input_size, dataset)
        data = json.load(open(json_path, "r", encoding="utf-8"), strict=False)

        for (img_name, temp_dict) in data.items():
            width = temp_dict['width']
            height = temp_dict['height']
            keypoint_list = [width // 2, height // 2, 2] * 5
            [x1, y1, x2, y2] = temp_dict['bbox']

            img_dict = {
                "license": 1,
                "file_name": img_name,
                "coco_url": "",
                "height": height,
                "width": width,
                "date_captured": "",
                "flickr_url": "",
                "id": img_id
            }

            ann_dict = {
                "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]],
                "true_points": np.array(temp_dict["new_keypoints_1"])[:, 1:].astype(np.int32).tolist(),
                "true_skeletons": temp_dict["new_skeletons_1"],
                # "true_points_2": np.array(temp_dict["new_keypoints_2"])[:, 1:].astype(np.int32).tolist(),
                # "true_skeletons_2": temp_dict["new_skeletons_2"],
                # "true_points_3": np.array(temp_dict["new_keypoints_3"])[:, 1:].astype(np.int32).tolist(),
                # "true_skeletons_3": temp_dict["new_skeletons_3"],
                "num_keypoints": 5,
                "area": int((x2 - x1) * (y2 - y1)),
                "iscrowd": 0,
                "keypoints": keypoint_list,
                "image_id": img_id,
                "bbox": [x1, y1, x2-x1, y2-y1],
                "category_id": 1,
                "id": ann_id
            }
            output_data['images'].append(img_dict)
            output_data['annotations'].append(ann_dict)

            img_id += 1
            ann_id += 1

        out_json_path =  r"{}/{}/anns/tower_keypoints_{}.json".format(src_dir, input_size, dataset)
        f = open(out_json_path, "w",encoding="utf-8")
        f.write(json.dumps(output_data, ensure_ascii=False, indent=4))
        f.close()


