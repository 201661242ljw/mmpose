import os
import json
import shutil

json_path = r"E:\LJW\Git\coco\person_detection_results/COCO_val2017_detections_AP_H_56_person.json"
data_ = json.load(open(json_path, "r", encoding="utf-8"), strict=False)
image_id_to_bbox_idx= {}
for (idx, temp_dict) in enumerate(data_):
    image_id_to_bbox_idx[temp_dict['image_id']] = idx



keep_list = []
for dataset in ["train", "val"]:
    temp_data = {}
    json_path = r"E:\LJW\Git\coco\annotations/person_keypoints_{}2017.json".format(dataset)
    data = json.load(open(json_path, "r", encoding="utf-8"), strict=False)

    temp_data['info'] = data['info']
    temp_data['licenses'] = data['licenses']
    temp_data['categories'] = data['categories']

    images = data['images']
    img_idx_to_id = {}
    img_id_to_idx = {}
    for idx, temp_dict in enumerate(images):
        id = temp_dict['id']
        img_idx_to_id[idx] = id
        img_id_to_idx[id] = idx
    ann_idx_to_id = {}
    ann_id_to_idx = {}
    annotations = data['annotations']
    for idx, temp_dict in enumerate(annotations):
        id = temp_dict['image_id']
        ann_idx_to_id[idx] = id
        if id not in ann_id_to_idx.keys():
            ann_id_to_idx[id] = []
        ann_id_to_idx[id].append(idx)

    for (img_id, ann_idxes) in ann_id_to_idx.items():
        temp_data['images'] = [data['images'][img_id_to_idx[img_id]]]
        temp_data['annotations'] = []


        src = os.path.join(r"E:\LJW\Git\coco\images\{}2017".format(dataset),data['images'][img_id_to_idx[img_id]]['file_name'])
        dst = data['images'][img_id_to_idx[img_id]]['file_name']

        keep_list.append(img_id)

        shutil.copy(src, dst)


        for ann_idx in ann_idxes:
            temp_data['annotations'].append(data['annotations'][ann_idx])
        break

    json_save_path = r"person_keypoints_{}2017.json".format(dataset)
    b = json.dumps(temp_data, ensure_ascii=False, indent=4)
    f2 = open(json_save_path, 'w', encoding='utf-8')
    f2.write(b)
    f2.close()

keep_data = []
for keep_img_id in keep_list:
    if keep_img_id in image_id_to_bbox_idx.keys():
        keep_data.append(data_[image_id_to_bbox_idx[keep_img_id]])

json_save_path = r"E:\LJW\Git\mmpose\tools\data\coco\person_detection_results\COCO_val2017_detections_AP_H_56_person.json"
b = json.dumps(keep_data, ensure_ascii=False, indent=4)
f2 = open(json_save_path, 'w', encoding='utf-8')
f2.write(b)
f2.close()



