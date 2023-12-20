import cv2
import numpy as np
import os
import json
import openpyxl


def compute_safe_region(box, other_boxes, img_width, img_height):
    temp_list = []
    for item in other_boxes:
        if item != box:
            temp_list.append(item)
    [x1, y1, x2, y2] = box

    other_boxes = temp_list

    x_c = (x1 + x2) // 2
    y_c = (y1 + y2) // 2

    x1 = x_c - 1
    x2 = x_c + 1
    y1 = y_c - 1
    y2 = y_c + 1

    # 拓展上方向
    assert y1 > 0
    low = 0
    high = y1
    medium = (high + low) / 2
    while high - low > 1:
        medium = (high + low) / 2
        if is_overlap((x1, medium, x2, y2), other_boxes):
            low = medium
        else:
            high = medium
    y1 = medium

    # 拓展下方向
    assert y2 <= img_width
    low = y2
    high = img_height
    medium = (high + low) / 2
    while high - low > 1:
        medium = (high + low) / 2
        if is_overlap((x1, y1, x2, medium), other_boxes):
            high = medium
        else:
            low = medium
    y2 = medium

    # 拓展左方向
    assert x1 >= 0
    left = 0
    right = x1
    medium = (left + right) / 2
    while right - left > 1:
        medium = (left + right) / 2
        if is_overlap((medium, y1, x2, y2), other_boxes):
            left = medium
        else:
            right = medium
    x1 = medium

    # 拓展右方向
    left = x2
    right = img_width
    medium = (left + right) / 2
    while right - left > 1:
        medium = (left + right) / 2
        if is_overlap((x1, y1, medium, y2), other_boxes):
            right = medium
        else:
            left = medium
    x2 = medium

    return [x1, y1, x2, y2]


def is_overlap(box1, other_boxes):
    [x1, y1, x2, y2] = box1
    overlap = False
    for [x3, y3, x4, y4] in other_boxes:
        if not (x1 >= x4 or x2 <= x3 or y1 >= y4 or y2 <= y3):
            overlap = True
    return overlap


def column_number_to_name(column_number):
    column_name = ""
    while column_number > 0:
        remainder = (column_number - 1) % 26
        column_name = chr(65 + remainder) + column_name
        column_number = (column_number - 1) // 26
    return column_name


def get_kt_gtoups():
    book = openpyxl.load_workbook(r"kt_groups.xlsx")
    sh = book['Sheet1']
    groups = {}
    row_num = 1
    while sh[f"A{row_num}"].value is not None:
        temp_list = []
        c_num = 2
        while sh[f"{column_number_to_name(c_num)}{row_num}"].value is not None:
            temp_list.append(sh[f"{column_number_to_name(c_num)}{row_num}"].value)
            c_num += 1
        groups[sh[f"A{row_num}"].value] = temp_list
        row_num += 1
    return groups


def main():
    groups_data = get_kt_gtoups()

    ann_dir = r"../data/tower_dataset_12456_train_val_test/annotations"
    for dataset in ["train", "val", "test"]:
        json_path = os.path.join(ann_dir, r"0_keypoints_{}.json".format(dataset))
        data = json.load(open(json_path, "r", encoding="utf-8"), strict=False)
        images = data['images']
        anns = data["annotations"]

        kt_name_list = data['categories'][0]['keypoint']

        for (image, ann) in zip(images, anns):
            assert image['id'] == ann['image_id']
            img_height = image['height']
            img_width = image['width']
            img_name = image['file_name']

            kts = np.array(ann['keypoints']).reshape([-1, 3])
            temp_dict = {}
            all_rect = []
            for (group_idx, group_kt_list) in groups_data.items():
                temp_dict[group_idx] = {"kts": {},
                                        "have": False,
                                        "outlines": [],
                                        "not cover": True}
                temp_lst = []
                for group_kt_name in group_kt_list:
                    [x, y, v] = kts[kt_name_list.index(group_kt_name)]
                    if v != 0:
                        temp_dict[group_idx]['kts'][group_kt_name] = [x, y]
                        temp_lst.append((x, y))
                if len(temp_lst) != 0:
                    temp_dict[group_idx]['have'] = True
                    temp_lst = np.array(temp_lst)
                    x1 = max(0, np.min(temp_lst[:, 0]) - 15)
                    y1 = max(0, np.min(temp_lst[:, 1]) - 15)
                    x2 = min(img_width, np.max(temp_lst[:, 0]) + 15)
                    y2 = min(img_height, np.max(temp_lst[:, 1] + 15))
                    temp_dict[group_idx]['outlines'] = [x1, y1, x2, y2]
                    all_rect.append([x1, y1, x2, y2])

            img_dave_dir = r"../data/partial_imgs/{}".format(img_name.split(".")[0])
            if not os.path.exists(img_dave_dir):
                os.makedirs(img_dave_dir)
            img = cv2.imread(
                os.path.join(r"E:\LJW\Git\mmpose\tools\data\tower_dataset_12456_train_val_test\imgs", img_name), 1)
            for (group_idx_1, group_data_1) in temp_dict.items():
                if group_data_1['have']:
                    outline = group_data_1['outlines']

                    outline_new = compute_safe_region(outline, all_rect, img_width, img_height)
                    [x1, y1, x2, y2] = outline_new
                    if x1 > outline[0] + 5 or y1 > outline[1] + 5 or x2 < outline[2] - 5 or y2 < outline[3] - 5:
                        group_data_1["not cover"] = False
                    else:
                        new_img = img[int(y1):int(y2), int(x1):int(x2), :]
                        cv2.imwrite(os.path.join(img_dave_dir, img_name.split(".")[0] + "_{}.JPG".format(group_idx_1)),
                                    new_img)
            exit()


if __name__ == '__main__':
    main()
