import copy

import cv2
import numpy as np
import os
import json
import openpyxl


# def compute_safe_region(box, other_boxes, img_width, img_height):
#     temp_list = []
#     for item in other_boxes:
#         if item != box:
#             temp_list.append(item)
#     [x1, y1, x2, y2] = box
#
#     other_boxes = temp_list
#
#
#     xc = (x1 + x2) / 2
#     yc = (y1 + y2) / 2
#
#     new_x1, new_y1, new_x2, new_y2 = 0, 0, img_width, img_height
#
#     for [x1_other, y1_other, x2_other, y2_other] in other_boxes:
#         xc_other = (x1_other + x2_other) / 2
#         if xc
#
#         yc_other = (y1_other + y2_other) / 2


# def compute_safe_region_(box, other_boxes, img_width, img_height):
#     temp_list = []
#     for item in other_boxes:
#         if item != box:
#             temp_list.append(item)
#     [x1, y1, x2, y2] = box
#
#     other_boxes = temp_list
#
#     x_c = (x1 + x2) // 2
#     y_c = (y1 + y2) // 2
#
#     x1 = x_c - 1
#     x2 = x_c + 1
#     y1 = y_c - 1
#     y2 = y_c + 1
#
#     # 拓展上方向
#     assert y1 > 0
#     low = 0
#     high = y1
#     medium = (high + low) / 2
#     while high - low > 1:
#         medium = (high + low) / 2
#         if is_overlap((x_c - 1, medium, x_c + 1, y_c + 1), other_boxes):
#             low = medium
#         else:
#             high = medium
#     y1 = medium
#
#     # 拓展下方向
#     assert y2 <= img_width
#     low = y2
#     high = img_height
#     medium = (high + low) / 2
#     while high - low > 1:
#         medium = (high + low) / 2
#         if is_overlap((x_c - 1, y_c - 1, x_c + 2, medium), other_boxes):
#             high = medium
#         else:
#             low = medium
#     y2 = medium
#
#     # 拓展左方向
#     assert x1 >= 0
#     left = 0
#     right = x1
#     medium = (left + right) / 2
#     while right - left > 1:
#         medium = (left + right) / 2
#         if_overlap = is_overlap((medium, y_c - 1, x_c + 2, y_c + 2), other_boxes)
#         if if_overlap:
#             left = medium
#         else:
#             right = medium
#     x1 = medium
#
#     # 拓展右方向
#     left = x2
#     right = img_width
#     medium = (left + right) / 2
#     while right - left > 1:
#         medium = (left + right) / 2
#         if is_overlap((x_c + 1, y_c + 1, medium, y_c + 2), other_boxes):
#             right = medium
#         else:
#             left = medium
#     x2 = medium
#
#     return [x1, y1, x2, y2]

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
        if is_overlap((x1, y1 + 1, x2, medium), other_boxes):
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
        if_overlap = is_overlap((medium, y1 + 1, x2, y2 - 1), other_boxes)
        if if_overlap:
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
        if is_overlap((x1 + 1, y1 + 1, medium, y2 - 1), other_boxes):
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

    new_dataset_img_id = 1200
    new_dataset_ann_id = 1200

    groups_series_2 = [
        ['1_12', '1_1', '1_2'],
        ['1_23', '1_2', '1_3'],
        ['1_34', '1_3', '1_4'],
        ['1_123', '1_1', '1_2', '1_3'],
        ['1_234', '1_2', '1_3', '1_4'],
        ['2_12', '2_1', '2_2'],
        ['2_23', '2_2', '2_3'],
        ['4_12', '4_1', '4_2'],
        ['4_23', '4_2', '4_3'],
        ['4_34', '4_3', '4_4'],
        ['4_123', '4_1', '4_2', '4_3'],
        ['4_234', '4_2', '4_3', '4_4'],
    ]

    ann_dir = r"../data/tower_dataset_12456_train_val_test/annotations"
    for dataset in ["train", "val", "test"]:
        json_path = os.path.join(ann_dir, r"0_keypoints_{}.json".format(dataset))
        data = json.load(open(json_path, "r", encoding="utf-8"), strict=False)
        images = data['images']
        anns = data["annotations"]

        new_data = copy.deepcopy(data)

        kt_name_list = data['categories'][0]['keypoint']

        new_data['images'] = []
        new_data['annotations'] = []

        for (image, ann) in zip(images, anns):
            assert image['id'] == ann['image_id']
            img_height = image['height']
            img_width = image['width']
            img_name = image['file_name']

            print("----------------------------------------")

            print(f"{image['id']}\t{img_name}")
            print("----------------------------------------")

            kts = np.array(ann['keypoints']).reshape([-1, 3])
            temp_dict = {}
            all_rect = []
            for (group_name, group_kt_list) in groups_data.items():
                temp_dict[group_name] = {"kts": {},
                                         "have": False,
                                         "outlines": [],
                                         "not cover": True}
                temp_lst = []
                for group_kt_name in group_kt_list:
                    [x, y, v] = kts[kt_name_list.index(group_kt_name)]
                    if v != 0:
                        temp_dict[group_name]['kts'][group_kt_name] = [x, y]
                        temp_lst.append((x, y))
                if len(temp_lst) != 0:
                    temp_dict[group_name]['have'] = True
                    temp_lst = np.array(temp_lst)
                    x1 = max(0, np.min(temp_lst[:, 0]) - 15)
                    y1 = max(0, np.min(temp_lst[:, 1]) - 15)
                    x2 = min(img_width, np.max(temp_lst[:, 0]) + 15)
                    y2 = min(img_height, np.max(temp_lst[:, 1] + 15))
                    temp_dict[group_name]['outlines'] = [x1, y1, x2, y2]
                    all_rect.append([x1, y1, x2, y2])
            if len(all_rect) < 2:
                continue
            img_dave_dir = r"../data/partial_imgs/{}".format(img_name.split(".")[0])
            if not os.path.exists(img_dave_dir):
                os.makedirs(img_dave_dir)
            img_dave_dir_2 = r"../data/partial_all_imgs"
            if not os.path.exists(img_dave_dir_2):
                os.makedirs(img_dave_dir_2)
            img = cv2.imread(
                os.path.join(r"E:\LJW\Git\mmpose\tools\data\tower_dataset_12456_train_val_test\imgs", img_name), 1)
            for (group_name, group_data) in temp_dict.items():
                if group_data['have']:
                    outline = group_data['outlines']

                    outline_new = compute_safe_region(outline, all_rect, img_width, img_height)
                    [x1, y1, x2, y2] = outline_new
                    if x1 > outline[0] + 5 or y1 > outline[1] + 5 or x2 < outline[2] - 5 or y2 < outline[3] - 5:
                        group_data["not cover"] = False
                    else:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        new_img = img[y1:y2, x1:x2, :]
                        new_img_name = img_name.split(".")[0] + f"_partial_{group_name}.JPG"

                        new_img_path = os.path.join(img_dave_dir, new_img_name)
                        cv2.imwrite(new_img_path, new_img)

                        new_img_path = os.path.join(img_dave_dir_2, new_img_name)
                        cv2.imwrite(new_img_path, new_img)

                        new_kts = copy.deepcopy(kts)
                        new_kt_names = group_data['kts'].keys()

                        for kt_idx in range(new_kts.shape[0]):
                            if kt_name_list[kt_idx] in new_kt_names and new_kts[kt_idx, 2] != 0:
                                new_kts[kt_idx, 0] -= x1
                                new_kts[kt_idx, 1] -= y1
                                new_kts[kt_idx, 2] = 2
                            else:
                                new_kts[kt_idx, 0] = 0
                                new_kts[kt_idx, 1] = 0
                                new_kts[kt_idx, 2] = 0

                        new_kts = np.reshape(new_kts, (-1)).tolist()

                        new_image = {
                            "height": y2 - y1,
                            "width": x2 - x1,
                            "id": new_dataset_img_id,
                            "file_name": new_img_name
                        }

                        new_ann = {
                            "id": new_dataset_ann_id,
                            "image_id": new_dataset_img_id,
                            'category_id': 1,
                            'iscrowd': 0,
                            'area': 1.0,
                            'num_keypoints': 0,
                            'segmentation': [
                                [
                                    int(max(outline[0] - x1, 0)),
                                    int(max(outline[1] - y1, 0)),
                                    int(min(outline[2] - x1, x2 - x1)),
                                    int(min(outline[3] - y1, y2 - y1))]

                            ],
                            'bbox': [
                                int(max(outline[0] - x1, 0)),
                                int(max(outline[1] - y1, 0)),
                                int(min(outline[2] - x1, x2 - x1)) - int(max(outline[0] - x1, 0)),
                                int(min(outline[3] - y1, y2 - y1)) - int(max(outline[1] - y1, 0))
                            ],
                            'keypoints': new_kts
                        }
                        new_data['images'].append(new_image)
                        new_data['annotations'].append(new_ann)
                        print("\t", new_dataset_img_id, new_img_name)
                        new_dataset_img_id += 1
                        new_dataset_ann_id += 1
                        a = 1

            for group in groups_series_2:
                group_name_new = group[0]
                group_item = group[1:]
                temp_need = []
                temp_not_need = []
                temp_need_kt_name = []
                for (group_name, group_data) in temp_dict.items():
                    if group_data['have']:
                        if group_name in group_item:
                            temp_need.append(temp_dict[group_name]['outlines'])
                            for g_name in group_data['kts']:
                                temp_need_kt_name.append(g_name)
                        else:
                            temp_not_need.append(temp_dict[group_name]['outlines'])
                if len(temp_need) > 1:
                    temp_need = np.array(temp_need)
                    outline = [
                        min(temp_need[:, 0]),
                        min(temp_need[:, 1]),
                        max(temp_need[:, 2]),
                        max(temp_need[:, 3])
                    ]
                    all_rect = temp_not_need

                    outline_new = compute_safe_region(outline, all_rect, img_width, img_height)
                    [x1, y1, x2, y2] = outline_new
                    if x1 > outline[0] + 5 or y1 > outline[1] + 5 or x2 < outline[2] - 5 or y2 < outline[3] - 5:
                        continue
                    else:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        new_img = img[y1:y2, x1:x2, :]
                        new_img_name = img_name.split(".")[0] + f"_partial_{group_name_new}.JPG"
                        new_img_path = os.path.join(img_dave_dir, new_img_name)
                        cv2.imwrite(new_img_path, new_img)

                        new_img_path = os.path.join(img_dave_dir_2, new_img_name)
                        cv2.imwrite(new_img_path, new_img)

                        new_kts = copy.deepcopy(kts)
                        new_kt_names = temp_need_kt_name

                        for kt_idx in range(new_kts.shape[0]):
                            if kt_name_list[kt_idx] in new_kt_names and new_kts[kt_idx, 2] != 0:
                                new_kts[kt_idx, 0] -= x1
                                new_kts[kt_idx, 1] -= y1
                                new_kts[kt_idx, 2] = 2
                            else:
                                new_kts[kt_idx, 0] = 0
                                new_kts[kt_idx, 1] = 0
                                new_kts[kt_idx, 2] = 0

                        new_kts = np.reshape(new_kts, (-1)).tolist()

                        new_image = {
                            "height": y2 - y1,
                            "width": x2 - x1,
                            "id": new_dataset_img_id,
                            "file_name": new_img_name
                        }

                        new_ann = {
                            "id": new_dataset_ann_id,
                            "image_id": new_dataset_img_id,
                            'category_id': 1,
                            'iscrowd': 0,
                            'area': 1.0,
                            'num_keypoints': 0,
                            'segmentation': [
                                [
                                    int(max(outline[0] - x1, 0)),
                                    int(max(outline[1] - y1, 0)),
                                    int(min(outline[2] - x1, x2 - x1)),
                                    int(min(outline[3] - y1, y2 - y1))]

                            ],
                            'bbox': [
                                int(max(outline[0] - x1, 0)),
                                int(max(outline[1] - y1, 0)),
                                int(min(outline[2] - x1, x2 - x1)) - int(max(outline[0] - x1, 0)),
                                int(min(outline[3] - y1, y2 - y1)) - int(max(outline[1] - y1, 0))
                            ],
                            'keypoints': new_kts
                        }
                        new_data['images'].append(new_image)
                        new_data['annotations'].append(new_ann)
                        print("\t", new_dataset_img_id, new_img_name)
                        new_dataset_img_id += 1
                        new_dataset_ann_id += 1

            # break
            #
            # exit()
        ann_save_dir = r"../data/partial_anns"
        if not os.path.exists(ann_save_dir):
            os.makedirs(ann_save_dir)
        save_path = os.path.join(ann_save_dir, f"0_keypoints_{dataset}.json")
        b = json.dumps(new_data, ensure_ascii=False, indent=4)
        f2 = open(save_path, 'w', encoding='utf-8')
        f2.write(b)
        f2.close()


def get_skeletons():
    skeletons = []
    s = ""
    for i in '12456':
        for row in open(r'E:\LJW\mmpose_\00_LJW\dataset/{}/keypoints_info/skeletons_{}.txt'.format(i, i), 'r',
                        encoding='utf-8').readlines():
            skeletons.append(row.strip().split())
            s = s + row
    f = open(r"skeletons_old.txt", "w", encoding="utf-8")
    f.write(s)
    f.close()
    return skeletons


def show_skeleton():
    skeletons = get_skeletons()
    partial_skeleton_dir = r"../data/partial_skeleton_imgs"
    if not os.path.exists(partial_skeleton_dir):
        os.makedirs(partial_skeleton_dir)
    for dataset in ['train', "val", "test"]:
        json_path = r"../data/partial_anns/0_keypoints_{}.json".format(dataset)
        data = json.load(open(json_path, "r", encoding="utf-8"), strict=False)
        images = data['images']
        anns = data["annotations"]

        kt_name_list = data['categories'][0]['keypoint']

        for (image, ann) in zip(images, anns):
            assert image['id'] == ann['image_id']
            img_height = image['height']
            img_width = image['width']
            img_name = image['file_name']
            img_path = r"../data/partial_all_imgs/{}".format(img_name)
            img = cv2.imread(img_path, 1) // 4
            kts = np.array(ann['keypoints']).reshape([-1, 3])
            [x1, y1, w, h] = ann['bbox']
            img = cv2.rectangle(img, (x1, y1), (w + x1, h + y1), color=(0, 0, 255), thickness=5)
            for [p1, p2] in skeletons:
                p1_idx = kt_name_list.index(p1)
                p2_idx = kt_name_list.index(p2)
                if kts[p1_idx, 2] * kts[p2_idx, 2] != 0:
                    x1 = kts[p1_idx, 0]
                    y1 = kts[p1_idx, 1]
                    x2 = kts[p2_idx, 0]
                    y2 = kts[p2_idx, 1]
                    img = cv2.line(img, (x1, y1), (x2, y2), thickness=4, color=(0, 255, 255))
                    img = cv2.circle(img, center=(x1, y1), radius=10, thickness=-1, color=(255, 255, 0))
                    img = cv2.circle(img, center=(x2, y2), radius=10, thickness=-1, color=(255, 255, 0))
            save_path = os.path.join(partial_skeleton_dir, img_name)
            cv2.imwrite(save_path, img)
            # exit()

    pass


if __name__ == '__main__':
    # main()
    show_skeleton()
