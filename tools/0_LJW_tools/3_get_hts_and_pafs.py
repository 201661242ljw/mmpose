import json
import os.path
import copy
import cv2
import numpy as np
import random
import shutil
import openpyxl
import colorsys


def main():
    # src_dir_img = r"E:\LJW\mmpose_\00_LJW\tower_dataset_12456_train_val_test\imgs"
    num = 0

    # ue_json_dir = r"E:\LJW\UE4\UAV\datasets\json"
    ue_json_dir = r"../data/UE4/json"
    ue_img_dir = r"../data/UE4\img"
    ue_img_list = os.listdir(ue_img_dir)
    ue_img_num = len(ue_img_list)
    rand = np.random.rand(ue_img_num)
    train_img_list = np.array(ue_img_list)[rand < 0.7].tolist()
    val_img_list = np.array(ue_img_list)[np.logical_and(rand < 0.8, rand >= 0.7)].tolist()
    test_img_list = np.array(ue_img_list)[rand >= 0.8].tolist()

    json_dir = r"../data/tower_dataset_12456_train_val_test/annotations"
    img_dir = r"../data/tower_dataset_12456_train_val_test/imgs"

    for dataset_idx, dataset in enumerate(['train', 'val', 'test']):
        temp_data = {}

        json_path = os.path.join(json_dir, "0_keypoints_{}.json".format(dataset))
        data = json.load(open(json_path, 'r', encoding='utf-8'), strict=False)

        keypoint_name_list = np.array(data['categories'][0]["keypoint"])

        images, annotations = data["images"], data["annotations"]

        for sheet_order in range(1, 4):
            points_data = {}
            book = openpyxl.load_workbook(r"new_points.xlsx")
            sh = book[f'Sheet{sheet_order * 2}']
            row_num = 2
            while sh[f"A{row_num}"].value is not None:
                new_name = sh[f"A{row_num}"].value
                type = sh[f"B{row_num}"].value
                points_data[new_name] = {"type": type,
                                         "points": []}
                c_num = 67
                while sh[f"{chr(c_num)}{row_num}"].value is not None:
                    points_data[new_name]["points"].append(sh[f"{chr(c_num)}{row_num}"].value)
                    c_num += 1
                row_num += 1
            b = json.dumps(points_data, ensure_ascii=False, indent=4)
            f2 = open(r"new_points.json", 'w', encoding='utf-8')
            f2.write(b)
            f2.close()

            skeleton_data = {}
            sh = book[f'Sheet{sheet_order * 2 + 1}']
            row_num = 2
            while sh[f"A{row_num}"].value is not None:
                p1 = sh[f"A{row_num}"].value
                p2 = sh[f"B{row_num}"].value
                t = sh[f"C{row_num}"].value
                skeleton_data[f"{row_num - 1}"] = {"p1": p1,
                                                   "p2": p2,
                                                   "type": t}
                row_num += 1
            b = json.dumps(skeleton_data, ensure_ascii=False, indent=4)
            f2 = open(r"new_skeletons.json", 'w', encoding='utf-8')
            f2.write(b)
            f2.close()

            if True:
                for image, annotation in zip(images, annotations):
                    # continue
                    assert image["id"] == annotation["image_id"]
                    keypoints = np.array(annotation["keypoints"]).reshape(-1, 3)
                    keypoints, real_keypoint_name_list = keypoints[keypoints[:, -1] != 0], keypoint_name_list[
                        keypoints[:, -1] != 0]
                    # if not real_keypoint_name_list[0][0] in ["1", "4", "6"]:
                    #     continue
                    image_name = image["file_name"]
                    if not image_name in temp_data.keys():
                        temp_data[image_name] = {}

                    temp_data[image_name]["file_path"] = os.path.join(img_dir, image_name)
                    temp_data[image_name]['width'] = image['width']
                    temp_data[image_name]['height'] = image['height']
                    temp_data[image_name]['old_keypoints'] = np.hstack(
                        [np.expand_dims(real_keypoint_name_list, axis=1), keypoints[:, :2]]).tolist()
                    temp_data[image_name][f'new_keypoints_{sheet_order}'] = []
                    temp_data[image_name][f'new_skeletons_{sheet_order}'] = []
                    temp_data[image_name]['if_ue'] = False
                    num += 1
                    real_keypoint_name_list = real_keypoint_name_list.tolist()
                    print(num)
                    xs = keypoints[:, 0]
                    ys = keypoints[:, 1]

                    # temp_data[image_name]['keypoints'] = []
                    # for keypoint_name, x, y in zip(real_keypoint_name_list, xs.tolist(), ys.tolist()):
                    #     keypoint_info = point_name_info[keypoint_name]
                    #     if keypoint_info["new_type"] is None:
                    #         continue
                    #     t = int(keypoint_info["new_type"][0])
                    #     temp_data[image_name]['keypoints'].append([keypoint_name, x, y, t])
                    #
                    # have_skeletons = []
                    # for [p1, p2, skeleton_type] in skeletons:
                    #     if not (p1 in real_keypoint_name_list and p2 in real_keypoint_name_list):
                    #         continue
                    #     index_1 = real_keypoint_name_list.index(p1)
                    #     index_2 = real_keypoint_name_list.index(p2)
                    #     have_skeletons.append(
                    #         [int(xs[index_1]), int(ys[index_1]), int(xs[index_2]), int(ys[index_2]), skeleton_type])
                    # temp_data[image_name]['skeletons'] = have_skeletons

                    # new_points
                    new_point_names = []
                    new_xs = []
                    new_ys = []
                    new_types = []

                    special_flag = 0

                    for (new_point_name, temp_dict) in points_data.items():
                        xs = []
                        ys = []

                        for p_name in temp_dict["points"]:
                            if p_name in real_keypoint_name_list:
                                xs.append(keypoints[real_keypoint_name_list.index(p_name)][0])
                                ys.append(keypoints[real_keypoint_name_list.index(p_name)][1])

                        if len(xs) != 0:
                            x = int(np.average(np.array(xs)))
                            y = int(np.average(np.array(ys)))
                            # for p_name in temp_dict["points"]:
                            #     if p_name in real_keypoint_name_list:
                            #         img = cv2.line(img, pt1=(x, y), pt2=(keypoints[real_keypoint_name_list.index(p_name)][0],
                            #                                              keypoints[real_keypoint_name_list.index(p_name)][1]),
                            #                        color=(255, 255, 255), thickness=2)

                            # text = new_point_name
                            # org = (x, y)
                            # fontFace = cv2.FONT_HERSHEY_SIMPLEX
                            # fontScale = 1
                            # color = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)][temp_dict['type'] - 1]
                            # thickness = 2

                            # img = cv2.circle(img, center=(x, y), radius=30, color=color, thickness=-1)
                            # cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
                            new_point_names.append(new_point_name)
                            new_xs.append(x)
                            new_ys.append(y)
                            new_types.append(temp_dict["type"])
                            temp_data[image_name][f'new_keypoints_{sheet_order}'].append(
                                [new_point_name, x, y, temp_dict["type"]])
                            if sheet_order == 2:
                                if new_point_name == "4_2_4":
                                    special_flag = 1
                            elif sheet_order == 3:
                                if new_point_name == "4_2_7":
                                    special_flag = 1


                    # skeleton
                    for (_, temp_dict) in skeleton_data.items():
                        p1 = temp_dict['p1']
                        p2 = temp_dict["p2"]
                        t = temp_dict["type"]
                        if special_flag and sheet_order == 2:
                            if p1 == "4_2_2" and p2 == "4_3_2":
                                continue
                        if special_flag and sheet_order == 3:
                            if p1 in ["4_2_2", "4_2_3", "4_2_4", "4_2_5"] and "4_3_" in p2:
                                continue

                        if p1 in new_point_names and p2 in new_point_names:
                            x1 = new_xs[new_point_names.index(p1)]
                            y1 = new_ys[new_point_names.index(p1)]
                            t1 = new_types[new_point_names.index(p1)]
                            x2 = new_xs[new_point_names.index(p2)]
                            y2 = new_ys[new_point_names.index(p2)]
                            t2 = new_types[new_point_names.index(p2)]
                            temp_data[image_name][f'new_skeletons_{sheet_order}'].append([x1, y1, t1, x2, y2, t2, t])

                            # color = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)][temp_dict['type'] - 1]

                    #         img = cv2.line(img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=5)
                    #
                    # cv2.imwrite(file_name.replace(".JPG", "_2.JPG"), img)

                    # img = cv2.imread(file_path, 1) // 8
                    # for keypoint, keypoint_name in zip(keypoints, real_keypoint_name_list):
                    #     x = keypoint[0]
                    #     y = keypoint[1]
                    # img = cv2.circle(img, center=(x, y), radius=2, color=(0, 255, 255), thickness=-1)
                    # text = keypoint_name
                    #
                    # org = (x, y)
                    # fontFace = cv2.FONT_HERSHEY_SIMPLEX
                    # fontScale = 0.5
                    # color = (0, 0, 255)
                    # thickness = 2
                    # cv2.putText(img, text, org, fontFace, fontScale, color, thickness)

                ue_img_dataset = [train_img_list, val_img_list, test_img_list][dataset_idx]

                for ue_img_name in ue_img_dataset:
                    num += 1
                    print(num)
                    ue_img_path = os.path.join(r"../data/UE4/IMG_", ue_img_name)

                    ue_img = cv2.imread(ue_img_path, 1) // 4
                    height = ue_img.shape[0]
                    width = ue_img.shape[1]

                    ue_ann_path = os.path.join(ue_json_dir, ue_img_name.replace("png", "json"))

                    ue_data = json.load(open(ue_ann_path, "r", encoding="utf-8"), strict=False)["points"]
                    if not ue_img_name in temp_data.keys():
                        temp_data[ue_img_name] = {}

                    temp_data[ue_img_name]["file_path"] = ue_img_path
                    temp_data[ue_img_name]['width'] = width
                    temp_data[ue_img_name]['height'] = height
                    temp_data[ue_img_name]['old_keypoints'] = []
                    temp_data[ue_img_name][f'new_keypoints_{sheet_order}'] = []
                    temp_data[ue_img_name][f'new_skeletons_{sheet_order}'] = []
                    temp_data[ue_img_name]['if_ue'] = True
                    new_point_names = []
                    new_xs = []
                    new_ys = []
                    new_types = []
                    for p_name in ue_data.keys():
                        temp_data[ue_img_name]['old_keypoints'].append(
                            [p_name, ue_data[p_name]['x'], ue_data[p_name]['y']])
                    for (new_point_name, temp_dict) in points_data.items():
                        xs = []
                        ys = []

                        for p_name in temp_dict["points"]:
                            if p_name in ue_data.keys():
                                xs.append(ue_data[p_name]['x'])
                                ys.append(ue_data[p_name]['y'])

                        if len(xs) != 0:
                            x = int(np.average(np.array(xs)))
                            y = int(np.average(np.array(ys)))
                            # for p_name in temp_dict["points"]:
                            #     if p_name in real_keypoint_name_list:
                            #         img = cv2.line(img, pt1=(x, y), pt2=(keypoints[real_keypoint_name_list.index(p_name)][0],
                            #                                              keypoints[real_keypoint_name_list.index(p_name)][1]),
                            #                        color=(255, 255, 255), thickness=2)

                            # text = new_point_name
                            # org = (x, y)
                            # fontFace = cv2.FONT_HERSHEY_SIMPLEX
                            # fontScale = 1
                            # color = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)][temp_dict['type'] - 1]
                            # thickness = 2

                            # img = cv2.circle(img, center=(x, y), radius=30, color=color, thickness=-1)
                            # cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
                            new_point_names.append(new_point_name)
                            new_xs.append(x)
                            new_ys.append(y)
                            new_types.append(temp_dict["type"])
                            temp_data[ue_img_name][f'new_keypoints_{sheet_order}'].append(
                                [new_point_name, x, y, temp_dict["type"]])

                    for (_, temp_dict) in skeleton_data.items():
                        p1 = temp_dict['p1']
                        p2 = temp_dict["p2"]
                        t = temp_dict["type"]
                        if p1 in new_point_names and p2 in new_point_names:
                            x1 = new_xs[new_point_names.index(p1)]
                            y1 = new_ys[new_point_names.index(p1)]
                            t1 = new_types[new_point_names.index(p1)]
                            x2 = new_xs[new_point_names.index(p2)]
                            y2 = new_ys[new_point_names.index(p2)]
                            t2 = new_types[new_point_names.index(p2)]
                            temp_data[ue_img_name][f'new_skeletons_{sheet_order}'].append([x1, y1, t1, x2, y2, t2, t])
                    a = 1

                #         ue_img = cv2.line(ue_img, pt1=(x1,y1),pt2=(x2,y2),color=(0,0,255), thickness=5)
                # cv2.imwrite(r"test3\{}".format(ue_img_name), ue_img)
                # exit()
        b = json.dumps(temp_data, ensure_ascii=False, indent=4)
        f2 = open(r"../data/00_new_dataset/tower_info_{}_2.json".format(dataset), 'w', encoding='utf-8')
        f2.write(b)
        f2.close()


def step2():
    # type_ = 2
    for new_size in [256, 384, 512, 640, 768, 1024, 1280]:
        # for new_size in [ 1280]:
        # ann_dir = r"E:\LJW\mmpose_\00_LJW\00_new_dataset"
        ann_dir = r"../data/00_new_dataset"
        # save_dir = r"E:\LJW\Git\00_Tower_Dataset/{}".format(new_size)
        save_dir = r"../data/00_Tower_Dataset/{}".format(new_size)

        if os.path.exists(os.path.join(save_dir, "imgs_show")):
            shutil.rmtree(os.path.join(save_dir, "imgs_show"))

        os.makedirs(os.path.join(save_dir, "imgs_show"))

        if not os.path.exists(save_dir):
            os.makedirs(os.path.join(save_dir, "anns"))
            os.makedirs(os.path.join(save_dir, "imgs"))

        already_draw_list = []
        num = 0
        for dataset in ["train", "val", "test"]:
            json_path = os.path.join(ann_dir, f"tower_info_{dataset}_2.json")
            data = json.load(open(json_path, "r", encoding="utf-8"), strict=False)
            for (img_name, img_data) in data.items():
                num += 1
                print(new_size, "  \t", num)
                img_path = img_data["file_path"]
                img_width = img_data["width"]
                img_height = img_data["height"]
                # if os.path.exists(save_img_path):
                #     continue
                old_points = np.array(img_data['old_keypoints'])
                first_point_name = old_points[0, 0]

                if "point" in first_point_name:
                    flag = "point"
                else:
                    flag = first_point_name[0]

                for type_ in range(1, 4):
                    new_points = np.array(img_data[f"new_keypoints_{type_}"])
                    new_skeletons = np.array(img_data[f"new_skeletons_{type_}"])

                    old_xs = old_points[:, 1].astype(int)
                    old_ys = old_points[:, 2].astype(int)

                    x1 = np.min(old_xs)
                    x2 = np.max(old_xs)
                    width = x2 - x1
                    y1 = np.min(old_ys)
                    y2 = np.max(old_ys)
                    height = y2 - y1

                    x1 = max(0, int(x1 - width * 0.05))
                    x2 = min(img_width - 1, int(x2 + width * 0.05))
                    y1 = max(0, int(y1 - height * 0.05))
                    y2 = min(img_height - 1, int(y2 + height * 0.05))

                    width = x2 - x1
                    height = y2 - y1
                    s = int(max(width, height) * 1.05)

                    new_width = int(img_width / s * new_size)
                    new_height = int(img_height / s * new_size)

                    old_xs = (old_xs / s * new_size).astype(int)
                    old_ys = (old_ys / s * new_size).astype(int)

                    new_points[:, 1:3] = (new_points[:, 1:3].astype(int) / s * new_size).astype(int)

                    new_skeletons[:, :2] = (new_skeletons[:, :2] / s * new_size).astype(int)
                    new_skeletons[:, 3:5] = (new_skeletons[:, 3:5] / s * new_size).astype(int)

                    img_data["width"] = new_width
                    img_data["height"] = new_height
                    img_data["bbox"] = [int(x1 / s * new_size), int(y1 / s * new_size), int(x2 / s * new_size),
                                        int(y2 / s * new_size)]
                    img_data['old_keypoints'] = np.vstack([old_points[:, 0], old_xs, old_ys]).T.tolist()
                    img_data[f"new_keypoints_{type_}"] = new_points.tolist()
                    img_data[f"new_skeletons_{type_}"] = new_skeletons.tolist()

                    # if flag in already_draw_list and not "scene" in img_name:
                    if flag in already_draw_list and  '04_4_020_head_no_0_0' not in img_name and "11_1_250_head_have_3_0" not in img_name:
                        continue

                    img = cv2.imread(img_path, 1)

                    new_img = cv2.resize(img, (new_width, new_height))
                    # cv2.imwrite(os.path.join(save_dir, "imgs", img_name), new_img)
                    new_img = new_img // 16
                    pt_colors = [[0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 0], [255, 0, 255]]

                    for [x1, y1, t1, x2, y2, t2, kt] in img_data[f"new_skeletons_{type_}"]:
                        new_img = cv2.circle(new_img, center=(x1, y1), color=pt_colors[t1 - 1], thickness=-1,
                                             radius=new_size // 256)
                        new_img = cv2.circle(new_img, center=(x2, y2), color=pt_colors[t2 - 1], thickness=-1,
                                             radius=new_size // 256)
                        new_img = cv2.arrowedLine(new_img, pt1=(x1, y1), pt2=((x2 + x1) // 2, (y1 + y2) // 2),
                                                  color=sk_colores[kt - 1], thickness=2)
                        new_img = cv2.line(new_img, pt1=(x1, y1), pt2=(x2, y2), color=sk_colores[kt - 1], thickness=1)
                    new_img = cv2.rectangle(new_img, pt1=(img_data["bbox"][0], img_data["bbox"][1]),
                                            pt2=(img_data["bbox"][2], img_data["bbox"][2]), color=(0, 255, 0),
                                            thickness=1)

                    save_img_path = os.path.join(save_dir, "imgs_show", img_name.split(".")[0] + f"_{type_}.JPG")
                    cv2.imwrite(save_img_path, new_img)
                already_draw_list.append(flag)
            save_json_path = os.path.join(save_dir, "anns", f"tower_info_{dataset}.json")
            b = json.dumps(data, ensure_ascii=False, indent=4)
            f2 = open(save_json_path, 'w', encoding='utf-8')
            f2.write(b)
            f2.close()


if __name__ == '__main__':
    scale = 8
    sigma = 2
    half_width = 1
    out_img_size = 512
    num_point_type = 4
    num_skeleton_type = 12
    ue_src_dir = r""
    bgr_colors = []
    hue_steps = 22  # 均分的色调数量
    saturation = 1.0  # 最大饱和度
    value = 1.0  # 最大明度
    for i in range(hue_steps):
        hue = i / hue_steps  # 计算色调
        rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)  # 转换为RGB颜色
        bgr_color = tuple([round(x * 255) for x in reversed(rgb_color)])  # 转换为BGR颜色
        bgr_colors.append(bgr_color)
    sk_colores = bgr_colors
    main()

    step2()
