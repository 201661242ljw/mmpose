import json
import os
import shutil

import cv2
import numpy as np



# type_ = 2
for new_size in [256, 384, 512, 640, 768, 1024, 1280]:
# for new_size in [ 1280]:
    ann_dir = r"../00_new_dataset"
    save_dir = r"E:\LJW\Git\00_Tower_Dataset/{}".format(new_size)
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

            for type_ in range(1,4):
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
                if flag in already_draw_list:
                    continue

                img = cv2.imread(img_path, 1)

                new_img = cv2.resize(img, (new_width, new_height))
                # cv2.imwrite(os.path.join(save_dir, "imgs", img_name), new_img)
                new_img = new_img // 4
                pt_colors = [[0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 0], [255, 0, 255]]
                sk_colores = [[0, 255, 128], [0, 128, 255], [255, 0, 128], [128, 0, 255], [255, 8128, 0]]
                for [x1, y1, t1, x2, y2, t2, kt] in img_data[f"new_skeletons_{type_}"]:
                    new_img = cv2.circle(new_img, center=(x1,y1), color=pt_colors[t1-1], thickness=-1,radius=new_size // 128)
                    new_img = cv2.circle(new_img, center=(x2,y2), color=pt_colors[t2-1], thickness=-1,radius=new_size // 128)
                    new_img = cv2.line(new_img, pt1=(x1, y1), pt2=(x2, y2), color=sk_colores[kt-1], thickness=2)
                new_img = cv2.rectangle(new_img, pt1=(img_data["bbox"][0], img_data["bbox"][1]),
                                        pt2=(img_data["bbox"][2], img_data["bbox"][2]), color=(0, 255, 0), thickness=1)

                save_img_path = os.path.join(save_dir, "imgs_show", img_name.split(".")[0] + f"_{type_}.JPG")
                cv2.imwrite(save_img_path, new_img)
            already_draw_list.append(flag)
        save_json_path = os.path.join(save_dir, "anns", f"tower_info_{dataset}.json")
        b = json.dumps(data, ensure_ascii=False, indent=4)
        f2 = open(save_json_path, 'w', encoding='utf-8')
        f2.write(b)
        f2.close()
