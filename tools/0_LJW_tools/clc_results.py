import json
import os

import openpyxl


def get_transform_json():
    src_dir = r"E:\LJW\Git\mmpose\tools\data\00_Tower_Dataset\384\anns"

    out_data = {
        'name_2_type': {},
        'name_2_id': {},
        "id_2_name": {},
        "id_2_type": {}
    }

    for dataset in ['train', "val", "test"]:
        json_path = os.path.join(src_dir, f"tower_info_{dataset}.json")
        data = json.load(open(json_path, "r", encoding="utf-8"), strict=False)
        for (img_name, temp_dict) in data.items():
            tower_type = temp_dict['new_keypoints_3'][0][0][0]
            out_data['name_2_type'][img_name] = int(tower_type)

    for dataset in ['train', "val", "test"]:
        json_path = os.path.join(src_dir, f"tower_keypoints_{dataset}_2.json")
        data = json.load(open(json_path, "r", encoding="utf-8"), strict=False)['images']
        for temp_dict in data:
            img_name = temp_dict['file_name']
            img_id = temp_dict['id']
            out_data['name_2_id'][img_name] = img_id
            out_data['id_2_name'][img_id] = img_name
            out_data['id_2_type'][img_id] = out_data['name_2_type'][img_name]

        a = 1

    save_path = r"img_name_2_type.json"
    b = json.dumps(out_data, ensure_ascii=False, indent=4)
    f2 = open(save_path, 'w', encoding='utf-8')
    f2.write(b)
    f2.close()


def get_results_json():
    json_path = r"img_name_2_type.json"

    data = json.load(open(json_path, "r", encoding="utf-8"), strict=False)
    id_2_name = data['id_2_name']
    id_2_type = data['id_2_type']

    out_dict = {
        "types": {},
        "imgs": {}
    }

    out_dir = r"E:\LJW\Git\mmpose\tools\0_LJW_tools\predict_show\00_json_out"
    for file_name in os.listdir(out_dir):
        file_path = os.path.join(out_dir, file_name)
        temp_dict = json.load(open(file_path, "r", encoding="utf-8"), strict=False)
        img_id = str(temp_dict['img_id'])
        img_name = id_2_name[img_id]
        tower_type = id_2_type[img_id]

        t_dict = {
            "img_name": img_name,
            "tower_type": tower_type,
            "img_id": img_id,
            "kt1": temp_dict['kt1'],
            "kt2": temp_dict['kt2'],
            "kt3": temp_dict['kt3'],
            "sk": temp_dict['sk'],
        }

        if not tower_type in out_dict['types'].keys():
            out_dict['types'][tower_type] = []
        out_dict['types'][tower_type].append(t_dict)
        out_dict['imgs'][img_name] = t_dict
    save_path = r"results.json"
    b = json.dumps(out_dict, ensure_ascii=False, indent=4)
    f2 = open(save_path, 'w', encoding='utf-8')
    f2.write(b)
    f2.close()


def judge_resluts():
    json_path = r"results.json"
    data = json.load(open(json_path, "r", encoding="utf-8"), strict=False)

    book = openpyxl.Workbook()
    sh = book.active
    sh.title = "Sheet1"

    titles = [
        'img_id',
        'img_name',
        'tower_type',
        'AP1',
        'AR1',
        'AF1',
        'AP2',
        'AR2',
        'AF2',
        'AP3',
        'AR3',
        'AF3',
        'AP_SK',
        'AR_SK',
        'AF_SK',
    ]
    for c_num, title_ in enumerate(titles):
        sh[f'{chr(c_num + 65)}1'] = title_
    r_num = 2
    for (img_name, temp_dict) in data['imgs'].items():
        img_id = temp_dict['img_id']
        tower_type = temp_dict['tower_type']

        AP1 = temp_dict['kt1']['tp'] / (max(1, temp_dict['kt1']['tp'] + temp_dict['kt1']['fp']))
        AR1 = temp_dict['kt1']['tp'] / (max(1, temp_dict['kt1']['tp'] + temp_dict['kt1']['fn']))
        AF1 = 2 * AP1 * AR1 / (AP1 + AR1)

        AP2 = temp_dict['kt2']['tp'] / (max(1, temp_dict['kt2']['tp'] + temp_dict['kt2']['fp']))
        AR2 = temp_dict['kt2']['tp'] / (max(1, temp_dict['kt2']['tp'] + temp_dict['kt2']['fn']))
        AF2 = 2 * AP2 * AR2 / (AP2 + AR2)

        AP3 = temp_dict['kt3']['tp'] / (max(1, temp_dict['kt3']['tp'] + temp_dict['kt3']['fp']))
        AR3 = temp_dict['kt3']['tp'] / (max(1, temp_dict['kt3']['tp'] + temp_dict['kt3']['fn']))
        AF3 = 2 * AP3 * AR3 / (AP3 + AR3)

        AP_SK = temp_dict['sk']['tp'] / (max(1, temp_dict['sk']['tp'] + temp_dict['sk']['fp']))
        AR_SK = temp_dict['sk']['tp'] / (max(1, temp_dict['sk']['tp'] + temp_dict['sk']['fn']))
        AF_SK = 2 * AP_SK * AR_SK / max((AP_SK + AR_SK), 1)

        lst = [
            img_id,
            img_name,
            tower_type,
            AP1,
            AR1,
            AF1,
            AP2,
            AR2,
            AF2,
            AP3,
            AR3,
            AF3,
            AP_SK,
            AR_SK,
            AF_SK
        ]
        for c_num, item_ in enumerate(lst):
            sh[f'{chr(65 + c_num)}{r_num}'] = item_
        r_num += 1
    sh = book.create_sheet('Sheet2')
    titles = [
        'tower_type',
        'AP1',
        'AR1',
        'AF1',
        'AP2',
        'AR2',
        'AF2',
        'AP3',
        'AR3',
        'AF3',
        'AP_SK',
        'AR_SK',
        'AF_SK',
        'ACC',
        'NUM',
        'ACC_RATE'
    ]
    for c_num, title_ in enumerate(titles):
        sh[f'{chr(c_num + 65)}1'] = title_
    r_num = 2
    thresh_kt = 0.9
    thresh_sk = 0.85
    for (tower_type, tower_list) in data['types'].items():
        tp1 = 0
        fp1 = 0
        fn1 = 0
        tp2 = 0
        fp2 = 0
        fn2 = 0
        tp3 = 0
        fp3 = 0
        fn3 = 0
        tp_sk = 0
        fp_sk = 0
        num = 0
        acc_num = 0
        fn_sk = 0
        for temp_dict in tower_list:
            tp1 += temp_dict['kt1']['tp']
            fp1 += temp_dict['kt1']['fp']
            fn1 += temp_dict['kt1']['fn']
            tp2 += temp_dict['kt2']['tp']
            fp2 += temp_dict['kt2']['fp']
            fn2 += temp_dict['kt2']['fn']
            tp3 += temp_dict['kt3']['tp']
            fp3 += temp_dict['kt3']['fp']
            fn3 += temp_dict['kt3']['fn']
            tp_sk += temp_dict['sk']['tp']
            fp_sk += temp_dict['sk']['fp']
            fn_sk += temp_dict['sk']['fn']

            AP3_ = temp_dict['kt3']['tp'] / (max(1, temp_dict['kt3']['tp'] + temp_dict['kt3']['fp']))
            AR3_ = temp_dict['kt3']['tp'] / (max(1, temp_dict['kt3']['tp'] + temp_dict['kt3']['fn']))
            AF3_ = 2 * AP3_ * AR3_ / (AP3_ + AR3_)

            AP_SK_ = temp_dict['sk']['tp'] / (max(1, temp_dict['sk']['tp'] + temp_dict['sk']['fp']))
            AR_SK_ = temp_dict['sk']['tp'] / (max(1, temp_dict['sk']['tp'] + temp_dict['sk']['fn']))
            AF_SK_ = 2 * AP_SK_ * AR_SK_ / max((AP_SK_ + AR_SK_), 1)

            if AF3_ > thresh_kt and AF_SK_ > thresh_sk:
                acc_num += 1
            num += 1

        AP1 = tp1 / max(1, (tp1 + fp1))
        AR1 = tp1 / max(1, (tp1 + fn1))
        AF1 = 2 * (AP1 * AR1) / max(1e-4, (AP1 + AR1))

        AP2 = tp2 / max(1, (tp2 + fp2))
        AR2 = tp2 / max(1, (tp2 + fn2))
        AF2 = 2 * (AP2 * AR2) / max(1e-4, (AP2 + AR2))

        AP3 = tp3 / max(1, (tp3 + fp3))
        AR3 = tp3 / max(1, (tp3 + fn3))
        AF3 = 2 * (AP3 * AR3) / max(1e-4, (AP3 + AR3))

        AP_sk = tp_sk / max(1, (tp_sk + fp_sk))
        AR_sk = tp_sk / max(1, (tp_sk + fn_sk))
        AF_sk = 2 * (AP_sk * AR_sk) / max(1e-4, (AP_sk + AR_sk))

        lst = [
            tower_type,
            AP1,
            AR1,
            AF1,
            AP2,
            AR2,
            AF2,
            AP3,
            AR3,
            AF3,
            AP_sk,
            AR_sk,
            AF_sk,
            acc_num,
            num,
            acc_num / num
        ]
        for c_num, item_ in enumerate(lst):
            if isinstance(item_, float):
                lst[c_num] = round(item_, 4)
            sh[f'{chr(65 + c_num)}{r_num}'] = item_
        r_num += 1
        print(lst)
    book.save("result.xlsx")


def get_num():
    json_path = r"img_name_2_type.json"

    data = json.load(open(json_path, "r", encoding="utf-8"), strict=False)
    id_2_name = data['id_2_name']
    id_2_type = data['id_2_type']
    name_2_type = data['name_2_type']

    src_dir = r"E:\LJW\Git\mmpose\tools\data\00_Tower_Dataset\384\anns"

    lst = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]

    for dataset_idx, dataset in enumerate(['train', "val", "test"]):
        json_path = os.path.join(src_dir, f"tower_info_{dataset}.json")
        data = json.load(open(json_path, "r", encoding="utf-8"), strict=False)
        for (img_name, temp_dict) in data.items():
            type = name_2_type[img_name] - 1
            lst[dataset_idx][type] += 1
    print(lst[0])
    print(lst[1])
    print(lst[2])


if __name__ == '__main__':
    get_transform_json()
    get_results_json()
    judge_resluts()
    get_num()
