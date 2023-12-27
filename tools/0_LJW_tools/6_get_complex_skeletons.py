import openpyxl


temp_dict = {
    1: {
        "description": "前层边缘点和下层支撑点",
        "floors": [
            [[11, 11], [12, 12], [13, 13], [21, 21], [22, 22], [31, 31], [32, 32], [33, 33], [41, 41], [44, 44]],
            [[42, 42]],
            [[51, 51], [61, 61]]
        ],
        "name_pairs": [
            [["sk_1", "e_1"], ["sk_2", "e_2"]],
            [["sk_1", "e_1"]],
            [["sk_1", "e_1"], ["sk_4", "e_2"]]
        ]
    },
    2: {
        "description": "后层边缘点和下层支撑点",
        "floors": [
            [[11, 11], [12, 12], [13, 13], [21, 21], [22, 22], [31, 31], [32, 32], [33, 33], [41, 41], [44, 44]],
            [[42, 42]],
            [[51, 51], [61, 61]]
        ],
        "name_pairs": [
            [["sk_3", "e_3"], ["sk_4", "e_4"]],
            [["sk_4", "e_4"]],
            [["sk_5", "e_3"], ["sk_8", "e_4"]]
        ]
    },
    3: {
        "description": "前层边缘点和上层支撑点",
        "floors": [
            [[11, 11], [12, 12], [13, 13], [21, 21], [22, 22], [31, 31], [32, 32], [33, 33], [41, 41], [44, 44]],
            [[42, 42]],
            [[52, 51], [62, 61]]
        ],
        "name_pairs": [
            [["sk_5", "e_1"], ["sk_6", "e_2"]],
            [["sk_5", "e_1"]],
            [["sk_1", "e_1"], ["sk_4", "e_2"]]
        ]
    },
    4: {
        "description": "后层边缘点和上层支撑点",
        "floors": [
            [[11, 11], [12, 12], [13, 13], [21, 21], [22, 22], [31, 31], [32, 32], [33, 33], [41, 41], [44, 44]],
            [[42, 42]],
            [[52, 51], [62, 61]]
        ],
        "name_pairs": [
            [["sk_7", "e_3"], ["sk_8", "e_4"]],
            [["sk_8", "e_4"]],
            [["sk_5", "e_3"], ["sk_8", "e_4"]]
        ]
    },
    5: {
        "description": "边缘点前后相连",
        "floors": [
            [[11, 11], [12, 12], [13, 13], [21, 21], [22, 22], [23, 23], [31, 31], [32, 32], [33, 33], [41, 41],
             [44, 44], [51, 51], [61, 61]],
            [[42, 42]]
        ],
        "name_pairs": [
            [["e_4", "e_1"], ["e_3", "e_2"]],
            [["e_4", "e_1"]]
        ]
    },
    6: {
        "description": "前面下层支撑点水平相连",
        # "floors": [
        #     [[11, 11], [12, 12], [13, 13], [21, 21], [22, 22], [23, 23], [31, 31], [32, 32], [33, 33], [41, 31],[52, 51], [62, 61]],
        #     [[42, 42]]
        # ],
        # "name_pairs": [
        #     [["e_4", "e_1"], ["e_3", "e_2"]],
        #     [["e_4", "e_1"]]
        # ]
    },
    7: {
        "description": "后面下层支撑点水平相连"
    },
    8: {
        "description": "前面上层支撑点水平相连"
    },
    9: {
        "description": "后面上层支撑点水平相连"
    },
    10: {
        "description": "下层支撑点前后相连"
    },
    11: {
        "description": "上层支撑点前后相连"
    },
    12: {
        "description": "前层同层支撑点上下相连"
    },
    13: {
        "description": "后层同层支撑点上下相连"
    },
    14: {
        "description": "前层隔层支撑点上下相连"
    },
    15: {
        "description": "后层隔层支撑点上下相连"
    },
    16: {
        "description": "前前上隔层支撑点水平相连"
    },
    17: {
        "description": "前后上隔层支撑点水平相连"}
    ,
    18: {
        "description": "后前上隔层支撑点水平相连"
    },
    19: {
        "description": "后后上隔层支撑点水平相连"
    },
    20: {""
         "description": "前前下隔层支撑点水平相连"
         },
    21: {
        "description": "前后下隔层支撑点水平相连"
    },
    22: {
        "description": "后前下隔层支撑点水平相连"
    },
    23: {
        "description": "后后下隔层支撑点水平相连"
    },
    24: {
        "description": "前面非边缘绝缘子点和支撑点相连"
    },
    25: {
        "description": "后面非边缘绝缘子点和支撑点相连"
    },
    26: {
        "description": "非边缘绝缘子点前后相连"
    },
    27: {
        "description": "前下支撑点与前避雷线点"
    },
    28: {
        "description": "前上支撑点与前避雷线点"},
    29: {
        "description": "后下支撑点与后避雷线点"
    },
    30: {
        "description": "后上支撑点与后避雷线点"
    },
    31: {
        "description": "下前特殊支撑点-前边缘点"
    },
    32: {
        "description": "上前特殊支撑点-前边缘点"
    },
    33: {
        "description": "下后特殊支撑点-后边缘点"
    },
    34: {
        "description": "上后特殊支撑点-后边缘点"
    },
    35: {
        "description": "上层特殊支撑点前后相连"
    },
    36: {
        "description": "下层特殊支撑点前后相连"
    },
    37: {
        "description": "前面特殊支撑点下上相连"
    },
    38: {
        "description": "后面特殊支撑点下上相连"
    },
}

book = openpyxl.load_workbook(r"new_points.xlsx")
sh = book.create_sheet("Sheet7")
sh['A1'] = 'p1'
sh['B1'] = 'p2'
sh['C1'] = 'skeleton_type'

r_num = 2
for (skeleton_type, skeleton_data) in temp_dict.items():
    if "floors" in skeleton_data.keys():
        if len(skeleton_data['floors']) != 0:
            for (sk_floors, sk_name_pairs) in zip(skeleton_data['floors'], skeleton_data['name_pairs']):
                for [pt1_floor, pt2_floor] in sk_floors:
                    for [pt1_name, pt2_name] in sk_name_pairs:
                        sh[f'A{r_num}'] = f"{str(pt1_floor)[0]}_{str(pt1_floor)[1]}_{pt1_name}"
                        sh[f'B{r_num}'] = f"{str(pt2_floor)[0]}_{str(pt2_floor)[1]}_{pt2_name}"
                        sh[f'C{r_num}'] = skeleton_type
                        r_num += 1

book.save(r"new_points.xlsx")
