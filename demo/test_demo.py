import os

img_dir = r"C:\Users\concrete\Desktop\0000_SCI\test"
for file_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, file_name)
    s = r"E:\LJW\Git\mmpose\openmmlab\python.exe E:\LJW\Git\mmpose\demo\image_demo.py --img {}".format(img_path)
    os.system(s)