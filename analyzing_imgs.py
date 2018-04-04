import os
import numpy as np
from libtiff import TIFF
from PIL import Image
import cv2
from pathlib import Path
from numpy import random


weathers = ["Blizzard", "Christmas", "Clear", "Clearing", "Clouds", "ExtraSunny", "Foggy", "Neutral", "Overcast",
            "Raining", "Smog", "Snowing", "Snowlight", "ThunderStorm"]
times_names = ["N1", "M1", "M2", "AN", "EV"]
times_num = {22, 6, 7, 13, 19}

def num_directories(tiffimage):
    count = 1
    TIFF.setdirectory(tiffimage, 0)
    while not TIFF.lastdirectory(tiffimage):
        TIFF.setdirectory(tiffimage, count)
        count += 1
    TIFF.setdirectory(tiffimage, 0)
    return count

def swapChannels(image):

    h, w, c = np.shape(image)
    tmp = np.zeros((h,w,3), dtype=image.dtype)
    tmp[:, :, 0] = image[:, :, 2]
    tmp[:, :, 1] = image[:, :, 1]
    tmp[:, :, 2] = image[:, :, 0]

    return tmp

def get_img_from_directory(tiff_tmp, dir_num, dst_img):
    TIFF.setdirectory(tiff_tmp, dir_num)
    TIFF.readencodedstrip(tiff_tmp, 0, dst_img.ctypes.data, -1)

def saveImgs(indices, imgList, nameList, w_index, t_index, saving_path):
    # sampledIndices, image_comp_path_list, image_names_only, w_index, t_index, f_path
    for idx in indices:
        imgPath = imgList[idx]
        f_name = nameList[idx]
        pg_idx = w_index*5+t_index
        tiffimg = TIFF.open(str(imgPath))
        img = Image.open(str(imgPath))
        w = img.width
        h = img.height

        image = np.empty((h, w, 4), dtype=np.uint8)

        get_img_from_directory(tiffimg, dir_num=pg_idx, dst_img=image)
        image = swapChannels(image)
        debug_img = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

        cv2.imwrite(str(saving_path/str(f_name+".jpg")), debug_img)

def save_video(image_comp_path_list, w_index, t_index, f_path, v_fname):
    video_name = v_fname+'.avi'

    img = Image.open(str(image_comp_path_list[0]))
    orig_width = img.width
    orig_height = img.height


    height = int(orig_height*0.4)
    width = int(orig_width*0.4)
    video = cv2.VideoWriter(str(f_path/video_name), -1, 1, (width, height))
    print(str(f_path/video_name))

    for imgPath in image_comp_path_list:
        pg_idx = w_index * 5 + t_index
        tiffimg = TIFF.open(str(imgPath))


        image = np.empty((orig_height, orig_width, 4), dtype=np.uint8)
        get_img_from_directory(tiffimg, dir_num=pg_idx, dst_img=image)
        image = swapChannels(image)
        debug_img = cv2.resize(image, (width, height))
        video.write(debug_img)
        # print("here")
    cv2.destroyAllWindows()
    video.release()

cur_dir = os.path.dirname(os.path.realpath(__file__))  +  "/analysis/"

archive_folder = "/home/muaz/archives/data1/"
folders_comp = [os.path.join(archive_folder, x) for x in os.listdir(archive_folder) if os.path.isdir(os.path.join(archive_folder, x))]
folders_only = [x for x in os.listdir(archive_folder) if os.path.isdir(os.path.join(archive_folder, x))]

for fold_name, fold_comp in zip(folders_only, folders_comp):
    pg_idx = 0
    image_comp_path_list = [os.path.join(fold_comp, f) for f in os.listdir(fold_comp) if os.path.isfile(os.path.join(fold_comp, f))]
    image_names_only = [os.path.splitext(f)[0] for f in os.listdir(fold_comp) if os.path.isfile(os.path.join(fold_comp, f))]

    img_indices = [x for x in range(len(image_comp_path_list))]

    if not len(image_comp_path_list)> 0:continue
    for weather, w_index in zip(weathers, range(len(weathers))):
        for time, t_index in zip(times_names, range(len(times_names))):
            f_path = Path(cur_dir) / Path(fold_name) / Path(weather) / Path(time)
            video_f_path = Path(cur_dir) / Path(fold_name)
            if not os.path.exists(str(f_path)):
                os.makedirs(str(f_path))

            # sampledIndices = random.choice(img_indices, 10)
            # saveImgs(sampledIndices, image_comp_path_list, image_names_only, w_index, t_index, f_path)
            save_video(image_comp_path_list, w_index, t_index, video_f_path, weather+"_"+time)
            pg_idx+=1
        exit()

exit()

folder = folders_comp[0]

images = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
images.sort()
image = images[0]

tiffimg = TIFF.open(str(image))
img = Image.open(str(image))
w = img.width
h = img.height

image = np.empty((h, w, 4), dtype=np.uint8)

lastdir = num_directories(tiffimg) - 1

get_img_from_directory(tiffimg, dir_num=4, dst_img=image)

image = swapChannels(image)
debug_img = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

print(np.mean(debug_img, axis=(0, 1)))
print(np.std(debug_img, axis=(0, 1)))
# cv2.imshow("tmp0", debug_img[:,:,0])
# cv2.imshow("tmp1", debug_img[:,:,1])
# cv2.imshow("tmp2", debug_img[:,:,2])
# # cv2.imshow("tmp3", debug_img[:,:,3])
# cv2.waitKey()