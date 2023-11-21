from utils import *
import cv2
from PIL import Image
import os

n_images = 100 # number of input1 images

results_matrix_none = [[0] * n_images for _ in range(n_images)]
fp_none = 0
results_matrix_crop = [[0] * n_images for _ in range(n_images)]
fp_crop = 0
offset1 = (300, 0, 350, 350)


wm_imgs = [] # to store watermarked images
wm_keys = [] # to store keys
S_arrays = []
for ith_input in range(n_images):
    src_img = cv2.imread(f'input4/{ith_input}.jpg') # read each input image
    key_base = WMKey(src_img)  # Generate base key
    wm_img = WaterMarker.embedWM(src_img, key_base, eps=3)  # watermark key_base to each images
    wm_imgs.append(wm_img)

    key = WMKey(src_img)   # Generate another key
    key.wm_keypoints = key_base.wm_keypoints
    key.wm_descriptors = key_base.wm_descriptors
    wm_keys.append(key)  # Add it to the list
    S_arrays.append(key.S)

np.savez('crosscheck/3/wmkeylist.npz', S_arrays)

for i in range(n_images):
    for j in range(n_images):
        if WaterMarker.extractWM(wm_imgs[i], wm_keys[j]):
            results_matrix_none[i][j] += 1
            fp_none += 1
        cropped_wm_img = Attacker.cropImage(wm_imgs[i], offset1)
        try:
            if WaterMarker.extractWM(cropped_wm_img, wm_keys[j]):
                results_matrix_crop[i][j] += 1
                fp_crop += 1
        except:
            cv2.imwrite('input4/crop_wm.jpg', cropped_wm_img)
            with Image.open('input4/crop_wm.jpg') as img:
                resized_img = img.resize((768, 640))
                resized_img.save('input4/crop_wm.jpg')
            resized_img = cv2.imread('input4/crop_wm.jpg')
            if WaterMarker.extractWM(resized_img, wm_keys[j]):
                results_matrix_crop[i][j] += 1
                fp_crop += 1


#for i in range(n_images):
#    print(results_matrix_none[i])
np.save('crosscheck/3/results_matrix_none.npy', results_matrix_none)
print(fp_none)
#print('-----------------------------------')
#for i in range(n_images):
#    print(results_matrix_crop[i])
np.save('crosscheck/3/results_matrix_crop.npy', results_matrix_crop)
print(fp_crop)


