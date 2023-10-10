from utils import *
import cv2

src_img_path = 'test_buf/image3.png'
wm_img_path = 'test_buf/wm.png'
crop_wm_img_path = 'test_buf/crop_wm.png'

src_img = cv2.imread(src_img_path)
wm_key = WMKey(src_img)



wm_img = WaterMarker.embedWM(src_img, wm_key, eps=3)
cv2.imwrite(wm_img_path, wm_img)

# compute and report PSNR after watermarking
psnr = Toolbox.computePSNR(src_img, wm_img)
print(f'PSNR is: {psnr:.2f} db')

print('Start extraction ...')
wm_img = cv2.imread(wm_img_path)

print('======================  Test 1: no attack  ======================')
print('------> extract from original wm_img')
WaterMarker.extractWM(wm_img, wm_key)
print('------> extract from original src_img')
WaterMarker.extractWM(src_img, wm_key)

print('======================  Test 2: crop attack ======================')
offset = (300, 300, 90, 90)
print('------> extract from cropped wm_img')
cropped_wm_img = Attacker.cropImage(wm_img, offset)
cv2.imwrite(crop_wm_img_path, cropped_wm_img)
WaterMarker.extractWM(cropped_wm_img, wm_key)
print('------> extract from cropped src_img')
cropped_src_img = Attacker.cropImage(src_img, offset)
WaterMarker.extractWM(cropped_src_img, wm_key)

print('======================  Test 3: scale attack ======================')
scale_ratio = 1.6
print('------> extract from scaled and recovered wm_img')
scaled_wm_img = Attacker.scaleImage(wm_img, scale_ratio)
WaterMarker.extractWM(scaled_wm_img, wm_key)
print('------> extract from scaled and recovered src_img')
scaled_src_img = Attacker.scaleImage(src_img, scale_ratio)
WaterMarker.extractWM(scaled_src_img, wm_key)