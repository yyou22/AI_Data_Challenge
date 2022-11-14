import os
from PIL import Image
import cv2
import numpy as np

path = "./video_frames/"
images_path = os.listdir(path)

# create a directory for cropped images
croped_dir= "./croped_framed/"
for i in range(1,7):
    if not os.path.exists(croped_dir + str(i) + "/"):
        os.makedirs(croped_dir + str(i) + "/")


for img_name in images_path:
    # take the image
    sample_image_path = path + img_name
    img = Image.open(sample_image_path)


    rois = [
        [502, 344, 61, 86],
        [627, 427, 193, 93],
        [522, 654, 90, 85],
        [704, 594, 90, 80],
        [545, 577, 81, 75],
        [460, 700, 52, 80]
    ]

    for r in [4]:

        width, height = rois[r][2], rois[r][3]
        x, y = rois[r][0], rois[r][1]

        # Select area to crop
        area = (x, y, x+width, y+height)

        # Crop, show, and save image
        cropped_img = img.crop(area)

        # crop unuseful area in ROI 2 and 5
        img_padding = np.array(cropped_img)

        if r == 1:
            pts = np.array([[0,0],[0,35],[193,88],[193,0]])
            pts = np.array([pts])
            mask = np.zeros(img_padding.shape[:2], np.uint8)

            cv2.polylines(mask, pts, 1, 255)    
            cv2.fillPoly(mask, pts, 255)   
            cropped_img_ = cv2.bitwise_and(img_padding, img_padding, mask=mask)
            cv2.imwrite(croped_dir + str(r+1) + "/"  + img_name, cropped_img_)

        elif r == 4:
            pts = np.array([[0,50],[0,75],[81,75],[81,50]])
            pts = np.array([pts])
            mask = np.zeros(img_padding.shape[:2], np.uint8)

            cv2.polylines(mask, pts, 1, 255)    
            cv2.fillPoly(mask, pts, 255)   
            cropped_img_ = cv2.bitwise_and(img_padding, img_padding, mask=mask)
            cv2.imwrite(croped_dir + str(r+1) + "/"  + img_name, cropped_img_)

        else:
            cropped_img.save(croped_dir + str(r+1) + "/"  + img_name)

