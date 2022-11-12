import os
from PIL import Image

path = "/content/frames/video_frames/"
images_path = os.listdir(path)


for img_path in images_path:
    # take the image
    sample_image_path = path + img_path
    img = Image.open(sample_image_path)


    roi_dic = {
        "ROI_1": [502, 344, 61, 86],
        "ROI_2": [627, 427, 193, 93],
        "ROI_3":  [522, 654, 90, 85],
        "ROI_4": [704, 594, 90, 80],
        "ROI_5": [545, 577, 81, 75],
        "ROI_6": [460, 700, 52, 80],
    }

    for r in roi_dic:

        width, height = roi_dic[r][2], roi_dic[r][3]
        x, y = roi_dic[r][0], roi_dic[r][1]

        # Select area to crop
        area = (x, y, x+width, y+height)

        # Crop, show, and save image
        cropped_img = img.crop(area)
        cropped_img.save("croped_frames/"+ r +"_"+ img_path)