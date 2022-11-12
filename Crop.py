import os
from PIL import Image

# create a directory for cropped images
croped_dir= "./croped_framed"
if not os.path.exists(croped_dir):
    os.makedirs(croped_dir)

path = "/content/frames/video_frames/"
images_path = os.listdir(path)


for img_path in images_path:
    # take the image
    sample_image_path = path + img_path
    img = Image.open(sample_image_path)


    roi_dic = {
        "ROI1_": [502, 344, 61, 86],
        "ROI2_": [627, 427, 193, 93],
        "ROI3_":  [522, 654, 90, 85],
        "ROI4_": [704, 594, 90, 80],
        "ROI5_": [545, 577, 81, 75],
        "ROI6_": [460, 700, 52, 80],
    }

    for r in roi_dic:

        width, height = roi_dic[r][2], roi_dic[r][3]
        x, y = roi_dic[r][0], roi_dic[r][1]

        # Select area to crop
        area = (x, y, x+width, y+height)

        # Crop, show, and save image
        cropped_img = img.crop(area)
        cropped_img.save(croped_dir + r + img_path)