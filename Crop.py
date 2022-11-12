import os
from PIL import Image



path = "/content/frames/video_frames/"
images_path = os.listdir(path)

# create a directory for cropped images
croped_dir= "../croped_framed/"
for i in range(1,7):
    if not os.path.exists(croped_dir + str(i) + "/"):
        os.makedirs(croped_dir + str(i) + "/")


for img_name in images_path[:10]:
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

    for r in range(0, 6):

        width, height = rois[r][2], rois[r][3]
        x, y = rois[r][0], rois[r][1]

        # Select area to crop
        area = (x, y, x+width, y+height)

        # Crop, show, and save image
        cropped_img = img.crop(area)
        cropped_img.save(croped_dir + str(r+1) + "/"  + img_name)