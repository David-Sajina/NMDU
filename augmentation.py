import os
import cv2
import numpy as np

def augment_images(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    for picName in os.listdir(input_folder):
        if not picName.endswith('.jpg'):
            continue 

        pic_dir = os.path.join(input_folder, picName)
        pic = cv2.imread(pic_dir)

        augmentedPics = []

        #Brightness
        augmentedPics.append(cv2.convertScaleAbs(pic, alpha=1.2, beta=10))  #Brightness +
        augmentedPics.append(cv2.convertScaleAbs(pic, alpha=0.8, beta=10))  #Brightness -

        #Saturation
        hsv_image = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * 1.5, 0, 255)  #Sat+
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * 0.7, 0, 255)  #Sat-
        augmentedPics.append(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR))

        #Zoom
        height, width = pic.shape[:2]
        ZoomRation = 0.8
        Zoomed = cv2.resize(pic, (int(width * ZoomRation), int(height * ZoomRation)))
        augmentedPics.append(Zoomed)

        #Save pic
        name = os.path.splitext(picName)[0]
        for i, augmented in enumerate(augmentedPics):
            picName = f"{name}_augmented_{i+1}.jpg"
            picOut_dir = os.path.join(output_folder, picName)
            cv2.imwrite(picOut_dir, augmented)

        print(f"Save za {picName}")



#augment_images("Data-validation/Up", "Augmented-validation/Up")
#augment_images("Data-validation/Left", "Augmented-validation/Left")
#augment_images("Data-validation/Right", "Augmented-validation/Right")
augment_images("Data-validation/Novo", "Augmented-validation/NovoRight")
