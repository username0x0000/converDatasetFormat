import cv2
import os
import numpy as np


class Image:
    def __init__(self, img_path, save_path):
        self.img_list = os.listdir(img_path)
        self.img_path = img_path
        self.save_path = save_path

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_path, self.img_list[idx]), 0)
        mask_img = {}
        for object_number in np.unique(img):
            mask_img[object_number] = np.zeros_like(img)
        for i in range(len(img)):
            for j in range(len(img[0])):
                object_num = img[i, j]
                mask_img[object_num][i, j] = 255
        for key in mask_img.keys():
            cv2.imwrite(os.path.join(self.save_path, f'{self.img_list[idx].split(".")[0]}_{key}.jpg'), mask_img[key])


def main():
    img_path = '/home/eagle/kyh_workspace/work/LGE_Segmentation_221008/label'
    image = Image(img_path, '.')
    image.draw_contour()


if __name__ == '__main__':
    main()