import cv2
import os
import numpy as np


class Image:
    def __init__(self, img_path, save_path):
        self.img_list = [f for f in os.listdir(img_path) if f[-3:] == 'jpg']
        self.img_path = img_path
        self.save_path = save_path

    def get_contour(self, img_name=None, file_name=None):
        if img_name:
            img = cv2.imread(img_name)
            _, thr = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            result = []
            for i in contours:
                for j in i:
                    result.append(j[0])
            return result
        else:
            result = []
            for img_name in self.img_list:
                print(img_name)
                result.append(img_name)
                img_path = os.path.join(self.img_path, img_name)
                img = cv2.imread(img_path, 0)
                _, thr = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                with open(os.path.join(self.save_path, img_name[:-3]+'txt'), 'w') as f:
                    for i in contours:
                        for j in i:
                            f.write(f'{j[0][0]}, {j[0][1]}\n')
            return result

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
    img_path = '.'
    image = Image(img_path, './label_data')
    image.get_contour()


if __name__ == '__main__':
    main()