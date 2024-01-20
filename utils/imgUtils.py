import os
import cv2
import pywt
import warnings
import numpy as np
import os.path as osp
import PIL.Image as Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.transforms import Compose,ToTensor,Resize

warnings.filterwarnings("ignore")

# histogram equalization
def his_equal(img: np.asarray, pixel_max=255):
    H, W, C = img.shape
    S = float(H * W * C)

    out = img.copy()

    sum_h = 0

    for i in range(0, pixel_max + 1):
        ind = np.where(img == i)
        sum_h += len(img[ind])

        z_prime = pixel_max / S * sum_h
        out[ind] = z_prime

    out = out.astype(np.uint8)

    return out

# histogram manipulation
def his_mani(img:np.asarray, s0:int, m0:int):
    m = np.mean(img)
    s = np.std(img)

    out = img.copy()

    # normalize
    out = s0 / s * (out - m) + m0
    out[out<0] = 0
    out[out>255] = 255

    out = out.astype(np.uint8)
    return out

# threshold (just one batch)
def threshold_denosing(img: np.asarray, mode='hard', value=None, substitute=None):
    img_ = img.reshape(1, -1)
    if value == None:
        value = (np.max(img_) + np.min(img_))/2
    print("threshold value:", value)
    if substitute == None:
        substitute = np.max(img_)
    print("threshold substitute:", substitute)
    if mode in ('soft', 'hard', 'greater', 'less'):
        soft_img = pywt.threshold(data=img, value=int(value), mode=mode, substitute=substitute)
    else:
        Exception("Threshold method just provide four mode 'soft', 'hard', 'greater', 'less'")

    return soft_img


# background subtraction
def back_subtraction(img, mode='MOG2'):
    if mode == 'MOG2':
        Sub = cv2.createBackgroundSubtractorMOG2()
    elif mode == 'KNN':
        Sub = cv2.createBackgroundSubtractorKNN()
    else:
        Exception("there is not this mode[MOG2, KNN]")

    afterProcess = Sub.apply(img.copy())

    return afterProcess

if __name__ == '__main__':
    def process_img(img: np.asarray):
        processing = Compose([
            Resize((512, 512)),
            ToTensor()
        ])
        return processing(img.copy()).unsqueeze(0)

    class dataset(Dataset):
        def __init__(self, img_path):
            self.img_list = os.listdir(img_path)
            self.img_path = img_path

        def __getitem__(self, index):
            img = Image.open(osp.join(self.img_path, self.img_list[index]))

            return np.asarray(img), self.img_list[index]

    data = dataset('/')

    for img, img_name in data:
        print("img shape:", img.shape)
        print("img_name:", img_name)
        plt.figure()
        img_ = his_mani(img, m0=128, s0=52)
        new_img = his_equal(img_, np.max(img.reshape(1, -1)))
        plt.imshow(new_img)
        plt.show()
