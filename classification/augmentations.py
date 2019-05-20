import cv2
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms 

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image 

class ConvertFromInts(object):
    def __call__(self, image):
        return image.astype(np.float32)

class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image

class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image

class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image

class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __call__(self, image):
        height, width, _ = image.shape
        while True:
            current_image = image

            w = random.uniform(0.6 * width, width)
            h = random.uniform(0.6 * height, height)

            # aspect ratio constraint b/t .5 & 2
            if h / w < 0.5 or h / w > 2:
                continue

            left = random.uniform(width - w)
            top = random.uniform(height - h)

            # convert to integer rect x1,y1,x2,y2
            rect = np.array([int(left), int(top), int(left+w), int(top+h)])

            # cut the crop from the image
            current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

            return current_image


class RandomMirror(object):
    def __call__(self, image):
        if random.randint(2):
            image = image[:, ::-1]
        return image

class Resize(object):
    def __init__(self, size=256):
        self.size = size

    def __call__(self, image):
        image = cv2.resize(image, (self.size, self.size))
        return image

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image):
        im = image.copy()
        im = self.rand_brightness(im)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im = distort(im)
        return self.rand_light_noise(im)

class Augmentation(object):
    def __init__(self, size=256):
        # self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            # ToAbsoluteCoords(),
            PhotometricDistort(),
            # Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            # ToPercentCoords(),
            Resize(self.size),
            # SubtractMeans(self.mean)
        ])

    def __call__(self, image):
        return self.augment(image)


if __name__ == '__main__':
    image = cv2.imread('000078.jpg')
    plt.figure()
    plt.subplot(121)
    plt.imshow(image[:,:,::-1])
    plt.xticks(())
    plt.yticks(())
    aug = Augmentation()
    image = aug(image)
    print(image.shape)
    image = image.astype(np.uint8)
    plt.subplot(122)
    plt.imshow(image[:,:,::-1])
    plt.xticks(())
    plt.yticks(())
    plt.show()



