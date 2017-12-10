import scipy.misc
import numpy as np
def get_image(file_name):
    image = scipy.misc.imread(file_name)
    image = image/127.5 - 1
    return image
    
def save_images(images, size, file_name):
    return scipy.misc.imsave(file_name, merge_images(images, size))

def merge_images(images, size):
    shape = images.shape
    h, w = shape[1], shape[2]
    imgs = np.zeros([h*size[0], w*size[1], 3])
    for idx, img in enumerate(images):
        i = idx % size[1]
        j = idx // size[0]
        imgs[j*h:j*h+h, i*w:i*w+w, :] = img
    return imgs
