from scipy import ndimage
from fastai.dataset import *
from fastai.transforms import *

def zero_borders(img):
    img[0, :] = 0.0
    img[:,  0] = 0.0
    img[-1, :] = 0.0
    img[:, -1] = 0.0
    return img

def binary_erosion(img):
    return ndimage.morphology.binary_erosion(img, border_value=0).astype(np.float32)

def calc_separators(agregated_mask, orig_masks):
    enlarged = [ndimage.binary_dilation(mask) for mask in orig_masks]
    separators = (np.sum(enlarged, axis=0) >= 2.0)
    agregated_mask[separators] = 0.0
    approx_weight = separators * 9.0 + 1.0
    return agregated_mask, approx_weight

def calc_weights(merged_mask, processed_masks, w0=10, q=5,):
    weight = np.zeros(merged_mask.shape)
    distances = np.array([cv2.distanceTransform((m == 0).astype(
        np.uint8), cv2.DIST_L2, 3) for m in processed_masks])
    shortest_dist = np.sort(distances, axis=0)
    # distance to the border of the nearest cell
    d1 = shortest_dist[0]
    # distance to the border of the second nearest cell
    d2 = shortest_dist[1] if len(shortest_dist) > 1 else np.zeros(d1.shape)
    weight = w0 * np.exp(-(d1+d2)**2/(2*q**2)).astype(np.float32)
    weight = 1 + (merged_mask == 0) * weight
    return weight

def pad_img(img, sz, mode='reflect'):
    #pad_width = [(int(np.ceil((sz-d)/2)),int(np.floor((sz-d)/2)))  for d in img.shape[:2]] +[(0,0)]
    pad_width = [(0, np.max([sz-d, 0])) for d in img.shape[:2]]
    if len(img.shape) == 3:
        pad_width += [(0,0)]
    return np.pad(img, pad_width, mode=mode)

class PadToSz(Transform): # suggestion to add this transformation to transfors.py
    """Pad image to have minimal size of sz"""
    def __init__(self, sz, tfm_y=TfmType.NO, sz_y=None, mode='reflect'):
        super().__init__(tfm_y)
        self.min_sz,self.sz_y = sz,sz_y
        if sz_y is None:
            self.sz_y = self.min_sz
        self.mode = mode

    def do_transform(self, x, is_y):
        return pad_img(x, self.sz_y if is_y else self.min_sz, mode=self.mode)