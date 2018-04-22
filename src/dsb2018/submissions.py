from functools import total_ordering

from dsb2018.datamodel import get_mask
from dsb2018.utils import show_image_list
from fastai.imports import *
from scipy import ndimage # TODO replace by cv2

from skimage import morphology
from dsb2018 import _flags
from fastai.transforms import scale_min
from numba import jit
from multiprocessing import Pool, TimeoutError

def get_mmask_from_pred(prediction, thr=0.5, dilation=True, debug=False):
    mmask = morphology.label(prediction > thr)
    if dilation:
        for i in range(1, mmask.max() + 1):
            mmask = np.maximum(mmask, ndimage.morphology.binary_dilation(mmask == i) * i)
        if debug:
            plt.imshow(mmask)
            plt.show()
    return mmask

def separate_masks(labeled_mask):
    n = int(labeled_mask.max())
    masks = list()
    for i in range(1, n + 1):
        m = labeled_mask == i
        if m.max() > 0:
            masks.append(m)
    if masks:
        return np.stack(masks)
    else:
        return np.zeros([0] + list(labeled_mask.shape))


def iou(masks1, masks2):
    max_ious1 = np.zeros((len(masks1)), dtype='float32')
    max_ious2 = np.zeros((len(masks2)), dtype='float32')

    bbox1 = [m.compute_bbox() for m in masks1]
    bbox2 = [m.compute_bbox() for m in masks2]
    area1 = [(m.mask > 0).sum() for m in masks1]
    area2 = [(m.mask > 0).sum() for m in masks2]
    for i, mask1 in enumerate(masks1):
        aarea = area1[i]
        abox = bbox1[i]
        for j, mask2 in enumerate(masks2):
            bbox = bbox2[j]
            if abox[0] > bbox[2] or abox[2] < bbox[0] or abox[1] > bbox[3] or abox[3] < bbox[1]:
                continue
            barea = area2[j]
            iarea = ((mask1.mask > 0) & (mask2.mask > 0)).sum()
            t = iarea / (aarea + barea - iarea)
            max_ious1[i] = max(t, max_ious1[i])
            max_ious2[j] = max(t, max_ious2[j])
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    avg = 0.0
    avg_tp, avg_fp, avg_fn = 0.0, 0.0, 0.0
    for threshold in thresholds:
        tp = (max_ious1 > threshold).sum()
        fp = len(masks1) - tp
        fn = len(masks2) - tp
        s = tp / (fp + fn + tp)
        avg += s
        avg_tp += tp
        avg_fp += fp
        avg_fn += fn
        # print(f"{threshold}: {tp}, {fp}, {fn}, {s:0.3f}")
    count = len(thresholds)
    avg /= count
    avg_tp /= count
    avg_fn /= count
    avg_tp /= count
    #     print(f"avg: {avg:0.3f} tp: {avg_tp:0.3f} tp: {avg_tp:0.3f} fn: {avg_fn:0.3f}")
    return avg


def acc(samples, gt_samples):
    assert len(samples) == len(gt_samples)
    avg_iou = 0.0
    for sample, gt_sample in zip(samples, gt_samples):
        assert sample.id == gt_sample.id
        #         print(sample.id)
        avg_iou += iou(sample.masks, gt_sample.masks)
    return avg_iou / len(samples)

#-------------------------------------------------------------------------------------------------
# submission
#



#
# test_image = learn.data.test_ds.get_x(0)
# squares, cuts, padded_shape, orig_shape = cut_to_squares(test_image, sz)
# tr_squares = to_nchw_norm(squares)
# nr_squares = [tfms[1].denorm(np.moveaxis(s, 0, -1)) for s in tr_squares]
# merged_image = merge_squares(nr_squares, cuts, padded_shape, orig_shape)
#
# print("Padding")
# print("Converted for model")
# show_image_list(tr_squares, NCHW=True)
# print("Converted back")
# show_image_list([test_image, merged_image, test_image - merged_image])
# print("sum(test_image - merged_image)", np.sum(test_image - merged_image))
#
# print("Scaling")
# squares, cuts, padded_shape, orig_shape = cut_to_squares(test_image, sz, reshape=scale_to_sz)
# tr_squares = to_nchw_norm(squares)
# nr_squares = [tfms[1].denorm(np.moveaxis(s, 0, -1)) for s in tr_squares]
# merged_image = merge_squares(nr_squares, cuts, padded_shape, orig_shape, reshape=scale_to_sz_r)
#
# print(merged_image.min(), merged_image.max())
# print("Converted for model")
# show_image_list(tr_squares, NCHW=True)
# print("Converted back")
# show_image_list([test_image, merged_image, np.clip(np.abs(test_image - merged_image), 0, 1)])
# print("sum(test_image - merged_image)", np.sum(test_image - merged_image))


# predict(list(TEST_PATH.glob("*")))
class MemoryDataset(object):
    def __init__(self, ids, images):
        self.ids = ids
        self.images = images

    def get_n(self):
        return len(self.images)

    def get_x(self, i):
        return self.images[i]


def pad_to_sz(img, sz):
    required_y = img.shape[1]
    required_y = (required_y // sz + 1) * sz

    flipped = img
    orig = img
    while flipped.shape[1] <= required_y:
        orig = np.flip(orig, axis=1)
        flipped = np.concatenate((flipped, orig), axis=1)
    flipped = flipped[:, :required_y]

    required_x = img.shape[0]
    required_x = (required_x // sz + 1) * sz
    orig = flipped
    while flipped.shape[0] <= required_x:
        orig = np.flip(orig, axis=0)
        flipped = np.concatenate((flipped, orig), axis=0)
    flipped = flipped[:required_x, :]
    return flipped, img.shape

def pad_to_sz2(img, sz, mode='reflect'):
    #pad_width = [(int(np.ceil((sz-d)/2)),int(np.floor((sz-d)/2)))  for d in img.shape[:2]] +[(0,0)]
    multi = [int(np.ceil(d//sz)) for d in img.shape[:2]]
    pad_width = [(0, np.max([m*sz-d, 0])) for m,d in zip(multi, img.shape[:2])]
    if len(img.shape) == 3:
        pad_width += [(0,0)]
    return np.pad(img, pad_width, mode=mode)

def pad_to_sz_r(img, shape):
    return img[:shape[0], :shape[1], :]


def scale_to_shape(img, shape):
    img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return img

class SubmissionPipline(object):
    def __init__(self, predict_batch, szs, scale=False):
        self.predict_batch = predict_batch
        self.szs = np.array([szs] if isinstance(szs, int) else szs)
        self.cutoff = 0.5

    def get_sz(self, shape):
        max_sz = max(shape[:2])
        candidates = self.szs[self.szs >= max_sz]
        if len(candidates) == 0:
            candidates = np.array([np.max(self.szs)])
        return np.min(candidates)

    def cut_to_squares(self, img, reshape=pad_to_sz):
        sz = self.get_sz(img.shape)
        img, orig_shape = reshape(img, sz)
        shape = img.shape[:2]
        max_x = shape[0] // sz + 1
        max_y = shape[1] // sz + 1
        images = []
        cuts = []
        for x in range(1, max_x):
            for y in range(1, max_y):
                cut = (slice((x - 1) * sz, x * sz), slice((y - 1) * sz, y * sz), slice(None))
                images.append(img[cut])
                cuts.append(cut)

        return images, [cuts, img.shape, orig_shape]

    def merge_squares(self, images, params, channels=None, reshape=pad_to_sz_r):
        cuts, padded_shape, orig_shape = params
        if channels is None:
            channels = padded_shape[2]
        img = np.zeros([padded_shape[0], padded_shape[1], channels])

        for i in range(len(images)):
            img[cuts[i]] = images[i]

        return reshape(img, orig_shape)

    def predict(self, ds, set_name='test'):
        mmasks = []
        for i in tqdm(range(0, ds.get_n())):
            sample = ds.samples[i]
            img = sample.image

            squares, params = self.cut_to_squares(img)
            preds = self.predict_batch(squares)

            mask = self.merge_squares(preds, params, channels=1)
            mmask = get_mmask_from_pred(mask, self.cutoff, dilation=_flags.USE_EROSION)

            mmask = sample.post_process_mask(mmask)
            mmasks.append(mmask)
        return Results(ds, mask, mmasks, set_name)

    # def predict(self, ds):
    #     preds_test = []
    #     for i in tqdm(ds.get_n()):
    #         img = ds.get_x(i)
    #         squares, cuts, corrected_shape, orig_shape = cut_to_squares(img, self.sz, reshape=scale_to_sz)
    #         preds = self.predict_batch(squares)
    #         mask = merge_squares(preds, cuts, corrected_shape, orig_shape, channels=1, reshape=scale_to_sz_r)
    #         preds_test.append(mask)
    #     return preds_test

# Run-length encoding from from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encode(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

# ref.: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    if type(mask_rle) == str:
        s = mask_rle.split()
    else:
        s = mask_rle
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape([shape[1], shape[0]]).T

#

class Results(object):

    def __init__(self, ds, preds, mmasks, set_name='test'):
        self.ds = ds
        self.preds = preds
        self.mmasks = mmasks
        self.set_name = set_name

    def show_prediction(self, id_):
        if id_ is str:
            index = self.ds.ids.index(id_)
        else:
            index = id_
        print(self.mmasks[index].shape)
        show_image_list([self.ds.get_x(index),  self.mmasks[index]])

    def convert_to_rle(self):
        new_test_ids = []
        rles = []
        for id_, mmask in tqdm(zip(self.ds.ids, self.mmasks)):
            rle = [rle_encode(mmask == i) for i in range(1, mmask.max() + 1)]
            rle = [r for r in rle if r]
            if len(rle) == 0:
                rle = [[1, 1]]
                print("No masks found on: ", id_)
            rles.extend(rle)
            new_test_ids.extend([id_] * len(rle))


        sub = pd.DataFrame()
        sub['ImageId'] = new_test_ids
        sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
        return sub

#
# def save_csv(kg, **params):
#     fname = kg.results_csv_fn(**params)
#     print("Submission file:", fname)
#
#     sub.to_csv(fname, index=False)