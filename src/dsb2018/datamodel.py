import hashlib
from functools import total_ordering

from torch.utils.data.sampler import WeightedRandomSampler, SequentialSampler

from dlutils.datasets import TrainingSplit

from scipy import ndimage
from fastai.dataset import *
from fastai.transforms import *
from skimage.measure import regionprops

from dsb2018.preprocessing import *

def mask_to_rle(self, mask):
    pixels = np.concatenate([[0], mask.T.flatten(), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)

def open_mask(fn):
    rgb_mask = open_image(fn)
    mask = rgb_mask[:,:,0]
    return mask

class Masks(object):
    def __init__(self, id_, masks=None, shape=None):
        if shape is None:
            assert masks is not None, "You need to provide shape if masks are empty"
            shape = masks[0].shape
        self.masks = masks
        self.shape = shape
        self.id = id_

    def __getitem__(self, item):
        return self.masks[item]

    def __len__(self): return len(self.masks)

    def to_rle(self):
        return [mask_to_rle(mask) for mask in self.masks]

    def color_aggregated_masks(self):
        v = np.zeros_like(self.shape)
        for i in range(len(self.masks)):
            v = np.maximum(v, self.masks[i].mask * (i + 1))
        return v

    def aggregated_masks(self):
        return np.stack([mask.mask for mask in self.masks]).max(0)

    def compute_bbox(self, i):
        regions = regionprops(self.masks[i])
        assert len(regions) == 1
        row1, col1, row2, col2 = regions[0].bbox
        return col1, row1, col2 - 1, row2 - 1

    @classmethod
    def from_files(cls, id, file_paths):
        return cls(id, masks=[open_mask(fn) for fn in file_paths])

class Sample(object):
    def __init__(self, id, x, y, frequency=1.0):
        self.id = id
        self.x = x
        self.y = y
        self.frequency = frequency

    def copy(self, x=None, y=None):
        x = x if x is not None else self.x
        y = y if y is not None else self.y
        return Sample(self.id, x, y)

def read_samples(path, has_masks=True, limit=None):
    data = []
    paths = list(path.glob('*'))
    if limit is not None:
        new_len = int(len(paths) * limit)
        print(f"Using limited dataset, {new_len} instead of {len(paths)}")
        paths = paths[:new_len]

    for image_dir in tqdm(paths):
        images = list(image_dir.glob('images/*'))
        assert len(images) == 1
        if has_masks:
            masks = list(image_dir.glob('masks/*'))
            assert len(masks) > 0
        else:
            masks = None
        id = image_dir.name
        data.append(Sample(id, open_image(images[0]), Masks.from_files(id, masks)))
    return data

class SamplesDataset(BaseDataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.ids = [s.id for s in samples]
        self.frequency = [s.frequency for s in self.samples]
        super().__init__(transform)

    def get_n(self): return len(self.samples)
    def get_sz(self): return self.transform.sz
    def get_x(self, i): return self.samples[i].x
    def get_y(self, i): return self.samples[i].y
    def is_reg(self): return True
    def get_c(self): return 1

def named_ds2tuple(ds): return (ds[0].trn, ds[1].val, ds[1].trn, ds[0].val, ds[1].tst, ds[0].tst)

class SamplesData(ImageData):
    @classmethod
    def from_samples(cls, samples, bs=64, tfms=(None, None), valid_size=0.1, test_size=0.0,
                  test_samples=None, num_workers=8, tmp_path='.'):
        assert isinstance(tfms[0], Transforms) and isinstance(tfms[1], Transforms), \
            "please provide transformations for your train and validation sets"
        paths = TrainingSplit.from_array(samples, valid_size=valid_size, test_size=test_size, key=lambda x: x.id)
        if test_samples is not None:
            paths = paths._replace(tst=test_samples)

        ds = [paths.map(SamplesDataset, transform=t) for t in tfms]
        datasets = named_ds2tuple(ds)
        return cls(tmp_path, datasets, bs, num_workers, classes=[])

    def get_dl(self, ds, shuffle):
        if ds is None: return None

        if shuffle:
            sampler = WeightedRandomSampler(ds.frequency, len(ds))
        else:
            sampler = SequentialSampler(ds)

        return DataLoader(ds, batch_size=self.bs, sampler=sampler,
            num_workers=self.num_workers, pin_memory=False)

    def resized(self, dl, targ, new_path):
        raise NotImplemented("yet")

    def to_gpu(self, squares): # todo change me
        def to_gpu1(im):
            nim = self.trn_ds.transform.norm(im, None)[0]
            return np.moveaxis(nim, -1, 0)
        return np.array([to_gpu1(im) for im in squares])

    def from_gpu(self, preds):
        return np.moveaxis(to_np(preds), 1, -1)