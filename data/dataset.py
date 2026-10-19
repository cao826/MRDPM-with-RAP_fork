import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import json
import os
import sys
from pathlib import Path
import torch
import numpy as np

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
 '.npy'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

def npy_loader(path):
    return np.load(path)


class NumpyDataset_mask(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[256, 256], loader=npy_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs_img = transforms.Compose([
                transforms.ToTensor(),
        ])
        self.tfs_mask = transforms.Compose([
                transforms.ToTensor(),
        ])
        self.loader = npy_loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]) + '.npy'
        
        img = self.tfs_img(self.loader('{}/{}/{}'.format(self.data_root, 'Tmax', file_name)))
        cond_image = self.tfs_img(self.loader('{}/{}/{}'.format(self.data_root, 'mCTA', file_name)))
        mask = self.tfs_mask(self.loader('{}/{}/{}'.format(self.data_root, 'mask', file_name)))
        mask = mask + 0

        img[img >1]=1
        img[img <0]=0
        cond_image[cond_image >1]=1
        cond_image[cond_image <0]=0

        cond_image = cond_image *np.concatenate([mask,mask,mask,mask], axis=0)
        mask_img = img*(1. - mask) + mask
        img = img * mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = file_name
        return ret 

    def __len__(self):
        return len(self.flist)


class MCTPDataset_mask(data.Dataset):
    """gan_bpm's mCTP slices (3 CTP phases, Tmax) as NumpyDataset_mask's dict (ctp_only_plan.md §5, §7, §10.1).

    Reads the root through gan_bpm's own `mctp` pipeline, so the input window, the brain
    envelope (and its cache) and the no-data rule are gan_bpm's, not re-derived here:

        cond_image  (3, H, W)  (gan_bpm input + 1)/2 = (clip(HU, 10, 325) - 10)/315, NOT masked:
                               the predictor reads the whole head, Network masks the U-Net's copy
        gt_image    (1, H, W)  clip(label, lo, hi) rescaled to [0, 1] (Tmax/12 s), 0 outside the envelope
        mask        (1, H, W)  gan_bpm's brain envelope: noise, the U-Net's condition, the sampling paste
        loss_mask   (1, H, W)  gan_bpm's `support`, the envelope minus no-data labels: the loss
        mask_image  (1, H, W)  for visualisation only
        path        str        '{case_id}_slice{idx:03d}.npy'

    `part` takes cases from a gan_bpm fold's `split.json` (§9):
        'train'    train_case_ids minus the monitor cases
        'monitor'  `monitor_cases_per_dataset` training patients per dataset (seeded), at most
                   `monitor_max_slices` slices, evenly spaced. For watching training only.
        'test'     val_case_ids, the fold gan_bpm held out

    gan_bpm's root is added to sys.path at run time, so DataLoader workers must fork (Linux's
    default). A spawned worker would not find `mctp`.
    """
    def __init__(self, mctp_root, gan_bpm_root, split_json, part, label_type='tmax', tmax_ceiling=12.0,
                 monitor_cases_per_dataset=1, monitor_max_slices=32, monitor_seed=0, mask_cache=True,
                 min_label_fg_frac=0.0, qc_drop_slice_idx=()):
        if part not in ('train', 'monitor', 'test'):
            raise ValueError("part must be 'train', 'monitor' or 'test', got {!r}".format(part))
        if str(gan_bpm_root) not in sys.path:
            sys.path.append(str(gan_bpm_root))  # appended, so MRDPM's own `models` still wins
        import pandas as pd
        from mctp.dataset import MCTPDataset
        from mctp.label_scale import label_bounds
        from mctp.mask_cache import MaskCache
        from mctp.transforms import LABEL_VALID_EPS_BY_TARGET, MASK_ENVELOPE, MASK_SUPPORT, build_transforms

        split = json.loads(Path(split_json).read_text())
        train_cases = [str(c) for c in split['train_case_ids']]
        test_cases = [str(c) for c in split['val_case_ids']]
        index = pd.read_parquet(Path(mctp_root) / '{}_index.parquet'.format(label_type))
        cases = index.groupby('case_id').first()
        dataset_of = {str(c): str(d) for c, d in cases['dataset'].items()}
        patient_of = ({str(c): str(p) for c, p in cases['patient_id'].items()} if 'patient_id' in cases
                      else {c: c for c in dataset_of})
        missing = sorted(set(train_cases + test_cases) - set(dataset_of))
        if missing:
            raise ValueError('{} lists {} case(s) the {} index lacks, e.g. {}'.format(split_json, len(missing), label_type, missing[:5]))
        shared = {patient_of[c] for c in train_cases} & {patient_of[c] for c in test_cases}
        if shared:
            raise ValueError('{}: patients in both train_case_ids and val_case_ids, e.g. {}'.format(split_json, sorted(shared)[:5]))

        rng = np.random.default_rng(monitor_seed)
        monitor_patients = set()
        for name in sorted({dataset_of[c] for c in train_cases}):
            patients = sorted({patient_of[c] for c in train_cases if dataset_of[c] == name})
            monitor_patients.update(patients[i] for i in rng.permutation(len(patients))[:monitor_cases_per_dataset])
        monitor_cases = [c for c in train_cases if patient_of[c] in monitor_patients]
        case_ids = {'train': [c for c in train_cases if patient_of[c] not in monitor_patients],
                    'monitor': monitor_cases, 'test': test_cases}[part]

        lo, hi = label_bounds(label_type, tmax_ceiling)
        # linear only: (y + 1)/2 is the [0, 1] label only under the linear label transform
        pre = {'label_transform': 'linear', 'label_lo': lo, 'label_hi': hi,
               'label_valid_eps': LABEL_VALID_EPS_BY_TARGET[label_type]}
        if mask_cache:
            pre['mask_cache'] = MaskCache(Path(mctp_root) / 'mask_cache')
        self.ds = MCTPDataset(str(mctp_root), case_ids, transform=build_transforms(pre, train=False),
                              label_type=label_type, min_label_fg_frac=min_label_fg_frac,
                              qc_drop_slice_idx=tuple(qc_drop_slice_idx))
        if part == 'monitor' and len(self.ds.rows) > monitor_max_slices:
            keep = np.unique(np.linspace(0, len(self.ds.rows) - 1, monitor_max_slices).round().astype(int))
            self.ds.rows = self.ds.rows.iloc[keep].reset_index(drop=True)
        self.part = part
        self.case_ids = sorted(set(self.ds.rows['case_id'].astype(str)))
        self.patient_ids = sorted({patient_of[c] for c in self.case_ids})
        self.mask_channels = (MASK_SUPPORT, MASK_ENVELOPE)

    def __getitem__(self, index):
        image, label, m = self.ds[index]        # (1, 3, H, W) and (1, H, W) in [-1, 1]; m (3, H, W)
        row = self.ds.rows.iloc[index]
        support, envelope = (m[i:i + 1] for i in self.mask_channels)
        gt = (label + 1.) / 2. * envelope

        ret = {}
        ret['gt_image'] = gt
        ret['cond_image'] = (image[0] + 1.) / 2.
        ret['mask_image'] = gt*(1. - envelope) + envelope
        ret['mask'] = envelope
        ret['loss_mask'] = support
        ret['path'] = '{}_slice{:03d}.npy'.format(row['case_id'], int(row['slice_idx']))
        return ret

    def __len__(self):
        return len(self.ds)