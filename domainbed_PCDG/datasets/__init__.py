import torch
import numpy as np

from domainbed.datasets import datasets
from domainbed.lib import misc
from domainbed.datasets import transforms as DBT

def set_transforms(dataset, data_type, hparams, algorithms_class=None):
    '''
    Params:
            -- data_type : ['train', 'valid', 'test', 'mnist']
    '''
    assert hparams['data_augmentation']

    additional_data =False
    if data_type == 'train':
        dataset.transforms = {'x': DBT.aug}
        additional_data = True

    elif data_type == 'valid':
        if hparams['val_augment'] is False:
            dataset.transforms = {'x': DBT.basic}
        
        else:
            dataset.transforms = {'x': DBT.aug}

    elif data_type == 'test':
        dataset.transforms = {'x': DBT.basic}

    elif data_type == 'mnist':
        dataset.transforms = {'x': lambda x: x}

    else:
        raise ValueError(data_type)

    if additional_data and algorithms_class is not None:
        for key, transform in algorithms_class.transforms.items():
            dataset.transforms[key] = transform


def get_dataset(test_envs, args, hparams, algorithm_class=None):
    is_mnist = 'MNIST' in args.dataset
    dataset = vars(datasets)[args.dataset](args.data_dir)

    in_splits = []
    out_splits = []

    for env_i, env in enumerate(dataset):
        out, in_ = split_dataset(
            env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i),
        )
        if env_i in test_envs:
            in_type = 'test'
            out_type = 'test'

        else:
            in_type = 'train'
            out_type = 'train'

        set_transforms(in_, in_type, hparams, algorithm_class)
        set_transforms(out, out_type, hparams, algorithm_class)

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)

        else:
            in_weights, out_weights = None, None

        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    return dataset, in_splits, out_splits


class _SplitDataset(torch.utils.data.Dataset):

    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transforms = {}

        self.direct_return = isinstance(underlying_dataset, _SplitDataset)

    def __getitem__(self, key):
        if self.direct_return:
            return self.underlying_dataset[self.keys[key]]

        x, y = self.underlying_dataset[self.keys[key]]
        ret = {'y': y}

        for key, transform in self.transforms.items():
            ret[key] = transform(x)

        return ret

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    '''
    Return a pair of datasets
    First dataset has n datapoints and rest in the last dataset
    '''

    assert n <= len(dataset)
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)