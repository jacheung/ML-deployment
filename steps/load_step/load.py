import tensorflow_datasets as tfds


def load_tensorflow_dataset(dataset_str: str, train_test_split: bool = True):
    if train_test_split is True:
        split = ['train', 'test']
    else:
        split = 'all'
    data, ds_info = tfds.load(
        dataset_str,
        split=split, shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return data