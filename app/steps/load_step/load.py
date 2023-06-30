import tensorflow_datasets as tfds


def load_tensorflow_dataset_training(dataset_str: str):
    (xy_train, xy_test), ds_info = tfds.load(
        dataset_str,
        split=['train', 'test'], shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return (xy_train, xy_test)

def load_tensorflow_dataset_production(dataset_str: str):
    data, ds_info = tfds.load(
        dataset_str,
        split='all',
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return data