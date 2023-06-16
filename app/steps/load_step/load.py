import tensorflow_datasets as tfds


def load_tensorflow_dataset(dataset_str: str):
    (xy_train, xy_test), ds_info = tfds.load(
        dataset_str,
        split=['train', 'test'], shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return (xy_train, xy_test)