from pathlib import Path
import numpy as np
from scripts.train_gtsrb import dataset_paths,split_stratified


def test_dataset_mapping_is_integer_class_order(tmp_path):
    for cls in range(43):
        folder=tmp_path/str(cls);folder.mkdir()
        for i in range(3):(folder/f'{i}.png').write_bytes(b'')
    files,labels=dataset_paths(tmp_path)
    assert len(files)==129
    assert set(labels)==set(range(43))
    a,b=split_stratified(labels,.25,42)
    assert not(set(a)&set(b))
    assert len(a)+len(b)==129
    assert len(set(labels[a]))==43 and len(set(labels[b]))==43
