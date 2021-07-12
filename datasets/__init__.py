from  .coco2014 import load_dataset as load_coco
from  .mirflickr25k import load_dataset as load_flickr
from  .nuswide import load_dataset as load_nuswide



_dst_factory = {
    'coco' : load_coco,
    'flickr25' : load_flickr,
    'nuswide' : load_nuswide

}


def load_data(name = 'coco', **kwargs):
    if name not in _dst_factory.keys():
        raise NotImplementedError("The dataset is not supported right now! please check the spelling")

    return _dst_factory[name](**kwargs)
