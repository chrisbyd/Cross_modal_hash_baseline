from  .coco2014 import load_dataset as load_coco
from  .mirflickr25k import load_dataset as load_flickr
from  .nuswide import load_dataset as load_nuswide
import torchvision.transforms as transforms


_dst_factory = {
    'coco' : load_coco,
    'flickr25' : load_flickr,
    'nuswide' : load_nuswide

}

mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


def load_data(name = 'coco', **kwargs):
    if name not in _dst_factory.keys():
        raise NotImplementedError("The dataset is not supported right now! please check the spelling")

    return _dst_factory[name](**kwargs)
