from pathlib import Path
from fastbook import *

path = Path('.')
def get_x(r): return path/r['fname']
def get_y(r): return r['labels'].split(' ')
dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock), 
                   get_x = get_x, 
                   get_y = get_y, 
                  item_tfms=RandomResizedCrop(460, min_scale=0.5),
                   batch_tfms=aug_transforms(size=224, min_scale=0.7))
dsets = dblock.datasets(multilables)

dls = dblock.dataloaders(multilables)
learn = cnn_learner(dls, resnet18, metrics=partial(accuracy_multi, thresh=0.2)).to_fp16()
learn.fine_tune(20)
