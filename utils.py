def get_id(fname):
    fname = fname.replace('COCO_train2014_', '')
    fname = fname.replace('COCO_val2014_', '')
    fname = fname.replace('.jpg', '')
    return int(fname)