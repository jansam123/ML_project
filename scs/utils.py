import os

def get_label_dict(files):
    classes = sorted({
        "_".join(os.path.basename(file).split("_")[:-1])
        for file in files
    })
    return {cls: i for i, cls in enumerate(classes)}