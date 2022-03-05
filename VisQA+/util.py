
MAX_WIDTH = [-1, 1066.666, 1169, 1069.25, 1600, 1066.666, 1066.666, 1023.28, 1066.666, 1142.67, 1035.22]
MAX_HEIGHT = [-1, 800, 800, 800, 774.98, 800, 800, 800, 800, 800, 800]

def compute_scale_factor(im, groupID):
    w, h = im.size
    print(w, h, groupID)
    if MAX_HEIGHT[groupID] / h < MAX_WIDTH[groupID] / w:
        scale_factor = MAX_HEIGHT[groupID] / h
    else:
        scale_factor = MAX_WIDTH[groupID] / w
    return scale_factor
