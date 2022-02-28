import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt

from os import makedirs
from glob import glob
from VisQA.preprocessing.parser.parse_element_labels import parse_element_label, combine_rows
from skimage.draw import polygon2mask
from PIL import Image, ImageDraw


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--element_labels_dir", type=str, default=None)
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--max_overlap_ratio", type=float, default=1.)
    args = vars(parser.parse_args())

    makedirs(os.path.join(os.path.dirname(args['images_dir']), 'src_bb'), exist_ok=True)

    with open('vis_whitelist', 'w') as f:
        for img_path in glob(os.path.join(args['images_dir'], '*.png')):
            vis = os.path.basename(img_path)[:-4]
            element_labels = parse_element_label(os.path.join(args['element_labels_dir'], vis))
            element_labels = combine_rows(element_labels)

            with Image.open(img_path) as im:
                width, height = im.size
                canvas = np.zeros((height, width), dtype=np.bool8)
                overlap = np.zeros((height, width), dtype=np.bool8)
                im_draw = ImageDraw.Draw(im)

                for row in element_labels.iterrows():
                    poly = np.array(row[1][3])
                    if len(poly) > 1:
                        im_draw.polygon(poly.flatten().tolist(), outline='blue')
                        mask = polygon2mask((height, width), poly[:, ::-1])
                        overlap |= (canvas & mask)
                        canvas |= mask

                overlap_ratio = overlap.sum() / (width*height)
                if overlap_ratio <= args['max_overlap_ratio']:
                    f.write(f'{vis}\n')
                    print(f'Ratio of overlapping pixels in {vis}: {overlap_ratio:.5f}')

                overlap = overlap.astype(float) * .5
                im = np.array(im)
                im[:, :, 1] = im[:, :, 1] * (1 - overlap)
                im[:, :, 2] = im[:, :, 2] * (1 - overlap)

                plt.imsave(os.path.join(os.path.dirname(args['images_dir']), 'src_bb', os.path.basename(img_path)), im)
