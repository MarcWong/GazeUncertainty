import argparse
import os.path
import numpy as np
from glob import glob
from VisQA.preprocessing.parser.parse_element_labels import parse_element_label, combine_rows
from skimage.draw import polygon, polygon2mask
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--element_labels_dir", type=str, default=None)
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--max_overlap_ratio", type=float, default=1.)
    args = vars(parser.parse_args())

    with open('no_overlap_vis', 'w') as f:
        for img_path in glob(os.path.join(args['images_dir'], '*.png')):
            vis = os.path.basename(img_path)[:-4]
            element_labels = parse_element_label(os.path.join(args['element_labels_dir'], vis))
            element_labels = combine_rows(element_labels)

            with Image.open(img_path) as im:
                width, height = im.size
                canvas = np.zeros((height, width), dtype=np.bool8)
                overlap_pixel = 0

                for row in element_labels.iterrows():
                    polygon = np.array(row[1][3])
                    mask = polygon2mask((height, width), polygon[:, ::-1])
                    overlap_pixel += (mask & canvas).sum()
                    canvas = (mask & canvas) | (np.logical_not(canvas) & mask)

                overlap_pixel /= width*height
                if overlap_pixel <= args['max_overlap_ratio']:
                    f.write(f'{vis}\n')
                    print(f'Ratio of overlapping pixels: {overlap_pixel:.5f} - {img_path}')

                canvas = canvas.astype(float)
                canvas[canvas == 0] = 0.75

                im = np.array(im)
                im[:, :, 0] = im[:, :, 0] * canvas
                im[:, :, 1] = im[:, :, 1] * canvas
                im[:, :, 2] = im[:, :, 2] * canvas

                plt.imshow(im)
                plt.show()