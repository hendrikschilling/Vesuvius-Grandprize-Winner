import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2
import argparse
from math import ceil
from joblib import Parallel, delayed

def tile_image(img, out_fn, x, w):
    tile = img[...,x:x+w]
    cv2.imwrite(out_fn, tile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='tile_imgs', usage='tile_imgs <cmd> <img1> <img2> ...')
    parser.add_argument('--split', nargs='+', help='split  imgs')
    # parser.add_argument('--merge', nargs='+', help='merge imgs into one')
    parser.add_argument('--overlap', default=1000)
    parser.add_argument('--tile-width', default=25000)
    parser.add_argument('--output', default=".")
    args = parser.parse_args()

    tile_w = args.tile_width
    eff_w = args.tile_width - args.overlap
    out = args.output
    
    os.makedirs(out, exist_ok=True)

    for fn in args.split:
        img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        print(img.shape)
        for n in range(ceil(img.shape[-1]/eff_w)):
            Parallel(n_jobs=-1)(delayed(tile_image)(img, out+"/"+os.path.basename(fn)+".split"+str(n*eff_w)+".tif", n*eff_w, tile_w) for n in range(ceil(img.shape[-1]/eff_w)))
