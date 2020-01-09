import os
from os import path
import glob
import imageio

def crop(x, patch_size=128):
    h, w, _ = x.shape
    hh = h // 2
    ww = w // 2
    pp = patch_size // 2

    tl = x[:hh + pp - 1, :ww + pp - 1]
    tr = x[:hh + pp - 1:, ww - pp:]
    bl = x[hh - pp:, :ww + pp - 1]
    br = x[hh - pp:, ww - pp:]

    return [tl, tr, bl, br]

def main():
    target_dir = 'REDS_deblur'
    dir_full = path.join(target_dir, 'train')
    dir_crop = path.join(target_dir, 'train_crop')

    for p, d, f in os.walk(dir_full):
        if not d:
            print(p)
            save_dir = p.replace(dir_full, dir_crop)
            os.makedirs(save_dir, exist_ok=True)
            for img_name in f:
                img = imageio.imread(path.join(p, img_name))
                imgs = crop(img)
                imgs_further = []
                for img in imgs:
                    imgs_further.extend(crop(img))

                name, _ = path.splitext(img_name)
                for idx, x in enumerate(imgs_further):
                    save_as = path.join(save_dir, name + '_{:0>2}.png'.format(idx))
                    imageio.imwrite(save_as, x)

if __name__ == '__main__':
    main()
