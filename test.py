from PIL import Image

from unet import Unet

if __name__ == "__main__":
    unet = Unet()
    name_classes    = ["background","cloud"]
    dir_origin_path = r"img"
    dir_save_path   = r"test_result"
    import os
    from tqdm import tqdm

    img_names = os.listdir(dir_origin_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path  = os.path.join(dir_origin_path, img_name)
            image       = Image.open(image_path)
            r_image     = unet.detect_image(image)
            r_image.save(os.path.join(dir_save_path, img_name))