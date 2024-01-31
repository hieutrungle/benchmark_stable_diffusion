import os
import shutil
import PIL.Image
import numpy as np

def mkdir_with_clear(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    
def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def display_sample(sample, i, title=""):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    # plt.figure(figsize=(10, 10))
    image_dir = "./images"
    mkdir_if_not_exists(image_dir)
    image_pil.save(image_dir/f"{title}_image_at_step_{i}.png")
    # plt.close()