import random

from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

IMAGE_SIZE = (255, 255)


def generate_solid_colour_image(filename):
    rgb = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    image = Image.new("RGB", IMAGE_SIZE, rgb)
    image.save(filename)


num_images = 5771
Parallel(n_jobs=7)(
    delayed(generate_solid_colour_image)(f"textures/solid_colours/{i:04}.png")
    for i in tqdm(range(num_images))
)
