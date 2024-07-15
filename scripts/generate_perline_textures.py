import functools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm


class TextureGenerator:
    def __init__(self, imageSize):
        self.imageSize = imageSize

    def __generateGradientVectors(self, gradient_number):
        gradients = []
        for i in range(gradient_number):
            while True:
                x, y = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
                if x * x + y * y < 1:
                    gradients.append([x, y])
                    break
        return gradients

    def __normalizeGradientVectors(self, gradient_number, gradients):
        for i in range(gradient_number):
            x, y = gradients[i][0], gradients[i][1]
            length = np.sqrt(x * x + y * y)
            gradients[i] = [x / length, y / length]
        return gradients

    # The modern version of the Fisher-Yates shuffle
    def __generatePermutationsTable(self, gradient_number):
        permutations = [i for i in range(gradient_number)]
        for i in reversed(range(gradient_number)):
            j = np.random.randint(0, i + 1)
            permutations[i], permutations[j] = permutations[j], permutations[i]
        return permutations

    def getGradientIndex(self, x, y, permutations, gradient_number):
        return permutations[(x + permutations[y % gradient_number]) % gradient_number]

    def perlinNoise(self, x, y, gradients, permutations, gradient_number):
        qx0 = int(np.floor(x))
        qx1 = qx0 + 1

        qy0 = int(np.floor(y))
        qy1 = qy0 + 1

        q00 = self.getGradientIndex(qx0, qy0, permutations, gradient_number)
        q01 = self.getGradientIndex(qx1, qy0, permutations, gradient_number)
        q10 = self.getGradientIndex(qx0, qy1, permutations, gradient_number)
        q11 = self.getGradientIndex(qx1, qy1, permutations, gradient_number)

        tx0 = x - np.floor(x)
        tx1 = tx0 - 1

        ty0 = y - np.floor(y)
        ty1 = ty0 - 1

        v00 = gradients[q00][0] * tx0 + gradients[q00][1] * ty0
        v01 = gradients[q01][0] * tx1 + gradients[q01][1] * ty0
        v10 = gradients[q10][0] * tx0 + gradients[q10][1] * ty1
        v11 = gradients[q11][0] * tx1 + gradients[q11][1] * ty1

        wx = tx0 * tx0 * (3 - 2 * tx0)
        v0 = v00 + wx * (v01 - v00)
        v1 = v10 + wx * (v11 - v10)

        wy = ty0 * ty0 * (3 - 2 * ty0)
        return (v0 + wy * (v1 - v0)) * 0.5 + 1

    def makeTexture(self, texture=None):
        gradient_number = np.random.randint(64, 256)
        gradients = self.__generateGradientVectors(gradient_number)
        gradients = self.__normalizeGradientVectors(gradient_number, gradients)
        permutations = self.__generatePermutationsTable(gradient_number)

        if texture is None:
            texture = np.random.choice([self.cloud, self.marble, self.wood])

        lows = np.random.uniform(0.01, 0.99, (3,))
        highs = np.random.uniform(lows, np.random.choice([-1.0, 1.0], (3,)), (3,))
        scale = highs - lows
        x_coords, y_coords = np.meshgrid(np.arange(255), np.arange(255))
        idxs = np.stack((x_coords, y_coords), axis=-1)
        noise = functools.partial(
            self.perlinNoise,
            gradients=gradients,
            permutations=permutations,
            gradient_number=gradient_number,
        )
        texture = functools.partial(texture, noise=noise)
        vfunc = np.vectorize(texture)
        noise = vfunc(idxs[:, :, 0], idxs[:, :, 1])
        minn, maxx = np.min(noise), np.max(noise)
        normed_values = np.expand_dims((noise - minn) / (maxx - minn), -1)
        rgb_values = np.clip(lows + scale * normed_values, 0, 1)
        return (rgb_values * 255).astype(np.uint8)

    def async_gen_textures(self, num_to_generate):
        executor = ThreadPoolExecutor(max_workers=np.minimum(num_to_generate, 10))
        self._futures = [
            executor.submit(self.makeTexture) for _ in range(num_to_generate)
        ]

    def get_textures(self):
        imgs = [f.result() for f in self._futures]
        self._futures = []
        return imgs

    def fractalBrownianMotion(self, x, y, func):
        octaves = np.random.randint(2, 16)
        amplitude = 1.0
        frequency = 1.0 / self.imageSize
        persistence = 0.5
        value = 0.0
        for k in range(octaves):
            value += func(x * frequency, y * frequency) * amplitude
            frequency *= 2
            amplitude *= persistence
        return value

    def cloud(self, x, y, noise):
        return self.fractalBrownianMotion(8 * x, 8 * y, noise)

    def wood(self, x, y, noise):
        frequency = 1.0 / self.imageSize
        n = noise(4 * x * frequency, 4 * y * frequency) * 10
        return n - int(n)

    def marble(self, x, y, noise):
        frequency = 1.0 / self.imageSize
        n = self.fractalBrownianMotion(8 * x, 8 * y, noise)
        return (np.sin(16 * x * frequency + 4 * (n - 0.5)) + 1) * 0.5

    def save_texture(self, filename):
        image = self.makeTexture()
        Image.fromarray(image).save(filename)


if __name__ == "__main__":
    # Run this file to see an example of a generated texture
    imageSize = 512
    noise = TextureGenerator(imageSize)
    num_textures = 5771
    Parallel(n_jobs=7)(
        delayed(noise.save_texture)(f"textures/perlin/{i:04}.png")
        for i in tqdm(range(num_textures))
    )
