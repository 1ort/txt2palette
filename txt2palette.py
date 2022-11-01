import modules.scripts as scripts
import gradio as gr

from modules import images
from modules.processing import process_images
from modules.shared import opts
import numpy as np


class Script(scripts.Script):

    def title(self):
        return "txt2palette"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        palette_size = gr.Slider(minimum=1, maximum=64, step=1, value=0,
            label="Palette size")
        method = gr.Radio(choices=['Median cut', 'KMeans'], value='Median cut', label='Palette extraction method')
        sort_by = gr.Radio(choices=["luminance", "hue", "saturation", "value", "lightness"], value="luminance", label="Sort colors by")
        overwrite = gr.Checkbox(False, label="Overwrite existing files")
        return [palette_size, method, sort_by, overwrite]

    def run(self, p, palette_size, method, sort_by, overwrite):
        import colorsys
        from PIL import Image
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            if method == 'KMeans':
                print('"sklearn" library is not installed, switching the extraction method to Median cut.')
                method = "Median cut"

        class Color:
            luminance_weights = np.array([0.2126, 0.7152, 0.0722])

            def __init__(self, RGB, frequency):
                self.rgb = tuple([c for c in RGB])
                self.freq = frequency

            def display(self, w=50, h=50):
                """
                Displays the represented color in a w x h window.
                :param w: width in pixels
                :param h: height in pixels
                """

                img = Image.new("RGB", size=(w, h), color=self.rgb)
                img.show()

            def __lt__(self, other):
                return self.freq < other.freq

            def get_colors(self, colorspace="rgb"):
                """
                Get the color in terms of a colorspace (string).
                :param colorspace: rgb/hsv/hls
                :return: corresponding color values
                """
                colors = {"rgb": self.rgb, "hsv": self.hsv, "hls": self.hls}
                return colors[colorspace]

            @property
            def hsv(self):
                return colorsys.rgb_to_hsv(*self.rgb)

            @property
            def hls(self):
                return colorsys.rgb_to_hls(*self.rgb)

            @property
            def luminance(self):
                return np.dot(self.luminance_weights, self.rgb)

        class ColorBox:
            """
            Represents a box in the RGB color space, with associated attributes, used in the Median Cut algorithm.
            """
            def __init__(self, colors):
                """
                Initialize with a numpy array of RGB colors.
                :param colors: np.ndarray (width * height, 3)
                """

                self.colors = colors
                self._get_min_max()

            def _get_min_max(self):
                min_channel = np.min(self.colors, axis=0)
                max_channel = np.max(self.colors, axis=0)

                self.min_channel = min_channel
                self.max_channel = max_channel

            def __lt__(self, other):
                """
                Compare cubes by volume
                :param other:
                """
                return self.size < other.size

            @property
            def size(self):
                return self.volume

            def _get_dominant_channel(self):
                dominant_channel = np.argmax(self.max_channel - self.min_channel)
                return dominant_channel

            @property
            def average(self):
                """
                Returns the average color contained in ColorBox
                :return: [R, G, B]
                """

                return np.mean(self.colors, axis=0)

            @property
            def volume(self):
                return np.prod(
                    self.max_channel - self.min_channel,
                )

            def split(self):
                """
                Splits the ColorBox into two ColorBoxes at the median of the dominant color channel.
                :return: [ColorBox1, ColorBox2]
                """

                # get the color channel with highest range
                dominant_channel = self._get_dominant_channel()

                # sorting colors by the dominant channel
                self.colors = self.colors[self.colors[:, dominant_channel].argsort()]

                median_index = len(self.colors) // 2

                return [
                    ColorBox(self.colors[:median_index]),
                    ColorBox(self.colors[median_index:]),
                ]

        class Palette:
            def __init__(self, colors):
                """
                Initializes a color palette with a list of Color objects.
                :param colors: a list of Color-objects
                """

                self.colors = colors
                self.frequencies = [c.freq for c in colors]
                self.number_of_colors = len(colors)

            def get_image(self, w=50, h=50):
                img = Image.new("RGB", size=(w * self.number_of_colors, h))
                arr = np.asarray(img).copy()
                for i in range(self.number_of_colors):
                    c = self.colors[i]
                    arr[:, i * h : (i + 1) * h, :] = c.rgb
                img = Image.fromarray(arr, "RGB")
                return img

        def k_means_extraction(arr, height, width, palette_size):
            """
            Extracts a color palette using KMeans.
            :param arr: pixel array (height, width, 3)
            :param height: height
            :param width: width
            :param palette_size: number of colors
            :return: a palette of colors sorted by frequency
            """
            arr = np.reshape(arr, (width * height, -1))
            model = KMeans(n_clusters=palette_size)
            labels = model.fit_predict(arr)
            palette = np.array(model.cluster_centers_, dtype=int)
            color_count = np.bincount(labels)
            color_frequency = color_count / float(np.sum(color_count))
            colors = []
            for color, freq in zip(palette, color_frequency):
                colors.append(Color(color, freq))
            return colors
        
        def median_cut_extraction(arr, height, width, palette_size):
            """
            Extracts a color palette using the median cut algorithm.
            :param arr:
            :param height:
            :param width:
            :param palette_size:
            :return:
            """
            arr = arr.reshape((width * height, -1))
            c = [ColorBox(arr)]
            full_box_size = c[0].size
            # Each iteration, find the largest box, split it, remove original box from list of boxes, and add the two new boxes.
            while len(c) < palette_size:
                largest_c_idx = np.argmax(c)
                # add the two new boxes to the list, while removing the split box.
                c = c[:largest_c_idx] + c[largest_c_idx].split() + c[largest_c_idx + 1 :]
            colors = [Color(map(int, box.average), box.size / full_box_size) for box in c]
            return colors

        sort_methods = {
            "luminance": lambda c: c.luminance,
            "hue": lambda c: c.hsv[0],
            "saturation": lambda c: c.hsv[1],
            "value": lambda c: c.hsv[2],
            "lightness": lambda c: c.hls[2],
        }

        def extract_colors(image, palette_size=5, resize=True, mode="Median cut", sort_mode=None):
            """
            Extracts a set of 'palette_size' colors from the given image.
            :param image: PIL.Image object of path to Image file
            :param palette_size: number of colors to extract
            :param resize: whether to resize the image before processing, yielding faster results with lower quality
            :param mode: the color quantization algorithm to use. Currently supports K-Means (KM) and Median Cut (MC)
            :param sort_mode: sort colors by luminance, or by frequency
            :return: a list of the extracted colors
            """
            if isinstance(image, Image.Image):
                img = image
            else:
                img = Image.open(image)
            img = img.convert("RGB")
            if resize:
                img = img.resize((256, 256))
            width, height = img.size
            arr = np.asarray(img)

            if mode == "KMeans":
                colors = k_means_extraction(arr, height, width, palette_size)
            elif mode == "Median cut":
                colors = median_cut_extraction(arr, height, width, palette_size)
            else:
                raise NotImplementedError("Extraction mode not implemented!")

            if sort_mode in sort_methods:
                colors.sort(key=sort_methods.get(sort_mode), reverse=False)
            else:
                raise NotImplementedError("Sorting mode not implemented!")
            return Palette(colors)  


        if(not overwrite):
            basename = f"_palette_{palette_size}x"
        else:
            p.do_not_save_samples = True

        proc = process_images(p)

        #do not make palettes out of grids

        if len(proc.images) > 1:
            iter_offset = 1
            iter_num = len(proc.images) - 1
        else:
            iter_offset = 0
            iter_num = 1

        for i in range(iter_num):
            pal = extract_colors(proc.images[i+iter_offset], palette_size=palette_size, sort_mode=sort_by, mode=method)
            proc.images[i+iter_offset] = pal.get_image()

            images.save_image(proc.images[i+iter_offset], p.outpath_samples, basename,
            proc.seed + i, proc.prompt, opts.samples_format, info= proc.info, p=p)

        return proc