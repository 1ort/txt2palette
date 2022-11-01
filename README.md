# txt2palette
Automatic1111' Stable Diffusion webui custom script to generate palettes by text description.
The script takes the generated images and converts them into color palettes.

![изображение](https://user-images.githubusercontent.com/83316072/199355532-19ca2f9e-5cd4-4a9e-b240-59f0da3c6a66.png)
![изображение](https://user-images.githubusercontent.com/83316072/199355588-6f774906-08dc-4282-ad20-3320f4969aea.png)

## Installation & usage
Сlone or download this repo and put the `txt2palette.py` file in a `/scripts` folder of your webui base dir.

To use the KMeans algorithm, add a `scikit-learn` to the `requirements_versions.txt` file in the base folder.

After launching the interface, select txt2palette from the list of scripts on the txt2img tab.

## Examples
"Tokyo neon" Steps: 20, Sampler: Euler a, CFG scale: 7
![изображение](https://user-images.githubusercontent.com/83316072/199354639-00d23c0f-97ca-45cd-be09-1f55eca8b211.png)
![изображение](https://user-images.githubusercontent.com/83316072/199354662-42a8b5aa-ce0d-4b94-9df2-cc21e4774608.png)

"Underwater disco" Steps: 20, Sampler: Euler a, CFG scale: 7
![изображение](https://user-images.githubusercontent.com/83316072/199354870-67a27306-1f06-45d9-9e97-c2eb4c139f25.png)
![изображение](https://user-images.githubusercontent.com/83316072/199354880-32d6f595-3443-4eac-8874-6426d58caf8a.png)

"van gogh starry night" Steps: 20, Sampler: Euler a, CFG scale: 7
![изображение](https://user-images.githubusercontent.com/83316072/199355231-15f4858c-4abc-4651-907d-3886a99622f5.png)
![изображение](https://user-images.githubusercontent.com/83316072/199355261-87a33205-7f5d-40a0-8124-38e277bc317f.png)

## Improvement Ideas:
- Somehow save hex codes of palettes

## Credits
The most important part of the script is based on the sources of Pylette color extractor: https://github.com/qTipTip/Pylette
