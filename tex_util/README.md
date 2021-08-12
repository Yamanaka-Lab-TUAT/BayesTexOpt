# Generation of synthetic crystallographic texture

## Overview
- These codes can be used for generating synthetic crystallographic texture containing Cube, Goss, S, Brass, Copper, and Random texture components. 
- Synthetic texture is produced by using the class named "Texture" in "tex.py". 

---
## Requirement
* numpy and Pillow are required. 

---
## Example
* Example of the class: tex_util.tex.Texture
```python
from tex_util.tex import Texture  # import
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

 if __name__ == '__main__':
    # 引数でvolumeを指定. 抽出する結晶方位数で, defaultで1000となる.
    original_texture = Texture(volume=1000)
    # add~(分散角度 degree, 体積分率 %) で優先方位を追加.
    original_texture.addBrass(10, 20)
    original_texture.addS(10, 20)
    original_texture.addCube(10, 30)
    # addRandom()では体積分率のみを指定.
    original_texture.addRandom(30)

    # 結晶方位を抽出. 抽出数は最初に指定した volume=1000
    initial_crystal_orientation = original_texture.sample()

    # 正極点図を保存. 拡張子まで入力. デフォルトでは{1 1 1}極点図. 画像サイズの指定も可能
    original_texture.savePoleFigure('file_name.png')
```
