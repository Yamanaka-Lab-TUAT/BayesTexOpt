# Deep Neural Network-based Numerical Material Test
-----

## Description
tensorflowとkerasで実装されたアルミニウム合金板材のr値を推定するdeep neural networkとBayesian optimizationを用いたr値面内異方性を低減するための集合組織最適化プログラム
最適化結果を可視化するプログラムもあるよ

## Publications
1. A. Yamanaka, R. Kamijyo, K. Koenuma, I. Watanabe and T. Kuwabara, "Deep neural network approach to estimate biaxial stress-strain curves of sheet metals", Materials & Design, Vol. 195 (2020), 108970. <a href="https://doi.org/10.1016/j.matdes.2020.108970">https://doi.org/10.1016/j.matdes.2020.108970</a>

2. K. Koenuma, A. Yamanaka, I. Watanabe and T. Kuwabara, "Estimation of texture-dependent stress－strain curve and r-value of aluminum alloy sheet using deep learning", Materials Transactions, Vol. 61 (2020), pp. 2276-2283 <a href="https://doi.org/10.2320/matertrans.P-M2020853">https://doi.org/10.2320/matertrans.P-M2020853</a>. 

3. K. Koenuma, A. Yamanaka, I. Watanabe and T. Kuwabara, "Estimation of texture-dependent stress－strain curve and r-value of aluminum alloy sheet using deep learning", Journal of Japan Society for Technology of Plasticity, Vol. 61 No. 709 (2020), pp. 48-55. (in Japanese) <a href="https://doi.org/10.9773/sosei.61.48">doi.org/10.9773/sosei.61.48</a>


## Requirements
- Anacondaのインストール後, 必要なライブラリをインストールするためのバッチファイルを下記コマンドで実行
```bat
tf_env
```

## Usage
#### Training DNN
1. Training dataを研究室HPからダウンロードし, 任意のディレクトリに解凍する. common/rawdata.pyの下記コードを編集し, ```ROOT_DIR```をtrainingdataの保存先と一致させてください.
```python : rawdata.py
ROOT_DIR = 'E:/'
```

2. Training dataを読み込み, DNNに学習させるdatasetを作成するscriptであるdataset.pyを実行する.
```bat
conda activate tf_env
python dataset.py
```

3. train_tfmodel.pyは作成したdatasetを用いてtensorflow kerasで実装されたDNNをtrainingするためのscriptです. 実行することで以下のディレクトリに実装されたDNNがtrainingされます. tf_models/dnn3d/model.py
```bat
python train_tfmodel.py
```

4. draw_rvalue.pyを実行することで, trained DNNの簡単なデモンストレーションを行うことができます.

#### Texture-optimizing calculation
1. DNNのtrainingを行った後, optimize_rvalue_BO.pyを実行すると, Bayesian optimization (BO) が実行される. 最適化の結果はOpt_result/ev_all.datに保存される. Opt_result/model.datはGaussian process regression (GPR) で使用するカーネルのパラメータである.

2. BO_result_visualizer.pyを実行してOpt_result/ev_all.datを読み込むと, webブラウザにBOで用いられるacquisition functionと, GPRによるobjective functionのpredictive mean, standard deviationの3Dグラフが表示される

## License
BSD License (3-clause BSD License)

## Author
[Yamanaka Research Group @ TUAT](http://web.tuat.ac.jp/~yamanaka/)
