# Bayesian Texture Optimization using Deep Neural Network-based Numerical Material Test 
-----

## Description
- The Bayesian Texture Optimization using Deep Neural Network-based Numerical Material Test (BayesTexOpt) project provides the neural network (NN) structure (named DNN-3D), the training parameters and the datasets for optimizing crystallographic texture in an aluminum alloy sheet to reduce in-plane anisotropy of Lankford value. 
- The DNN-3D was constructed on TensorFlow and Keras. 
- This repository also provides 3D-viewer to visualize the optization results. 
- This project is related to the Deep Neural Network-based Numerical Material Test (DNN-NMT) project (click <a href="https://github.com/Yamanaka-Lab-TUAT/DNN-NMT">here</a>). 
- The detailed methodology is reported in the following publications. 

## Publications 
1. R. Kamijyo, A. Ishii, and A. Yamanaka, Bayesian Texture Optimization using deep neural network-based numerical material test, (2021), submitted. 

2. A. Yamanaka, R. Kamijyo, K. Koenuma, I. Watanabe and T. Kuwabara, "Deep neural network approach to estimate biaxial stress-strain curves of sheet metals", Materials & Design, Vol. 195 (2020), 108970. <a href="https://doi.org/10.1016/j.matdes.2020.108970">https://doi.org/10.1016/j.matdes.2020.108970</a>

3. K. Koenuma, A. Yamanaka, I. Watanabe and T. Kuwabara, "Estimation of texture-dependent stress－strain curve and r-value of aluminum alloy sheet using deep learning", Materials Transactions, Vol. 61 (2020), pp. 2276-2283 <a href="https://doi.org/10.2320/matertrans.P-M2020853">https://doi.org/10.2320/matertrans.P-M2020853</a>. 

## Requirements 
- Anaconda enviroment is required. 
- Install some python libraries required for the BayesTexOpt by executing the following batch file. 
```bat
tf_env
```

## Usage
### Training DNN
1. Download the training data from <a href="http://web.tuat.ac.jp/~yamanaka/opendata.html">Website of Yamanaka Laboratory, TUAT</a>. 
2. Save the downloaded files to any directory. For example, 'E:/'.  
3. Edit "common/rawdata.py" so that ```ROOT_DIR``` is the same directory as that where the training data is saved. 
```python : rawdata.py
ROOT_DIR = 'E:/'
```

4. Run "dataset.py" to load the training data and make the training dataset.  
```bat
conda activate tf_env
python dataset.py
```

4. Run "train_tfmodel.py" to train the DNN-3D using the training dataset. The trained DNN is saved in the directroy "tf_models/dnn3d/model.py". 
```bat
python train_tfmodel.py
```

5. You can estimate Lankford values by executing "draw_rvalue.py". 

### Texture optimization using the trained DNN-3D
1. Run "Bayesian Texture Optimization" by executing "optimize_rvalue_BO.py". The results are saved to the file named "Opt_result/ev_all.dat". 
2. The parameters of Kernel function used in the Gaussian process regression (GPR) are saved in the file named "Opt_result/model.dat". 
3. The optimization results (i.e., 3D distribution of predictive mean, standard deviation, and acquisition function) can be visualize by executing "BO_result_visualizer.py" and open your web browser。 

## License
BSD License (3-clause BSD License)

## Author
[Yamanaka Research Group @ TUAT](http://web.tuat.ac.jp/~yamanaka/)
