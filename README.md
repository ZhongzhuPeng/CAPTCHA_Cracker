# CAPTCHA_Cracker

Implement convolutional neural network(CNN) to extract the features from the CAPTCHA images.

Use [CAPTCHA](https://github.com/lepture/captcha/) librabry to generate CAPTCHA images.

Train the model with 640,000 images.

Highest achieved test acurracy: 98.7%.

Version information:
- python: 3.5.2
- Keras: 2.1.2
- TensorFlow: 1.4.0


# Model

The size of input image is 80x140. We use CNN for feature extraction and use four independant dense layers to output the prediction of four digits.

Python Code | Layers
---|---
input_layer = Input(shape = (img_height, img_width, 3)) | Input Layer (size: 3x80x140)
layers = Conv2D(32, (3, 3), activation='relu')(input_layer) | Convolutional layer (size: 32x78x138)
layers = Conv2D(32, (3, 3), activation='relu')(layers) |  Convolutional layer (size: 32x76x136)
layers = MaxPooling2D((2, 2))(layers) | Pooling Layer (size: 32x38x68)
layers = Conv2D(64, (3, 3), activation='relu')(layers) | Convolutional layer (size: 64x36x66)
layers = Conv2D(64, (3, 3), activation='relu')(layers) | Convolutional layer (size: 32x34x64)
layers = MaxPooling2D((2, 2))(layers) | Pooling Layer (size: 64x17x32)
layers = Conv2D(128, (3, 3), activation='relu')(layers) | Convolutional layer (size: 128x15x30)
layers = Conv2D(128, (3, 3), activation='relu')(layers) | Convolutional layer (size: 128x13x28)
layers = MaxPooling2D((2, 2))(layers) | Pooling Layer (size: 128x7x14)
layers = Conv2D(256, (3, 3), activation='relu')(layers) | Convolutional layer (size: 256x5x12)
layers = Conv2D(256, (3, 3), activation='relu')(layers) | Convolutional layer (size: 256x3x10)
layers = MaxPooling2D((2, 2))(layers) | Pooling Layer (size: 256x2x5)
layers = Flatten()(layers) | Flatten Layer (size: 2560)
layers = Dropout(0.2)(layers) | Dropout Layer (size: 2560)
output_layers = [Dense(n_class, activation='softmax',  name='digit_%d'%i)(layers) for i in (0, 1, 2, 3)] | Output Layer for four digit (size: 36x4)

# Training

![](https://github.com/ZhongzhuPeng/CAPTCHA_Cracker/blob/master/training.png?raw=true)

The model is trained with 20 epochs. It takes about 170s per epochs.

The total training time is about 55 minites on my laptop with the Nvidia 1050Ti.


# Samples of Predictions
![](https://github.com/ZhongzhuPeng/CAPTCHA_Cracker/blob/master/sample_images_of_results/7N43.png?raw=true)
![](https://github.com/ZhongzhuPeng/CAPTCHA_Cracker/blob/master/sample_images_of_results/7PGH.png?raw=true)
![](https://github.com/ZhongzhuPeng/CAPTCHA_Cracker/blob/master/sample_images_of_results/9FP4.png?raw=true)
![](https://github.com/ZhongzhuPeng/CAPTCHA_Cracker/blob/master/sample_images_of_results/YQZD.png?raw=true)

# Wrong Predictions

In rare cases, the model cannot distinguish '0' and 'O'

![](https://github.com/ZhongzhuPeng/CAPTCHA_Cracker/blob/master/test_img/N0DU.jpg?raw=true)    
Real: N0DU    Predict: NODU

![](https://github.com/ZhongzhuPeng/CAPTCHA_Cracker/blob/master/test_img/YOPJ.jpg?raw=true)    
Real: YOPJ    Predict: Y0PJ
