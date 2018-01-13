# CAPTCHA_Cracker

Implement convolutional neural network to extract the features from the CAPTCHA images.

Train the model with 640,000 images.

Test acurracy: 99.8%.

Version information:
- python: 3.5.2
- Keras: 2.1.2
- TensorFlow: 1.4.0

# Samples
![](https://github.com/ZhongzhuPeng/CAPTCHA_Cracker/blob/master/sample_images_of_results/7N43.png?raw=true)
![](https://github.com/ZhongzhuPeng/CAPTCHA_Cracker/blob/master/sample_images_of_results/7PGH.png?raw=true)
![](https://github.com/ZhongzhuPeng/CAPTCHA_Cracker/blob/master/sample_images_of_results/9FP4.png?raw=true)
![](https://github.com/ZhongzhuPeng/CAPTCHA_Cracker/blob/master/sample_images_of_results/YQZD.png?raw=true)

# Wrong predictions

In rare cases, the model cannot disting '0' and 'O'

![](https://github.com/ZhongzhuPeng/CAPTCHA_Cracker/blob/master/test_img/N0DU.jpg?raw=true)    
Real: N0DU    Predict: NODU

![](https://github.com/ZhongzhuPeng/CAPTCHA_Cracker/blob/master/test_img/YOPJ.jpg?raw=true)    
Real: YOPJ    Predict: Y0PJ
