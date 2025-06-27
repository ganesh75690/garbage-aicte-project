HI MY NAME IS B SAI GANESH , BTECH

Its about garbage collection using AI.
There are 'cardboard',  'glass', 'metal', 'paper', 'plastic','trash' total different types of waste materials which are use for recycling.

This model which help us to classify waste with 6 different waste materials and it will show you the details of that particular waste materials. This will help to raise awareness for people to reduce and recycle waste.

THESE ARE THE WASTE MATERIALS.................

![Screenshot 2025-06-26 185303](https://github.com/user-attachments/assets/934a10be-64da-4b93-8af6-d52d76147c4e)

ALSO AFTER BUILDING THE MODEL RESULT WILL BE

![Screenshot 2025-06-26 184814](https://github.com/user-attachments/assets/c4d17dde-8a84-4ded-aee8-936371a4c01b)


AND THE PREDICTION RESULT OF WORKING WILL BE.............

![Screenshot 2025-06-26 184847](https://github.com/user-attachments/assets/47a608ea-f5bc-4132-ba73-96b1ef955f87)


IT WILL GIVE BETTER PREDICTION RESULTS AT THE END FOR THE GIVEN IMAGE AND DATA.

+-----------------------------+
|  📷  Input Layer            |
|  Image (128 × 128 × 3)     |
+-----------------------------+
             |
             ↓
+-----------------------------+
|  🔍 MobileNetV3Large        |
|  Feature Extraction         |
|  Lightweight CNN Backbone   |
+-----------------------------+
             |
             ↓
+-----------------------------+
|  🧮 Global Average Pooling  |
|  Reduces Feature Dimensions |
+-----------------------------+
             |
             ↓
+-----------------------------+
|  🔢 Dense Layer             |
|  6 Neurons (Softmax)        |
+-----------------------------+
             |
             ↓
+-----------------------------+
|  🗑️ Output Classes          |
|  Paper, Plastic, Metal,     |
|  Glass, Organic, Others     |
+-----------------------------+



