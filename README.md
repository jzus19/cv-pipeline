# cv-pipeline
Image woof classification
Arch: XResNet50 + Mish + MaxBlurPool + SA + Ranger Optimizer (RAdam + LookAHead)
* 4 bs / 15 epochs
Accuracy@1: 0.775 
Accuracy@5: 0.975

Graphics and Metrics in visualization.ipynb

* weights https://drive.google.com/file/d/1jT1Wpp5rB7Q3GNVQctHyVWJc66ZJq_Zy/view?usp=sharing
# Running Flask app with Docker #
``` 
git clone git@github.com:jzus19/cv-pipeline.git
```
```
docker-compose build
```
```
docker run -it -p 5000:5000 cv-pipeline_flask 
```

* Flask Interface 

![Flask](apps_interface/flask_interface.png)

# Telegram bot # 
* https://t.me/imagewoof_classificator_bot

* Telegram Interface
![TG](apps_interface/tg_interface.png)

# Run trainning #
```
python install -r requirements.txt
```
```
python train.py
```

## You can run tg bot yourself 
``` 
cd bot_app
```
```
python bot.py
```