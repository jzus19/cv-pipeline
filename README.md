# cv-pipeline
Image woof classification
Arch: XResNet50 + Mish + MaxBlurPool + SA

Accuracy@1: 0.78
Accuracy@5: 0.99

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