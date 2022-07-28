from telegram.ext import Updater, Filters, CommandHandler, MessageHandler
from PIL import Image
import numpy as np
import torch
import sys
import os
import cv2
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "../")
from src.model import get_model
from src.dataset import get_transformations

TOKEN = "5521876792:AAE1nSu9DQiUZdyh-9BND0-PHUOe9q6Dw34"

model = get_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load("../Loss1.0944_epoch13.pt", map_location=device))
model.eval()
transformations = get_transformations()
pseudo2real = dict(
                        n02086240= 'Shih-Tzu',
                        n02087394= 'Rhodesian ridgeback',
                        n02088364= 'Beagle',
                        n02089973= 'English foxhound',
                        n02093754= 'Australian terrier',
                        n02096294= 'Border terrier',
                        n02099601= 'Golden retriever',
                        n02105641= 'Old English sheepdog',
                        n02111889= 'Samoyed',
                        n02115641= 'Dingo'
                        )

def start(updater, context): 
	updater.message.reply_text("Welcome to the classification bot!")

def help_(updater, context): 
	updater.message.reply_text("Just send the image you want to classify.")

def message(updater, context):
	msg = updater.message.text
	print(msg)
	updater.message.reply_text(msg)

def image(updater, context):
    photo = updater.message.photo[-1].get_file()
    photo.download("img.jpg")
    image = cv2.imread("img.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transformations["valid"](image=image)["image"].unsqueeze(0)
    prediction = model(image.to(device))
    prediction = torch.argmax(prediction, dim=1)
    le = LabelEncoder()
    le.classes_ = np.load('../classes.npy')
    prediction = pseudo2real[le.inverse_transform(prediction.cpu().detach().numpy())[0]]

    updater.message.reply_text(prediction)

def main():
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_))

    dispatcher.add_handler(MessageHandler(Filters.text, message))

    dispatcher.add_handler(MessageHandler(Filters.photo, image))

    updater.start_polling()
    updater.idle()

if __name__=="__main__":
    main()
