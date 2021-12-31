import file_path_adder
from pygame import init, mixer
import pyttsx3
from random import randint
from os.path import realpath
from os import listdir
from time import sleep

convertor = pyttsx3.init()
voices = convertor.getProperty("voices")
convertor.setProperty("voice", voices[1].id)
convertor.say("Loading..")
print("Loading..")
convertor.say("Please wait..")
print("Please wait..")
convertor.runAndWait()

sleep(1)

path4bgms = realpath(r"bgms")
bgm = list(listdir(path4bgms))                           # changed
print(bgm)
sng = bgm[randint(0, len(bgm)-1)]

init()
mixer.init()
song = mixer.Sound(f"{path4bgms}\\{sng}")
song.set_volume(1)
song.play(loops=100)
