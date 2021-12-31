import file_path_adder
from start_theme import song, mixer
from os import listdir, remove, system
from os.path import join, realpath
from pickle import load, dump
from random import randint
import cv2
from numpy import array
from cvlib.object_detection import YOLO
from easygui import enterbox
from easyocr import Reader
from face_recognition import face_locations, face_encodings, compare_faces
from keyboard import is_pressed, wait
from pyttsx3 import init
from speech_recognition import Microphone, Recognizer, HTTPError, WaitTimeoutError, UnknownValueError, RequestError
from thirukkural import Kural
from sys import exit
from pyaudio import PyAudio, paInt16
from struct import unpack
from pvporcupine import create
from pyjokes import get_joke
from wikipedia import summary
from datetime import datetime
from time import time, sleep
from PyDictionary import PyDictionary
from email.message import EmailMessage
from smtplib import SMTP
from gnews import GNews

print("realpath = ", realpath(""))

path_of_images = realpath(r"face recognition\coded images")
path_of_face_encodings = realpath(r"face recognition\face_encodings")
path_for_names = realpath(r"face recognition\names")
path_for_non_coded_images = realpath(r"face recognition\non coded images")
path_for_configurationFile = realpath(r"object detection\yolov3.cfg")
path_for_weights_file = realpath(r"object detection\yolov3.weights")
path_for_yolo_names = realpath(r"object detection\yolov3_classes.txt")
path_for_known_people_details = realpath(r"user\known_people_details.txt")
path_for_user_details = realpath(r"user\user_details.txt")
wake_word_path = realpath(r"wake words\windows")
win_name = "Division By Zero"
url = "http://172.28.6.63:81/stream"
start_time_of_detected_people = 0
detected_people_before_nmin = []
start_time_of_detected_objects = 0
previous_obj = []


# cnn Convolutional Neural Network


def sayAndPrint(text="ok"):  # support
    convertor.say(text)
    print(text)
    convertor.runAndWait()


def face_location_function(img):  # support
    try:
        locations = face_locations(img)
    except TypeError:
        locations = [[]]
    return locations


def draw_rectangle(img, locations):  # support
    for (a, b, c, d) in locations:
        cv2.rectangle(img, (d, a), (b, c), color=(250, 0, 250), thickness=2)  # doubt


def add_image(img, folder=False):
    location = face_location_function(img)
    draw_rectangle(img, location)
    length = len(location)
    encodings = face_encodings(face_image=img, model="large")
    if length != 1:
        sayAndPrint("Number of people in the image is not 1...\nPlease try again...")
        return
    elif not encodings:
        sayAndPrint("Ask them to face the camera properly and try again..")
        return
    if folder:
        cv2.imshow(win_name, img)
        cv2.waitKey(0)
        msg = "Enter their name:"
        sayAndPrint(msg)
        name = enterbox(msg=msg, title=win_name)
        if name in ["", None]:
            sayAndPrint("Given name is invalid...Can't add this image..")
            return
    else:
        try:
            name = getTxtFromAudio("tell me their name..")
            again = getTxtFromAudio("tell me their name again..")
            assert name == again
            sayAndPrint("names matched..")

        except TypeError:
            sayAndPrint("Image not added..")
            return

        except AssertionError:
            sayAndPrint("Names didn't match..Image not added..")
            return
    try:
        known = load(open(path_for_names, "rb"))
    except FileNotFoundError:
        known = []
    img_path = r"{path}\{count}.{name}.jpg".format(path=path_of_images, count=len(known) + 1, name=name)
    known += [name]
    try:
        old_encoding = load(open(path_of_face_encodings, "rb"))

    except FileNotFoundError:
        dump(encodings, open(path_of_face_encodings, "wb"))
        sayAndPrint("Image saved")
        return

    else:
        new_encodings = old_encoding + encodings
        dump(new_encodings, open(path_of_face_encodings, "wb"))
        sayAndPrint("Image saved")

    finally:
        dump(known, open(path_for_names, "wb"))
        cv2.imwrite(img_path, img)
        cv2.destroyAllWindows()


def add_images_from_folder():  # main
    files = listdir(path_for_non_coded_images)
    if len(files) == 0:
        return
    mixer.pause()
    sleep(1)
    sayAndPrint("Adding images from folder")
    for file in files:
        path = join(path_for_non_coded_images, file)
        img = cv2.imread(path)
        add_image(img, folder=True)
        remove(path)
    sleep(1)
    mixer.unpause()


def detect_for_now():  # main
    try:
        known_face_encodings = load(file=open(path_of_face_encodings, "rb"))
    except FileNotFoundError:
        sayAndPrint("Please add images and try again...")
        return None
    img = start_camera()
    face_encodings_cam = face_encodings(img, model="large")
    if not face_encodings_cam:
        sayAndPrint("Ask them to face the camera properly and try again.. ")
        return
    known_people = load(open(path_for_names, "rb"))
    for i in face_encodings_cam:
        comparison = compare_faces(known_face_encodings, i, tolerance=0.5)
        print(comparison)
        if comparison.count(True) > 1:
            detect_for_now()
        try:
            result = comparison.index(True)
            sayAndPrint(known_people[result])
        except ValueError:
            sayAndPrint("unknown..")


def reset():  # main
    try:
        remove(path_of_face_encodings)
        remove(path_for_names)
        for i in listdir(path_of_images):
            remove(join(path_of_images, i))
    except IOError as error:
        print(error.with_traceback(error.__traceback__))
        pass
    sayAndPrint("successfully reseted...")


def start_camera():  # support
    cam = cv2.VideoCapture(0)
    if cam.isOpened():
        sayAndPrint("started camera..")
    else:
        sayAndPrint("Can't start the camera...")
        return
    img = None
    while not is_pressed("shift"):
        ret, img = cam.read()
        if not ret:
            print("fatal error...")
            sayAndPrint("Can run camera..")
            break
        img = array(img)
        cv2.imshow(win_name, img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    cam.release()
    return img


def start_live_camera(function):  # support
    cam = cv2.VideoCapture(0)
    if cam.isOpened():
        sayAndPrint("started camera..")
    else:
        sayAndPrint("Can't start the camera...")
        return
    while not is_pressed("shift"):
        ret, img = cam.read()
        if not ret:
            print("fatal error...")
            sayAndPrint("Can run camera..")
            break
        img = array(img)
        exit_code = function(img)
        if exit_code == 0:
            break
        cv2.imshow(win_name, img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    cam.release()


def ocr_live(img):  # main
    text_data = read.readtext(img)
    for (location, text, prob) in text_data:
        if prob > 0.5:
            sayAndPrint(text)


def detect_live(img):  # main
    global start_time_of_detected_people, detected_people_before_nmin
    try:
        known_face_encodings = load(file=open(path_of_face_encodings, "rb"))
    except FileNotFoundError:
        sayAndPrint("Please add images and try again...")
        return 0
    known_people = load(open(path_for_names, "rb"))
    if start_time_of_detected_people == 0:
        start_time_of_detected_people = time()
    face_encodings_cam = face_encodings(img, model="large")
    for i in face_encodings_cam:
        comparison = compare_faces(known_face_encodings, i, tolerance=0.5)
        if comparison.count(True) > 1:
            continue
        if time() - start_time_of_detected_people > 300:
            start_time_of_detected_people = time()
            detected_people_before_nmin = []
        try:
            result = comparison.index(True)
            detected_person = known_people[result]
            assert detected_person not in detected_people_before_nmin, detected_person  # get minutes from user
            sayAndPrint(detected_person)
            detected_people_before_nmin.append(detected_person)
            detected_people_before_nmin.append(detected_person)
        except ValueError:
            pass
        except AssertionError as err:
            print(err, "already detected..")


def obj_detector_live(img):  # main
    global previous_obj, start_time_of_detected_objects
    bbox, classes, confidence = model.detect_objects(image=img, confidence=0.5)
    model.draw_bbox(img=img, bbox=bbox, labels=classes, confidence=confidence, write_conf=True)
    if len(classes) == 0:
        return None
    if time() - start_time_of_detected_objects > 300:
        start_time_of_detected_objects = time()
        previous_obj.clear()
    for i in classes:
        if i not in previous_obj and i != "person":
            sayAndPrint(i)
            previous_obj.append(i)
        else:
            print(i, "already detected")


def play():  # main
    filter_command("play")
    vid = command
    if vid.isspace() or vid == "":
        sayAndPrint("sorry dear..Can't catch you..")
        return
    sayAndPrint("playing..")
    playonyt(vid)
    wait("shift")
    system("TASKKILL /F /IM chrome.exe /T")


def joke():
    sayAndPrint(get_joke(category="all"))


def email():
    subject = getTxtFromAudio("what's the subject?")
    mail_body = getTxtFromAudio("what's the body of the mail?")
    gmail = EmailMessage()
    gmail["From"] = user_email
    gmail["To"] = support_email
    gmail["Subject"] = subject
    gmail.set_content(mail_body)
    try:
        assert mail_body != "None" and subject != "None"
        server = SMTP(port=587, host="smtp.gmail.com")
        server.starttls()  # transport layer security
        server.login(user=user_email, password=user_password)
        server.send_message(gmail)

    except AssertionError:
        sayAndPrint("Mail not sent")

    except Exception as expt:
        print(expt.with_traceback(expt.__traceback__))
        sayAndPrint("Something went wrong..Email not sent...")

    else:
        sayAndPrint("Mail sent successfully..")


def search_wiki():
    try:
        filter_command("search", "who is")
        data = summary(command, auto_suggest=False, sentences=2)
        sayAndPrint(data)
    except Exception as error:
        print(error)
        sayAndPrint("sorry dear Please search with different name")


def shut_down():
    system("shutdown/sg")


def say_meaning():
    filter_command("what is", "define", "meaning")
    dict_obj = dictionary.meaning(command)
    if dict_obj is None:
        sayAndPrint("sorry dear..can't catch the word..")
        return
    for ty, val in zip(dict_obj, dict_obj.values()):
        sayAndPrint(f"as {ty}:\n{val[0]}")


def repeat():  # need support word..
    filter_command("repeat")
    sayAndPrint(command)


def say_thirukkural():
    random_kural_number = randint(1, 1330)
    kural_data = Kural(no=random_kural_number)
    kural_in_en = kural_data.get_kural_in_en()
    sayAndPrint(kural_in_en)


def filter_command(*rem):
    global command
    for i in rem:
        command = command.replace(i + " ", "").replace("" + i, "").strip()
        print(command)


def getTxtFromAudio(message):
    try:
        sayAndPrint(message)
        with mic as s:
            voice = speech_recognizer.listen(source=s, timeout=30, phrase_time_limit=15)
        speech_2_txt = speech_recognizer.recognize_google(audio_data=voice).lower()
        print(speech_2_txt)

    except WaitTimeoutError as error:
        print(error.with_traceback(error.__traceback__))
        sayAndPrint("sorry dear..please try again..")
        return "None"

    except UnknownValueError as error:
        print(error.with_traceback(error.__traceback__))
        sayAndPrint("sorry dear..can't catch you..")
        return "None"

    except HTTPError as error:
        print(error.with_traceback(error.__traceback__))
        sayAndPrint("sorry dear..Can't start assistant now..")
        return "None"

    except RequestError as error:
        print(error.with_traceback(error.__traceback__))
        sayAndPrint("please check your internet connection..")
        return "None"

    except TimeoutError as error:
        print(error.with_traceback(error.__traceback__))
        sayAndPrint("please check your internet connection..")
        return "None"

    else:
        return speech_2_txt


def action():  # support
    if command == "None":
        return

    elif "add image" in command:
        sayAndPrint()
        add_image(start_camera())

    elif "detect people" in command:
        sayAndPrint()
        start_live_camera(detect_live)
        sayAndPrint("Detection has been stopped...")

    elif "detect for now" in command:
        sayAndPrint()
        detect_for_now()

    elif "read text" in command:
        sayAndPrint()
        start_live_camera(ocr_live)
        sayAndPrint("Reading has been stopped...")

    elif "detect objects" in command:
        sayAndPrint()
        start_live_camera(obj_detector_live)
        sayAndPrint("object detection has been stopped..")

    elif "search" in command or "who is" in command:  # need support word
        search_wiki()

    elif "reset" in command:
        reset()

    elif "thirukkural" in command or "thirukural" in command:
        say_thirukkural()

    elif "play" in command:  # need support word
        play()

    elif "joke" in command:
        joke()

    elif "email" in command:
        email()

    elif "time" in command:
        date_time = datetime.now()
        now = date_time.strftime("%I %M %p")  # time in 12 hours format
        sayAndPrint(now)

    elif "what is" in command or "define" in command or "meaning" in command:  # need support word
        say_meaning()

    elif "news" in command:
        news = GNews(country="In", period="1d", max_results=3)
        for new in news.get_top_news():
            sayAndPrint(new["description"])

    elif "shut down" in command or "shutdown" in command:
        sayAndPrint("Shutting down....")
        shut_down()

    elif "repeat" in command:
        repeat()

    elif "date" in command:
        date_time = datetime.now()
        date = date_time.strftime("%d %B %Y %A")  # B month name A week day
        sayAndPrint(date)

    else:
        sayAndPrint("please say again..")


convertor = init(debug=True)
voices = convertor.getProperty("voices")
convertor.setProperty("voice", voices[1].id)

try:
    from pywhatkit import playonyt, send_mail

except Warning:
    song.stop()
    sleep(2)
    sayAndPrint("Internet is too slow..can't start the engine..")
    exit(0)

except Exception:
    song.stop()
    sleep(2)
    sayAndPrint("please check your internet connection and try again")
    exit(0)

add_images_from_folder()

with open(path_for_known_people_details, "r") as f:
    support_name, support_email, support_ph = f.readline().strip().split(",")

with open(path_for_user_details, "r") as f:
    user_name, user_email, user_password, user_ph = f.readline().strip().split(",")

# print(support_name, support_email, support_ph, user_name, user_email, user_password, user_ph)

read = Reader(lang_list=["en"], gpu=False)

model = YOLO(config=path_for_configurationFile, weights=path_for_weights_file, labels=path_for_yolo_names)

speech_recognizer = Recognizer()
speech_recognizer.dynamic_energy_threshold = False

mic = Microphone()  # can use device index

audio = PyAudio()
key_words_paths = []
for word in listdir(wake_word_path):
    key_words_paths.append(join(wake_word_path, word))
    print(join(wake_word_path, word))

assistant_names = ["hello iris"]
print(assistant_names)
sensitivities = [0.9]
print(key_words_paths)
wake_up = create(keyword_paths=key_words_paths, sensitivities=sensitivities,
                 access_key="wWINo25gtAQFzRzY2me010qwCKz1icZY8Zf7Wqpk143F/ZBZ7dnhkQ==")

dictionary = PyDictionary()

song.stop()

sayAndPrint("Hi my dear friend,I am ready...")
wake_mic = audio.open(frames_per_buffer=wake_up.frame_length, channels=1, rate=wake_up.sample_rate, input=True,
                      format=paInt16)

pre_index = 0
while 1:
    pcm = wake_mic.read(num_frames=wake_up.frame_length)
    pcm = unpack("h" * wake_up.frame_length, pcm)
    index = wake_up.process(pcm=pcm)  # pcm audio sample
    if index == -1 or pre_index != -1:
        pre_index = index
        continue
    pre_index = index
    print(assistant_names[index])
    print("hot word detected..")
    command = getTxtFromAudio("Listening..")
    action()
