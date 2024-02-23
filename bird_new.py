import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow
from tensorflow import expand_dims
from keras.models import load_model
import google.generativeai as palm
import pyttsx3
palm.configure(api_key="") #insert_PALM_API_KEY
model = load_model('trainedmodel.h5',compile=False)



lab = {
    0: 'ABBOTTS BOOBY', 1: 'ABYSSINIAN GROUND HORNBILL', 2: 'ASHY THRUSHBIRD', 
    3: 'ASIAN CRESTED IBIS', 4: 'ASIAN DOLLARD BIRD', 5: 'ASIAN GREEN BEE EATER', 
    6: 'ASIAN OPENBILL STORK', 7: 'AUCKLAND SHAQ', 8: 'AUSTRAL CANASTERO', 
    9: 'AUSTRALASIAN FIGBIRD', 10: 'AVADAVAT', 11: 'AZARAS SPINETAIL', 
    12: 'AZURE BREASTED PITTA', 13: 'AZURE JAY', 14: 'AZURE TANAGER', 
    15: 'AZURE TIT', 16: 'BAIKAL TEAL', 17: 'BALD EAGLE', 18: 'BALD IBIS', 
    19: 'BROWN CREPPER', 20: 'BROWN HEADED COWBIRD', 21: 'BROWN NOODY', 
    22: 'BROWN THRASHER', 23: 'BUFFLEHEAD', 24: 'BULWERS PHEASANT', 
    25: 'BURCHELLS COURSER', 26: 'BUSH TURKEY', 27: 'CAATINGA CACHOLOTE', 
    28: 'CABOTS TRAGOPAN', 29: 'CACTUS WREN', 30: 'CALIFORNIA CONDOR', 
    31: 'CALIFORNIA GULL', 32: 'CRIMSON CHAT', 33: 'CRIMSON SUNBIRD', 
    34: 'CROW', 35: 'CUBAN TODY', 36: 'CUBAN TROGON', 37: 'CURL CRESTED ARACURI', 
    38: 'D-ARNAUDS BARBET', 39: 'DALMATIAN PELICAN', 40: 'DARJEELING WOODPECKER', 
    41: 'DUNLIN', 42: 'DUSKY LORY', 43: 'DUSKY ROBIN', 44: 'EARED PITA', 
    45: 'EASTERN BLUEBIRD', 46: 'EMU', 47: 'ENGGANO MYNA', 48: 'EURASIAN BULLFINCH', 
    49: 'EURASIAN GOLDEN ORIOLE', 50: 'EURASIAN MAGPIE', 51: 'EUROPEAN GOLDFINCH', 
    52: 'EUROPEAN TURTLE DOVE', 53: 'EVENING GROSBEAK', 54: 'FAIRY BLUEBIRD', 
    55: 'FAIRY PENGUIN', 56: 'FAIRY TERN', 57: 'FLAME TANAGER', 58: 'FOREST WAGTAIL', 
    59: 'FRIGATE', 60: 'FRILL BACK PIGEON', 61: 'GAMBELS QUAIL', 62: 'GANG GANG COCKATOO', 
    63: 'GILA WOODPECKER', 64: 'GILDED FLICKER', 65: 'GLOSSY IBIS', 66: 'GREY HEADED FISH EAGLE', 
    67: 'GREY PLOVER', 68: 'GROVED BILLED ANI', 69: 'GUINEA TURACO', 70: 'GUINEAFOWL', 
    71: 'GURNEYS PITTA', 72: 'GYRFALCON', 73: 'HAMERKOP', 74: 'HARLEQUIN DUCK', 
    75: 'HARLEQUIN QUAIL', 76: 'HARPY EAGLE', 77: 'HORNED LARK', 78: 'HORNED SUNGEM', 
    79: 'HOUSE FINCH', 80: 'HOUSE SPARROW', 81: 'HYACINTH MACAW', 82: 'IBERIAN MAGPIE', 
    83: 'IBISBILL', 84: 'IMPERIAL SHAQ', 85: 'INCA TERN', 86: 'INDIAN BUSTARD', 
    87: 'INLAND DOTTEREL', 88: 'IVORY BILLED ARACARI', 89: 'IVORY GULL', 90: 'IWI', 
    91: 'JABIRU', 92: 'JACK SNIPE', 93: 'JACOBIN PIGEON', 94: 'JAVA SPARROW', 
    95: 'JOCOTOCO ANTPITTA', 96: 'KAGU', 97: 'KAKAPO', 98: 'KILLDEAR', 99: 'KING EIDER', 
    100: 'KING VULTURE'
}

def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    # Make predictions
    predictions = model.predict(img_array)
    class_labels = lab
    score =tensorflow.nn.softmax(predictions[0])
    res=(f"{class_labels[np.argmax(score)]}")
    return res


def run():
    img1 = Image.open('logo.jpg')
    img1 = img1.resize((450,350))
    st.image(img1,use_column_width=False)
    st.title("Birds Image Classification")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'></h4>''',
                unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image of Bird", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file,use_column_width=False)
        save_image_path = 'Birds/upload_images'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            res = processed_img(save_image_path)
            st.success("Predicted Bird is: "+res)
            engine = pyttsx3.init()
            engine.say("Predicted Bird is: "+res)
            engine.runAndWait()
            engine.stop()
            display=palm.generate_text(prompt="short description about "+res+" bird in 200 words")
            st.success(display.result)
            engine.say(display.result)
            engine.runAndWait()
            engine.stop()
run()