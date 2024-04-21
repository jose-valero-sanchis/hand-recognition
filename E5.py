import sys
import os
import json
import base64
import numpy as np
from tabnanny import verbose
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

train_dir = sys.argv[1]
test_dir = sys.argv[2]

def extract_images(json_path, jpeg_path):
    os.mkdir(jpeg_path)
    jsonFiles = [fJson for fJson in os.listdir(json_path) if fJson.endswith('.json')]

    for fJ in jsonFiles:
        with open(json_path+"/"+fJ) as infile:
            data = json.load(infile)
            imdata = base64.b64decode(data["imageData"])
            with open(f"./{jpeg_path}/{fJ.replace('.json', '')}.jpeg", "wb") as f:
                f.write(imdata)

def ampliar_imagenes(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    jpeg_files = [fjpeg for fjpeg in os.listdir(input_path) if fjpeg.endswith('.jpeg')]

    for jpeg_file in jpeg_files:
        image_path = os.path.join(input_path, jpeg_file)
        input_img = Image.open(image_path)

        imagen_rotada_90 = input_img.rotate(90)
        imagen_rotada_180 = input_img.rotate(180)
        imagen_rotada_250 = input_img.rotate(250)

        output_path_org = os.path.join(output_path, f"{jpeg_file.replace('.jpeg', '')}_original.jpeg")
        output_path_rot1 = os.path.join(output_path, f"{jpeg_file.replace('.jpeg', '')}_rot1.jpeg")
        output_path_rot2 = os.path.join(output_path, f"{jpeg_file.replace('.jpeg', '')}_rot2.jpeg")
        output_path_rot3 = os.path.join(output_path, f"{jpeg_file.replace('.jpeg', '')}_rot3.jpeg")

        input_img.save(output_path_org)
        imagen_rotada_90.save(output_path_rot1)
        imagen_rotada_180.save(output_path_rot2)
        imagen_rotada_250.save(output_path_rot3)

def get_train_labels(json_path):
    jsonFiles = [fJson for fJson in os.listdir(json_path) if fJson.endswith('.json')]

    aux_Y = {}
    for fJ in jsonFiles:
        with open(json_path+"/"+fJ) as infile:
            data = json.load(infile)
            for item in data["shapes"]:
                label = item["label"].upper().replace("2", "").replace(".JPG", "")
                aux_Y[fJ.replace('.json', '')] = label
    
    return aux_Y

def get_features(jpeg_path):
    model_tf = VGG16()
    model_tf = Model(inputs=model_tf.inputs, outputs=model_tf.layers[-2].output)

    X = []
    jpegFiles = [fjpeg for fjpeg in os.listdir(jpeg_path) if fjpeg.endswith('.jpeg')]

    for img in jpegFiles:   
        imagen = load_img(f'./{jpeg_path}/{img}', target_size=(224, 224))
        imagen = img_to_array(imagen)
        imagen = imagen.reshape((1, imagen.shape[0], imagen.shape[1], imagen.shape[2]))
        imagen = preprocess_input(imagen)
        feature = model_tf.predict(imagen, verbose=0)
        X.append(feature[0])
    X = np.array(X)

    return X

def nn_model(X, Y, label_encoder):
    Y_encoded = label_encoder.fit_transform(Y)
    Y_onehot = to_categorical(Y_encoded)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(128, input_dim=4096, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.2, verbose = 0)

    return model

if __name__ == "__main__":
    extract_images(train_dir, "fotos_train")
    extract_images(test_dir, "fotos_test")
    ampliar_imagenes("fotos_train", "fotos_train_ampliadas")

    aux_Y = get_train_labels(train_dir)
    Y_train = np.array(list(aux_Y.values()))
    Y_train = np.repeat(Y_train, 4)

    X_train = get_features("fotos_train_ampliadas")
    X_test = get_features("fotos_test")

    label_encoder = LabelEncoder()
    model = nn_model(X_train, Y_train, label_encoder)
    predictions = model.predict(X_test, verbose = 0)
    decoded_predictions = np.argmax(predictions, axis=1)
    decoded_predictions_labels = label_encoder.inverse_transform(decoded_predictions)

    for i, pred in enumerate(decoded_predictions_labels):
        linea = "{:04d} {}".format(i, pred)
        print(linea)