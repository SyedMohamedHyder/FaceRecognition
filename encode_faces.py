import os
import cv2
import pickle
import face_recognition

dataset = 'dataset_images'
dataset_images = os.listdir(dataset)

def encode():
    encodings_data = dict()

    for image_dir in dataset_images:

        #print(f'[INFO] : Encoding {image_dir}')
        if image_dir not in encodings_data:
            encodings_data[image_dir] = list()

        images = os.path.join(dataset, image_dir)
        for image in os.listdir(images):

            image_path = os.path.join(images, image)

            img = cv2.imread(image_path)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb, model='hog')
            face_encodings = face_recognition.face_encodings(rgb, face_locations)

            for encoding in face_encodings:
                encodings_data[image_dir].append(encoding)

    with open('encodings.pickle', 'wb') as file:
        file.write(pickle.dumps(encodings_data))

