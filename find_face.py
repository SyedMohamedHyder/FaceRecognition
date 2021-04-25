import os
import cv2
import pickle
import face_recognition

with open('encodings.pickle', 'rb') as file:
    encodings_data = pickle.loads(file.read())

def recognize(img):

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb, model='hog')
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        match_info = dict()
        for name, known_encodings in encodings_data.items():
            num_matches = sum( 1 for _ in filter(None, face_recognition.compare_faces(known_encodings, encoding)))
            match_info[name] = num_matches
        person, num_matches = sorted(match_info.items(), key=lambda item: item[1], reverse=True)[0]
        if num_matches:
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(img, (left, top), (right, bottom-100), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name.title(), (left+5, top-5), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (0, 0, 255), 2)


if __name__ == '__main__':
    image = os.path.join('sample_images', 'obama.jpg')
    img = cv2.imread(image)
    recognize(img)
    cv2.imshow('Recognized Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()