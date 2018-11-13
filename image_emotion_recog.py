import cv2
import os.path

faceCascade = cv2.CascadeClassifier('/Users/tonycmooz/PycharmProjects/MyJourney/'
                                    'facifier/src/models/haarcascade_frontalface_alt.xml')


def find_faces(image):
    coordinates = locate_faces(image)
    cropped_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in coordinates]
    normalized_faces = [normalize_face(face) for face in cropped_faces]
    return zip(normalized_faces, coordinates)


def normalize_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (350, 350))

    return face


def locate_faces(image):
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(70, 70)
    )

    return faces


def analyze_picture(model_emotion, model_gender, path, i):

    emotions_dict = {}

    image = cv2.imread(path)
    for normalized_face, (x, y, w, h) in find_faces(image):

        emotion_prediction = model_emotion.predict(normalized_face)
        gender_prediction = model_gender.predict(normalized_face)
        if (gender_prediction[0] == 0):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, emotions[emotion_prediction[0]], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        emotions_dict['index'] = i
        emotions_dict['path'] = path
        emotions_dict['prediction'] = emotions[emotion_prediction[0]]
        # print(emotions[emotion_prediction[0]])

    # return emotions_dict
    print(emotions_dict)


if __name__ == '__main__':
    emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"]

    # Load model
    fisher_face_emotion = cv2.face.FisherFaceRecognizer_create()
    fisher_face_emotion.read('/Users/tonycmooz/PycharmProjects/MyJourney/'
                             'facifier/src/models/emotion_classifier_model.xml')

    fisher_face_gender = cv2.face.FisherFaceRecognizer_create()
    fisher_face_gender.read('/Users/tonycmooz/PycharmProjects/MyJourney/'
                            'facifier/src/models/gender_classifier_model.xml')

    directory_path = '/Users/tonycmooz/PycharmProjects/MyJourney/stanford_ai_emotions/saved_images'
    for i, filename in enumerate(os.listdir(directory_path)):
        if filename:
            file_path = os.path.join(directory_path, filename)
            analyze_picture(fisher_face_emotion, fisher_face_gender, file_path, i)
