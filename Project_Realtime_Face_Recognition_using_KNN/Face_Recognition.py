import cv2
import numpy as np
import os

# Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = './data/'
class_id = 0
labels = []
names = {}


# KNN
def distance(x1, x2):
    return np.sqrt(sum((x1 - x2) ** 2))


def knn(X, Y, querypoint, k=5):
    vals = []
    m = X.shape[0]
    for i in range(m):
        d = (distance(querypoint, X[i]), Y[i])
        vals.append(d)

    vals = sorted(vals)
    vals = vals[:k]
    vals = np.array(vals)

    new_vals = np.unique(vals[:, 1], return_counts=True)
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    print(pred)


def KNN(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on the distance & the get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency & Corresponding label
    index = np.argmax(output[1])
    return output[0][index]


# Data Preparation
for file in os.listdir(dataset_path):
    if file.endswith('.npy'):
        # Create the mapping between class id & file name.
        names[class_id] = file[:-4]
        print('Loaded ' + file)
        data_item = np.load(dataset_path + file)
        # print(type(data_item))
        # print(data_item.shape)
        face_data.append(data_item)

        # Create labels for the class
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
print(type(face_dataset), face_dataset.shape)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
print(type(face_labels), face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

while True:
    ret, frame = cap.read()

    if ret == False:
        continue

    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if len(faces) == 0:
        continue

    # Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Extract (Crop out the required face) : Region of Interest
        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))
        # Predicted label
        out = KNN(trainset, face_section.flatten())
        # Display name & Rectangle on Screen.
        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow('faces',frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

