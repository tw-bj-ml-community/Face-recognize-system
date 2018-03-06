import numpy as np
import cv2


def main():
    # show_camera_image()
    # find_face_on_image()
    extract_face_from_camera()


def show_camera_image():
    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


'''
    Use cascadeClassifier to extract face, find more details in:
    https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html
'''


def find_face_on_image():
    # cap = cv2.VideoCapture(0)
    # ret, img = cap.read()
    # img = cv2.imread('resources/face_demo.jpg')
    img = cv2.imread('resources/full_body_zh1.jpg')
    face_cascade = cv2.CascadeClassifier(
        '/usr/local/Cellar/opencv/3.4.0_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(
        '/usr/local/Cellar/opencv/3.4.0_1/share/OpenCV/haarcascades/haarcascade_eye.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_face_from_camera():
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()

        face_cascade = cv2.CascadeClassifier(
            '/usr/local/Cellar/opencv/3.4.0_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_images = [];
        for (x, y, w, h) in faces:
            face_images.append(frame[y:y + h, x:x + w]);

        for i in range(len(face_images)):
            cv2.imwrite('tmp/test{}.jpg'.format(i), face_images[i])

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
