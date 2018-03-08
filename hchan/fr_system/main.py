import tkinter as tk
import tkinter.filedialog

import time
from PIL import Image
from PIL import ImageTk

from hchan.fr_system.db_connector import DBManager
from hchan.fr_system.tf_serving_connector import ModelService
from scipy import misc
import numpy as np
import threading
import cv2
import face_recognition


class MainFrame:
    def __init__(self, db_connector, model_service):
        self.db_connector = db_connector
        self.model_service = model_service
        self.video_capture = cv2.VideoCapture(0)

        self.root = tk.Tk()
        self.root.wm_protocol("WM_DELETE_WINDOW", self.on_close)

        self.monitor_frame = tk.Frame(self.root, height=180, width=320)
        self.monitor_frame.pack_propagate(0)  # don't shrink
        self.monitor = tk.Label(self.monitor_frame)
        self.monitor.pack(fill=tk.BOTH)
        self.monitor_frame.pack(side="left")
        self.name_tag = tk.Label(self.root, text="User Name", bg="green", fg="black")
        self.name_tag.pack(fill=tk.X)
        self.name_entry = tk.Entry(self.root, text='Name')
        self.name_entry.pack(fill=tk.X)
        self.file_label = tk.Label(self.root, text="File Name", bg="green", fg="black")
        self.file_label.pack(fill=tk.X)

        btn_select = tk.Button(self.root, text="Select an image", command=self.select_image, height=1)
        btn_select.pack(expand="yes")
        btn_upload = tk.Button(self.root, text="Upload", command=self.upload, height=1)
        btn_upload.pack(expand="yes")
        btn_pause = tk.Button(self.root, text="Pause", command=self.pause, height=1)
        btn_pause.pack(expand="yes")
        self.feature_dict = self.load_existing_users()

        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        # self.thread.start()

    def load_existing_users(self):
        result = {}
        responses = self.db_connector.execute_select(
            'select user_info.user_name,user_info.id,face_feature.face_feature '
            'from face_feature join user_info on face_feature.user_id == user_info.id')
        for res in responses:
            user_unique_name = res[0] + str(res[1])
            face_feature = np.frombuffer(res[2])
            result[user_unique_name] = face_feature
        print(result)
        similarity = np.linalg.norm(result['hchchchc1'] - result['wangyi2'])
        print(similarity)
        return result

    def upload(self):
        user_name = self.name_entry.get()
        image_path = self.file_label['text']
        print(user_name, image_path)
        if user_name != '' and image_path != '':
            input_size = 160
            picture = misc.imread(image_path)
            locations = face_recognition.face_locations(picture)
            print('selected picture size', picture.shape)
            print('find {} face on {}'.format(len(locations), image_path))
            (top, right, bottom, left) = locations[0]
            face = picture[top:bottom, left:right, :]
            face = cv2.resize(face, (160, 160))
            # cv2.imshow('Video', face)

            processed_face = face.reshape([1, input_size, input_size, 3]).astype(np.float32)
            face_feature = self.model_service.predict(processed_face)
            # To-do insert user
            user_id = self.db_connector.execute_insert(
                "INSERT INTO user_info (user_name) VALUES ('{}')".format(user_name))
            # TO-DO insert feature
            self.db_connector.execute_insert("INSERT INTO face_feature (user_id,face_feature) VALUES (?,?)", user_id,
                                             memoryview(face_feature))

    def select_image(self):
        # open a file chooser dialog and allow the user to select an input
        # image
        path = tk.filedialog.askopenfilename()
        if path != '':
            self.file_label.config(text=path)

    def videoLoop(self):
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                ret, frame = self.video_capture.read()
                frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(frame)

                if len(face_locations) > 0:
                    top, right, bottom, left = face_locations[0]
                    face0 = frame[top:bottom, left:right, :]
                    face0 = cv2.resize(face0, (160, 160))
                    face0 = face0.reshape([1, 160, 160, 3]).astype(np.float32)
                    feature = self.model_service.predict(face0)
                    print('feature shape', feature.shape)
                    for k, v in self.feature_dict.items():
                        similarity = np.linalg.norm(v - feature)
                        print('similarity with {} is {}'.format(k, similarity))

                # Display the results
                for top, right, bottom, left in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    # Draw a label with a name below the face
                    # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

                image = Image.fromarray(frame)
                image = image.resize((320, 180), Image.ANTIALIAS)
                image = ImageTk.PhotoImage(image)

                self.monitor.configure(image=image)
                self.monitor.image = image
                time.sleep(1 / 30)

        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

    def on_close(self):
        print("[INFO] closing...")
        self.stopEvent.set()
        self.video_capture.release()
        self.root.quit()

    def pause(self):
        if self.stopEvent.is_set():
            self.stopEvent.clear()
            self.thread.start()
        else:
            self.stopEvent.set()


if __name__ == '__main__':
    main_frame = MainFrame(db_connector=DBManager(), model_service=ModelService())

    tk.mainloop()
