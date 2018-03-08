import tkinter as tk
import tkinter.filedialog
from hchan.fr_system.db_connector import DBManager
from hchan.fr_system.tf_serving_connector import ModelService
from scipy import misc
import numpy as np


class MainFrame:
    def __init__(self, db_connector, model_service):
        self.db_connector = db_connector
        self.model_service = model_service

        self.root = tk.Tk()
        self.monitor_frame = tk.Frame(self.root, height=200, width=200)
        self.monitor_frame.pack_propagate(0)  # don't shrink
        self.monitor = tk.Label(self.monitor_frame, text="Monitor", bg="red", fg="white")
        self.monitor.pack(fill=tk.BOTH, expand=1)
        self.monitor_frame.pack(side="left")
        self.name_tag = tk.Label(self.root, text="User Name", bg="green", fg="black")
        self.name_tag.pack(fill=tk.X)
        self.name_entry = tk.Entry(self.root, text='Name')
        self.name_entry.pack(fill=tk.X)
        self.file_label = tk.Label(self.root, text="File Name", bg="green", fg="black")
        self.file_label.pack(fill=tk.X)

        btn = tk.Button(self.root, text="Select an image", command=self.select_image, height=1)
        btn.pack(expand="yes")
        btn = tk.Button(self.root, text="Upload", command=self.upload, height=1)
        btn.pack(expand="yes")

    def upload(self):
        user_name = self.name_entry.get()
        image_path = self.file_label['text']
        print(user_name, image_path)
        if user_name != '' and image_path != '':
            input_size = 160
            picture = misc.imread(image_path)[:input_size, :input_size, :]
            print('picture size', picture.shape)
            picture = picture.reshape([1, input_size, input_size, 3]).astype(np.float32)
            face_feature = self.model_service.predict(picture)
            print(face_feature)
            # To-do insert user
            user_id = self.db_connector.execute_insert(
                "INSERT INTO user_info (user_name) VALUES ('{}')".format(user_name))
            # TO-DO insert feature
            self.db_connector.execute_insert("INSERT INTO face_feature (user_id,face_feature) VALUES (?,?)", user_id,
                                             memoryview(face_feature))

            res = self.db_connector.execute_select('select (user_id ,face_feature) from face_feature')
            # print(np.frombuffer(res[5][2]))

    def select_image(self):
        # open a file chooser dialog and allow the user to select an input
        # image
        path = tk.filedialog.askopenfilename()
        if path != '':
            self.file_label.config(text=path)


if __name__ == '__main__':
    main_frame = MainFrame(db_connector=DBManager(), model_service=ModelService())

    tk.mainloop()
