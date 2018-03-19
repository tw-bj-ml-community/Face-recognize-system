import face_recognition
import time

known_image = face_recognition.load_image_file("../resources/full_body_zh1.jpg")
unknown_image = face_recognition.load_image_file("../resources/zh2.jpg")

print(time.clock())

biden_encoding = face_recognition.face_encodings(known_image)[0]
print(time.clock())
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
print(time.clock())
results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
print(time.clock())
print(results)