# Face-recognize-system

We are learning machine leaning techs and intend to create a face recognize check in system.

#### Software requirement:
1. Python 3.6.x
2. OpenCV 3.4
3. face_recognition(this lib depends on dlib)



pip3 install face_recognition
pip3 install protobuf
 




[Install dlib on mac](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)

Install opencv with python3.6
===========================
[参考1](https://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/)
[参考2](https://www.learnopencv.com/install-opencv3-on-macos/)
1. install xcode
2. sudo xcodebuild -license
3. brew install opencv3 --with-python3 --without-python
4. 将库连接到我们使用的python包里，下面路径可能需要修改
echo /usr/local/opt/opencv/lib/python3.6/site-packages >> 
/usr/local/lib/python3.6/site-packages/opencv3.pth


Pre-trained model 
=================
This system need a pretrained model from [here](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk),
more information about this pretrained model can be found [here](https://github.com/davidsandberg/facenet).

You need download the pretrained model, unzip it and put all the files into 'pretrained_model' folder, The folder structure sould like:
```
    - pretrained_model
        - 20170512-110547.pb
        - model-20170512-110547.ckpt-250000.data-00000-of-00001
        - model-20170512-110547.ckpt-250000.index
        - model-20170512-110547.meta
```
 
 

Serve model with tensorflow-serving
===================
More details about TF serving and client demo can be found [here](https://a7744hsc.github.io/machine/learning/2018/03/06/Tensorflow-Serving-101.html), a chinese toturial about TF serving.





A simple way to start a tf-serving docker.
=========================
1. Download the latest model from [here](https://drive.google.com/drive/folders/11O5O0pHGy1LrEgLitV6cAceHBJb8nlhZ) 
2. Unzip it into  dockers/all_in_one/models
3. Build the image use following command
   ``` 
   cd ./dockers/all_in_one  
   docker build --pull -t $USER/all_in_one .
   ```
4. Start the tf-serving server `docker run -p 9000:9000 $USER/all_in_one` 


TO DO
===================
1. extract face feature with different regions


