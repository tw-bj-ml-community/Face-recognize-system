# Face-recognize-system

We are learning machine leaning techs and intend to create a face recognize check in system.

#### Software requirement:
1. Python 3.6.x
2. OpenCV 3.4





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
