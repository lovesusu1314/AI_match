import os
import sys
import cv2
from PyQt5.Qt import *
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QFileInfo
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from PIL import Image
import threading
# import tensorflow as tf
import numpy as np
# from ssd import SSD
from paddel_version.infer import *

import shutil




def cvImgtoQtImg(cvImg):  # 定义opencv图像转PyQt图像的函数
    QtImgBuf = cv2.cvtColor(cvImg, cv2.COLOR_BGR2BGRA)
    QtImg = QtGui.QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0], QtGui.QImage.Format_RGB32)
    return QtImg
# 模型作为全局变量,随时切换
# 目前只训练了ssd，yolov3
global model
global video_path
model = {"ssd":"C:\\Users\\HYF\\Documents\\GitHub\\PaddleDetection\\configs\\ssd\\ssd_vgg16_300_240e_voc.yml"}
class mainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.set_ui()
        self.setWindowIcon(QtGui.QIcon("colonoscopy.png"))
        self.model_dir = None
        self.video_file = None
        self.video_file_out = None
        self.img_file = None

        # 创建一个关闭事件并设为未触发
        self.stopEvent = threading.Event()
        self.stopEvent.clear()


    def set_ui(self):
        loadUi(r'D:\xiangmu\AI_match\GUI_paddle\main.ui', self)
        self.InputImgPushButton.clicked.connect(self.inputImg)
        self.OutputImgPushButton.clicked.connect(self.outputImg)
        self.IntputVideoPushButton.clicked.connect(self.inputVideo)
        self.OutputVideoPushButton.clicked.connect(self.outputVideo)
        self.ImageRecognize.clicked.connect(self.recognizeImg)
        # self.DoExchange.clicked.connect(self.importmodel)
        self.VideoRecognize.clicked.connect(self.recognizeVideo)#连接到 实时检测
        self.VideoRecognize_2.clicked.connect(self.recognize)
        self.VideoRecognize_3.clicked.connect(self.Close)

    
    
    
    
    def yolo(self):
        self.model_dir = 'PaddleDetection/inference_model_yolo/yolov3_darknet53_270e_coco'
        print(self.model_dir)
        
    def ssd(self):
        self.model_dir = 'PaddleDetection/output_inference/ssd_vgg16_300_240e_voc'
        print(self.model_dir)
        

    def inputImg(self):
        """
        文件读取与显示
        """
        file = QFileDialog.getOpenFileName(self,'打开图片文件','./','Image Files(*.png *.jpg *.bmp *.tif)')
        self.img_file = file[0]
        self.InputImgLineEdit.setText(file[0])
        self.pix = QtGui.QPixmap(file[0])
        self.img = file[0]
        self.ImageBox.setPixmap(self.pix)
        self.ImageBox.setScaledContents(True)


    def outputImg(self):
        """
        将读取的文件进行保存
        """
        file = QFileDialog.getSaveFileName(self,'文件保存','./','Image Files(*.png *.jpg *.bmp )')
        self.OutputImgLineEdit.setText(file[0])
        self.pix.save(file[0],"png",100)


    def inputVideo(self):
        self.video_file = QFileDialog.getOpenFileName(self, '打开视频文件', './', 'Video Files(*.mp4 *.avi *.mkv *.mpg)')
        print(type(self.video_file))
        self.video_file = list(self.video_file)[0]
        print(type(self.video_file))
        print(self.video_file)

        self.video = cv2.VideoCapture(self.video_file)
        ret,frame = self.video.read()
        frame = cv2.resize(frame, (926, 788), interpolation=cv2.INTER_AREA)
                
        # # video = cv2.VideoCapture('D:\\xiangmu\\voc2coco\\PaddleDetection\\dataset\\12345.mp4')
        
        

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.Qframe = QtGui.QImage(frame.data,frame.shape[1],frame.shape[0],frame.shape[1]*3,QtGui.QImage.Format_RGB888)
        self.ImageBox_2.setPixmap(QtGui.QPixmap.fromImage(self.Qframe))
        # cv2.waitKey(int(1000 / self.frameRate))
        # self.update()
        _video_file = self.video_file.split('/')[-1]
        self.video_file_out =  _video_file
        print(self.video_file_out)
        # self.video_file = self.video_file(0)
        # print(self.video_file)
        self.InputVideoLineEdit.setText(self.video_file)

    def Close(self):
        # 关闭事件设为触发，关闭视频播放
        self.stopEvent.set()

    def video_play(self):
        model_dir = 'D:\\xiangmu\\AI_match\\output_inference\\ssd_vgg16_300_240e_voc'
        camera_id = 0
        out_dir = 'D:\\xiangmu\\AI_match\\GUI_paddle\\output'


        pred_config = PredictConfig(model_dir)  #准备预测参数
        detector = Detector(pred_config, model_dir)  #创建 检测类对象
        video_name = 'output.mp4'
        fps = 30
        frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        print('frame_count', frame_count)
        # width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # yapf: disable
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'mpeg')
        # yapf: enable
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, video_name)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (926, 788))
        index = 1

        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break
            print('detect frame:%d' % (index))
            frame = cv2.resize(frame, (926, 788), interpolation=cv2.INTER_AREA)
            index += 1
            results = detector.predict([frame], 0.5)

            im = visualize_box_mask(
                frame,
                results,
                detector.pred_config.labels,
                threshold=0.7)

            im = np.array(im)
            # frame_in = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.write(im)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            self.Qframe = QtGui.QImage(im.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3,
                                       QtGui.QImage.Format_RGB888)
            self.ImageBox_2.setPixmap(QtGui.QPixmap.fromImage(self.Qframe))
            cv2.waitKey(int(1000 / self.frameRate))

            if True == self.stopEvent.is_set():
                # 关闭事件置为未触发，清空显示label
                self.stopEvent.clear()
                self.ImageBox_2.clear()
                # self.ui.Open.setEnabled(True)
                self.video.release()
                writer.release()
                break
        writer.release()
        self.video.release()

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break


        # 设置视频保存对象，参数
        # out_path = 'D:\\xiangmu\\AI_match\\GUI\\output\\'+self.video_file_out
        # print(out_path)

        # self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # self.writer = cv2.VideoWriter(out_path, self.fourcc, fps, (926, 788))


        # while self.video.isOpened():
            # ret,frame = self.video.read()
            #
            # # video = cv2.VideoCapture('D:\\xiangmu\\voc2coco\\PaddleDetection\\dataset\\12345.mp4')
            #
            #
            # frame = cv2.resize(frame,(926,788),interpolation=cv2.INTER_AREA)
            # #----------调用部署好的模型 一帧一帧的预测 --------------#
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = Image.fromarray(np.uint8(frame))
            # frame = np.array(ssd.detect_image(frame))
            # # ----------------------------------------------------
            # # 把一帧帧写入视频
            # frame_in = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # self.writer.write(frame_in)
            # frame =



            # 转化为QT格式输出


        #     if True == self.stopEvent.is_set():
        #         # 关闭事件置为未触发，清空显示label
        #         self.stopEvent.clear()
        #         self.ImageBox_2.clear()
        #         # self.ui.Open.setEnabled(True)
        #         self.video.release()
        #         self.writer.release()
        #         break
        # self.writer.release()
        # self.video.release()

            

    def recognizeVideo(self): # 实时检测
        #os.system()
        # command = 'python PaddleDetection/deploy/python/infer.py --model_dir=' +self.model_dir +' --camera_id=0 --device=gpu --output_dir=output'
        # print(command)
        # os.system(command)


        self.video = cv2.VideoCapture(0)
        self.frameRate = self.video.get(cv2.CAP_PROP_FPS)

        th = threading.Thread(target=self.video_play)
        th.start()



    def recognize(self):    # 视频检测
        #os.system()
        # command = 'python PaddleDetection/deploy/python/infer.py --model_dir='+self.model_dir+ ' --video_file='+self.video_file+' --device=gpu --output_dir=output'
        # os.system(command)
        self.video = cv2.VideoCapture(self.video_file)
        self.frameRate = self.video.get(cv2.CAP_PROP_FPS)

        th = threading.Thread(target=self.video_play)
        th.start()
        

        pass

    def outputVideo(self):
        video = QFileDialog.getSaveFileName(self,'视频保存','./','Video Files(*.mp4 )')
        self.OutputVideoLineEdit.setText(video[0])
        src = 'D:/xiangmu/AI_match/GUI_paddle/output/output.mp4'
        print(src)
        print(video[0])
        shutil.move(src,video[0])



    def recognizeImg(self):
        '''
        单张图片识别
        '''
        # D:\xiangmu\AI_match\PaddleDetection\output_inference\ssd_vgg16_300_240e_voc
        command = 'python ../PaddleDetection/deploy/python/infer.py --model_dir=../PaddleDetection/output_inference/ssd_vgg16_300_240e_voc --image_file='+self.img_file+' --device=gpu --output_dir=output'
        os.system(command)
        # os.system("python ..\\PaddleDetection\\deploy\\python\\infer.py -c ..\\PaddleDetection\\configs\\ssd\\ssd_vgg16_300_240e_voc.yml \
        #             --infer_img={} --output_dir=D:\\xiangmu\\AI_match\\GUI\\output -o weights=D:\\xiangmu\\AI_match\\PaddleDetection\\output\\ssd_vgg16_300_240e_voc\\best_model.pdparams".format(self.img))
        print(self.img)

        self.pix = QtGui.QPixmap("output\\{}".format(self.img.split('/')[-1]))
        self.ImageBox.setPixmap(self.pix)
        self.ImageBox.setScaledContents(True)



    def importmodel(self):
        model_file = QFileDialog.getOpenFileName(self, '选择一个paddle模型', './', 'Image Files(*.pdparams)')
        self.model = model_file[0]      # 指定模型
        self.ModelPath.setText(self.model)

    def Mass_production(self):
        # 批量文件生成
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec_())
    shutil.rmtree("C:\\Users\\HYF\\Documents\\GitHub\\PaddleDetection\\output")

