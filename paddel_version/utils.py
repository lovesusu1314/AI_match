import time
import os
import ast
import argparse


class Times(object):
    def __init__(self):
        self.time = 0.
        # start time
        self.st = 0.
        # end time
        self.et = 0.

    def start(self):
        self.st = time.time()

    def end(self, repeats=1, accumulative=True):
        self.et = time.time()
        if accumulative:
            self.time += (self.et - self.st) / repeats
        else:
            self.time = (self.et - self.st) / repeats

    def reset(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def value(self):
        return round(self.time, 4)


class Timer(Times):
    def __init__(self):
        super(Timer, self).__init__()
        self.preprocess_time_s = Times()
        self.inference_time_s = Times()
        self.postprocess_time_s = Times()
        self.img_num = 0

    def info(self, average=False):
        total_time = self.preprocess_time_s.value(
        ) + self.inference_time_s.value() + self.postprocess_time_s.value()
        total_time = round(total_time, 4)
        print("------------------ Inference Time Info ----------------------")
        print("total_time(ms): {}, img_num: {}".format(total_time * 1000,
                                                       self.img_num))
        preprocess_time = round(
            self.preprocess_time_s.value() / max(1, self.img_num),
            4) if average else self.preprocess_time_s.value()
        postprocess_time = round(
            self.postprocess_time_s.value() / max(1, self.img_num),
            4) if average else self.postprocess_time_s.value()
        inference_time = round(self.inference_time_s.value() /
                               max(1, self.img_num),
                               4) if average else self.inference_time_s.value()

        average_latency = total_time / max(1, self.img_num)
        print("average latency time(ms): {:.2f}, QPS: {:2f}".format(
            average_latency * 1000, 1 / average_latency))
        print(
            "preprocess_time(ms): {:.2f}, inference_time(ms): {:.2f}, postprocess_time(ms): {:.2f}".
            format(preprocess_time * 1000, inference_time * 1000,
                   postprocess_time * 1000))

    def report(self, average=False):
        dic = {}
        dic['preprocess_time_s'] = round(
            self.preprocess_time_s.value() / max(1, self.img_num),
            4) if average else self.preprocess_time_s.value()
        dic['postprocess_time_s'] = round(
            self.postprocess_time_s.value() / max(1, self.img_num),
            4) if average else self.postprocess_time_s.value()
        dic['inference_time_s'] = round(
            self.inference_time_s.value() / max(1, self.img_num),
            4) if average else self.inference_time_s.value()
        dic['img_num'] = self.img_num
        total_time = self.preprocess_time_s.value(
        ) + self.inference_time_s.value() + self.postprocess_time_s.value()
        dic['total_time_s'] = round(total_time, 4)
        return dic
