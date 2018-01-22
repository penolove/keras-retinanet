#! /usr/bin/env python
"""Run a Retina object detection model on test images."""
import io
import os
import time
import imghdr
import sqlite3
import argparse
import colorsys
import pathlib
import arrow

import pickle
import keras
import numpy as np
import tensorflow as tf
from bottle import route, run, request
from PIL import Image, ImageDraw, ImageFont
from keras_retinanet.models.resnet import custom_objects


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
keras.backend.tensorflow_backend.set_session(get_session())



parser = argparse.ArgumentParser(
    description='Run a Retina style detection model on test images..')
parser.add_argument(
    'model_path',
    help='path to h5 model file containing body'
    'of a Retina model')
parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to coco_classes.txt',
    default='snapshots/classMap_val2017.pkl')
parser.add_argument(
    '-t',
    '--test_path',
    help='path to directory of test images, defaults to images/',
    default='images')
parser.add_argument(
    '-od',
    '--output_path_det',
    help='path to output test images, defaults to images/out',
    default='out')
parser.add_argument(
    '-o',
    '--output_path',
    help='path to output test images, defaults to images/out',
    default='')
parser.add_argument(
    '-s',
    '--score_threshold',
    type=float,
    help='threshold for bounding box scores, default .3',
    default=.3)
parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .5',
    default=.5)
parser.add_argument(
    '--ip',
    help='ip address used to listen',
    default='localhost:5566')
parser.add_argument(
    '-db_path',
    '--db_path',
    help='datapath of db',
    default='default_retina.sqlite3')


def _main(args):
    retina_model = modelWrapper(args)
    host, port = args.ip.split(':')
    # run test images
    retina_model.detect_test_folder()

    @route('/echo', method='POST')
    def echo():
        im_path = request.headers.get('image_path', 'temp.jpg')
        raw_image_path2save = os.path.join(retina_model.output_path, im_path)
        det_image_path2save = os.path.join(retina_model.output_path_det, im_path)
        tzinfo = request.headers.get('tzinfo', '+08:00')
        arrive_timestamp = arrow.now(tzinfo).datetime
        retina_model.insert_image_info(im_path, arrive_timestamp)  # insert into image_info

        image_data = retina_model.readImage(raw_image_path2save)

        print("[upload_image] get post_image with file_name :", im_path)

        # detect image with retina
        detected_results = retina_model.image_datect_draw_save(image_data, im_path,
                                                             det_image_path2save)

    @route('/folder_detection', method='POST')
    def folder_detection():
        dir_path = request.headers['dir_path']
        tzinfo = request.headers.get('tzinfo', '+08:00')
        arrive_timestamp = arrow.now(tzinfo).datetime
        # will insert image_infos with all same timestamp, and detection
        retina_model.detect_images_in_folder(dir_path, arrive_timestamp)

    @route('/upload_images', method='POST')
    def upload_image():
        """make sure you add im_path in post header"""
        data = request.body.read()
        file_name = request.headers.get('image_path', 'temp.jpg')
        raw_image_path2save = os.path.join(retina_model.output_path, file_name)
        det_image_path2save = os.path.join(retina_model.output_path_det, file_name)
        tzinfo = request.headers.get('tzinfo', '+08:00')

        # read posted image from bytes, and save it.
        image_data_raw = io.BytesIO(bytearray(data))
        image_data = retina_model.readImage(image_data_raw)
        retina_model.saveImage(image_data, raw_image_path2save)

        print("[upload_image] get post_image with file_name :", file_name)
        # inser image info into image_info table
        arrive_timestamp = arrow.now(tzinfo).datetime
        retina_model.insert_image_info(file_name, arrive_timestamp)

        # detect image with retina
        detected_results = retina_model.image_datect_draw_save(image_data, file_name,
                                                             det_image_path2save)

    run(host=host, port=port, debug=True)

class RetinaWrapper(object):
    def __init__(self, args):
        # snapshots/resnet50_coco_best_v1.2.2.h5
        
        model_path = os.path.expanduser(args.model_path)
        self.model = keras.models.load_model(model_path,
                                             custom_objects=custom_objects)

        # snapshots/classMap_val2017.pkl
        classes_path = os.path.expanduser(args.classes_path)
        self.class2idx = pickle.load(open(classes_path,"rb"))
        
        self.class_names = [class_ for class_,_ in self.class2idx.items()]
        self.idx2class = dict((idx, class_) for class_, idx in self.class2idx.items())

        self.score = args.score_threshold
        # colors for class display
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))

        color_pair = zip(self.class_names, colors)
        #print([i for i in color_pair])
        self._colors = dict([(class_name, (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)))
                             for class_name, x in color_pair])

    @property
    def colors(self):
        """
        dict map class to colors.
        """
        return self._colors

    def image_detection(self, image):
        """image_detection
            given image_path, detected with retina model
            write detected_image, save to db, return detections(bbox info)

        Parameters
        ----------
        image: Pil.

        Returns
        ---------
        detected_result_list: List[((int, int, int, int), str, float)]
            detected result List[(left, top, right, bottom), predicted_class, score)]
        """

        # start to detect image
        start_time = time.time()
        
        I = np.asarray(image.convert("RGB"))
        I = I[:,:,::-1] # rbg2bgr

        detected_result_list = []
        _, _, detections = self.model.predict_on_batch(np.expand_dims(I, axis=0))
        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

        for idx, (label_, score) in enumerate(zip(predicted_labels, scores)):
            if score < self.score:
                continue
            b = detections[0, idx, :4].astype(int)
            top, left, bottom, right = (b[1],b[0],b[3], b[2])
            top = int(max(0, np.floor(top + 0.5).astype('int32')))
            left = int(max(0, np.floor(left + 0.5).astype('int32')))
            bottom = int(min(image.size[1], np.floor(bottom + 0.5).astype('int32')))
            right = int(min(image.size[0], np.floor(right + 0.5).astype('int32')))
            detected_result_list.append(((left, top, right, bottom), self.idx2class[label_], score))

        print("--- %s seconds ---" % (time.time() - start_time))
        return detected_result_list

class modelWrapper(object):
    def __init__(self, args):
        self.core_detector = RetinaWrapper(args)
        self.output_path = os.path.expanduser(args.output_path)
        # path of output det image
        self.output_path_det = os.path.join(self.output_path, args.output_path_det)
        db_path = os.path.join(self.output_path, args.db_path)
        self.test_path = os.path.expanduser(args.test_path)
        self.conn = sqlite3.connect(db_path)
        self.create_db_table()

    def create_db_table(self):
        """ connect_create_db
            create table in the db file in db if table not exist
        """
        print('connet/create image_info table')

        self.conn.execute('''CREATE TABLE IF NOT EXISTS image_info(
                          image_path TEXT PRIMARY KEY,
                          timestamp TIMESTAMP
                          )''')

        print('connet/create image_annotation table')
        self.conn.execute('''CREATE TABLE IF NOT EXISTS image_annotation(
                          ID INTEGER PRIMARY KEY AUTOINCREMENT,
                          image_path TEXT,
                          x1 INTEGER,
                          y1 INTEGER,
                          x2 INTEGER,
                          y2 INTEGER,
                          label TEXT,
                          FOREIGN KEY(image_path) REFERENCES image_info(image_path)
                          )''')
        self.conn.commit()

    def insert_image_info(self, image_path, arrive_timestamp):
        try:
            self.conn.execute('''INSERT INTO image_info(image_path, timestamp)
                              VALUES(?,?)''', (image_path, arrive_timestamp))
        except:
            pass

    def insert_image_annotation(self, annoation_info):
        """ insert_image_annoation
        insert image annoation into db
        Parameters
        ----------
        annoation_info: (str, int, int, int, int, str)
            annoation_info = (test_image_path, left, top, right, bottom, predicted_class)
        """
        (test_image_path, (left, top, right, bottom), predicted_class) = annoation_info
        try:
            self.conn.execute('''INSERT INTO image_annotation(image_path, x1, y1, x2, y2
                                ,label) VALUES (?,?,?,?,?,?)''',
                              (test_image_path, left, top, right, bottom, predicted_class))
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass

    def saveImage(self, image, image_path):
        """
        save Image to path, if path dont exit, create it.

        Parameters
        ----------
        image: PIL.Image.Image
            image to save
        image_path: str
            image_path.
        """
        directory = os.path.dirname(image_path)
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        image.save(image_path, quality=90)

    def readImage(self, test_image_path):
        """
        Parameters
        ----------
        image_path. current using relative path "YYYY/MM/DD/HH/mm-ss.jpg"
            (can pass io.bytesio object, take a look upload_image in the main)

        Returns
        ---------
        image: PIL.Image.Image
        """
        image = Image.open(test_image_path)
        return image

    def image_detection(self, image):
        """image_detection
            given image_path, detected with retina model
            write detected_image, save to db, return detections(bbox info)

        Parameters
        ----------
        image: PIL.Image.Image

        Returns
        ---------
        detected_result_list: List[((int, int, int, int), str, float)]
            detected result List[((left, top, right, bottom), predicted_class, score)]
        """
        return self.core_detector.image_detection(image)

    def draw_bbox(self, image, detected_result_list):
        """draw_bbox
            give folder_path and arrvie_timestmap to test images in the folder

        Parameters
        ----------
        image: PIL.Image.Image
            image to be added bbox

        detected_result_list: List[((int, int, int, int), str, float)]
            detected result List[((left, top, right, bottom), predicted_class, score)]
        """
        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for ((left, top, right, bottom), predicted_class, score) in detected_result_list:

            label = '{} {:.2f}'.format(predicted_class, score)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            # creating bbox on images
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.core_detector.colors[predicted_class])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.core_detector.colors[predicted_class])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

    def image_datect_draw_save(self, image_data, test_image_path, save_output_image_path):
        """
        wrapper of pipeline of detect, draw, save

        """
        # detect image with retina
        detected_results = self.image_detection(image_data)
        if save_output_image_path:
            self.draw_bbox(image_data, detected_results)
            self.saveImage(image_data, save_output_image_path)

        detected_results = [(test_image_path, (left, top, right, bottom), predicted_class)
                            for ((left, top, right, bottom), predicted_class, score)
                            in detected_results]

        for detected_reuslt in detected_results:
            self.insert_image_annotation(detected_reuslt)

    def detect_test_folder(self):
        """image_detection
            function used to test demo images in images folder
        """
        arrive_timestamp = arrow.now("+08:00").datetime
        self.detect_images_in_folder(self.test_path, arrive_timestamp)

    def detect_images_in_folder(self, folder_path, arrive_timestamp):
        """detect_images_in_folder
            give folder_path and arrvie_timestmap to test images in the folder

        Parameters
        ----------
        folder_path: str
            folder path that contains images

        arrive_timestamp: datetime.datetime
            the post requtest arrive time

        """

        for image_file in os.listdir(folder_path):
            try:
                image_type = imghdr.what(os.path.join(folder_path, image_file))
                if not image_type:
                    continue
            except IsADirectoryError:
                continue
            test_image_path = os.path.join(folder_path, image_file)
            save_output_image_path = os.path.join(self.output_path_det, image_file)

            self.insert_image_info(test_image_path, arrive_timestamp)
            image_data = self.readImage(test_image_path)

            # detect image with retina
            self.image_datect_draw_save(image_data, test_image_path, save_output_image_path)

    def __del__(self):
        self.conn.close()


if __name__ == '__main__':
    _main(parser.parse_args())
