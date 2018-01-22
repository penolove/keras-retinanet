import os
import time
import argparse
import pathlib
import requests
import arrow
import shutil

DEBUG = True
TZINFO = '+08:00'

parser = argparse.ArgumentParser(
    description='post image to bottle yolo server')

parser.add_argument(
    '-o',
    '--BaseDir',
    help='used for write_image_sent_path',
    default='')
parser.add_argument(
    '--ip',
    help='ip address used to post image',
    default='localhost:5566')

parser.add_argument(
    '--timesleep',
    type=int,
    help='ip address used to post image',
    default=10)


parser.set_defaults(save=True)

class WbcamCaputrefswebcam(object):
    def __init__(self, args):
        self.ip = args.ip
        self.BaseDir = args.BaseDir
        self.timesleep = args.timesleep
        self.sess = requests.Session()


    def img_write_DVR(self, file_name, ch):
        try:
            resp = self.sess.get('http://192.168.1.110/cgi-bin/web_jpg.cgi?ch='+str(ch),
                                 auth=('admin','123456'), stream=True)

            if resp.status_code == 200:
                with open(file_name, 'wb') as f:
                    resp.raw.decode_content = True
                    shutil.copyfileobj(resp.raw, f)
            else:
                print("connection err due to ", resp.status_code)
        except Exception as e:
            print(e)


    def post_yolo_path(self, img_path):
        print("posting path to yolo server")
        try:
            headers = {'image_path': img_path,
                       'tzinfo': TZINFO}
            requests.post("http://%s/echo"%(self.ip), headers=headers)

        except Exception as e:
            print(e)
            print("[post_yolo_path] post somewhat fails")

        print("post done")


    def write_image_sent_path(self):
        while True:
            # get time
            now = arrow.now(TZINFO)

            target_date = now.format('YYYY/MM/DD/HH')
            target_dir = os.path.join(args.BaseDir, target_date)
            pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

            image_name = now.format('mm_ss')
            for ch in range(8):
                img_path = '%s_%s.jpg' % (os.path.join(target_dir, image_name), ch)
                # write to basedir/target_date/image_name

                if DEBUG: print("[write_image_sent_path] current writing to", img_path)
                self.img_write_DVR(img_path, ch)

                if os.path.exists(img_path):
                    # post target_date/image_name to yolo server
                    img_path = '%s_%s.jpg' % (os.path.join(target_date, image_name), ch)
                    self.post_yolo_path(img_path
                    )
            time.sleep(self.timesleep)


if __name__ == '__main__':
    args = parser.parse_args()
    WCPG = WbcamCaputrefswebcam(args)
    WCPG.write_image_sent_path()
