import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import datetime
import logging
import os
import json
from ultralytics import YOLO
import time

class ParkOff():
    
    def __init__(self,log_level=logging.INFO,settings_json=""):

        """initialising class with dataframe
        args:
            log_level(logging.level): log level for the class
        """

        self.log = logging.getLogger("parkoff-logger")
        self.log.setLevel(log_level)
        self._load_settings(settings_json)

    def _read_image_from_url(self,url):
        # Fetch the image from the URL
        self.log.debug("Fetching image from %s" % (url))
        response = requests.get(url)
        if response.status_code == 200:
            # Convert the response content to a NumPy array
            image_array = np.array(Image.open(BytesIO(response.content)))
            # Convert RGB image to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            return image_bgr
        else:
            self.log.error("Failed to fetch the image from the URL")
    

    def _load_settings(self,json_file_path=''):
        """load json with settings
        args:
            json_file_path: metric sheet in json
        """
        
        # try :a
        #     if not os.path.exists(json_file_path):
        #             self.log.error("Json Doesnt Exist")
        #     else :
        f = open(json_file_path,"r")
        self.settings_json = json.loads(f.read())
        f.close()
        self.log.info(f"Loading Settings from '{json_file_path}'" )


    def cache_image_from_url(self,url="", cache_dir=""):
       
        # url = 'https://www.seattle.gov/trafficcams/images/1_Seneca_EW.jpg?619'
        image = self._read_image_from_url(url)

        if cache_dir == "":
            cache_dir = os.path.join(os.getcwd()+'sample_data','input')

        if not os.path.exists(cache_dir):
            self.log.info(f"Output Directory Doesnt Exist at'{cache_dir}'")
        else :
            self.log.info(f"Creating Output Directory at'{cache_dir}'")
            os.makedirs(cache_dir, exist_ok=True)

        output_path = cache_dir+str(datetime.datetime.now())+'.png' 
        self.log.info(f"Saving Image: '{output_path}'")
        cv2.imwrite(output_path, image)

    def analyse_image(self, model='',input_img_dir="", output_img_dir=""):

        model = YOLO(self.settings_json['settings']['model'])  # load a pretrained model (recommended for training)
        # Use the model
        # model.train(data="coco8.yaml", epochs=3)  # train the model
        # metrics = model.val()  # evaluate model performance on the validation set
        # img_path= "/Users/riteshtekriwal/Work/GitClones/adhoc/sample_data/input/test1_2024-07-03 22:47:24.336888.png"
        input_img_dir = self.settings_json['img_cache']['input_dir']
        output_img_dir = self.settings_json['img_cache']['output_dir']
        os.makedirs(output_img_dir, exist_ok=True)
        
        for filename in os.listdir(input_img_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(input_img_dir, filename)
                # image = cv2.imread(image_path)
                results = model(image_path)
                output_path = os.path.join(output_img_dir, filename)
            for result in results:
                # boxes = result.boxes  # Boxes object for bounding box outputs
                # masks = result.masks  # Masks object for segmentation masks outputs
                # keypoints = result.keypoints  # Keypoints object for pose outputs
                # probs = result.probs  # Probs object for classification outputs
                # obb = result.obb  # Oriented boxes object for OBB outputs
                # # result.show()  # display to screen
                # # output_img_path= "/Users/riteshtekriwal/Work/GitClones/adhoc/sample_data/output/test1_1_2024-07-03 22:47:24.336888.png"
                result.save(filename=output_path)  # save to disk
    

    def cache_all_images(self):
        for iLink in self.settings_json['img_source']:
            iurl = iLink['url']
            self.cache_image_from_url(iurl,self.settings_json['img_cache']['input_dir'])
            time.sleep(1)

     
        


