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

import subprocess
import platform
import sys

class ParkOff():
    
    def __init__(self,settings_json="", logger = "", log_level=logging.INFO):

        """initialising class 
        args:
            log_level(logging.level): log level for the class
        """
        if logger == "":
            self.log = logging.getLogger("parkoff-logger")
            self.log.setLevel(log_level)
            
        else:
            self.log = logger
        
        self._load_settings(settings_json)
        self.tempC = -1

    def _read_image_from_url(self,url):

        """supplimentary function to read an image from url
        args:
            url(str): url pointing to an image
        """


        # Fetch the image from the URL
        self.log.debug("Fetching image from %s" % (url))
        response = requests.get(url,verify=False )
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
        
        try:
            if json_file_path == '':
                json_file_path = "../settings.json"
        
            self.log.info(f"Loading Settings from '{json_file_path}'" )

            if not os.path.exists(json_file_path):
                self.log.info(f"File {json_file_path} Not Found")
                json_file_path = "../settings.json"
                self.log.info(f"Attempting to load default path {json_file_path}")
        

            f = open(json_file_path,"r")
            self.settings_json = json.loads(f.read())
            f.close()

            self.log.info("Settings Loaded!")
            
            
        except Exception as e:
            self.log.error(e)
            sys.exit()
            
        
        


    def _check_model_format(self):
        """exports model to NCNN for embedded devices
        args:
            
        """
        model_name = self.settings_json['settings']['model']["name"]
        
        if self.settings_json['settings']["model"]['format'] == "ncnn":
            model_name,ext =  os.path.splitext(model_name)
            print(model_name)
            model_name = f"{model_name}_ncnn_model"
            if not os.path.exists(model_name):
                self.log.info(f"Did Not Find Model'{model_name}', Generating....")
                model = YOLO(self.settings_json['settings']['model']["name"])
                model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'

        
        self.log.info(f"Loading Model'{model_name}'")
        return model_name


    def cache_image_from_url(self,url="", cache_dir=""):
        
        """reads and saves an image from a given url 
        args:
            url(str): url pointing to an image
            cache_dir(str): filepath to save the image
            
        """

       
        # url = 'https://www.seattle.gov/trafficcams/images/1_Seneca_EW.jpg?619'
        image = self._read_image_from_url(url)

        if cache_dir == "":
            cache_dir = os.path.join(os.getcwd(),'sample_data','input')

        if not os.path.exists(cache_dir):
            self.log.info(f"Output Directory Doesnt Exist at'{cache_dir}'")
            self.log.info(f"Creating Output Directory at'{cache_dir}'")
            os.makedirs(cache_dir, exist_ok=True)

        output_path = os.path.join(cache_dir,str(datetime.datetime.now())+'.png') 
        self.log.info(f"Saving Image: '{output_path}'")
        cv2.imwrite(output_path, image)

    def analyse_image(self, model='',input_img_dir="", output_img_dir="", debug = False):
        
        """performs detection on a given image
            - Yolo Object Detection
            - Canny Edge Detection for Curbs
            - Filter Parked Vehicles

        args:
            model(str): Yolo model to use
            input_img_dir(str): filepath to read the input image from
            output_img_dir(str): filepath to save the result
            debug(boolean): If True it saves an output at every step
            
        """
        

        model_name = self._check_model_format()
        
        self.model = YOLO(model_name)  # load a pretrained model (recommended for training)

        
        # Use the model
        # model.train(data="coco8.yaml", epochs=3)  # train the model
        # metrics = model.val()  # evaluate model performance on the validation set
        # img_path= "/Users/riteshtekriwal/Work/GitClones/adhoc/sample_data/input/test1_2024-07-03 22:47:24.336888.png"
        if output_img_dir == "":
            output_img_dir = self.settings_json['img_cache']['output_dir']
        
        if input_img_dir == "":
            input_img_dir = self.settings_json['img_cache']['input_dir']
        
        os.makedirs(output_img_dir, exist_ok=True)

        for filename in os.listdir(input_img_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):

                
             
                
                image_path = os.path.join(input_img_dir, filename)
                
                output_path = os.path.join(output_img_dir, filename)
                
                
                if not os.path.exists(output_path):

                    try:
                        self.tempC = self.get_cpu_temperature()

                    except Exception as e:
                        self.log.info(e)

                    if self.tempC > 75:
                        self.log.info(f"CPU Temperature is high[{self.tempC}], will analyse results later..")
                        return

                    self.log.info(f"Running Analysis on Input Image'{image_path}'")
                    self.log.debug(f"Output Image'{output_path}'")
                    # image = cv2.imread(image_path)
                    image = cv2.imread(image_path)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # results = self.model(image_path)

                    # TODO: Find a better way to find size of image and pass image size closed to yolo acceptable dimension
                    results = self.model.predict(
                        source= image_rgb, 
                        conf = self.settings_json['settings']["model"]['confidence'],
                        iou = self.settings_json['settings']["model"]['iou'],
                        # imgsz = (height,width),
                        imgsz = (480,736),
                        augment =True,
                        classes = self._get_yolo_class_ids(self.model, ['car','truck','bus','bicycle','fire hydrant','tree']),
                        # visualize = True,
                        # max_det = 3
                        # visualise
                        )

                    self.log.debug(f"Found {len(results)} elements in the image")
                    
                    vehicles = self._get_vehicles(results)
                    curbs = self._get_curbs(image_rgb,debug=debug,output_img_dir=output_img_dir,filename=filename)
                    filtered_curbs = self._filter_curbs(image=image_rgb,curbs=curbs)
                    parked_vehicles = self._find_parked_vehicles(vehicles, filtered_curbs)

                    if debug == True:
                        image = cv2.imread(image_path)
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        result_image = self._draw_image(image=image_rgb, curbs=[], vehicles=vehicles)
                        output_path = os.path.join(output_img_dir,"vehicle_"+filename)
                        cv2.imwrite(img=result_image,filename=output_path)

                        image = cv2.imread(image_path)
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        result_image2 = self._draw_image(image=image_rgb,curbs=curbs,vehicles= [])
                        output_path = os.path.join(output_img_dir,"curbs_"+filename)
                        cv2.imwrite(img=result_image2,filename=output_path)

                        image = cv2.imread(image_path)
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        result_image2 = self._draw_image(image=image_rgb,curbs=filtered_curbs,vehicles= [])
                        output_path = os.path.join(output_img_dir,"filtered_curbs_"+filename)
                        cv2.imwrite(img=result_image2,filename=output_path)

                        image = cv2.imread(image_path)
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        result_image3 = self._draw_image(image=image_rgb,curbs=curbs,vehicles=parked_vehicles)
                        output_path = os.path.join(output_img_dir,"parked_vehicles_"+filename)
                        cv2.imwrite(img=result_image3,filename=output_path)


                        
                    if debug == False:
                        result_image = self._draw_image(image=image_rgb, curbs= curbs, vehicles=parked_vehicles)
                        cv2.imwrite(img=result_image,filename=output_path)
    

    def cache_all_images(self):
        """Loop through urls to save image
    
        args:
            
        """
        for iLink in self.settings_json['img_source']:
            iurl = iLink['url']
            self.cache_image_from_url(iurl,self.settings_json['img_cache']['input_dir'])
            time.sleep(1)

    def _get_yolo_class_ids(self, model, class_names=''):
        """Returns a list of class IDs for given class names in YOLO."""
        class_dict = model.names  # Get class ID-to-name mapping
        class_ids = [class_id for class_id, name in class_dict.items() if name.lower() in [n.lower() for n in class_names]]
        return class_ids
    
    def get_cpu_temperature(self):
        """Get the CPU temperature for the target hardware

            args:
            
        """

        system = platform.system()
        self.log.info(f"Executing on a {system} system")
        if system == "Linux":
            return self._get_cpu_temp_rpi() or self._get_cpu_temp_linux()
        elif system == "Darwin":  # macOS
            return 50 # Darwin needs root perms. Skip for now
            # return self._get_cpu_temp_mac()
        else:
            return None  # Unsupported platform


    def _get_cpu_temp_rpi(self):
        """Get the CPU temperature for RPI

            args:
            
        """
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = int(f.read()) / 1000.0  # Convert from millidegree
                self.log.info(f"RPI Core Temp {temp} degC")
            return temp
        except Exception as e:
            self.log.info(e)
            return None


    def _get_cpu_temp_linux(self):
        """Get the CPU temperature for linux

            args:
            
        """
        try:
            output = subprocess.check_output(["sensors"]).decode("utf-8")
            for line in output.split("\n"):
                if "Core" in line:  # Looks for lines like "Core 0: +45.0°C"
                    temp = float(line.split("+")[1].split("°C")[0])
                    return temp
        except Exception as e:
            self.log.info(e)
            return None
    
    def _get_cpu_temp_mac(self):
        """Get the CPU temperature for Darwin/Mac
                Needs root access
            args:
            
        """
        try:
            output = subprocess.check_output(["osx-cpu-temp"]).decode("utf-8")
            temp = float(output.split("°")[0])  # Extracts temperature
            return temp
        except Exception as e:
            self.log.info(e)
            return None
        
       # Convert lines into slope-intercept form
    def _get_slope_intercept(self, x1, y1, x2, y2):
            if x2 - x1 == 0:
                return float('inf'), x1  # Vertical line
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            return slope, intercept
    

    def _merge_similar_lines(self, lines, angle_threshold=10, distance_threshold=30):
        """ Merges lines that have a similar slope and are close to each other. """
        if lines is None:
            return []

        grouped_lines = []  # Store merged lines
        used = set()  # Track processed lines

        for i, line1 in enumerate(lines):
            if i in used:
                continue  # Skip if already merged

            x1, y1, x2, y2 = line1[0]
            slope1, intercept1 = self._get_slope_intercept(x1, y1, x2, y2)
            merged = [line1]  # Start a group

            for j, line2 in enumerate(lines):
                if i == j or j in used:
                    continue

                x3, y3, x4, y4 = line2[0]
                slope2, intercept2 = self._get_slope_intercept(x3, y3, x4, y4)

                # Compare slopes and distance
                angle_diff = abs(np.degrees(np.arctan(slope1)) - np.degrees(np.arctan(slope2)))
                intercept_diff = abs(intercept1 - intercept2)

                if angle_diff < angle_threshold and intercept_diff < distance_threshold:
                    merged.append(line2)
                    used.add(j)

            # Find min/max points to create a long merged line
            all_x = [p[0] for line in merged for p in [line[0][:2], line[0][2:]]]
            all_y = [p[1] for line in merged for p in [line[0][:2], line[0][2:]]]

            grouped_lines.append([[min(all_x), min(all_y), max(all_x), max(all_y)]])  # Store merged line

        return grouped_lines


    def _is_unwanted_color(self,image, x1, y1, x2, y2):
        """ 
            Check if a line is a specific color (e.g., white/yellow road markings) 
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define unwanted color ranges (adjust as needed)
        yellow_lower = np.array([90, 0, 0])
        yellow_upper = np.array([100, 200, 200])

        # white_lower = np.array([0, 0, 200])
        # white_upper = np.array([180, 50, 255])

        # Sample 5 points along the line
        # for alpha in np.linspace(0, 1, 50):
            # x = int(x1 * (1 - alpha) + x2 * alpha)
            # y = int(y1 * (1 - alpha) + y2 * alpha)
        x = int((x1+x2)/2)
        y = int((y1+y2)/2)
            # y = int(y1 * (1 - alpha) + y2 * alpha)

        pixel_hsv = hsv_image[y, x]
        # print(f"X:{x},Y:{y}, HSV:{pixel_hsv}")
            # Check if the pixel falls within the unwanted color ranges
        if (yellow_lower <= pixel_hsv).all() and (pixel_hsv <= yellow_upper).all():
            # if (cv2.inRange(pixel_hsv, (20, 100, 100), (40, 255, 255))): 
                # cv2.inRange(pixel_hsv, white_lower, white_upper)):
            return True  # Line should be ignored
        else:
            return False  # Keep the line


    def _get_vehicles(self, results):

        """Filters vehiceles from yolo results
           
            args:
        """
        # Get detected vehicles
        vehicles = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])  # Class ID
                if cls in self._get_yolo_class_ids(self.model, ['car','truck','bus','bicycle']):  # Car, truck, bus, motorcycle
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    vehicles.append((x1, y1, x2, y2))
        
        return vehicles
    
    def _filter_curbs(self,image,curbs):

        """Filters curbs by color
           
            args:
        """

         # Apply Hough Transform and Filter by Color
        filtered_curb_lines = []

        if curbs is not None:
            for line in curbs:
                x1, y1, x2, y2 = line[0]

                if not self._is_unwanted_color(image, x1, y1, x2, y2):
                    filtered_curb_lines.append(line)  # Keep only valid curb lines
    

        return filtered_curb_lines

    
    
    def _get_curbs(self,image, debug=False, output_img_dir = "",filename=""):


        """Finds the curbs in a given image
           
            args:
                debug(boolean): if True, it saves the output at every process
                output_img_dir(str): filepath to save the output to
                filename(str): base filename to use
        """

        # Detect curb using Hough Transform
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (self.settings_json['settings']['curb_detection']["blur_box_edge"], self.settings_json['settings']['curb_detection']["blur_box_edge"]), 0)
        
        edges = cv2.Canny(blurred, self.settings_json['settings']['curb_detection']["canny_thres_0"], self.settings_json['settings']['curb_detection']["canny_thres_1"])
        # edges = cv2.Canny(blurred, 100, 75)
        # curb_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=220, minLineLength=50, maxLineGap=15)
        curb_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=self.settings_json['settings']['curb_detection']["color_gradient_thrs"], minLineLength=self.settings_json['settings']['curb_detection']["min_straight_line"], maxLineGap=self.settings_json['settings']['curb_detection']["max_gap_bw_lines"])
        
        if debug == True:
            output_path = os.path.join(output_img_dir,"blurred_"+filename)
            cv2.imwrite(img=blurred,filename=output_path)

            output_path = os.path.join(output_img_dir,"edges_"+filename)
            cv2.imwrite(img=edges,filename=output_path)

        
    
        return curb_lines

    def _draw_image(self, image,curbs,vehicles):
        # Draw detected curbs in green
        """Draw the image with curbs, vehicles
           
            args:
                image(cv2.image): if True, it saves the output at every process
                curbs(list): list of coordinates pointing curbs
                vehicles(list): bounding box for vehicles from yolos
        """

        if (curbs is not None): 
            for line in curbs:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green curb lines
        

        if (vehicles is not None) and len(vehicles) != 0  : 

            if len(vehicles[0]) == 5:
                for (x1, y1, x2, y2,curb_id) in vehicles:
                    if curb_id is np.nan:
                        color = (255, 0, 0)
                    else:
                        color = (0,255,0)
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

            elif len(vehicles[0]) == 4:
                for (x1, y1, x2, y2) in vehicles:
                    color = (0,255,0)
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        
        out_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return out_image



    def _find_parked_vehicles(self, vehicles, curbs):
        
        """Find Parked Vehicle next to curb
           
            args:
                curbs(list): list of coordinates pointing curbs
                vehicles(list): bounding box for vehicles from yolos
        """
        
        if len(vehicles) > 0 and len(curbs)>0:
        
            new_row = np.zeros((len(vehicles),1)) 
            new_row[:] = np.nan
            # new_row

            vehicles = np.hstack((vehicles, new_row))
            for line_number, line in enumerate(curbs):
                lx1, ly1, lx2, ly2 = line[0]

                for (itr, iCar ) in enumerate(vehicles):

                    x1 = iCar[0]
                    y1 = iCar[1]
                    x2 = iCar[2]
                    y2 = iCar[3]

                    car_center_x = (x1 + x2) // 2
                    car_center_y = (y1 + y2) // 2
        
                    distance = abs((ly2 - ly1) * car_center_x - (lx2 - lx1) * car_center_y + lx2 * ly1 - ly2 * lx1) / np.sqrt((ly2 - ly1)**2 + (lx2 - lx1)**2)
                    self.log.debug(f"Line:{line_number}, Car:{itr}, Dist:{distance}")
                    if distance < self.settings_json['settings']['curb_detection']["dist_bw_curb_car"]:  # Threshold for being "near curb"
                        # color = (255,255,random.randint(100,200))
                        vehicles[itr][4]= line_number
                        # color = (255,255,random.randint(100,200)) # Green for detected and near curb
                    # else:
                        # cars[itr][4]= np.nan
                        # color = (0, 0, 255)  # Red for detected but not near curb


            return vehicles

        # Convert to RGB for display
        # image_with_detections_rgb = cv2.cvtColor(image_with_detections, cv2.COLOR_BGR2RGB)


     
        


