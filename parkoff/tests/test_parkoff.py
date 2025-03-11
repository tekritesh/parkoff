import pandas as pd
from parkoff.base import ParkOff
import logging
import cv2
import os
import shutil
import glob


def test_clean_all():
    
    try:
        shutil.rmtree("tests/sample/")
    except FileNotFoundError as e:
        print(e)
    except OSError as e:
        print(e)
    except PermissionError  as e:
        print(e)

def test_init_load():
    try:
        inst = ParkOff(log_level=logging.DEBUG,settings_json="settings.json")
    except Exception as e:
        logging.info(e)

    
    assert inst is not None 


def test_cache_image_from_url():
    

    inst = ParkOff(log_level=logging.DEBUG,settings_json="settings.json")
    
    url = 'https://www.seattle.gov/trafficcams/images/1_Seneca_EW.jpg?619'
    cache_dir="tests/sample/"
    inst.cache_image_from_url(url=url, cache_dir=cache_dir)

    img_file = glob.glob(os.path.join(cache_dir, "*.png"))

    assert img_file is not None, f"Failed to load image from {img_file}"


def test_analyse_image():

    inst = ParkOff(log_level=logging.DEBUG,settings_json="settings.json")
    cache_dir="tests/sample/"
    # img_file = glob.glob(os.path.join(cache_dir, "*.png"))[0]
    out_file = "tests/sample/out/"
    inst.analyse_image(input_img_dir=cache_dir,output_img_dir=out_file,debug=True)





