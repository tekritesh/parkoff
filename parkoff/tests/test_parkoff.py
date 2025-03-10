import pandas as pd
from parkoff.base import ParkOff
import logging

def init_load():
    try:
        inst = ParkOff(og_level=logging.DEBUG,settings_json="../settings.json")
    except Exception as e:
        logging.info(e)

    
    assert inst is not None 

