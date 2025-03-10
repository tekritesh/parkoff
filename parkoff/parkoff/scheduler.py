import schedule
import time
import logging
from base import ParkOff
from logging.handlers import RotatingFileHandler
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logpath', help='filepath to cache logs',default="")

args = parser.parse_args()



# Configure logger
def setup_logger(file_path=""):
    logger = logging.getLogger("app_parkoff")
    logger.setLevel(logging.INFO)

    # Create a rotating file handler (5 MB per file, keep last 3)
    logfile = "app_parkoff.log"
    if os.path.exists(file_path):
        logfile = os.path.join(file_path, logfile)
    else:
        logfile = os.path.join(os.getcwd(), logfile)

    
    handler = RotatingFileHandler(
        logfile, maxBytes=1 * 1024 * 1024, backupCount=3
    )
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger

logger = setup_logger(args.logpath)

def task():
    
    logger.info(f"Running Task Scheduled Task")
    try:
        inst = ParkOff(logger =logger)
        inst.get_cpu_temperature()
        inst.cache_all_images()
        inst.analyse_image()
        
        logger.info(f"Finished Task")
    
    except Exception as e:
        logger.error(e)
    
    

# Schedule the task to run every 10 seconds
schedule.every(60).seconds.do(task)

# Keep the scheduler running
while True:
    schedule.run_pending()
    time.sleep(1)
