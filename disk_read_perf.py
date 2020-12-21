import time
import math
import argparse
import os
import logging
import gc

logging.basicConfig(
     filename='log_file_name.log',
     level=logging.INFO,
     format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
     datefmt='%H:%M:%S'
 )

# Set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# Set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
# Add the handler to the root logger
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Disk Read Perf")
parser.add_argument("--path", type=str, default="")

args = parser.parse_args()
path = args.path

iteration_count = 1
iteration = 1

start = int(round(time.time()*1000))
size = 0
dirs = os.listdir(path)

for i in dirs:
    i_start = int(round(time.time()*1000))
    d =  os.path.join(path, i)
    i_size = 0
    for f in os.listdir(d):
        file_path = os.path.join(d, f)
        f = open(file_path, mode='rb')
        content = f.read()
        f.close()
        i_size = i_size + os.path.getsize(file_path)
        size = size + i_size
    i_end = int(round(time.time()*1000))
    i_time = i_end - i_start
    i_speed = math.ceil(i_size*1000/i_time)
    logger.info("folder: {}\tsize: {:,}\ttime: {}ms\tspeed:{:,}Bps".format(
        i, i_size, i_time, i_speed))
    gc.collect()
end = int(round(time.time()*1000))
time = end-start
speed = math.ceil(size*1000/time)
logger.info(
    "Total\tsize: {:,}\ttime: {}ms\tspeed:{:,}Bps".format(size, time, speed))
