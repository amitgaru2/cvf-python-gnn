import logging
import datetime


logger = logging.getLogger()

filename = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S_log.txt")
fh = logging.FileHandler(filename)

# logger.addHandler(logging.StreamHandler())
logger.addHandler(fh)
logger.setLevel(logging.INFO)
