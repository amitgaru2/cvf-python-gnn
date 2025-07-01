import logging
import datetime


logger = logging.getLogger()


logger.addHandler(logging.StreamHandler())
# logger.addHandler(fh)
logger.setLevel(logging.INFO)
