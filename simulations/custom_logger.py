import logging


logger = logging.getLogger()

# filename = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S_log.txt")
# filename = os.path.join("logs", filename)
# fh = logging.FileHandler(filename)
# logger.addHandler(fh)

logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
