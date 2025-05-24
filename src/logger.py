import logging
import os
from datetime import datetime

LOG_FILE_NAME = f'{datetime.now().strftime(format='%m_%d_%Y')}.log' 
LOG_FILE_PATH = os.path.join(os.getcwd(),"logs",LOG_FILE_NAME)
os.makedirs(LOG_FILE_PATH,exist_ok=True)


FILE_PATH = os.path.join(LOG_FILE_PATH,LOG_FILE_NAME)


logging.basicConfig(
    filename= FILE_PATH,
    filemode='w',
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


if __name__ =='__main__':
    logging.info("logging has started..")
