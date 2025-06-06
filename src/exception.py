import sys
from src.logger import logging


def error_message_detail(error, error_detail: sys):
    _, _, exec_tb = error_detail.exc_info()
    file_name = exec_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python in file [{0}] line_number [{1}] and error message [{2}]".format(
        file_name, exec_tb.tb_lineno, str(error)
    )
    
    return error_message
   

class CustomException(Exception) :
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail)
        
    def __str__(self):
        return self.error_message

if __name__ =="__main__":
    try:
        div = 1/0
    
    except Exception as e:
        logging.error("Divide by Zero error")
        raise CustomException(e,sys)
    
        