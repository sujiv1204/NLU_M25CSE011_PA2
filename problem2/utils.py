import logging
import os

def get_logger(name="name_generation_pipeline"):
    """
    Creates and configures a centralized logger that outputs to both 
    the console and a unified log file.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        os.makedirs("outputs", exist_ok=True)
        
        # Log everything to one file
        file_handler = logging.FileHandler("outputs/experiment.log")
        console_handler = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Prevent propagation to root logger to avoid duplicate prints
        logger.propagate = False
        
    return logger
