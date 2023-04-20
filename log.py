import logging

def info_logger(fname):
    logging.basicConfig(filename=fname, filemode="a", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
    logger = logging.getLogger()
    return logger

if __name__ == '__main__':
    logger = info_logger('log.log')
    logger.info("hello")
