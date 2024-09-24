
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

env = os.environ["environment"]


def lambda_handler(event, context):
    try:
        print("Inside mlops-lambda Lambda !!!", event)
        print("Environment = ", env)
        return 'Lambda execution success!'
    except Exception as e:
        print('Error Occurred in lambda  Record :: ', e)
        logger.exception(e)
    finally:
        logger.info('End of mlops-lambda lambda')


