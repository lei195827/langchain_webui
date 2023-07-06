import logging
import os
import openai

openai.proxy = "代理"
openai.api_key = 'API_KEY'
os.environ["OPENAI_API_KEY"] = 'API_KEY'
VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")
# 文本分句长度
SENTENCE_SIZE = 100
ZH_TITLE_ENHANCE = False

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)
