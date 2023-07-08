import logging
import os
import openai
# 正常应该是127.0.0.1
HOST = "0.0.0.0"
PORT = 8888
SHARE = False
openai.proxy = "127.0.0.1:7890"
os.environ["OPENAI_API_KEY"] = ''
VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")
# 文本分句长度
SENTENCE_SIZE = 100
ZH_TITLE_ENHANCE = False
# 匹配后单段上下文长度
CHUNK_SIZE = 250
# 知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，经测试设置为小于500时，匹配结果更精准
VECTOR_SEARCH_SCORE_THRESHOLD = 0

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)
