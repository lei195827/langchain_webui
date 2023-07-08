import shutil
from typing import List

import pypinyin
import torch
from .MyFAISS import MyFAISS

from configs.model_config import *
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from pydantic import BaseModel
from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    SystemMessage
)
from configs.model_config import logger
from loader import UnstructuredPaddlePDFLoader, UnstructuredPaddleImageLoader
from textsplitter import ChineseTextSplitter


def singleton(cls):
    instance = None

    def wrapper(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(*args, **kwargs)
        return instance

    return wrapper


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        try:
            from torch.mps import empty_cache
            empty_cache()
        except Exception as e:
            print(e)
            print(
                "如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")


def get_pinyin(word):
    return "".join(pypinyin.lazy_pinyin(word))


class ChatMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = []

    def add_user_message(self, message: str) -> None:
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))

    def add_system_message(self, message: str) -> None:
        self.messages.append(SystemMessage(content=message))

    def clear(self) -> None:
        self.messages = []


def load_file(filepath, sentence_size=100):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".txt"):
        loader = TextLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredPaddlePDFLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
        loader = UnstructuredPaddleImageLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    elif filepath.lower().endswith(".csv"):
        loader = CSVLoader(filepath)
        docs = loader.load()
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    return docs


@singleton
class VectorStore:
    def __init__(self):
        self.db = None
        self.old_db = None
        self.embeddings = OpenAIEmbeddings()
        self.vs_path = None
        self.old_db_path = None
        self.source_dict = {}

    def load_old_vector_store(self, vs_path=None, kb_name="知识库"):
        if vs_path is None:
            vs_path = os.path.join(VS_ROOT_PATH, get_pinyin(kb_name))
        self.old_db_path = os.path.join(vs_path, "all_old")
        if os.path.exists(self.old_db_path):
            try:
                self.old_db = MyFAISS.load_local(self.old_db_path, self.embeddings)
            except Exception as e:
                logger.info(f"Failed to load  the vector store: {str(e)}")
                if os.path.isdir(self.old_db_path):
                    os.makedirs(self.old_db_path)
                    self.old_db = None
        return self.old_db

    def create_vector_store(self, documents=None, source="tmp", embeddings=OpenAIEmbeddings(), kb_name="知识库"):
        vs_path = os.path.join(VS_ROOT_PATH, get_pinyin(kb_name))
        self.vs_path = vs_path
        self.old_db = self.load_old_vector_store(vs_path=self.vs_path)
        if documents is not None:
            try:
                if self.old_db is not None:
                    db_tmp = MyFAISS.from_documents(documents, embeddings)
                    db_tmp_path = os.path.join(self.vs_path, get_pinyin(source))
                    db_tmp.save_local(db_tmp_path)
                    self.old_db.add_documents(documents)
                    self.db = self.old_db
                else:
                    db_tmp = MyFAISS.from_documents(documents, embeddings)
                    db_tmp_path = os.path.join(self.vs_path, get_pinyin(source))
                    db_tmp.save_local(db_tmp_path)
                    self.db = MyFAISS.from_documents(documents, embeddings)
            except Exception as e:
                logger.info(f"Failed to create the vector store: {str(e)}")
        else:
            self.db = self.old_db
        if self.db is not None:
            self.db.save_local(self.old_db_path)

    def delete_vector_store(self):
        if os.path.exists(self.vs_path):
            try:
                shutil.rmtree(self.vs_path)
                self.db = None
                self.old_db = None
                info = "已清除数据库"
                return info
            except Exception as e:
                info = f"由于{e}删除数据库失败"
                return info
        else:
            info = "不存在数据库"
            return info

    def delete_one_vector_store(self, source):
        """
        :param source: 删除的文件名basename
        :return: 文件删除信息
        """
        info = ''
        delete_path = os.path.join(self.vs_path, get_pinyin(source.split('.')[0]))
        if os.path.exists(delete_path):
            try:
                info += f"成功删除{source}文件"
                shutil.rmtree(delete_path)
            except Exception as e:
                info += f"无法删除{source}文件，错误码{e}\n"
        self.source_dict = self.get_docs_dict()
        if source in self.source_dict:
            del self.source_dict[source]
        else:
            info += f"文件{source}不存在"
        first_db = True
        if len(self.source_dict) > 0:
            for key, value in self.source_dict.items():
                tmp_db_path = os.path.join(self.vs_path, get_pinyin(str(key).split('.')[0]))
                try:
                    if first_db is True:
                        self.db = MyFAISS.load_local(tmp_db_path, self.embeddings)
                        first_db = False
                    else:
                        tmp_db = MyFAISS.load_local(tmp_db_path, self.embeddings)
                        self.db.merge_from(tmp_db)
                    self.db.save_local(self.old_db_path)
                except Exception as e:
                    info += f"{key}重新加载失败，错误码:{e}\n"
        else:
            self.db = None
            shutil.rmtree(self.old_db_path)
        return info

    def get_docs_dict(self):
        # 创建一个空列表
        source_list = []
        if not isinstance(self.source_dict, dict):
            self.source_dict = {}
        if self.db is not None:
            for key, value in self.db.docstore._dict.items():
                # 获取每个文档的元数据
                metadata = value.metadata
                # 获取每个文档的source属性
                source = metadata.get('source')
                # 将source添加到列表中
                source_list.append(source)
            source_set = set(source_list)
            # 遍历列表中的路径
            for path in source_set:
                # 使用os.path.basename()函数获取路径的文件名
                filename = os.path.basename(path)
                # 使用文件名作为键，路径作为值，添加到字典中
                self.source_dict[filename] = path
            return self.source_dict
        else:
            return {}
