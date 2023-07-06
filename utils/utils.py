import shutil
from typing import List

import pypinyin
import torch
from langchain import FAISS

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

    def load_old_vector_store(self, vs_path=None, kb_name="知识库"):
        if vs_path is None:
            vs_path = os.path.join(VS_ROOT_PATH, get_pinyin(kb_name))
        if os.path.exists(vs_path):
            try:
                self.old_db = FAISS.load_local(os.path.join(vs_path), self.embeddings)
            except Exception as e:
                logger.info(f"Failed to load  the vector store: {str(e)}")
                if not os.path.isdir(vs_path):
                    os.makedirs(vs_path)
                    self.old_db = None
        return self.old_db

    def create_vector_store(self, documents=None, embeddings=OpenAIEmbeddings(), kb_name="知识库"):
        vs_path = os.path.join(VS_ROOT_PATH, get_pinyin(kb_name))
        self.vs_path = vs_path
        self.old_db = self.load_old_vector_store(vs_path=vs_path)
        if documents is not None:
            try:
                self.old_db = self.load_old_vector_store()
                self.db = FAISS.from_documents(documents, embeddings)
                if self.old_db is not None:
                    self.db.merge_from(self.old_db)
            except Exception as e:
                logger.info(f"Failed to create the vector store: {str(e)}")
        else:
            self.db = self.old_db
        if self.db is not None:
            self.db.save_local(os.path.join(vs_path))

    def delete_vector_store(self):
        shutil.rmtree(self.vs_path)
        self.db = None
        self.old_db = None
        return "已清除数据库"
