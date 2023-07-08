import os.path
import shutil
import time
import gradio as gr
from configs.model_config import *
from langchain.chat_models import ChatOpenAI
from utils.utils import ChatMessageHistory, load_file, VectorStore
from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    SystemMessage
)

# 初始化记忆池
chat_history = ChatMessageHistory()
chatkn_history = ChatMessageHistory()
# 初始化向量数据库
vector_store = VectorStore()
vector_store.create_vector_store(documents=None)
vs_file_dict = vector_store.get_docs_dict()


def delete_one_file(vs_file_choice_dropdown):
    global vs_file_dict
    info = ""
    vs_file_dict = vector_store.get_docs_dict()
    if isinstance(vs_file_dict, dict) and len(vs_file_dict) > 0:
        try:
            info = vector_store.delete_one_vector_store(source=vs_file_choice_dropdown)
        except Exception as e:
            info += f"dict is not exist,{e}"
            logger.exception(f"dict is not exist,{e}")
    else:
        info += "数据库为空"
        return info, gr.Dropdown.update(choices=None)
    vs_file_dict = vector_store.get_docs_dict()
    if len(vs_file_dict) == 0:
        info = vector_store.delete_vector_store()
    return info, gr.Dropdown.update(choices=list(vs_file_dict.keys()))


def delete_all_file():
    global vs_file_dict
    info = vector_store.delete_vector_store()
    vs_file_dict = vector_store.get_docs_dict()
    return info, gr.Dropdown.update(choices=list(vs_file_dict.keys()))


def move_file(source_path, destination_path):
    try:
        shutil.move(source_path, destination_path)
        print(f"移动文件{str(source_path)}到{str(destination_path)}")
        return f"已增加文件{os.path.basename(destination_path)}至数据库"
    except Exception as e:
        print(f"移动文件{str(source_path)}失败: {str(e)}")
        return f"增加文件{os.path.basename(destination_path)}至数据库失败: {str(e)}"


def build_vs_by_file(files, sentence_size):
    """
        建立数据库的回调函数，先读取文件，再
    """
    vector_store = VectorStore()
    directory_path = os.path.join(VS_ROOT_PATH, "docs")
    os.makedirs(directory_path, exist_ok=True)  # 创建保存文件的目录
    try:
        if isinstance(files, list):
            result = []
            for file in files:
                filebasename = os.path.basename(file.name)
                file_path = os.path.join(directory_path, filebasename)
                result.append(move_file(file.name, file_path))
                try:
                    documents = load_file(file_path, sentence_size=sentence_size)
                    vector_store.create_vector_store(documents=documents, source=filebasename.split('.')[0])
                except Exception as e:
                    logger.warning(f"failed to load file {file},{e}")
                    result = f"failed to load file {file},{e}"
        else:
            filebasename = os.path.basename(files.name)
            file_path = os.path.join(directory_path, filebasename)
            result = move_file(files.name, file_path)
            try:
                documents = load_file(file_path, sentence_size=sentence_size)
                vector_store.create_vector_store(documents=documents,source=filebasename.split('.')[0])
            except Exception as e:
                logger.warning(f"failed to load file {files},{e}")
                result = f"failed to load file {files},{e}"
    except Exception as e:
        logger.warning("failed to build vector_store")
        result = f"failed to add file {str(files)} vector_store,error:{e}"
    vs_file_dict = vector_store.get_docs_dict()
    return result, gr.Dropdown.update(choices=list(vs_file_dict.keys()))


def chat(query, chat_chatbot, system_prompt, temperature):
    if len(chat_history.messages) > 0:
        if type(chat_history.messages[0]) is SystemMessage:
            if system_prompt:
                chat_history.messages[0] = SystemMessage(content=system_prompt)
        else:
            if system_prompt:
                chat_history.messages[0].insert(0, SystemMessage(content=system_prompt))
    else:
        if system_prompt:
            chat_history.add_system_message(system_prompt)

    chat_history.add_user_message(query)
    chat = ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo")

    response = chat(chat_history.messages)
    chat_history.add_ai_message(response.content)
    chat_chatbot.append((query, response.content))
    time.sleep(1)
    return "", chat_chatbot


def kn_chat(know_ask_input, chat_chatbot, kv_num=4, min_score=0.3):
    vector_store = VectorStore()
    db = vector_store.db
    if db is None:
        chat_chatbot.append((know_ask_input, "请先加载或者上传知识库"))
        return "", chat_chatbot
    # try:
    docs_and_scores = db.similarity_search_with_score(query=know_ask_input, k=kv_num)
    # except Exception as e:
    #     info = f"数据库搜索出错，请检查是否有上传文件或所有文件已被删除,错误码:{e}"
    #     chat_chatbot.append((know_ask_input, info))
    #     return "", chat_chatbot
    kn_vector = ""
    kn = []
    # 假设docs_and_scores是db.similarity_search_with_score(query)返回的文档和分数列表
    for i, (doc, score) in enumerate(docs_and_scores):
        # 将文档的索引，内容，元数据和分数添加到字符串中，用换行符分隔
        if score >= min_score:
            kn_vector += f"<details><summary>来自文件{os.path.basename(doc.metadata['source'])}</summary>\n\n"
            kn_vector += doc.page_content + "\n"
            kn_vector += f"Document {i + 1}:\t"
            kn_vector += "Score:"
            kn_vector += str(f"{score:.3f}") + "\n"
            kn_vector += "</details>"
            kn.append(f"Document{i + 1}:doc.page_content\n")
    query = f"""
            Please response in Chinese,
            I will ask you questions based on the following context:
            - Start of Context -
            {kn}
            - End of Context-
            My question is:“{know_ask_input}"

            """
    chatkn_history.add_user_message(query)
    chat = ChatOpenAI(model_name="gpt-3.5-turbo")
    response = chat(chatkn_history.messages)
    chatkn_history.add_ai_message(response.content)
    chat_chatbot.append((know_ask_input, response.content + "\n" + kn_vector))
    time.sleep(1)
    return "", chat_chatbot


with gr.Blocks() as demo:
    gr.Markdown("知识库测试")
    with gr.Tab("普通聊天模式"):
        with gr.Row():
            with gr.Column(scale=0.7):
                chatbot = gr.Chatbot(label="聊天记录")
                ask_input = gr.Textbox(label="提问")
                clear = gr.ClearButton([ask_input, chatbot])
            with gr.Column(scale=0.3):
                with gr.Accordion("参数"):
                    temperature = gr.Slider(step=0.01, minimum=0, maximum=1, label="temperature", value=0.7)
                    text_sysprompt_input = gr.Textbox(label="AI人设")

    with gr.Tab("知识库问答"):
        with gr.Row():
            with gr.Column(scale=0.7):
                chatbot_kn = gr.Chatbot(label="聊天记录")
                know_ask_input = gr.Textbox(label="询问")
                clear_kn = gr.ClearButton([know_ask_input, chatbot_kn])
            with gr.Column(scale=0.3):
                gr.Markdown("分数越高检索到的答案越少")
                kv_num = gr.Slider(step=1, minimum=0, maximum=20, label="知识库检索数目", value=4)
                min_score = gr.Slider(step=0.01, minimum=0, maximum=1, label="分数阈值", value=0.25)
                sentence_size = gr.Slider(step=1, minimum=50, maximum=1000, label="分割长度", value=200)
                docs_input = gr.File()
                info_docx_text = gr.Textbox(label="通知栏")
                with gr.Row():
                    vs_creat_button = gr.Button("新建或增加知识库")
                    vs_delete_button = gr.Button("删除数据库")
                with gr.Row():
                    vs_file_choice_dropdown = gr.Dropdown(choices=vs_file_dict,
                                                          label="删除指定文件")
                    vs_file_delete_button = gr.Button("删除指定文件")

    with gr.Accordion("Open for More!"):
        gr.Markdown("Look at  me...")

    ask_input.submit(chat, inputs=[ask_input, chatbot, text_sysprompt_input, temperature], outputs=[ask_input, chatbot])
    know_ask_input.submit(kn_chat, inputs=[know_ask_input, chatbot_kn, kv_num, min_score],
                          outputs=[know_ask_input, chatbot_kn])
    vs_creat_button.click(build_vs_by_file, inputs=[docs_input, sentence_size],
                          outputs=[info_docx_text, vs_file_choice_dropdown])
    # 删除整个知识库
    vs_delete_button.click(delete_all_file, outputs=[info_docx_text, vs_file_choice_dropdown])
    vs_file_delete_button.click(delete_one_file, inputs=[vs_file_choice_dropdown],
                                outputs=[info_docx_text, vs_file_choice_dropdown])

demo.launch(server_name=HOST, server_port=PORT, share=SHARE)
