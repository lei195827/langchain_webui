import time
import gradio as gr
from utils import *
from langchain.chat_models import ChatOpenAI


# 初始化记忆池
chat_history = ChatMessageHistory()
chatkn_history = ChatMessageHistory()


def chat(query, chat_chatbot, system_prompt, temperature):
    """
        query是必须的
    """
    if len(chat_history.messages) > 0:
        if type(chat_history.messages[0]) is SystemMessage:
            if system_prompt:
                chat_history.messages[0] = SystemMessage(content=system_prompt)
        else:
            if system_prompt:
                chat_history.messages[0].insert(0, AIMessage(content=system_prompt))
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


def save_file(files, sentence_size):
    """
        建立数据库的回调函数，先读取文件，再
    """
    directory_path = "/root/langchain-chatbot/docs"
    os.makedirs(directory_path, exist_ok=True)  # 创建保存文件的目录

    def move_file(source_path, destination_path):
        try:
            shutil.move(source_path, destination_path)
            print(f"移动文件{source_path}到{destination_path}")
            return f"移动文件{source_path}到{destination_path}\n" \
                   f"已创建数据库"
        except Exception as e:
            print(f"移动文件{source_path}失败: {str(e)}")
            return f"移动文件{source_path}失败: {str(e)}"

    if isinstance(files, list):
        result = []
        for file in files:
            filebasename = os.path.basename(file.name)
            file_path = os.path.join(directory_path, filebasename)
            result.append(move_file(file.name, file_path))
    else:
        filebasename = os.path.basename(files.name)
        file_path = os.path.join(directory_path, filebasename)
        result = move_file(files.name, file_path)

    documents = load_file(file_path, sentence_size=sentence_size)
    vector_store.create_vector_store(documents=documents)
    return result


def kn_chat(know_ask_input, chat_chatbot, kv_num=4, min_score=0.3):
    db = vector_store.db
    if db is None:
        chat_chatbot.append((know_ask_input, "请先加载或者上传知识库"))
        return "", chat_chatbot
    docs_and_scores = db.similarity_search_with_score(query=know_ask_input, k=kv_num)
    kn_vector = ""
    kn = []
    # 假设docs_and_scores是db.similarity_search_with_score(query)返回的文档和分数列表
    for i, (doc, score) in enumerate(docs_and_scores):
        # 将文档的索引，内容，元数据和分数添加到字符串中，用换行符分隔
        if score >= min_score:
            kn_vector += f"<details><summary>来自文件{doc.metadata['filename']}</summary>\n\n"
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


# 初始化向量数据库
vector_store = VectorStore()
vector_store.create_vector_store(documents=None)

with gr.Blocks() as demo:
    gr.Markdown("知识库测试")
    with gr.Tab("普通聊天模式"):
        with gr.Row():
            with gr.Column(scale=0.7):
                chatbot = gr.Chatbot(label="聊天记录")
                msg = gr.Textbox(label="提问")
                clear = gr.ClearButton([msg, chatbot])
            with gr.Column(scale=0.3):
                with gr.Accordion("参数"):
                    temperature = gr.Slider(step=0.01, minimum=0, maximum=1, label="temperature", value=0.7)
                    text_sysprompt_input = gr.Textbox(label="AI人设")

    with gr.Tab("知识库问答"):
        with gr.Row():
            with gr.Column(scale=0.7):
                chatbot_kn = gr.Chatbot(label="聊天记录")
                know_ask_input = gr.Textbox(label="询问")
                clear_kn = gr.ClearButton([know_ask_input, chatbot])
            with gr.Column(scale=0.3):
                gr.Markdown("分数越高检索到的答案越少")
                kv_num = gr.Slider(step=1, minimum=0, maximum=20, label="知识库检索数目", value=4)
                min_score = gr.Slider(step=0.01, minimum=0, maximum=1, label="分数阈值", value=0.25)
                sentence_size = gr.Slider(step=1, minimum=50, maximum=1000, label="分数阈值", value=200)
                docs_input = gr.File()

                docx_text = gr.Textbox(label="通知栏")
                vector_store_creat_button = gr.Button("新建或知识库")
                vector_store_delete_button = gr.Button("删除数据库")

    with gr.Accordion("Open for More!"):
        gr.Markdown("Look at  me...")

    msg.submit(chat, inputs=[msg, chatbot, text_sysprompt_input, temperature], outputs=[msg, chatbot])
    know_ask_input.submit(kn_chat, inputs=[know_ask_input, chatbot_kn, kv_num, min_score],
                          outputs=[know_ask_input, chatbot_kn])
    vector_store_creat_button.click(save_file, inputs=[docs_input, sentence_size], outputs=docx_text)
    vector_store_delete_button.click(vector_store.delete_vector_store, outputs=docx_text)
demo.launch(server_name='0.0.0.0', server_port=8888)
