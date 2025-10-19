# app.py
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import gradio as gr

PERSIST_DIR = "storage"
COLLECTION = "proj_docs"

def format_docs(docs):
    return "\n\n".join(
        [f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)]
    )

def build_chain():
    model_id = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    llm = ChatGoogleGenerativeAI(model=model_id)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    )
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    system = (
        "Aşağıdaki kullanıcı sorusunu sadece verilen bağlamdan yararlanarak yanıtla. "
        "Kaynak yoksa 'Verilerimde buna dair bilgi yok' de."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Soru: {question}\n\nBağlam:\n{context}")
        ]
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain

chain = build_chain()

def chat_fn(message, history):
    try:
        response = chain.invoke(message)
        return str(response.content)
    except Exception as e:
        return f"Hata: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# 📚 RAG Chatbot (Gemini + Chroma + LangChain)")
    chat = gr.ChatInterface(fn=chat_fn)
    gr.Markdown("> Not: Yanıtlar yalnızca indekslenen belgelere dayanır.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
