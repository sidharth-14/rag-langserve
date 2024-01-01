import pinecone
from langchain.prompts import ChatPromptTemplate
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, OnlinePDFLoader, DuckDBLoader
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.vectorstores.pinecone import Pinecone
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
cwd = os.getcwd()

langserve_facts = [
    """Introduction
Language models have transformed the way we interact with computers. From chatbots to virtual assistants, these models are the brains behind many of our favorite applications. But taking an idea from a prototype to a real-world application can be a daunting task, especially if youre not a machine learning expert. Thats where LangServe comes in.

What is LangServe?
LangServe is a powerful tool that simplifies the deployment of language models. Its like turning your language model prototype into a real, working application. Think of it as the bridge that connects your brilliant idea with the people who can benefit from it.

Why LangServe Matters
Imagine youve built a chatbot that can help answer questions. Youve created it in a notebook, tested it, and it works like a charm. But now, you want to share it with the world. Thats where LangServe comes in. It takes your prototype and turns it into a full-fledged application.

Heres why this is a game-changer:

1. Fast and Easy Deployment

LangServe makes deploying your language model quick and painless. You can go from a simple prototype to a real application thats ready for users in no time.

2. No Coding Hassles

You dont need to be a coding guru to use LangServe. Its designed to be user-friendly and doesnt require a deep understanding of complex programming.

3. Scaling Made Simple

LangServe ensures your application can handle multiple requests simultaneously. Its ready for production-level usage without any extra headaches.

4. Intermediate Results Access

Sometimes, you might want to see whats happening inside your application, even before the final result is ready. LangServe offers a way to check these intermediate steps, making it a great tool for debugging and improving your application.
"""]
loader = DuckDBLoader("""SELECT * FROM read_parquet('./docs/*.parquet')""")
# loader = PyPDFLoader("./docs/field-guide-to-data-science.pdf")
# loader = OnlinePDFLoader("https://wolfpaulus.com/wp-content/uploads/2017/05/field-guide-to-data-science.pdf")
data=loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
texts = text_splitter.split_documents(data)
print(f"New document number: {len(texts)}")


hf_api_token="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"
repo_id= "TinyLlama/TinyLlama-1.1B-Chat-v1.0" #"codellama/CodeLlama-7b-hf"
# embedding = OpenAIEmbeddings()
embedding=HuggingFaceHubEmbeddings(huggingfacehub_api_token=hf_api_token,)
pinecone.init(api_key="489869ae-cdee-4522-9efe-ab76364c72a9", 
              environment="gcp-starter")
index_name="langchain"

vectorstore = FAISS.from_texts(
    # texts=texts,
    [t.page_content for t in texts], 
    embedding=embedding,
    # index_name=index_name
)


retriever = vectorstore.as_retriever()

llm = HuggingFaceHub(
    repo_id=repo_id,
    huggingfacehub_api_token=hf_api_token,
)


template = """Answer the question based only on the following context:
{context}
Question: {question}.
"""
prompt = ChatPromptTemplate.from_template(template)


# RAG
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)