import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_DB_PATH = "./knowledge_base_ptit"
os.makedirs(CHROMA_DB_PATH, exist_ok=True)


class RAGChatbot:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=self.embeddings
        )
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3
        )

        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 4}  # lấy 4 đoạn liên quan nhất
        )

        template = """
        Bạn là trợ lý ảo của Học viện Công nghệ Bưu chính Viễn thông (PTIT).
        Nhiệm vụ của bạn là chỉ trả lời các câu hỏi liên quan đến PTIT.
        Nếu người dùng hỏi về các trường khác, hoặc chủ đề không liên quan tới PTIT,
        hãy trả lời: "Tôi chỉ có thể cung cấp thông tin liên quan đến Học viện Công nghệ Bưu chính Viễn thông (PTIT)."

        Dưới đây là thông tin lấy được từ tài liệu nội bộ của PTIT (nếu có):
        ----------------
        {context}
        ----------------
        Dựa trên thông tin trên, hãy trả lời câu hỏi:
        {question}

        Nếu không tìm thấy câu trả lời trong tài liệu PTIT, hãy nói rõ rằng bạn chưa có dữ liệu cụ thể.
        """

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

    def get_answer(self, question: str):
        """Truy vấn câu hỏi qua RAG"""
        try:
            response = self.qa_chain.invoke({"query": question})
            return response["result"].strip()
        except Exception as e:
            return f"Lỗi khi truy vấn RAG: {str(e)}"
