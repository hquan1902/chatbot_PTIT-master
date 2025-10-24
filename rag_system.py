# import glob
# import os
# import shutil
# from dotenv import load_dotenv
# from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings

# load_dotenv()
# if "OPENAI_API_KEY" not in os.environ:
#     print("Lỗi: OPENAI_API_KEY chưa được thiết lập. Vui lòng tạo file .env và thêm key vào.")
#     exit()

# EMBEDDING_MODEL = "text-embedding-3-small"
# CHROMA_DB_PATH = "./knowledge_base_ptit"
# OLD_DOCS_DIR = "./old_docs"
# NEW_DOCS_DIR = "./new_docs"


# def load_and_process_documents(docs_dir: str):
#     """Load và xử lý tài liệu từ thư mục"""
#     try:
#         loader = DirectoryLoader(
#             docs_dir,
#             glob="**/*",
#             loader_cls=UnstructuredFileLoader,
#             show_progress=True,
#             use_multithreading=True
#         )
#         documents = loader.load()
#         if not documents:
#             return [], []

#         for doc in documents:
#             if "source" in doc.metadata:
#                 doc.metadata["file_name"] = os.path.basename(doc.metadata["source"])

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
#         chunks = text_splitter.split_documents(documents)
#         source_files = list(set([doc.metadata.get("source") for doc in documents]))
#         return chunks, source_files
#     except Exception as e:
#         print(f"Lỗi khi load tài liệu: {e}")
#         return [], []


# def initialize_vector_store(db_path: str, embedding_model: str, docs_dir: str):
#     """Khởi tạo hoặc load Vector Store"""
#     try:
#         embeddings = OpenAIEmbeddings(model=embedding_model)

#         if os.path.exists(db_path):
#             print(f"Đang load Knowledge Base từ '{db_path}'...")
#             vector_store = Chroma(
#                 persist_directory=db_path,
#                 embedding_function=embeddings
#             )
#         else:
#             print(f"Knowledge Base chưa tồn tại. Đang tạo mới từ '{docs_dir}'...")
#             chunks, _ = load_and_process_documents(docs_dir)

#             if not chunks:
#                 print("Không có tài liệu ban đầu, tạo một Knowledge Base rỗng.")
#                 vector_store = Chroma(
#                     embedding_function=embeddings,
#                     persist_directory=db_path
#                 )
#             else:
#                 print(f"Đang embedding {len(chunks)} chunks...")
#                 vector_store = Chroma.from_documents(
#                     chunks,
#                     embedding=embeddings,
#                     persist_directory=db_path
#                 )

#             print(f"Đã tạo và lưu Knowledge Base vào '{db_path}'.")

#         return vector_store
#     except Exception as e:
#         print(f"Lỗi khi khởi tạo Vector Store: {e}")
#         exit(1)


# def check_and_update_database(vector_store: Chroma, new_docs_dir: str, old_docs_dir: str):
#     """Kiểm tra và cập nhật database với tài liệu mới"""
#     try:
#         if not os.path.exists(new_docs_dir) or not os.listdir(new_docs_dir):
#             print("Thư mục 'new_docs' rỗng, không có gì để cập nhật.")
#             return

#         new_chunks, processed_files = load_and_process_documents(new_docs_dir)
#         if not new_chunks:
#             print("Không tìm thấy tài liệu mới hợp lệ.")
#             return

#         # Lấy tên file từ metadata
#         new_file_names = set()
#         for chunk in new_chunks:
#             if "file_name" in chunk.metadata:
#                 new_file_names.add(chunk.metadata["file_name"])

#         old_files = [
#             os.path.basename(f)
#             for f in glob.glob(os.path.join(old_docs_dir, "**", "*.*"), recursive=True)
#         ]

#         existing = [f for f in old_files if f in new_file_names]

#         if existing:
#             print(f"Đã phát hiện {len(existing)} file cũ, đang cập nhật...")
#             for chunk in new_chunks:
#                 if "file_name" in chunk.metadata and chunk.metadata["file_name"] in existing:
#                     vector_store.delete(where={"file_name": chunk.metadata["file_name"]})
#             print(f"Đã xóa {len(existing)} file cũ khỏi Knowledge Base.")

#         print(f"Đang thêm {len(new_chunks)} chunks mới vào Knowledge Base...")
#         vector_store.add_documents(new_chunks)

#         # Di chuyển file từ new_docs sang old_docs
#         for file_path in processed_files:
#             file_name = os.path.basename(file_path)
#             destination_path = os.path.join(old_docs_dir, file_name)
#             if os.path.exists(destination_path):
#                 os.remove(destination_path)
#             shutil.move(file_path, destination_path)

#         print(f"Đã di chuyển {len(processed_files)} file sang 'old_docs'.")
#         print("--- Hoàn tất quy trình cập nhật ---")

#     except Exception as e:
#         print(f"Lỗi khi cập nhật database: {e}")


# # ✅ HÀM THÊM MỚI CHO FLASK
# def update_knowledge_base_auto():
#     """Hàm tự động cập nhật Knowledge Base mà không cần nhập tay (dùng cho Flask)."""
#     os.makedirs(OLD_DOCS_DIR, exist_ok=True)
#     os.makedirs(NEW_DOCS_DIR, exist_ok=True)

#     db = initialize_vector_store(CHROMA_DB_PATH, EMBEDDING_MODEL, OLD_DOCS_DIR)
#     try:
#         check_and_update_database(db, NEW_DOCS_DIR, OLD_DOCS_DIR)
#     except Exception as e:
#         print(f"Lỗi khi tự động cập nhật: {e}")


# if __name__ == "__main__":
#     print("=== Bắt đầu quá trình xây dựng/cập nhật Knowledge Base ===\n")
#     os.makedirs(OLD_DOCS_DIR, exist_ok=True)
#     os.makedirs(NEW_DOCS_DIR, exist_ok=True)
#     db = initialize_vector_store(CHROMA_DB_PATH, EMBEDDING_MODEL, OLD_DOCS_DIR)
#     check_and_update_database(db, NEW_DOCS_DIR, OLD_DOCS_DIR)
#     print("\n=== Hoàn tất ===")





import glob
import os
import shutil
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# --- Load OpenAI API key ---
load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    print("Lỗi: OPENAI_API_KEY chưa được thiết lập. Vui lòng tạo file .env và thêm key vào.")
    exit()

# --- Cấu hình ---
EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_DB_PATH = "./knowledge_base_ptit"
OLD_DOCS_DIR = "./old_docs"
NEW_DOCS_DIR = "./new_docs"

# === TIỆN ÍCH HASH ĐỂ PHÁT HIỆN TRÙNG LẶP ===
def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# === LOAD FILE ĐA ĐỊNH DẠNG ===
def load_text_from_file(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".csv":
            # Chỉ dùng nếu CSV chứa văn bản (không phải dữ liệu bảng)
            loader = CSVLoader(file_path)
        else:
            print(f"❌ Bỏ qua file không hỗ trợ: {file_path}")
            return []
        return loader.load()
    except Exception as e:
        print(f"⚠️ Lỗi đọc file {file_path}: {e}")
        return []

# === XỬ LÝ VĂN BẢN ===
def load_and_process_documents(docs_dir: str):
    """Đọc & chia nhỏ tài liệu"""
    documents = []
    for filename in os.listdir(docs_dir):
        path = os.path.join(docs_dir, filename)
        if os.path.isfile(path):
            docs = load_text_from_file(path)
            for d in docs:
                d.metadata["file_name"] = filename
            documents.extend(docs)

    if not documents:
        return [], []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks, [os.path.join(docs_dir, f) for f in os.listdir(docs_dir)]

# === KHỞI TẠO VECTOR STORE ===
def initialize_vector_store(db_path: str, embedding_model: str, docs_dir: str):
    embeddings = OpenAIEmbeddings(model=embedding_model)

    if os.path.exists(db_path):
        print(f"Đang load Knowledge Base từ '{db_path}'...")
        vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
    else:
        print(f"Tạo mới Knowledge Base từ '{docs_dir}'...")
        chunks, _ = load_and_process_documents(docs_dir)
        vector_store = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=db_path)
        print(f"✅ Knowledge Base đã được tạo và lưu vào '{db_path}'.")

    return vector_store

# === CẬP NHẬT DATABASE ===
def check_and_update_database(vector_store: Chroma, new_docs_dir: str, old_docs_dir: str):
    """Cập nhật DB khi có file mới (loại trùng nội dung theo hash)"""
    if not os.path.exists(new_docs_dir) or not os.listdir(new_docs_dir):
        print("📂 Không có file mới trong 'new_docs'.")
        return

    new_chunks, processed_files = load_and_process_documents(new_docs_dir)
    if not new_chunks:
        print("⚙️ Không có tài liệu hợp lệ.")
        return

    # Tạo set hash của các đoạn đã có
    existing_hashes = set()
    try:
        existing_docs = vector_store.get(include=["metadatas"])
        for meta in existing_docs["metadatas"]:
            if "hash" in meta:
                existing_hashes.add(meta["hash"])
    except Exception:
        pass

    # Gắn hash mới và lọc trùng
    unique_chunks = []
    for chunk in new_chunks:
        h = compute_hash(chunk.page_content)
        if h not in existing_hashes:
            chunk.metadata["hash"] = h
            unique_chunks.append(chunk)
            existing_hashes.add(h)

    if not unique_chunks:
        print("✅ Không có đoạn mới (toàn bộ nội dung đã tồn tại).")
        return

    print(f"🧠 Đang thêm {len(unique_chunks)} đoạn mới...")
    vector_store.add_documents(unique_chunks)
    vector_store.persist()

    os.makedirs(old_docs_dir, exist_ok=True)
    for f in processed_files:
        shutil.move(f, os.path.join(old_docs_dir, os.path.basename(f)))

    print(f"🎉 Đã thêm {len(unique_chunks)} đoạn mới và di chuyển file sang 'old_docs'.")

# === HÀM GỌI TỪ FLASK ===
def update_knowledge_base_auto():
    """Hàm tự động cập nhật khi upload từ Flask"""
    os.makedirs(OLD_DOCS_DIR, exist_ok=True)
    os.makedirs(NEW_DOCS_DIR, exist_ok=True)
    db = initialize_vector_store(CHROMA_DB_PATH, EMBEDDING_MODEL, OLD_DOCS_DIR)
    check_and_update_database(db, NEW_DOCS_DIR, OLD_DOCS_DIR)

# === MAIN CHẠY ĐỘC LẬP ===
if __name__ == "__main__":
    print("=== BẮT ĐẦU CẬP NHẬT CƠ SỞ TRI THỨC ===\n")
    os.makedirs(OLD_DOCS_DIR, exist_ok=True)
    os.makedirs(NEW_DOCS_DIR, exist_ok=True)
    db = initialize_vector_store(CHROMA_DB_PATH, EMBEDDING_MODEL, OLD_DOCS_DIR)
    check_and_update_database(db, NEW_DOCS_DIR, OLD_DOCS_DIR)
    print("\n=== HOÀN TẤT ===")
