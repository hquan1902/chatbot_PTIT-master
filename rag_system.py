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


# === CẬP NHẬT DATABASE (PHIÊN BẢN XÓA FILE LỖI/TRÙNG LẶP) ===
def check_and_update_database(vector_store: Chroma, new_docs_dir: str, old_docs_dir: str):
    """
    Cập nhật DB.
    - File mới, hợp lệ -> Chuyển vào old_docs.
    - File lỗi hoặc nội dung trùng lặp -> Xóa vĩnh viễn.
    """
    if not os.path.exists(new_docs_dir) or not os.listdir(new_docs_dir):
        print("📂 Không có file mới trong 'new_docs'.")
        return

    new_chunks, processed_files = load_and_process_documents(new_docs_dir)
    
    # --- XỬ LÝ TRƯỜNG HỢP FILE BỊ LỖI, KHÔNG ĐỌC ĐƯỢC ---
    if not new_chunks:
        print("⚙️ Không có nội dung hợp lệ hoặc không thể đọc được file.")
        for f in processed_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"🗑️ Đã xóa file lỗi: {os.path.basename(f)}")
            except Exception as e:
                print(f"⚠️ Lỗi khi xóa file {os.path.basename(f)}: {e}")
        return

    # --- XỬ LÝ KIỂM TRA TRÙNG LẶP NỘI DUNG ---
    existing_hashes = set()
    try:
        existing_docs = vector_store.get(include=["metadatas"])
        for meta in existing_docs["metadatas"]:
            if "hash" in meta:
                existing_hashes.add(meta["hash"])
    except Exception:
        pass

    unique_chunks = []
    for chunk in new_chunks:
        h = compute_hash(chunk.page_content)
        if h not in existing_hashes:
            chunk.metadata["hash"] = h
            unique_chunks.append(chunk)
            existing_hashes.add(h)

    # --- QUYẾT ĐỊNH XÓA HAY DI CHUYỂN DỰA TRÊN KẾT QUẢ ---
    
    # Trường hợp 1: Có nội dung mới, hợp lệ
    if unique_chunks:
        print(f"🧠 Đang thêm {len(unique_chunks)} đoạn mới...")
        vector_store.add_documents(unique_chunks)
        
        # Di chuyển file gốc vào old_docs để lưu trữ
        os.makedirs(old_docs_dir, exist_ok=True)
        for f in processed_files:
            if os.path.exists(f):
                shutil.move(f, os.path.join(old_docs_dir, os.path.basename(f)))
        
        print(f"🎉 Đã thêm tri thức mới và di chuyển {len(processed_files)} file gốc sang 'old_docs'.")

    # Trường hợp 2: Toàn bộ nội dung đều đã tồn tại (trùng lặp)
    else:
        print("✅ Toàn bộ nội dung trong file đã tồn tại trong tri thức.")
        for f in processed_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"🗑️ Đã xóa file trùng lặp: {os.path.basename(f)}")
            except Exception as e:
                print(f"⚠️ Lỗi khi xóa file {os.path.basename(f)}: {e}")


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
