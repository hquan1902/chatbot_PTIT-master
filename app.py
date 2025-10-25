from flask import Flask, render_template, request, jsonify
import json, os, shutil
from dotenv import load_dotenv

# THÊM HÀM initialize_vector_store TỪ FILE rag_system
from rag_system import initialize_vector_store 

from rag_chatbot import RAGChatbot

# 1. Cấu hình Flask và API
app = Flask(__name__)
load_dotenv()

# Khởi tạo chatbot ban đầu
rag_chatbot = RAGChatbot()

# Các hằng số đường dẫn
CHAT_HISTORY_FILE = "chat_history.json"
OLD_DOCS_DIR = "./old_docs"
CHROMA_DB_PATH = "./knowledge_base_ptit"
EMBEDDING_MODEL = "text-embedding-3-small"


# 2. Quản lý lịch sử chat (Không đổi)
# ... (Giữ nguyên các hàm load_history và save_message)
if not os.path.exists(CHAT_HISTORY_FILE):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

def load_history():
    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_message(role, text):
    history = load_history()
    history.append({"role": role, "text": text})
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# --- CÁC ROUTE CỦA FLASK ---

# 3. Giao diện chính (Không đổi)
@app.route("/")
def index():
    return render_template("index.html", history=load_history())


# 4. Chat API (Không đổi)
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Tin nhắn trống."}), 400
    save_message("user", user_message)
    try:
        rag_reply = rag_chatbot.get_answer(user_message)
        if "Tôi không tìm thấy thông tin" in rag_reply or "Lỗi khi truy vấn" in rag_reply:
            bot_reply = "Xin lỗi, tôi chỉ có thể trả lời các thông tin liên quan đến PTIT và hiện chưa có dữ liệu cho câu hỏi này."
        else:
            bot_reply = rag_reply
    except Exception as e:
        bot_reply = f"Lỗi khi xử lý: {str(e)}"
    save_message("bot", bot_reply)
    return jsonify({"reply": bot_reply})


# 5. Giao diện Admin (Không đổi)
@app.route("/admin")
def admin_page():
    knowledge_files = []
    if os.path.exists(OLD_DOCS_DIR):
        knowledge_files = [f for f in os.listdir(OLD_DOCS_DIR) if os.path.isfile(os.path.join(OLD_DOCS_DIR, f))]
    return render_template("admin.html", knowledge_files=knowledge_files)


# 6. Upload file (Không đổi)
@app.route("/upload", methods=["POST"])
def upload():
    admin_password = os.getenv("ADMIN_PASSWORD")
    submitted_password = request.form.get("password")
    if not admin_password or submitted_password != admin_password:
        return jsonify({"error": "Mật khẩu không đúng hoặc chưa được thiết lập."}), 403
    if "file" not in request.files:
        return jsonify({"error": "Không có file được gửi."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Tên file không hợp lệ."}), 400
    try:
        # Import hàm update ở đây để tránh lỗi circular import
        from rag_system import update_knowledge_base_auto
        os.makedirs("new_docs", exist_ok=True)
        path = os.path.join("new_docs", file.filename)
        file.save(path)
        update_knowledge_base_auto()
        return jsonify({"success": True, "filename": file.filename})
    except Exception as e:
        return jsonify({"error": f"Lỗi khi xử lý file: {str(e)}"}), 500


# 7. Endpoint kiểm tra mật khẩu (Không đổi)
@app.route("/check-admin-password", methods=["POST"])
def check_admin_password():
    admin_password = os.getenv("ADMIN_PASSWORD")
    submitted_password = request.json.get("password")
    if admin_password and submitted_password == admin_password:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False}), 401


# 8. HÀM RESET ĐÃ ĐƯỢC SỬA LỖI FILE LOCK
@app.route("/reset-knowledge", methods=["POST"])
def reset_knowledge_base():
    """Xóa và tự động xây dựng lại toàn bộ tri thức từ các file trong old_docs."""
    admin_password = os.getenv("ADMIN_PASSWORD")
    submitted_password = request.form.get("password")

    if not admin_password or submitted_password != admin_password:
        return jsonify({"error": "Mật khẩu không đúng."}), 403

    global rag_chatbot # 1. Báo cho Python biết ta muốn thay đổi biến global

    try:
        # 2. QUAN TRỌNG: Gỡ bỏ chatbot cũ để giải phóng file lock
        rag_chatbot = None 

        # 3. Xóa thư mục ChromaDB cũ nếu tồn tại
        if os.path.exists(CHROMA_DB_PATH):
            print(f"Đang xóa Knowledge Base cũ tại: {CHROMA_DB_PATH}")
            shutil.rmtree(CHROMA_DB_PATH)
        
        # 4. Gọi hàm để xây dựng lại Knowledge Base từ các file trong old_docs
        print(f"Bắt đầu xây dựng lại Knowledge Base từ thư mục: {OLD_DOCS_DIR}")
        if os.path.exists(OLD_DOCS_DIR) and os.listdir(OLD_DOCS_DIR):
            initialize_vector_store(CHROMA_DB_PATH, EMBEDDING_MODEL, OLD_DOCS_DIR)
        else:
            print("Thư mục old_docs rỗng, chỉ tạo một Knowledge Base trống.")
            from langchain_openai import OpenAIEmbeddings
            from langchain_chroma import Chroma
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)

        # 5. Tải lại chatbot để nó dùng DB mới
        rag_chatbot = RAGChatbot()
        
        print("Hoàn tất reset và xây dựng lại Knowledge Base.")
        return jsonify({"success": True, "message": "Đã reset và tự động nạp lại tri thức thành công."})

    except Exception as e:
        # Trong trường hợp có lỗi, hãy cố gắng khởi tạo lại chatbot để ứng dụng không chết
        if rag_chatbot is None:
            rag_chatbot = RAGChatbot()
        print(f"Lỗi khi reset và xây dựng lại: {e}")
        return jsonify({"error": f"Lỗi khi reset: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)