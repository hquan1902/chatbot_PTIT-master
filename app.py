from flask import Flask, render_template, request, jsonify
import json, os
from dotenv import load_dotenv
from openai import OpenAI
from rag_system import update_knowledge_base_auto
from rag_chatbot import RAGChatbot


# 1. Cấu hình Flask và API
app = Flask(__name__)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
rag_chatbot = RAGChatbot()

CHAT_HISTORY_FILE = "chat_history.json"



# 2. Quản lý lịch sử chat
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



# 3. Giao diện chính
@app.route("/")
def index():
    return render_template("index.html", history=load_history())


# 4. Chat API – kết hợp RAG + GPT
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Tin nhắn trống."}), 400

    save_message("user", user_message)

    try:
        # Gọi RAG để lấy câu trả lời dựa trên knowledge base
        rag_reply = rag_chatbot.get_answer(user_message)

        # Nếu không có thông tin, chỉ trả lời rằng không tìm thấy
        if "Tôi không tìm thấy thông tin" in rag_reply or "Lỗi khi truy vấn" in rag_reply:
            bot_reply = "Xin lỗi, tôi chỉ có thể trả lời các thông tin liên quan đến PTIT và hiện chưa có dữ liệu cho câu hỏi này."
        else:
            bot_reply = rag_reply


    except Exception as e:
        bot_reply = f"Lỗi khi xử lý: {str(e)}"

    save_message("bot", bot_reply)
    return jsonify({"reply": bot_reply})



# 5. Upload file & tự động cập nhật
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Không có file được gửi."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Tên file không hợp lệ."}), 400

    os.makedirs("new_docs", exist_ok=True)
    path = os.path.join("new_docs", file.filename)
    file.save(path)

    update_knowledge_base_auto()  # cập nhật RAG DB

    return jsonify({"success": True, "filename": file.filename})



if __name__ == "__main__":
    app.run(debug=True)
