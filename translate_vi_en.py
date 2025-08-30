import google.generativeai as genai

# Cấu hình API key của bạn
# Thay 'YOUR_API_KEY' bằng key bạn đã lấy từ Google AI Studio
GOOGLE_API_KEY = 'AIzaSyBzg_T5DxMotDoA2SWRonYza8nS3YedTFU'
genai.configure(api_key=GOOGLE_API_KEY)

# Khởi tạo mô hình Gemini 1.5 Flash
# Sử dụng 'gemini-1.5-flash-latest' để luôn dùng phiên bản mới nhất
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Gửi prompt (câu hỏi) của bạn đến mô hình
prompt = f"Translate this text into 2 English version : '{input()}'. only return the text in double quotation marks, texts are separated by a comma and do not use any special symbols except the symbols in the original text"
response = model.generate_content(prompt)

# In ra câu trả lời của mô hình
print(response.text)