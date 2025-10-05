import gradio as gr
from transformers import pipeline

# Load Granite pipeline
chatbot_pipeline = pipeline(
    "text-generation",
    model="ibm-granite/granite-3.3-8b-instruct",  # Model IBM Granite 8B
    device_map="auto"
)

def respond(message, history, max_tokens, temperature, top_p):
    try:
        # Input Format
        conversation = ""
        for user_msg, bot_reply in history:
            conversation += f"User: {user_msg}\nAssistant: {bot_reply}\n"
        conversation += f"User: {message}\nAssistant:"

        response = chatbot_pipeline(
            conversation,
            max_new_tokens=int(max_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p)
        )[0]["generated_text"]

        # Mengambil jawaban terakhir
        reply = response.split("Assistant:")[-1].strip()

        # Fallback kalau tidak ada jawaban
        if not reply:
            reply = "(Maaf, saya masih tidak bisa menjawab permintaan ini.)"

        return reply

    except Exception as e:
        # Error handling biar tidak crash
        return f"[Error saat memproses: {str(e)}]"

# UI OmonBot
demo = gr.ChatInterface(
    fn=respond,
    title="🤖 IBM Granite OmonBot",
    description="Chatbot sederhana dengan model IBM Granite 8B. Halo, silahkan prompt apapun!",
    examples=[
        ["Halo, apakah kamu punya ijazah SMA?"],
        ["Apa yang terjadi jika gorong-gorong itu ditutup."],
        ["Cara cepat memiliki ijazah S1."]
    ],
    theme="soft",
    additional_inputs=[
        gr.Slider(50, 256, value=128, step=10, label="Max Tokens"),  # default lebih rendah
        gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
    ]
)

if __name__ == "__main__":
    demo.launch()
