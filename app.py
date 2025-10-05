import gradio as gr
from transformers import pipeline

# Load Granite pipeline
chatbot_pipeline = pipeline(
    "text-generation",
    model="ibm-granite/granite-3.3-8b-instruct",  # model ibm granite
    device_map="auto"
)

def respond(message, history, max_tokens, temperature, top_p):
    # input fromat
    conversation = ""
    for user_msg, bot_reply in history:
        conversation += f"User: {user_msg}\nAssistant: {bot_reply}\n"
    conversation += f"User: {message}\nAssistant:"

    # genrate
    response = chatbot_pipeline(
        conversation,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p
    )[0]["generated_text"]

    # mengambil jawaban terakhir
    reply = response.split("Assistant:")[-1].strip()

    return reply

# UI omon-omonbot
demo = gr.ChatInterface(
    fn=respond,
    title="🤖 IBM Granite OmonBot",
    description="Chatbot sederhana dengan model IBM Granite. Tersedia untuk siapapun!",
    examples=[
        ["Halo, apakah kamu punya ijazah SMA?"],
        ["Apa yang terjadi jika gorong-gorong itu ditutup."],
        ["Cara cepat memiliki ijazah S1."]
    ],
    theme="soft",
    additional_inputs=[
        gr.Slider(50, 512, value=256, step=10, label="Max Tokens"),
        gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
    ]
)

if __name__ == "__main__":
    demo.launch()
