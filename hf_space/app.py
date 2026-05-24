import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = "You are a helpful, harmless assistant. Answer concisely."
MAX_NEW_TOKENS = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()


def chat(messages: list[dict]) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS)
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _prepend_system(history: list[dict]) -> list[dict]:
    if history and history[0]["role"] == "system":
        return history
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history


def _chatbot_view(history: list[dict]) -> list[dict]:
    return [m for m in history if m["role"] != "system"]


def handle_message(user_input: str, history: list[dict]):
    if not user_input.strip():
        return _chatbot_view(history), history, ""
    history = _prepend_system(history)
    history = history + [{"role": "user", "content": user_input}]
    try:
        reply = chat(history)
        history = history + [{"role": "assistant", "content": reply}]
    except Exception as exc:
        reply = f"Error: {type(exc).__name__}: {exc}"
        history = history + [{"role": "assistant", "content": reply}]
    return _chatbot_view(history), history


def handle_clear():
    return [], []


with gr.Blocks(title="Ollive OSS Assistant") as demo:
    gr.Markdown("## Ollive OSS Assistant\nQwen2.5-0.5B-Instruct running on CPU.")
    chatbot = gr.Chatbot(label="Conversation", height=420, type="messages")
    history_state = gr.State([])
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type your message here...",
            show_label=False,
            scale=4,
        )
        submit_btn = gr.Button("Submit", scale=1)
    clear_btn = gr.Button("Clear conversation")

    handler_cfg = dict(
        fn=handle_message,
        inputs=[user_input, history_state],
        outputs=[chatbot, history_state],
    )
    submit_btn.click(**handler_cfg).then(lambda: "", outputs=[user_input])
    user_input.submit(**handler_cfg).then(lambda: "", outputs=[user_input])
    clear_btn.click(fn=handle_clear, outputs=[chatbot, history_state])


if __name__ == "__main__":
    demo.launch()
