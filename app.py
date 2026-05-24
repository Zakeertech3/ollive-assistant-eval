import gradio as gr

import assistants.oss as oss
import assistants.frontier as frontier

SYSTEM_PROMPT = "You are a helpful, harmless assistant. Answer concisely."
OSS_LABEL = "OSS (Qwen2.5-0.5B via Ollama)"
FRONTIER_LABEL = "Frontier (GPT-OSS-120B via Groq)"


def _prepend_system(history: list[dict]) -> list[dict]:
    if history and history[0]["role"] == "system":
        return history
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history


def _chatbot_view(history: list[dict]) -> list[dict]:
    return [m for m in history if m["role"] != "system"]


def _format_status(result: dict) -> str:
    blocked = result.get("blocked_by")
    guardrail_line = (
        f"blocked by {blocked}: {result.get('block_reason', '')}"
        if blocked
        else "guardrail: passed"
    )
    return "\n".join([
        f"provider: {result.get('provider', '-')}",
        f"model: {result.get('model', '-')}",
        f"latency: {result.get('latency_ms', 0):.1f} ms",
        f"tokens in / out: {result.get('tokens_in', 0)} / {result.get('tokens_out', 0)}",
        guardrail_line,
    ])


def handle_message(user_input: str, history: list[dict], assistant_choice: str):
    if not user_input.strip():
        return _chatbot_view(history), history, ""
    history = _prepend_system(history)
    history = history + [{"role": "user", "content": user_input}]
    try:
        if assistant_choice == OSS_LABEL:
            result = oss.chat_safe(history)
        else:
            result = frontier.chat_safe(history)
        history = history + [{"role": "assistant", "content": result["reply"]}]
        status = _format_status(result)
    except Exception as exc:
        error_msg = f"Error: {type(exc).__name__}: {exc}"
        history = history + [{"role": "assistant", "content": error_msg}]
        status = error_msg
    return _chatbot_view(history), history, status


def handle_clear():
    return [], [], ""


with gr.Blocks(title="Ollive AI Assistant") as demo:
    gr.Markdown("## Ollive AI Assistant")
    assistant_radio = gr.Radio(
        choices=[OSS_LABEL, FRONTIER_LABEL],
        value=OSS_LABEL,
        label="Assistant",
    )
    chatbot = gr.Chatbot(label="Conversation", height=420)
    history_state = gr.State([])
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type your message here...",
            show_label=False,
            scale=4,
        )
        submit_btn = gr.Button("Submit", scale=1)
    clear_btn = gr.Button("Clear conversation")
    status_panel = gr.Textbox(
        label="Last call status",
        interactive=False,
        lines=5,
    )

    handler_cfg = dict(
        fn=handle_message,
        inputs=[user_input, history_state, assistant_radio],
        outputs=[chatbot, history_state, status_panel],
    )
    submit_btn.click(**handler_cfg).then(lambda: "", outputs=[user_input])
    user_input.submit(**handler_cfg).then(lambda: "", outputs=[user_input])
    clear_btn.click(fn=handle_clear, outputs=[chatbot, history_state, status_panel])


if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
