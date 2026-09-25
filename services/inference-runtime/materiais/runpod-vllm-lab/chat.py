"""Chat leve para o servidor vLLM existente. Execute: python chat.py."""
import getpass
import json
import os

import gradio as gr
import httpx

BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
MODEL = os.getenv("VLLM_MODEL", "qwen14b-int8")


def text_content(content):
    """Gradio pode representar texto como string ou blocos de conteúdo."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        )
    return ""


async def respond(message, history, temperature, max_tokens):
    messages = [
        {"role": item["role"], "content": text_content(item.get("content"))}
        for item in history if item.get("role") in ("user", "assistant")
    ]
    messages.append({"role": "user", "content": message})
    headers = {}
    if os.getenv("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]
    answer = ""
    try:
        # O contexto fecha a conexão inclusive se o usuário interromper a resposta.
        async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10)) as client:
            async with client.stream(
                "POST", BASE_URL + "/chat/completions", headers=headers,
                json={"model": MODEL, "messages": messages, "stream": True,
                      "temperature": float(temperature), "max_tokens": int(max_tokens)},
            ) as response:
                if response.is_error:
                    body = (await response.aread()).decode(errors="replace")
                    if response.status_code == 400:
                        raise gr.Error("O vLLM recusou a pergunta. Se o contexto de 2048 tokens foi excedido, limpe a conversa ou reduza a saída. Detalhe: " + body[:700])
                    raise gr.Error(f"vLLM retornou HTTP {response.status_code}: {body[:700]}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        raise gr.Error("Erro do vLLM: " + str(event["error"])[:700])
                    choices = event.get("choices", [])
                    if not choices:
                        continue
                    answer += choices[0].get("delta", {}).get("content") or ""
                    yield answer
                if not answer:
                    raise gr.Error("O servidor encerrou a resposta sem texto. Confira o log do vLLM.")
    except httpx.ConnectError as exc:
        raise gr.Error("Não consegui conectar ao vLLM. Confirme que o servidor está rodando na porta 8000, no mesmo Pod.") from exc
    except httpx.TimeoutException as exc:
        raise gr.Error("O vLLM demorou demais para responder. Confira o terminal do servidor e tente novamente.") from exc
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        raise gr.Error("A resposta foi interrompida ou inválida. Confira o log do vLLM e tente novamente.") from exc


def build_app():
    return gr.ChatInterface(
        fn=respond,
        title="Chat com Qwen · vLLM",
        description="Converse com o modelo do Arthur. Use o botão de limpar para começar uma nova conversa.",
        chatbot=gr.Chatbot(height=480),
        textbox=gr.Textbox(placeholder="Digite sua pergunta…", lines=2,
                           submit_btn="Enviar", stop_btn="Parar"),
        additional_inputs=[
            gr.Slider(0, 1, value=0, step=0.1, label="Temperatura"),
            gr.Slider(32, 1024, value=256, step=32, label="Máximo de tokens da resposta"),
        ],
        concurrency_limit=1, save_history=False,
    )


if __name__ == "__main__":
    # O proxy HTTP do RunPod torna a interface acessível externamente.
    username = os.getenv("CHAT_USER", "yan")
    password = os.getenv("CHAT_PASSWORD") or getpass.getpass("Crie uma senha para entrar no chat: ")
    if not password.strip():
        raise SystemExit("A senha não pode estar vazia. Execute novamente e escolha uma senha.")
    build_app().launch(
        server_name="0.0.0.0", server_port=7860,
        auth=(username, password), share=False,
    )
