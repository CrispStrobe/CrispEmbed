# Serving default

The bundled `chat_template.jinja` defaults to `enable_thinking=false` when the
caller omits the chat-template kwarg. Serving code should still pass the
explicit vLLM flag for clarity:

```bash
--default-chat-template-kwargs '{"enable_thinking": false}'
```

Passing `enable_thinking=true` explicitly remains an opt-in override.
