import gradio as gr
from transformers import pipeline
import torch
from huggingface_hub import login
import os

login(token = os.getenv('MODEL_REPO_TK'))
model_id = "meta-llama/Llama-3.2-1B-Instruct"
model_loaded = False
while not model_loaded:
    try:
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model_loaded = True
    except Exception as e:
        print('Error loading model: ', e)

def rewrite(query):
    outputs = pipe(
        [
            {"role": "system", "content": "Eres un experto en historia latinoamericana. Tu tarea es tomar consultas relacionadas con la historia de América Latina y reformularlas para hacerlas más precisas, claras y detalladas. Deberías asegurarte de que la nueva consulta sea más específica, completa y comprensible. Responde solo con la consulta reformulada, sin agregar respuestas completas ni contenido adicional. No uses listas, títulos ni formato Markdown."},
            {"role": "user", "content": query},
        ],
        max_new_tokens=256,
    )
    
    return outputs[0]["generated_text"][-1]['content']

demo = gr.Interface(fn=rewrite, inputs="text", outputs="text")
demo.launch()
