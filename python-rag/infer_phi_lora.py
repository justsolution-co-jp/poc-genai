from transformers import pipeline

pipe = pipeline("text-generation", model="./phi-lora-checkpoint", tokenizer=tokenizer)
print(pipe("The Eiffel Tower is located in", max_new_tokens=50))
