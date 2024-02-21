from ctransformers import AutoModelForCausalLM
import time

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf"
)


def get_prompt(instruction: str) -> str:
    system = "You are an AI assistant that gives helpful answers. You provide answers in a short and consize way."
    prompt = f"[INST] <<SYS>>{system}<</SYS>>{instruction}[/INST]"
    return prompt


start = time.time()

question = "Which city is the capital of India?"

for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)

stop = time.time()

print("\nTotal execution time: {}.".format(round(stop - start)))
