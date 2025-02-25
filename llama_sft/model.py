# from huggingface_hub import snapshot_download
import os

# model_id = 'hfl/llama-3-chinese-8b'
save_dir = os.path.expanduser("~/autodl-tmp/LLMs/")

# snapshot_download(repo_id=model_id, local_dir=save_dir, local_dir_use_symlinks=False)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # llm_int8_threshold=6.0
)

tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(save_dir, quantization_config=bnb_config, device_map='auto')
# model.to('cuda')

def chat_with_model(prompt, max_new_tokens=200):
    # 编码输入文本
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成回复
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,  # 生成的最大 token 数
        temperature=1,  # 控制输出随机性，越低越确定
        # top_p=0.9,  # 采样时保留概率最高的 top_p 部分
        # do_sample=True  # 开启采样
    )

    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 测试对话
user_input = "hello, who are you? answer: "
response = chat_with_model(user_input)
print(response)