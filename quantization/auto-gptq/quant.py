import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig


model_id = 'THUDM/chatglm3-6b'

# add quantization config
quantize_4_bit_config = GPTQConfig(
    bits=4,  # 量化到 4-bit 精度
    group_size=128,  
    dataset="c4",  # default dataset among ['wikitext2','c4','c4-new','ptb','ptb-new']
    desc_act=False
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
quant_4_bit_model = AutoModelForCausalLM.from_pretrained(
     model_id, quantization_config=quantize_4_bit_config,
     device_map='auto', trust_remote_code=True
)


# check the mode is accuratly quantized from the attributes of the linear layers,
# they should contain `qweight` and `qzeros` and in `torch.int32` dtype

print("check quantize")
print(quant_4_bit_model.model.decoder.layers[0].self_attn.q_proj.__dict__)


# inference
text = "Hello my name is"
inputs = tokenizer(text, return_tensors='pt').to(0)

out = quant_model.generate(**inputs)
print(tokenizer.decode(out[0], skip_special_tokens=True))


# train the quantized model with peft



from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["k_proj","o_proj","q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()