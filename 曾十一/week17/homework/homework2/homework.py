# --- 1. 安装必要的库 ---
# 如果您尚未安装这些库，请取消下面这行注释并运行它
# !pip install transformers torch bitsandbytes accelerate sentencepiece

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- 2. 定义模型ID和设置量化配置 ---
# 我们选用 Qwen2 的 0.5B（5亿参数）版本
model_id = "/mnt/data_1/zfy/agent/modlezoo/Qwen3-0.6B"

# 配置4-bit量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # 启用4-bit量化加载
    bnb_4bit_compute_dtype=torch.bfloat16  # 设置计算时的数据类型，以保持精度
)

print(f"正在加载模型: {model_id}...")

# --- 3. 加载量化模型和分词器 ---
# 加载模型
# device_map="auto" 会自动将模型加载到可用的硬件上（例如单个GPU）
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True # Qwen2模型需要信任远程代码
)

# 加载与模型匹配的分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True # 同样需要信任远程代码
)

print("模型和分词器加载完成！")

# --- 4. 准备输入并进行推理 ---
# 定义一个中文提示
prompt = "你好，请你用中文讲一个笑话："

# 使用分词器将文本提示转换为模型可以理解的token ID
# 注意：对于Qwen2这类模型，推荐使用它们特定的聊天模板来获得最佳效果，
# 但为了简化，我们这里直接使用提示文本。
inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # 将输入数据移动到模型所在的设备

print("\n--- 开始生成文本 ---")
print(f"输入提示: {prompt}")

# 使用模型生成文本
# max_new_tokens: 指定最多生成多少个新词
# do_sample=True: 启用采样，可以让回答更多样
# temperature: 控制随机性，值越小回答越确定，值越大越随机
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id # 设置填充token ID为结束符ID，避免警告
)

# --- 5. 解码并打印结果 ---
# 将模型输出的token ID转换回人类可读的文本
# skip_special_tokens=True 会在解码时移除特殊的标记（如<|endoftext|>）
result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- 模型输出 ---")
print(result_text)