from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import traceback
from pathlib import Path
import os


def local_model_inference(prompt):
    """
    使用本地模型进行推理。
    :param prompt: 用户输入的文本
    :return: 模型生成的输出
    """
    # 定义本地模型路径
    local_model_path = "C:/Users/nashk/Documents/nashknight/search_agent/Qwen3-0.6B"

    print(f"加载模型路径: {local_model_path}")
    # 确认本地路径存在，避免被 HF 误判为 repo id（会触发远程校验）
    if not Path(local_model_path).exists():
        print(f"错误：模型路径不存在: {local_model_path}")
        raise FileNotFoundError(local_model_path)

    # 加载分词器和模型（明确指定 local_files_only=True 强制只从本地加载）
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    try:
        # 若安装 accelerate，可使用 device_map="auto" 将模型分配到可用设备
        try:
            import accelerate  # type: ignore
            have_accelerate = True
        except Exception:
            have_accelerate = False

        dtype_arg = {"dtype": (torch.float16 if torch.cuda.is_available() else torch.float32)}

        if have_accelerate:
            model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                device_map="auto",
                local_files_only=True,
                **dtype_arg,
            )
        else:
            # 没有 accelerate：在 CPU 上加载或直接加载到单 GPU（如可用）
            if torch.cuda.is_available():
                # 加载到 CPU 然后移动到 cuda，以避免要求 accelerate
                model = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    local_files_only=True,
                    **dtype_arg,
                    device_map=None,
                )
                try:
                    model.to('cuda')
                except Exception:
                    print("无法将模型移动到 CUDA；保持在加载时的设备上")
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    local_files_only=True,
                    **dtype_arg,
                    device_map=None,
                )
    except Exception:
        print("模型加载失败，打印异常：")
        traceback.print_exc()
        raise

    # 准备模型输入
    messages = [
        {"role": "user", "content": prompt}
    ]
    # 有些 tokenizer 版本没有 apply_chat_template 方法，提供回退
    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
    else:
        # 回退为直接使用用户内容
        text = messages[0]['content']
    model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    # 生成文本
    try:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100  # 限制生成的最大 token 数
        )
        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    except Exception:
        print("生成失败，打印异常：")
        traceback.print_exc()
        raise

    return output

# 测试用例
if __name__ == "__main__":
    test_prompt = "介绍一下大型语言模型。"
    print("开始推理...")
    try:
        result = local_model_inference(test_prompt)
        print("生成结果:", result)
    except Exception:
        print("主流程异常，详细 traceback：")
        traceback.print_exc()