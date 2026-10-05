"""Local Qwen inference and robust JSON extraction."""
import gc
import json
from .prompts import QWEN_SYSTEM_PROMPT

def extract_json_from_text(text):
    # Avoid changing strings when repairing trailing commas: remove only commas
    # outside quoted strings that are followed by ] or }.
    start = text.find('{')
    if start < 0:
        return None
    raw = text[start:]
    try:
        graph, _ = json.JSONDecoder().raw_decode(raw)
        return graph
    except json.JSONDecodeError:
        pass
    chars, quoted, escaped = [], False, False
    for i, ch in enumerate(raw):
        if quoted:
            chars.append(ch)
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                quoted = False
        else:
            if ch == '"':
                quoted = True
            if ch == ',':
                j = i+1
                while j < len(raw) and raw[j].isspace():
                    j += 1
                if j < len(raw) and raw[j] in ']}':
                    continue
            chars.append(ch)
    try:
        graph, _ = json.JSONDecoder().raw_decode(''.join(chars))
        return graph
    except json.JSONDecodeError:
        return None


class LocalQwenParser:
    def __init__(self, model_path, device='cuda:0', dtype='auto'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable in this environment')
            with torch.cuda.device(self.device):
                use_bfloat16 = torch.cuda.is_bf16_supported()
            chosen = torch.bfloat16 if use_bfloat16 else torch.float16
        else:
            chosen = torch.float32
        if dtype != 'auto':
            chosen = getattr(torch, dtype)
        print(f'Loading local Qwen: {model_path}; device={device}; dtype={chosen}', flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, use_fast=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), trust_remote_code=True, torch_dtype=chosen,
            device_map={'': str(self.device)}, local_files_only=True).eval()
        self.model.requires_grad_(False)

    def parse(self, expression, max_new_tokens, attempt):
        messages = [{'role': 'system', 'content': QWEN_SYSTEM_PROMPT},
                    {'role': 'user', 'content': f'Input: "{expression}"\nOutput:'}]
        if attempt > 0:
            messages[-1]['content'] += ('\nReturn one complete JSON object with target '
                                       '(compound_words, attributes) and relation. '
                                       'Do not omit any closing brackets or include explanations.')
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors='pt').to(self.device)
        raw = ''
        try:
            with self.torch.no_grad():
                output = self.model.generate(
                    **inputs, max_new_tokens=min(max_new_tokens*(attempt+1), 2048),
                    do_sample=False, temperature=None, top_p=None, top_k=None,
                    repetition_penalty=1.05, pad_token_id=self.tokenizer.eos_token_id)
            raw = self.tokenizer.decode(output[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            del output
            return extract_json_from_text(raw), raw
        finally:
            del inputs
            gc.collect()
            if self.device.type == 'cuda':
                self.torch.cuda.empty_cache()
