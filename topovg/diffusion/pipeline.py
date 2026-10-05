"""One noisy U-Net forward per image-expression sample, without GT evaluation."""
import gc
from pathlib import Path
from ..utils.text import normalize_text
from ..grounding.semantic_executor import SemanticExecutor
from .ptp_utils import AttentionAccumulator, CaptureAttentionProcessor

class DiffusionEngine:
    def __init__(self, args, graphs):
        self.args, self.graphs = args, graphs
        self.pipe = None
        self.accumulator = AttentionAccumulator()
        self.latent = self.context = None

    def load(self):
        global torch, np, Image
        import torch
        torch.set_num_threads(1)
        import numpy as np
        from PIL import Image
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        device = torch.device(self.args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Use a CUDA PyTorch build, or --device cpu for debugging.')
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                  beta_schedule='scaled_linear', clip_sample=False,
                                  set_alpha_to_one=False, steps_offset=1)
        path = str(Path(self.args.model_path).expanduser().resolve())
        self.pipe = StableDiffusionPipeline.from_pretrained(
            path, scheduler=scheduler, torch_dtype=dtype, local_files_only=True,
            safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(device)
        for module in (self.pipe.vae, self.pipe.text_encoder, self.pipe.unet):
            module.eval().requires_grad_(False)
        if int(self.pipe.tokenizer.model_max_length) != 77:
            raise ValueError('This implementation expects SD1.5 / CLIP with 77 tokens.')
        processors = {}
        for name in self.pipe.unet.attn_processors:
            if not name.startswith(('down_blocks.', 'mid_block.', 'up_blocks.')):
                raise ValueError(f'Unexpected attention layer: {name}')
            processors[name] = CaptureAttentionProcessor(self.accumulator, self.args.query_chunk)
        self.pipe.unet.set_attn_processor(processors)
        print(f'Model loaded: {path}, {device}, {dtype}. No fixed random seed.', flush=True)

    def prepare(self, sample):
        if self.pipe is None:
            self.load()
        self.prompt = 'a photograph of ' + normalize_text(sample['expression'])
        tokens = self.pipe.tokenizer.encode(self.prompt, add_special_tokens=True)
        if len(tokens) > 77:
            raise ValueError(f"{sample['id']}: prompt exceeds 77 tokens; refusing silent truncation")
        with torch.no_grad():
            with Image.open(sample['_image_path']) as image:
                # Same PIL resize default and VAE scaling as the original scripts.
                pixels = np.array(image.convert('RGB').resize((512, 512)))
            tensor = torch.from_numpy(pixels).float() / 127.5 - 1.0
            tensor = tensor.permute(2, 0, 1)[None].to(self.pipe.device, dtype=self.pipe.dtype)
            self.latent = self.pipe.vae.encode(tensor).latent_dist.mean * 0.18215
            text = self.pipe.tokenizer([self.prompt], padding='max_length', max_length=77,
                                       truncation=True, return_tensors='pt')
            uncond = self.pipe.tokenizer([''], padding='max_length', max_length=77,
                                         return_tensors='pt')
            emb = self.pipe.text_encoder(text.input_ids.to(self.pipe.device))[0]
            uncond_emb = self.pipe.text_encoder(uncond.input_ids.to(self.pipe.device))[0]
            self.context = torch.cat([uncond_emb, emb])
            del tensor, emb, uncond_emb

    def draw(self, sample):
        self.accumulator.reset()
        try:
            with torch.no_grad():
                # Do not call manual_seed, seed_everything, or recreate a generator.
                noise = torch.randn_like(self.latent)
                timestep = torch.tensor([self.args.timestep], dtype=torch.long, device=self.pipe.device)
                noisy = self.pipe.scheduler.add_noise(self.latent, noise, timestep)
                self.pipe.unet(torch.cat([noisy]*2), timestep, encoder_hidden_states=self.context)
                del noise, noisy, timestep
                features = {}
                for res in (8, 16):
                    cross = self.accumulator.result(('cross', res)).reshape(res, res, 77).to(self.pipe.device)
                    executor = SemanticExecutor(self.pipe.tokenizer, cross)
                    executor.set_prompt(self.prompt)
                    features[f'semantic_mask_{res}'] = executor.execute_graph(
                        self.graphs[sample['id']]).detach().cpu().float().contiguous()
                    del cross, executor
                features['self_64'] = self.accumulator.result(('self', 64)).contiguous()
                return features
        finally:
            self.accumulator.reset()

    def release(self, features):
        # Saved feature tensors are on CPU; release references after committing.
        features.clear()
        gc.collect()
        if self.pipe is not None and self.pipe.device.type == 'cuda':
            torch.cuda.empty_cache()

    def finish(self):
        self.latent = self.context = None
        self.accumulator.reset()
        gc.collect()
        if self.pipe is not None and self.pipe.device.type == 'cuda':
            torch.cuda.empty_cache()
