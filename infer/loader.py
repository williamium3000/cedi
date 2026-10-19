
from functools import partial
import torch


def _is_api_model(model_path: str) -> bool:
    return model_path.startswith(("openai/", "gemini/", "zhipu/", "minimax/", "parity/", "uniapi/"))


def _load_api_model(args):
    from .api_vlm import (
        OpenAIVisionProvider, GeminiVisionProvider, ZhipuVisionProvider,
        MiniMaxVisionProvider, ParityVisionProvider,
    )

    if args.model_path.startswith("openai/"):
        model = args.model_path.split("/", 1)[1]
        return OpenAIVisionProvider(model=model)
    if args.model_path.startswith("gemini/"):
        model = args.model_path.split("/", 1)[1]
        return GeminiVisionProvider(model=model)
    if args.model_path.startswith("zhipu/"):
        model = args.model_path.split("/", 1)[1]
        return ZhipuVisionProvider(model=model)
    if args.model_path.startswith("minimax/"):
        model = args.model_path.split("/", 1)[1]
        return MiniMaxVisionProvider(model=model)
    if args.model_path.startswith("parity/"):
        # Legacy gateway prefix; not supported in this release. Use a
        # standard OpenAI-compatible endpoint via the openai/ or
        # uniapi/ prefixes instead.
        raise RuntimeError(
            "parity/ prefix is no longer supported. "
            "Use 'openai/<model>' or 'uniapi/<model>' (or set OPENAI_BASE_URL "
            "to your OpenAI-compatible gateway)."
        )
    if args.model_path.startswith("uniapi/"):
        model = args.model_path.split("/", 1)[1]
        return ParityVisionProvider(model=model,
                                    api_key_env="UNIAPI_API_KEY",
                                    api_base_env="UNIAPI_API_BASE")
    raise NotImplementedError(f"Unknown API model prefix in model_path={args.model_path!r}")


def load_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _is_api_model(args.model_path):
        return _load_api_model(args)
    if "opera" in args.model_path:
        from .infer_opera import eval_model as eval_model_opera, setup_seeds, MODEL_EVAL_CONFIG_PATH, load_preprocess
        from minigpt4.common.config import Config
        from minigpt4.common.registry import registry
        
        model_name = args.model_path.split("/")[-1]
        args = type('Args', (), {
                                "model": model_name,
                                "options": [],
                            })()
        args.cfg_path = MODEL_EVAL_CONFIG_PATH[model_name]
        args.model = model_name
        args.gpu_id = "0"
        args.batch_size = 1
        cfg = Config(args)
        setup_seeds(cfg)

        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(device)
        model.eval()
        processor_cfg = cfg.get_config().preprocess
        processor_cfg.vis_processor.eval.do_normalize = False
        vis_processors, txt_processors = load_preprocess(processor_cfg)
        return partial(eval_model_opera, model=model, args=args, image_processor=vis_processors)
    if "llava-1.5" in args.model_path:
        from .infer_llava import eval_model as eval_model_llava
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(0)

        processor = AutoProcessor.from_pretrained(args.model_path)

        # Conditionally use conversation wrapper based on use_conversation flag.
        # Without this, v18conv / v19conv ran llava as single-shot per round
        # (no chat memory) because partial(...) doesn't hold history state.
        use_conversation = getattr(args, 'use_conversation', False)

        if use_conversation:
            import threading as _threading
            class LlavaConversationWrapper:
                # History is per-thread: v19conv runs `parallel` worker threads
                # behind a single VLM lock, so `__call__` is serialized, but the
                # wrapper instance is shared. Storing history on `self` lets
                # one thread's reset()/append() clobber another thread's
                # mid-conversation state and produces garbage prompts.
                def __init__(self, processor, model):
                    self.processor = processor
                    self.model = model
                    self._tls = _threading.local()

                def _history(self):
                    h = getattr(self._tls, 'history', None)
                    if h is None:
                        h = []
                        self._tls.history = h
                    return h

                def __call__(self, image_file, query):
                    output, new_history = eval_model_llava(
                        self.processor, self.model, image_file, query, self._history()
                    )
                    self._tls.history = new_history
                    return output

                def reset(self):
                    self._tls.history = []

            return LlavaConversationWrapper(processor=processor, model=model)
        else:
            return partial(eval_model_llava, model=model, processor=processor)
    elif "blip2" in args.model_path:
        from .infer_blip2 import eval_model as eval_model_blip2
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained(args.model_path)
        model = Blip2ForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float32
        )
        model.to(device)
        return partial(eval_model_blip2, model=model, processor=processor)
    elif "Qwen3-VL" in args.model_path:
        from .infer_qwenvl3 import eval_model as eval_model_qwenvl3
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        return partial(eval_model_qwenvl3, processor=processor, model=model)
    elif "Qwen2.5-VL" in args.model_path:
        from .infer_qwenvl2d5 import eval_model as eval_model_qwenvl2d5
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        # --attn_implementation flash_attention_2 is recommended: the default attention materialises
        # a full vision attention matrix (~20 GB for the largest SVG images), which can run out of memory.
        attn_implementation = getattr(args, "attn_implementation", None)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype="auto", device_map="auto",
            **({"attn_implementation": attn_implementation} if attn_implementation else {})
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
        
        # Conditionally use conversation wrapper based on use_conversation flag
        use_conversation = getattr(args, 'use_conversation', False)
        
        if use_conversation:
            # Return wrapper that manages conversation history per-thread.
            import threading as _threading
            class Qwen25VLConversationWrapper:
                def __init__(self, processor, model):
                    self.processor = processor
                    self.model = model
                    self._tls = _threading.local()

                def _history(self):
                    h = getattr(self._tls, 'history', None)
                    if h is None:
                        h = []
                        self._tls.history = h
                    return h

                def __call__(self, image_file, query):
                    output, new_history = eval_model_qwenvl2d5(
                        self.processor, self.model, image_file, query, self._history()
                    )
                    self._tls.history = new_history
                    return output

                def reset(self):
                    self._tls.history = []

            return Qwen25VLConversationWrapper(processor=processor, model=model)
        else:
            # Return simple function without conversation history
            return partial(eval_model_qwenvl2d5, processor=processor, model=model, conversation_history=None)
    elif "Qwen2-VL" in args.model_path:
        from .infer_qwenvl2 import eval_model as eval_model_qwenvl2
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.float16)
        processor = AutoProcessor.from_pretrained(args.model_path)
        model.to(device)
        return partial(eval_model_qwenvl2, model=model, processor=processor)
    elif "paligemma" in args.model_path:
        from .infer_pali_gemma import eval_model as eval_model_pali_gemma
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
        model = PaliGemmaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        processor = PaliGemmaProcessor.from_pretrained(args.model_path)
        model.to(device)
        return partial(eval_model_pali_gemma, model=model, processor=processor)
    elif "instructblip" in args.model_path:
        from .infer_instruct_blip import eval_model as eval_model_instruct_blip
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
        model = InstructBlipForConditionalGeneration.from_pretrained(args.model_path)
        processor = InstructBlipProcessor.from_pretrained(args.model_path)
        model.to(device)
        return partial(eval_model_instruct_blip, model=model, processor=processor)
    elif "cogagent" in args.model_path:
        from .infer_cogvlm import eval_model as eval_model_cogvlm
        from transformers import AutoModelForCausalLM, LlamaTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=False,
            trust_remote_code=True
        ).to(device).eval()
        tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        
        return partial(eval_model_cogvlm, model=model, tokenizer=tokenizer, torch_type=torch.bfloat16)
    elif "Ovis2" in args.model_path:
        from .infer_ovis2 import eval_model as eval_model_ovis2
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).cuda()
        model.to(device)
        return partial(eval_model_ovis2, model=model)
    elif "Phi-3.5" in args.model_path:
        from .infer_phi3d5vl import eval_model as eval_model_phi3d5vl
        from transformers import AutoModelForCausalLM, AutoProcessor
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            device_map=device, 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='flash_attention_2'    
            )

        # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        processor = AutoProcessor.from_pretrained(args.model_path, 
            trust_remote_code=True, 
            num_crops=4
            ) 
        return partial(eval_model_phi3d5vl, model=model, processor=processor)
    elif "gemma-3" in args.model_path:
            from .infer_gemma3 import eval_model as eval_model_gemma3
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration
            # Avoid FailOnRecompileLimitHit: Gemma3 forward can trigger many dynamo recompilations
            try:
                import torch._dynamo as dynamo
                if hasattr(dynamo, 'config'):
                    dynamo.config.cache_size_limit = 512
                    if hasattr(dynamo.config, 'recompile_limit'):
                        dynamo.config.recompile_limit = 128
            except Exception:
                pass
            # Eager attention: torch's fused SDPA kernel raises "CUDA error: misaligned address" /
            # CUBLAS_STATUS_EXECUTION_FAILED at some sequence lengths (torch 2.5.1, fp32, H100).
            # Eager is numerically equivalent and is what the Gemma-3 docs recommend.
            model = Gemma3ForConditionalGeneration.from_pretrained(
                args.model_path, device_map="auto", attn_implementation="eager").eval()
            processor = AutoProcessor.from_pretrained(args.model_path)
            
            # Conditionally use conversation wrapper based on use_conversation flag
            use_conversation = getattr(args, 'use_conversation', False)
            
            if use_conversation:
                # Return wrapper that manages conversation history per-thread.
                import threading as _threading
                class Gemma3ConversationWrapper:
                    def __init__(self, processor, model):
                        self.processor = processor
                        self.model = model
                        self._tls = _threading.local()

                    def _history(self):
                        h = getattr(self._tls, 'history', None)
                        if h is None:
                            h = []
                            self._tls.history = h
                        return h

                    def __call__(self, image_file, query):
                        output, new_history = eval_model_gemma3(
                            self.processor, self.model, image_file, query, self._history()
                        )
                        self._tls.history = new_history
                        return output

                    def reset(self):
                        self._tls.history = []

                return Gemma3ConversationWrapper(processor=processor, model=model)
            else:
                # Return simple function without conversation history
                return partial(eval_model_gemma3, processor=processor, model=model, conversation_history=None)
    elif "Intern" in args.model_path:
        from .infer_internvl3 import eval_model as eval_model_internvl3, split_model
        from transformers import AutoModel, AutoTokenizer
        device_map = split_model(args.model_path)
        # use_flash_attn requires the flash_attn pip package; the qwenvl3 env
        # doesn't have it (only a precompiled wheel-less source build is
        # available), so default to off. Override with INTERNVL_USE_FLASH_ATTN=1.
        import os as _os
        _use_fa = _os.environ.get("INTERNVL_USE_FLASH_ATTN", "0") == "1"
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=_use_fa,
            trust_remote_code=True,
            device_map=device_map).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
        
        # Conditionally use conversation wrapper based on use_conversation flag
        use_conversation = getattr(args, 'use_conversation', False)
        
        if use_conversation:
            # Return wrapper that manages conversation history per-thread.
            import threading as _threading
            class InternVL3ConversationWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                    self._tls = _threading.local()

                def _history(self):
                    h = getattr(self._tls, 'history', None)
                    if h is None:
                        h = []
                        self._tls.history = h
                    return h

                def __call__(self, image_file, query):
                    output, new_history = eval_model_internvl3(
                        self.model, self.tokenizer, image_file, query, self._history()
                    )
                    self._tls.history = new_history
                    return output

                def reset(self):
                    self._tls.history = []

            return InternVL3ConversationWrapper(model=model, tokenizer=tokenizer)
        else:
            # Return simple function without conversation history
            return partial(eval_model_internvl3, model=model, tokenizer=tokenizer, conversation_history=None)
    elif "lpoi" in args.model_path:
        from .infer_lpoi import eval_model as eval_model_lpoi
        from transformers import AutoModelForVision2Seq, AutoProcessor
        if "idefics2" in args.model_path:
            model_name = "HuggingFaceM4/idefics2-8b"
        else:
            raise NotImplementedError
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            load_in_8bit=False,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False)
        model.load_adapter(args.model_path)
        return partial(eval_model_lpoi, model=model, processor=processor)
    elif "LLaVA-RLHF" in args.model_path:
        from .infer_llava_rlhf import eval_model as eval_model_llava_rlhf
        from .infer_llava_rlhf import load_pretrained_model
        from peft import PeftModel
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            f"{args.model_path}/sft_model", f"{args.model_path}/rlhf_lora_adapter_model", "llava-rlhf-13b-v1.5-336", True)
        
        model = PeftModel.from_pretrained(model, f"{args.model_path}/rlhf_lora_adapter_model")

        return partial(eval_model_llava_rlhf, model=model, tokenizer=tokenizer, image_processor=image_processor)
    
    else:
        raise NotImplementedError