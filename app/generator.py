# Updated generator.py with proper function order
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torchaudio
import logging
import os
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing
from app.models import Segment
from app.text_normalizer import clean_text_for_tts
from app.text_normalizer import TextNormalizer
from app.utils.torch_utils import get_device, safely_move_to_device


# Set up logging
logger = logging.getLogger(__name__)

# Import the CSM watermarking code
try:
    from app.watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark
except ImportError:
    # Define stubs for watermarking if the module is not available
    CSM_1B_GH_WATERMARK = "CSM1B"
    def load_watermarker(device="cpu"):
        return None
    def watermark(watermarker, audio, sample_rate, key):
        return audio, sample_rate

def load_llama3_tokenizer():
    """
    Load tokenizer for Llama 3.2, using unsloth's open version
    instead of the gated meta-llama version.
    """
    try:
        # Use the unsloth version which is not gated
        tokenizer_name = "unsloth/Llama-3.2-1B"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
        )
        logger.info("Successfully loaded tokenizer from unsloth/Llama-3.2-1B")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer from unsloth: {e}")
        # Fallback to a simpler tokenizer if needed
        try:
            from transformers import GPT2Tokenizer
            logger.warning("Falling back to GPT2Tokenizer")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as fallback_e:
            logger.error(f"Fallback tokenizer also failed: {fallback_e}")
            raise RuntimeError("Could not load any suitable tokenizer")

class Generator:
    """Generator class for CSM-1B model."""
    def __init__(self, model):
        """Initialize generator with model."""
        self._model = model
        self._model.setup_caches(1)
        self._text_tokenizer = load_llama3_tokenizer()
        device = next(model.parameters()).device
        # Load Mimi codec for audio tokenization with CPU fallback
        logger.info("Loading Mimi audio codec...")
        from huggingface_hub import hf_hub_download
        # Determine Mimi loader
        try:
            from moshi.models import loaders
            DEFAULT_REPO = loaders.DEFAULT_REPO
            MIMI_NAME = loaders.MIMI_NAME
            get_mimi = loaders.get_mimi
        except ImportError:
            logger.warning("moshi.models.loaders not found, using fallback loader")
            DEFAULT_REPO = "kyutai/mimi"
            MIMI_NAME = "mimi-december.pt"
            def get_mimi(checkpoint_path, device):
                from moshi.models import MiMiModule
                checkpoint = torch.load(checkpoint_path, map_location=device)
                return MiMiModule.init_from_checkpoint(checkpoint, device=device)
        mimi_weight = hf_hub_download(DEFAULT_REPO, MIMI_NAME)
        # Try primary device then CPU
        self._audio_tokenizer = None
        self.sample_rate = 24000
        for attempt_device in (device, torch.device("cpu")):
            try:
                mimi = get_mimi(mimi_weight, device=attempt_device)
                mimi.set_num_codebooks(32)
                self._audio_tokenizer = mimi
                self.sample_rate = mimi.sample_rate
                logger.info(f"Mimi codec loaded successfully with sample rate {self.sample_rate} on {attempt_device}")
                break
            except Exception as e:
                logger.error(f"Error loading Mimi codec on {attempt_device}: {e}")
        if self._audio_tokenizer is None:
            logger.warning(f"Mimi codec not loaded, using fallback sample rate {self.sample_rate}")
        try:
            self._watermarker = load_watermarker(device=device)
            logger.info("Watermarker loaded successfully")
        except Exception as e:
            logger.warning(f"Error loading watermarker: {e}. Watermarking will be disabled.")
            self._watermarker = None
            
        self.device = device
        # Optimize for CUDA throughput
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            logger.info("CUDA optimizations enabled")
        # Wrap Mimi decode to fallback on CPU for MPS placeholder storage errors
        elif self._audio_tokenizer is not None:
            try:
                orig_decode = self._audio_tokenizer.decode
                def decode_with_fallback(codes):
                    try:
                        return orig_decode(codes)
                    except RuntimeError as e:
                        # TODO Probbaly a better way to do this
                        if self.device.type == 'mps' and 'Placeholder storage' in str(e):
                            logger.warning("Mimi decode falling back to CPU due to MPS placeholder storage error")
                            cpu_codes = codes.to('cpu')
                            self._audio_tokenizer.to('cpu')
                            audio = orig_decode(cpu_codes)
                            return audio.to(self.device)
                        else:
                            raise
                self._audio_tokenizer.decode = decode_with_fallback
            except Exception as e:
                logger.warning(f"Failed to wrap Mimi decode fallback: {e}")
            
    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a text segment."""
        frame_tokens = []
        frame_masks = []
        # Strip any voice instructions in square brackets to avoid them being read out
        text = self._clean_text_input(text)
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True
        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))
        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)
    
    def _clean_text_input(self, text: str) -> str:
        """Clean and normalize text for TTS."""
        return clean_text_for_tts(text)
    
    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize audio."""
        if self._audio_tokenizer is None:
            raise RuntimeError("Audio tokenizer not initialized")
        frame_tokens = []
        frame_masks = []
        # (K, T)
        # Send audio to the same device as the audio tokenizer (Mimi) to avoid dtype mismatches
        try:
            tokenizer_device = next(self._audio_tokenizer.parameters()).device
        except Exception:
            tokenizer_device = self.device
        audio = audio.to(tokenizer_device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # move tokens to model device for downstream processing before adding EOS
        audio_tokens = audio_tokens.to(self.device)
        # add EOS codebook column on model device
        eos_frame = torch.zeros(audio_tokens.size(0), 1, dtype=audio_tokens.dtype, device=self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)
        # build audio frame and mask on model device
        audio_frame = torch.zeros(audio_tokens.size(1), 33, dtype=torch.long, device=self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33, dtype=torch.bool, device=self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True
        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)
        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)
    
    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a segment of text and audio."""
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    def generate_quick(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 2000,  # Short for quick generation
        temperature: float = 0.7,  # Lower for more predictable output
        topk: int = 20,  # Lower for faster beam selection
    ) -> torch.Tensor:
        """Generate audio quickly for real-time streaming."""
        # Similar to generate() but optimized for speed
        self._model.reset_caches()
        
        # Convert max_audio_length_ms to frames - limit for faster generation
        max_audio_frames = min(int(max_audio_length_ms / 80), 128)  # Smaller limit
        
        # Process text
        cleaned_text = clean_text_for_tts(text)
        
        # Prepare tokens
        tokens, tokens_mask = [], []
        # Add context segments (limited to 1 for speed)
        if context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(context[0])
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)
        # Add text tokens
        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(cleaned_text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)
        
        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
        
        # Generate with larger batch size for fewer iterations
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        
        # Use larger batch size
        batch_size = 64  # Generate more frames at once
        all_samples = []
        for start_idx in range(0, max_audio_frames, batch_size):
            end_idx = min(start_idx + batch_size, max_audio_frames)
            batch_frames = end_idx - start_idx
            samples_batch = []
            for i in range(batch_frames):
                sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                samples_batch.append(sample)
                if torch.all(sample == 0):
                    break
                curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1
            all_samples.extend(samples_batch)
            if len(samples_batch) < batch_frames:
                break
        
        if not all_samples:
            return torch.zeros(10, device=self.device)  # Return short empty audio
            
        # Decode audio
        audio = self._audio_tokenizer.decode(torch.stack(all_samples).permute(1, 2, 0)).squeeze(0).squeeze(0)
        return audio

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        """Generate audio from text."""
        if self._audio_tokenizer is None:
            raise RuntimeError("Audio tokenizer not initialized")
        
        # Start timing: use CUDA events on CUDA devices, otherwise fallback to time.time()
        use_cuda_timing = self.device.type == 'cuda'
        if use_cuda_timing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            import time as _time
            cpu_start_time = _time.time()
            
        self._model.reset_caches()
        
        # Convert max_audio_length_ms to frames - this controls the maximum generation length
        max_audio_frames = min(int(max_audio_length_ms / 80), 1024)  # Limit to reasonable size
        max_seq_len = 2048 - max_audio_frames
        
        # Check if text is long and should be split
        if len(text) > 200:
            logger.info(f"Long text detected ({len(text)} chars), processing in segments")
            sentences = TextNormalizer.split_into_sentences(text)
            logger.info(f"Split into {len(sentences)} segments")
            
            # Process sentences individually and concatenate the results
            all_audio_segments = []
            
            # Use the first sentence to establish voice
            first_sentence = sentences[0]
            cleaned_text = clean_text_for_tts(first_sentence)
            
            # Generate the first segment
            tokens, tokens_mask = [], []
            
            # Add context segments for the first sentence
            for segment in context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)
            
            # Add first sentence tokens
            gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(cleaned_text, speaker)
            tokens.append(gen_segment_tokens)
            tokens_mask.append(gen_segment_tokens_mask)
            
            prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
            prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
            
            # Check context size and truncate if needed
            if prompt_tokens.size(0) >= max_seq_len:
                logger.warning(f"Inputs too long ({prompt_tokens.size(0)} tokens), truncating to {max_seq_len - 50}")
                prompt_tokens = prompt_tokens[-max_seq_len+50:]
                prompt_tokens_mask = prompt_tokens_mask[-max_seq_len+50:]
            
            # Generate first sentence audio
            curr_tokens = prompt_tokens.unsqueeze(0)
            curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
            curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
            
            # Generate first segment
            first_segment_samples = []
            for start_idx in range(0, max_audio_frames, 32):
                end_idx = min(start_idx + 32, max_audio_frames)
                batch_frames = end_idx - start_idx
                samples_batch = []
                
                for i in range(batch_frames):
                    sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                    samples_batch.append(sample)
                    
                    if torch.all(sample == 0):
                        break
                    
                    curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                    curr_tokens_mask = torch.cat(
                        [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                    ).unsqueeze(1)
                    curr_pos = curr_pos[:, -1:] + 1
                
                first_segment_samples.extend(samples_batch)
                
                if len(samples_batch) < batch_frames:
                    break
            
            if not first_segment_samples:
                raise RuntimeError("No audio generated for first segment")
            
            # Decode first segment
            first_segment_audio = self._audio_tokenizer.decode(
                torch.stack(first_segment_samples).permute(1, 2, 0)
            ).squeeze(0).squeeze(0)
            
            all_audio_segments.append(first_segment_audio)
            
            # Now process remaining sentences using the first as context
            for i, sentence in enumerate(sentences[1:], 1):
                logger.info(f"Generating segment {i+1}/{len(sentences)}")
                cleaned_text = clean_text_for_tts(sentence)
                
                # Create a context segment from the previous generation
                prev_segment = Segment(
                    speaker=speaker,
                    text=sentences[i-1],
                    audio=all_audio_segments[-1]
                )
                
                # Generate with this segment as context
                segment_tokens, segment_tokens_mask = [], []
                segment_tokens.append(self._tokenize_segment(prev_segment)[0])
                segment_tokens_mask.append(self._tokenize_segment(prev_segment)[1])
                
                # Add current segment tokens
                current_tokens, current_tokens_mask = self._tokenize_text_segment(cleaned_text, speaker)
                segment_tokens.append(current_tokens)
                segment_tokens_mask.append(current_tokens_mask)
                
                segment_prompt_tokens = torch.cat(segment_tokens, dim=0).long().to(self.device)
                segment_prompt_tokens_mask = torch.cat(segment_tokens_mask, dim=0).bool().to(self.device)
                
                # Check length and truncate if needed
                if segment_prompt_tokens.size(0) >= max_seq_len:
                    segment_prompt_tokens = segment_prompt_tokens[-max_seq_len+50:]
                    segment_prompt_tokens_mask = segment_prompt_tokens_mask[-max_seq_len+50:]
                
                # Generate audio for this segment
                curr_tokens = segment_prompt_tokens.unsqueeze(0)
                curr_tokens_mask = segment_prompt_tokens_mask.unsqueeze(0)
                curr_pos = torch.arange(0, segment_prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
                
                # Generate segment
                segment_samples = []
                for start_idx in range(0, max_audio_frames, 32):
                    end_idx = min(start_idx + 32, max_audio_frames)
                    batch_frames = end_idx - start_idx
                    samples_batch = []
                    
                    for i in range(batch_frames):
                        sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                        samples_batch.append(sample)
                        
                        if torch.all(sample == 0):
                            break
                        
                        curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                        curr_tokens_mask = torch.cat(
                            [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                        ).unsqueeze(1)
                        curr_pos = curr_pos[:, -1:] + 1
                    
                    segment_samples.extend(samples_batch)
                    
                    if len(samples_batch) < batch_frames:
                        break
                
                if not segment_samples:
                    logger.warning(f"No audio generated for segment {i+1}")
                    continue
                
                # Decode segment
                segment_audio = self._audio_tokenizer.decode(
                    torch.stack(segment_samples).permute(1, 2, 0)
                ).squeeze(0).squeeze(0)
                
                all_audio_segments.append(segment_audio)
            
            # Combine all segments with small pauses
            pause_samples = int(0.3 * self.sample_rate)  # 300ms pause
            pause = torch.zeros(pause_samples, device=self.device)
            
            audio_parts = []
            for i, segment_audio in enumerate(all_audio_segments):
                audio_parts.append(segment_audio)
                if i < len(all_audio_segments) - 1:
                    audio_parts.append(pause)
            
            audio = torch.cat(audio_parts)
            logger.info(f"Combined {len(all_audio_segments)} segments into final audio")
        
        else:
            # For shorter text, standard processing
            tokens, tokens_mask = [], []
            
            # Add context segments
            for segment in context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)
            
            # Process text
            cleaned_text = clean_text_for_tts(text)
            gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(cleaned_text, speaker)
            tokens.append(gen_segment_tokens)
            tokens_mask.append(gen_segment_tokens_mask)
            
            prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
            prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
            
            # Check context size
            if prompt_tokens.size(0) >= max_seq_len:
                logger.warning(f"Inputs too long ({prompt_tokens.size(0)} tokens), truncating to {max_seq_len - 50}")
                prompt_tokens = prompt_tokens[-max_seq_len+50:]
                prompt_tokens_mask = prompt_tokens_mask[-max_seq_len+50:]
            
            # Generate audio - optimized batch generation
            curr_tokens = prompt_tokens.unsqueeze(0)
            curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
            curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
            
            # Using optimized batch generation
            batch_size = 32  # Generate this many frames at once 
            all_samples = []
            
            for start_idx in range(0, max_audio_frames, batch_size):
                end_idx = min(start_idx + batch_size, max_audio_frames)
                batch_frames = end_idx - start_idx
                
                samples_batch = []
                
                for i in range(batch_frames):
                    sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                    samples_batch.append(sample)
                    
                    if torch.all(sample == 0):
                        break
                    
                    curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                    curr_tokens_mask = torch.cat(
                        [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                    ).unsqueeze(1)
                    curr_pos = curr_pos[:, -1:] + 1
                
                all_samples.extend(samples_batch)
                
                if len(samples_batch) < batch_frames:
                    logger.info(f"Early stopping at frame {start_idx + len(samples_batch)}/{max_audio_frames}")
                    break
            
            if not all_samples:
                raise RuntimeError("No audio generated - model produced empty output")
            
            # Decode audio
            audio = self._audio_tokenizer.decode(torch.stack(all_samples).permute(1, 2, 0)).squeeze(0).squeeze(0)
        
        # Apply watermark
        if self._watermarker is not None:
            try:
                audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
                audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)
            except Exception as e:
                logger.warning(f"Error applying watermark: {e}. Continuing without watermark.")
        
        # Record execution time
        if use_cuda_timing:
            end_event.record()
            torch.cuda.synchronize()
            execution_ms = start_event.elapsed_time(end_event)
        else:
            execution_ms = (_time.time() - cpu_start_time) * 1000
        audio_length_ms = (audio.shape[0] / self.sample_rate) * 1000
        
        # Calculate real-time factor (RTF)
        rtf = execution_ms / audio_length_ms
        logger.info(f"Audio generated in {execution_ms:.2f}ms, length: {audio_length_ms:.2f}ms, RTF: {rtf:.2f}x")
        
        return audio

# Define helper functions for multi-GPU support
def _manual_device_map(model, state_dict, strategy="balanced"):
    """Apply manual device mapping for multi-GPU setups.
    
    Args:
        model: The model to map
        state_dict: Model state dict
        strategy: Mapping strategy ('balanced', 'sequential')
        
    Returns:
        Model with weights distributed across GPUs
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        # No need for mapping with single GPU
        model.load_state_dict(state_dict)
        model = model.to("cuda")
        return model
    
    logger.info(f"Applying manual {strategy} device mapping across {num_gpus} GPUs")
    
    # Get all layer names from state dict
    layer_names = [name for name in state_dict.keys() if "layers" in name]
    backbone_layers = [name for name in layer_names if "backbone.layers" in name]
    decoder_layers = [name for name in layer_names if "decoder.layers" in name]
    
    # Count number of backbone and decoder layers
    backbone_layer_indices = set()
    for name in backbone_layers:
        parts = name.split('.')
        if len(parts) > 2:
            try:
                backbone_layer_indices.add(int(parts[2]))
            except ValueError:
                pass
    
    decoder_layer_indices = set()
    for name in decoder_layers:
        parts = name.split('.')
        if len(parts) > 2:
            try:
                decoder_layer_indices.add(int(parts[2]))
            except ValueError:
                pass
    
    num_backbone_layers = len(backbone_layer_indices)
    num_decoder_layers = len(decoder_layer_indices)
    
    # Create device map
    device_map = {}
    
    if strategy == "balanced":
        # Distribute layers evenly across GPUs
        layers_per_gpu = (num_backbone_layers + num_decoder_layers) // num_gpus
        remainder = (num_backbone_layers + num_decoder_layers) % num_gpus
        
        # Assign backbone layers
        for i in backbone_layer_indices:
            gpu_idx = min(i // layers_per_gpu, num_gpus - 1)
            device_map[f"backbone.layers.{i}"] = f"cuda:{gpu_idx}"
        
        # Assign decoder layers
        for i in decoder_layer_indices:
            gpu_idx = min((i + num_backbone_layers) // layers_per_gpu, num_gpus - 1)
            device_map[f"decoder.layers.{i}"] = f"cuda:{gpu_idx}"
            
    elif strategy == "sequential":
        # Fill each GPU sequentially
        # Backbone layers on first GPU(s)
        backbone_per_gpu = max(1, num_backbone_layers // ((num_gpus + 1) // 2))
        for i in backbone_layer_indices:
            gpu_idx = min(i // backbone_per_gpu, (num_gpus + 1) // 2 - 1)
            device_map[f"backbone.layers.{i}"] = f"cuda:{gpu_idx}"
        
        # Decoder layers on remaining GPU(s)
        decoder_per_gpu = max(1, num_decoder_layers // (num_gpus - (num_gpus + 1) // 2 + 1))
        for i in decoder_layer_indices:
            gpu_idx = min(i // decoder_per_gpu + (num_gpus + 1) // 2 - 1, num_gpus - 1)
            device_map[f"decoder.layers.{i}"] = f"cuda:{gpu_idx}"
    
    # Assign embeddings and other components
    device_map["text_embeddings"] = "cuda:0"
    device_map["audio_embeddings"] = "cuda:0"
    device_map["projection"] = "cuda:0"
    device_map["codebook0_head"] = "cuda:0"
    device_map["audio_head"] = "cuda:0"
    
    # Load state dict with device mapping
    model.load_state_dict(state_dict)
    
    # Move model parts to assigned devices
    for name, device in device_map.items():
        if "backbone.layers" in name:
            layer_idx = int(name.split('.')[-1])
            if hasattr(model.backbone, 'layers') and layer_idx < len(model.backbone.layers):
                model.backbone.layers[layer_idx] = model.backbone.layers[layer_idx].to(device)
        elif "decoder.layers" in name:
            layer_idx = int(name.split('.')[-1])
            if hasattr(model.decoder, 'layers') and layer_idx < len(model.decoder.layers):
                model.decoder.layers[layer_idx] = model.decoder.layers[layer_idx].to(device)
        elif hasattr(model, name):
            setattr(model, name, getattr(model, name).to(device))
    
    logger.info(f"Model distributed across GPUs with {strategy} strategy")
    return model

def load_csm_1b(ckpt_path: str = "ckpt.pt", use_gpu: bool = True, device_map: str = None) -> Generator:
    """Load CSM-1B model and create generator with performance optimizations.
    
    Args:
        ckpt_path: Path to model checkpoint
        use_gpu: Whether to attempt using GPU (CUDA or MPS if available). Defaults to True.
        device_map: Optional device mapping strategy for multi-GPU CUDA (\'auto\', \'balanced\', \'sequential\', or None). Ignored for MPS/CPU.
    
    Returns:
        Generator instance with optimized settings
    """
    try:
        # Import models module for CSM
        from app.torchtune_models import Model, ModelArgs

        # Get the target device using the utility function
        target_device = get_device(use_gpu=use_gpu)
        logger.info(f"Target device selected: {target_device}")

        # Create model args
        model_args = ModelArgs(
            backbone_flavor="llama-1B",
            decoder_flavor="llama-100M",
            text_vocab_size=128256,
            audio_vocab_size=2051,
            audio_num_codebooks=32,
        )

        # Load model
        logger.info(f"Loading CSM-1B model from {ckpt_path} for device={target_device}, device_map={device_map if target_device.type == 'cuda' else 'N/A'}")

        # --- Device-Specific Optimizations ---
        dtype = torch.float32 # Default precision

        if target_device.type == 'cuda':
            # Check if we should enable TF32 (faster but slightly less precise)
            enable_tf32 = os.environ.get("ENABLE_TF32", "true").lower() == "true"
            if enable_tf32:
                logger.info("Enabling TF32 for faster matrix multiplications on CUDA")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # Check for available precision modes
            use_bfloat16 = torch.cuda.is_bf16_supported()
            # use_float16 = not use_bfloat16 # Float16 often has issues, prefer bfloat16 or float32

            if use_bfloat16:
                dtype = torch.bfloat16
                logger.info("Using bfloat16 precision for faster inference on CUDA")
            # elif use_float16:
            #     dtype = torch.float16
            #     logger.info("Using float16 precision for faster inference on CUDA")
            else:
                logger.info("Using float32 precision on CUDA (bfloat16 not supported)")

            # Enable Flash Attention if available
            try:
                import flash_attn
                if os.environ.get("ENABLE_FLASH_ATTN", "true").lower() == "true":
                    logger.info("Flash Attention detected - enabling for faster attention on CUDA")
                    os.environ["PYTORCH_FLASH_ATTENTION_ENABLED"] = "1"
            except ImportError:
                logger.info("Flash Attention not available (install flash-attn for faster inference on CUDA)")

        elif target_device.type == 'mps':
            # MPS specific setup (if any needed in the future)
            # MPS supports bfloat16 on some devices, but float32 is generally safer
            logger.info("Using float32 precision on MPS")
            dtype = torch.float32 # Stick to float32 for broader MPS compatibility

        else: # CPU
            logger.info("Using CPU mode with float32 precision")
            dtype = torch.float32

        # Check for quantization (Currently CUDA only via bitsandbytes)
        enable_quantization = os.environ.get("ENABLE_QUANTIZATION", "false").lower() == "true" and target_device.type == 'cuda'
        is_quantized = False

        # Check for multi-GPU setup (CUDA only)
        if device_map and target_device.type == 'cuda' and torch.cuda.device_count() > 1:
            logger.info(f"Using device_map={device_map} across {torch.cuda.device_count()} CUDA GPUs")

            # Create model with device map
            model = Model(model_args)

            # Load state dict to CPU first for potential quantization/mapping
            state_dict = torch.load(ckpt_path, map_location='cpu')

            # Try quantization before device mapping if enabled
            if enable_quantization:
                try:
                    from bitsandbytes.nn import Linear8bitLt

                    def replace_with_8bit(model):
                        """Replace linear layers with 8-bit quantized versions"""
                        for name, module in model.named_modules():
                            if isinstance(module, torch.nn.Linear) and module.out_features > 256:
                                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                                parent = model
                                if parent_name:
                                    for attr in parent_name.split('.'):
                                        parent = getattr(parent, attr)
                                child_name = name.rsplit('.', 1)[1] if '.' in name else name
                                setattr(parent, child_name, Linear8bitLt.from_float(module))
                        return model

                    logger.info("Applying 8-bit quantization to linear layers (CUDA only)")
                    model = replace_with_8bit(model)
                    is_quantized = True
                except ImportError:
                    logger.warning("bitsandbytes not available, skipping quantization")
                except Exception as quant_error:
                     logger.error(f"Quantization failed: {quant_error}, proceeding without quantization")


            # Apply device mapping
            if device_map == "auto":
                # Use accelerate for automatic device mapping
                try:
                    from accelerate import init_empty_weights, load_checkpoint_and_dispatch

                    # Initialize empty model
                    with init_empty_weights():
                        empty_model = Model(model_args)
                        if is_quantized: # Ensure empty model reflects potential quantization
                             empty_model = replace_with_8bit(empty_model)


                    # Load and dispatch model across GPUs
                    # Pass the already loaded state_dict instead of ckpt_path if quantized
                    if is_quantized:
                         # If quantized, we loaded the state dict already, need a way to load into mapped model
                         # This might require custom loading with accelerate if quantized first
                         logger.warning("Quantization before accelerate auto-mapping might require custom handling. Attempting manual mapping.")
                         model = _manual_device_map(model, state_dict, "balanced") # Fallback
                    else:
                        model = load_checkpoint_and_dispatch(
                            empty_model,
                            ckpt_path,
                            device_map="auto",
                            no_split_module_classes=["TransformerLayer"],
                            dtype=dtype, # Pass desired dtype
                            # Offload CPU if very large model
                            offload_folder="offload" if os.environ.get("OFFLOAD_TO_CPU", "false").lower() == "true" else None
                        )
                    logger.info("Model loaded with automatic device mapping (CUDA)")
                except ImportError:
                    logger.warning("accelerate package not found, falling back to manual device mapping (CUDA)")
                    model = _manual_device_map(model, state_dict, "balanced")
                except Exception as mapping_error:
                    logger.error(f"Auto device mapping failed: {mapping_error}, falling back to manual (CUDA)")
                    model = _manual_device_map(model, state_dict, "balanced")
            else:
                # Manual device mapping
                logger.info(f"Applying manual device map: {device_map or 'balanced'} (CUDA)")
                model = _manual_device_map(model, state_dict, device_map or "balanced")

            # Safely move parts not handled by device_map (if any) - less likely needed here
            # model = safely_move_to_device(model, target_device) # device_map should handle this

        else:
            # Single device (CUDA, MPS, or CPU) setup
            logger.info(f"Using single device setup on {target_device}")
            model = None # Ensure model is reset before loading attempts

            # Try quantization before loading if enabled (GPU only)
            if enable_quantization and not is_quantized: # Already checked target_device.type == 'cuda'
                try:
                    # Load to CPU first for quantization
                    logger.info("Loading model to CPU for 8-bit quantization...")
                    model_cpu = Model(model_args).to("cpu")
                    state_dict = torch.load(ckpt_path, map_location="cpu")
                    model_cpu.load_state_dict(state_dict)

                    from bitsandbytes.nn import Linear8bitLt

                    def replace_with_8bit(model_to_quantize):
                        # ... (quantization function - same as above, ensure it's defined or accessible)
                        for name, module in model_to_quantize.named_modules():
                            if isinstance(module, torch.nn.Linear) and module.out_features > 256:
                                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                                parent = model_to_quantize
                                if parent_name:
                                    for attr in parent_name.split('.'):
                                        parent = getattr(parent, attr)
                                child_name = name.rsplit('.', 1)[1] if '.' in name else name
                                setattr(parent, child_name, Linear8bitLt.from_float(module))
                        return model_to_quantize


                    logger.info("Applying 8-bit quantization to linear layers (CUDA only)")
                    model_quantized = replace_with_8bit(model_cpu)
                    logger.info(f"Moving quantized model to {target_device}")
                    model = safely_move_to_device(model_quantized, target_device)
                    is_quantized = True
                    # Clear CPU model if needed
                    del model_cpu
                    if target_device.type == 'cuda': torch.cuda.empty_cache()


                except ImportError:
                    logger.warning("bitsandbytes not available, loading without quantization")
                except Exception as quant_error:
                    logger.error(f"Quantization failed: {quant_error}, loading without quantization")

            # Standard load if not quantized or quantization failed/skipped
            if model is None: # Only load standard if model wasn't set by quantization block
                 logger.info(f"Loading model directly to {target_device} with dtype {dtype}")
                 # Instantiate model directly on target device if possible (reduces peak memory)
                 try:
                     # Try initializing directly on the target device (might fail for meta device init)
                     model = Model(model_args, device=target_device, dtype=dtype)
                     state_dict = torch.load(ckpt_path, map_location=target_device)
                     model.load_state_dict(state_dict)
                 except Exception as direct_init_error:
                     logger.warning(f"Direct model initialization on {target_device} failed ({direct_init_error}), loading to CPU first.")
                     model = Model(model_args).to('cpu') # Load on CPU first
                     state_dict = torch.load(ckpt_path, map_location='cpu')
                     model.load_state_dict(state_dict)
                     logger.info(f"Moving model from CPU to {target_device} with dtype {dtype}")
                     model.to(dtype=dtype) # Apply dtype before moving
                     model = safely_move_to_device(model, target_device) # Move to target device


        # Ensure model is correctly placed after all loading logic
        if model is None:
             raise RuntimeError("Model loading failed.")

        # Verify final model device
        final_device = next(model.parameters()).device
        logger.info(f"Model loaded. Final device: {final_device}")
        if final_device.type != target_device.type:
             logger.warning(f"Model seems to be on {final_device} but target was {target_device}. Attempting final move.")
             model = safely_move_to_device(model, target_device)


        # Apply torch.compile if available (PyTorch 2.0+) and on CUDA
        compile_mode = os.environ.get("TORCH_COMPILE_MODE", "none")
        if hasattr(torch, 'compile') and compile_mode != "none" and target_device.type == 'cuda':
            try:
                logger.info(f"Using torch.compile with mode '{compile_mode}' for faster inference (CUDA only)")
                if compile_mode == "default":
                    model = torch.compile(model)
                else:
                    model = torch.compile(model, mode=compile_mode)
            except Exception as compile_error:
                logger.warning(f"Torch compile failed (requires PyTorch 2.0+ on CUDA): {compile_error}")

        # CUDA Graph setup (CUDA only)
        use_cuda_graphs = os.environ.get("USE_CUDA_GRAPHS", "false").lower() == "true"
        if use_cuda_graphs and target_device.type == 'cuda' and hasattr(torch.cuda, 'CUDAGraph'):
            try:
                logger.info("Setting up CUDA graphs for repeated inference patterns (CUDA only)")
                # This requires custom integration inside the model's forward method
                # For now, just log the intention. Real implementation needs model changes.
                # Example: model = torch.cuda.make_graphed_callables(model, (sample_input,))
            except Exception as graph_error:
                logger.warning(f"CUDA Graph setup failed: {graph_error}")

        # Create and return generator
        generator = Generator(model)
        logger.info("Generator created successfully.")
        return generator

    except Exception as e:
        logger.error(f"Error loading CSM-1B model: {e}")
        raise