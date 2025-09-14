import importlib.metadata
import torch
import logging
from tqdm import tqdm
import types, collections
from comfy.utils import ProgressBar, copy_to_param, set_attr_param
from comfy.model_patcher import get_key_weight, string_to_seed
from comfy.lora import calculate_weight
from comfy.model_management import cast_to_device
from comfy.float import stochastic_rounding
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def check_device_same(first_device, second_device):
    if first_device.type != second_device.type:
        return False

    if first_device.type == "cuda" and first_device.index is None:
        first_device = torch.device("cuda", index=0)

    if second_device.type == "cuda" and second_device.index is None:
        second_device = torch.device("cuda", index=0)

    return first_device == second_device

# simplified version of the accelerate function https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/modeling.py
def set_module_tensor_to_device(module, tensor_name, device, value=None, dtype=None):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).

    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        tensor_name (`str`):
            The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`, *optional*):
            The value of the tensor (useful when going from the meta device to any other device).
        dtype (`torch.dtype`, *optional*):
            If passed along the value of the parameter will be cast to this `dtype`. Otherwise, `value` will be cast to
            the dtype of the existing parameter in the model.
    """
    # Recurse if needed
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    param = module._parameters[tensor_name] if tensor_name in module._parameters else None
    param_cls = type(param)

    if value is not None:
        if dtype is None:
            value = value.to(old_value.dtype)
        elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            value = value.to(dtype)

    device_quantization = None
    with torch.no_grad():
        if value is None:
            new_value = old_value.to(device)
            if dtype is not None and device in ["meta", torch.device("meta")]:
                if not str(old_value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
                    new_value = new_value.to(dtype)

                if not is_buffer:
                    module._parameters[tensor_name] = param_cls(new_value, requires_grad=old_value.requires_grad)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = torch.tensor(value, device=device)
        if device_quantization is not None:
            device = device_quantization
        if is_buffer:
            module._buffers[tensor_name] = new_value
        elif value is not None or not check_device_same(torch.device(device), module._parameters[tensor_name].device):
            param_cls = type(module._parameters[tensor_name])
            new_value = param_cls(new_value, requires_grad=old_value.requires_grad).to(device)
            module._parameters[tensor_name] = new_value

    #if device != "cpu":
    #    mm.soft_empty_cache()

def check_diffusers_version():
    try:
        version = importlib.metadata.version('diffusers')
        required_version = '0.31.0'
        if version < required_version:
            raise AssertionError(f"diffusers version {version} is installed, but version {required_version} or higher is required.")
    except importlib.metadata.PackageNotFoundError:
        raise AssertionError("diffusers is not installed.")

def print_memory(device):
    memory = torch.cuda.memory_allocated(device) / 1024**3
    max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    log.info(f"Allocated memory: {memory=:.3f} GB")
    log.info(f"Max allocated memory: {max_memory=:.3f} GB")
    log.info(f"Max reserved memory: {max_reserved=:.3f} GB")
    #memory_summary = torch.cuda.memory_summary(device=device, abbreviated=False)
    #log.info(f"Memory Summary:\n{memory_summary}")

def get_module_memory_mb(module):
    memory = 0
    for param in module.parameters():
        if param.data is not None:
            memory += param.nelement() * param.element_size()
    return memory / (1024 * 1024)  # Convert to MB

def get_tensor_memory(tensor):
    memory_bytes = tensor.element_size() * tensor.nelement()
    return f"{memory_bytes / (1024 * 1024):.2f} MB"

def patch_weight_to_device(self, key, device_to=None, inplace_update=False, backup_keys=False, scale_weight=None):
    if key not in self.patches:
        return
    
    weight, set_func, convert_func = get_key_weight(self.model, key)
    inplace_update = self.weight_inplace_update or inplace_update

    if backup_keys and key not in self.backup:
        self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(weight.to(device=self.offload_device, copy=inplace_update), inplace_update)

    if device_to is not None:
        temp_weight = cast_to_device(weight, device_to, torch.float32, copy=True)
    else:
        temp_weight = weight.to(torch.float32, copy=True)
    if convert_func is not None:
        temp_weight = convert_func(temp_weight, inplace=True)

    if scale_weight is not None:
        temp_weight = temp_weight * scale_weight.to(temp_weight.device, temp_weight.dtype)

    out_weight = calculate_weight(self.patches[key], temp_weight, key)
    
    if set_func is None:
        out_weight = stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key))
        if inplace_update:
            copy_to_param(self.model, key, out_weight)
        else:
            set_attr_param(self.model, key, out_weight)
    else:
        set_func(out_weight, inplace_update=inplace_update, seed=string_to_seed(key))

def apply_lora(model, device_to, transformer_load_device, params_to_keep=None, dtype=None, base_dtype=None, state_dict=None, low_mem_load=False, control_lora=False, scale_weights={}):
        model.patch_weight_to_device = types.MethodType(patch_weight_to_device, model)
        to_load = []
        for n, m in model.model.named_modules():
            params = []
            skip = False
            for name, param in m.named_parameters(recurse=False):
                params.append(name)
            for name, param in m.named_parameters(recurse=True):
                if name not in params:
                    skip = True # skip random weights in non leaf modules
                    break
            if not skip and (hasattr(m, "comfy_cast_weights") or len(params) > 0):
                to_load.append((n, m, params))

        to_load.sort(reverse=True)
        pbar = ProgressBar(len(to_load))
        for x in tqdm(to_load, desc="Loading model and applying LoRA weights:", leave=True):
            name = x[0]
            m = x[1]
            params = x[2]
            if hasattr(m, "comfy_patched_weights"):
                if m.comfy_patched_weights == True:
                    continue
            for param in params:
                name = name.replace("._orig_mod.", ".") # torch compiled modules have this prefix
                if low_mem_load:
                    dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
                    if "patch_embedding" in name:
                        dtype_to_use = torch.float32
                    key = f"{name.replace('diffusion_model.', '')}.{param}"
                    try:
                        set_module_tensor_to_device(model.model.diffusion_model, key, device=transformer_load_device, dtype=dtype_to_use, value=state_dict[key])
                    except:
                        continue
                key = f"{name}.{param}"
                if scale_weights is not None:
                    scale_key = key.replace("weight", "scale_weight").replace("diffusion_model.", "") if "weight" in key else None
                if low_mem_load:
                    model.patch_weight_to_device(f"{name}.{param}", device_to=device_to, inplace_update=True, backup_keys=control_lora, scale_weight=scale_weights.get(scale_key, None))
                else:
                    model.patch_weight_to_device(f"{name}.{param}", device_to=device_to, backup_keys=control_lora, scale_weight=scale_weights.get(scale_key, None))
                    if device_to != transformer_load_device:
                        set_module_tensor_to_device(m, param, device=transformer_load_device)
                if low_mem_load:
                    try:
                        set_module_tensor_to_device(model.model.diffusion_model, key, device=transformer_load_device, dtype=dtype_to_use, value=model.model.diffusion_model.state_dict()[key])
                    except:
                        continue
            m.comfy_patched_weights = True
            pbar.update(1)
      
        model.current_weight_patches_uuid = model.patches_uuid
        if low_mem_load:
            for name, param in model.model.diffusion_model.named_parameters():
                if param.device != transformer_load_device:
                    dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
                    if "patch_embedding" in name:
                        dtype_to_use = torch.float32
                    try:
                        set_module_tensor_to_device(model.model.diffusion_model, name, device=transformer_load_device, dtype=dtype_to_use, value=state_dict[name])
                    except:
                        continue
        return model


# from https://github.com/cubiq/ComfyUI_IPAdapter_plus/blob/9d076a3df0d2763cef5510ec5ab807f6632c39f5/utils.py#L181
def split_tiles(embeds, num_split):
    _, H, W, _ = embeds.shape
    out = []
    for x in embeds:
        x = x.unsqueeze(0)
        h, w = H // num_split, W // num_split
        x_split = torch.cat([x[:, i*h:(i+1)*h, j*w:(j+1)*w, :] for i in range(num_split) for j in range(num_split)], dim=0)    
        out.append(x_split)
    
    x_split = torch.stack(out, dim=0)
    
    return x_split

def merge_hiddenstates(x, tiles):
    chunk_size = tiles*tiles
    x = x.split(chunk_size)

    out = []
    for embeds in x:
        num_tiles = embeds.shape[0]
        tile_size = int((embeds.shape[1]-1) ** 0.5)
        grid_size = int(num_tiles ** 0.5)

        # Extract class tokens
        class_tokens = embeds[:, 0, :]  # Save class tokens: [num_tiles, embeds[-1]]
        avg_class_token = class_tokens.mean(dim=0, keepdim=True).unsqueeze(0)  # Average token, shape: [1, 1, embeds[-1]]

        patch_embeds = embeds[:, 1:, :]  # Shape: [num_tiles, tile_size^2, embeds[-1]]
        reshaped = patch_embeds.reshape(grid_size, grid_size, tile_size, tile_size, embeds.shape[-1])

        merged = torch.cat([torch.cat([reshaped[i, j] for j in range(grid_size)], dim=1) 
                            for i in range(grid_size)], dim=0)
        
        merged = merged.unsqueeze(0)  # Shape: [1, grid_size*tile_size, grid_size*tile_size, embeds[-1]]
        
        # Pool to original size
        pooled = torch.nn.functional.adaptive_avg_pool2d(merged.permute(0, 3, 1, 2), (tile_size, tile_size)).permute(0, 2, 3, 1)
        flattened = pooled.reshape(1, tile_size*tile_size, embeds.shape[-1])
        
        # Add back the class token
        with_class = torch.cat([avg_class_token, flattened], dim=1)  # Shape: original shape
        out.append(with_class)
    
    out = torch.cat(out, dim=0)

    return out

from comfy.clip_vision import clip_preprocess, ClipVisionModel

def clip_encode_image_tiled(clip_vision, image, tiles=1, ratio=1.0):
    embeds = encode_image_(clip_vision, image)
    tiles = min(tiles, 16)

    if tiles > 1:
        # split in tiles
        image_split = split_tiles(image, tiles)

        # get the embeds for each tile
        embeds_split = {}
        for i in image_split:
            encoded = encode_image_(clip_vision, i)
            if not hasattr(embeds_split, "last_hidden_state"):
                embeds_split["last_hidden_state"] = encoded
            else:
                embeds_split["last_hidden_state"] = torch.cat(embeds_split["last_hidden_state"], encoded, dim=0)

        embeds_split['last_hidden_state'] = merge_hiddenstates(embeds_split['last_hidden_state'], tiles)

        if embeds.shape[0] > 1: # if we have more than one image we need to average the embeddings for consistency
            embeds = embeds * ratio + embeds_split['last_hidden_state']*(1-ratio)
        else: # otherwise we can concatenate them, they can be averaged later
            embeds = torch.cat([embeds * ratio, embeds_split['last_hidden_state']])

    return embeds

def encode_image_(clip_vision, image):
    if isinstance(clip_vision, ClipVisionModel):
        out = clip_vision.encode_image(image).last_hidden_state
    else:
        pixel_values = clip_preprocess(image, size=224, crop=True).float()
        out = clip_vision.visual(pixel_values)

    return out

# Code based on https://github.com/WikiChao/FreSca (MIT License)
import torch
import torch.fft as fft

def fourier_filter(x, scale_low=1.0, scale_high=1.5, freq_cutoff=20):
    """
    Apply frequency-dependent scaling to an image tensor using Fourier transforms.

    Parameters:
        x:           Input tensor of shape (B, C, H, W)
        scale_low:   Scaling factor for low-frequency components (default: 1.0)
        scale_high:  Scaling factor for high-frequency components (default: 1.5)
        freq_cutoff: Number of frequency indices around center to consider as low-frequency (default: 20)

    Returns:
        x_filtered: Filtered version of x in spatial domain with frequency-specific scaling applied.
    """
    # Preserve input dtype and device
    dtype, device = x.dtype, x.device

    # Convert to float32 for FFT computations
    x = x.to(torch.float32)

    # 1) Apply FFT and shift low frequencies to center
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    # 2) Create a mask to scale frequencies differently
    C, B, H, W = x_freq.shape
    crow, ccol = H // 2, W // 2

    # Initialize mask with high-frequency scaling factor
    mask = torch.ones((C, B, H, W), device=device) * scale_high

    # Apply low-frequency scaling factor to center region
    mask[
        ...,
        crow - freq_cutoff : crow + freq_cutoff,
        ccol - freq_cutoff : ccol + freq_cutoff,
    ] = scale_low

    # 3) Apply frequency-specific scaling
    x_freq = x_freq * mask

    # 4) Convert back to spatial domain
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    # 5) Restore original dtype
    x_filtered = x_filtered.to(dtype)

    return x_filtered

def is_image_black(image, threshold=1e-3):
    if image.min() < 0:
        image = (image + 1) / 2
    return torch.all(image < threshold).item()

def add_noise_to_reference_video(image, ratio=None):
    sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * ratio 
    image_noise = torch.randn_like(image) * sigma[:, None, None, None]
    image_noise = torch.where(image==-1, torch.zeros_like(image), image_noise)
    image = image + image_noise
    return image

def optimized_scale(positive_flat, negative_flat):

    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm
    
    return st_star

def find_closest_valid_dim(fixed_dim, var_dim, block_size):
    for delta in range(1, 17):
        for sign in [-1, 1]:
            candidate = var_dim + sign * delta
            if candidate > 0 and ((fixed_dim * candidate) // 4) % block_size == 0:
                return candidate
    return var_dim

 # Radial attention setup
def setup_radial_attention(transformer, transformer_options, latent, seq_len, latent_video_length, context_options=None):
    if context_options is not None:
        context_frames =  (context_options["context_frames"] - 1) // 4 + 1

    dense_timesteps = transformer_options.get("dense_timesteps", 1)
    dense_blocks = transformer_options.get("dense_blocks", 1)
    dense_vace_blocks = transformer_options.get("dense_vace_blocks", 1)
    decay_factor = transformer_options.get("decay_factor", 0.2)
    dense_attention_mode = transformer_options.get("dense_attention_mode", "sageattn")
    block_size = transformer_options.get("block_size", 128)

    # Calculate closest valid latent sizes
    if latent.shape[2] % (block_size/8) != 0 or latent.shape[3] % (block_size/8) != 0:
        block_div = int(block_size // 8)
        closest_h = round(latent.shape[2] / block_div) * block_div
        closest_w = round(latent.shape[3] / block_div) * block_div
        raise Exception(
            f"Radial attention mode only supports image size divisible by block size. "
            f"Got {latent.shape[3] * 8}x{latent.shape[2] * 8} with block size {block_size}.\n"
            f"Closest valid sizes: {closest_w * 8}x{closest_h * 8} (width x height in pixels)."
        )
    tokens_per_frame = (latent.shape[2] * latent.shape[3]) // 4
    if tokens_per_frame % block_size != 0:
        closest_latent_h = find_closest_valid_dim(latent.shape[3], latent.shape[2], block_size)
        closest_latent_w = find_closest_valid_dim(latent.shape[2], latent.shape[3], block_size)
        raise Exception(
            f"Radial attention mode requires tokens per frame ((latent_h * latent_w) // 4) to be divisible by block size ({block_size}).\n"
            f"Current size in latent space:{latent.shape[3]}x{latent.shape[2]}, pixel space: {latent.shape[3]*8}x{latent.shape[2]*8} tokens_per_frame={tokens_per_frame}.\n"
            f"Try adjusting to one of these latent sizes (in pixels):\n"
            f"  Height: {latent.shape[2]*8} -> {closest_latent_h * 8}\n"
            f"  Width: {latent.shape[3]*8} -> {closest_latent_w * 8}\n"
            f"Or choose another resolution so that (latent_h * latent_w) // 4 is divisible by {block_size}."
        )

    from .wanvideo.radial_attention.attn_mask import MaskMap
    for i, block in enumerate(transformer.blocks):
        block.self_attn.mask_map = block.dense_attention_mode = block.dense_timesteps = block.self_attn.decay_factor = None
        if isinstance(dense_blocks, list):
            block.dense_block = i in dense_blocks
        else:
            block.dense_block = i < dense_blocks
        block.self_attn.mask_map = MaskMap(video_token_num=seq_len, num_frame=latent_video_length if context_options is None else context_frames, block_size=block_size)
        block.dense_attention_mode = dense_attention_mode
        block.dense_timesteps = dense_timesteps
        block.self_attn.decay_factor = decay_factor
    if transformer.vace_layers is not None:
        for i, block in enumerate(transformer.vace_blocks):
            block.self_attn.mask_map = block.dense_attention_mode = block.dense_timesteps = block.self_attn.decay_factor = None
            if isinstance(dense_vace_blocks, list):
                block.dense_block = i in dense_vace_blocks
            else:
                block.dense_block = i < dense_vace_blocks
            block.self_attn.mask_map = MaskMap(video_token_num=seq_len, num_frame=latent_video_length if context_options is None else context_frames, block_size=block_size)
            block.dense_attention_mode = dense_attention_mode
            block.dense_timesteps = dense_timesteps
            block.self_attn.decay_factor = decay_factor
                    
    log.info(f"Radial attention mode enabled.")
    log.info(f"dense_attention_mode: {dense_attention_mode}, dense_timesteps: {dense_timesteps}, decay_factor: {decay_factor}")
    log.info(f"dense_blocks: {[i for i, block in enumerate(transformer.blocks) if getattr(block, 'dense_block', False)]})")



def list_to_device(tensor_list, device, dtype=None):
    """
    Move all tensors in a list to the specified device and optionally cast to dtype.
    """
    return [t.to(device, dtype=dtype) if dtype is not None else t.to(device) for t in tensor_list]

def dict_to_device(tensor_dict, device, dtype=None):
    """
    Move all tensors (and tensor lists) in a dict to the specified device and optionally cast to dtype.
    Supports values that are tensors or lists of tensors.
    """
    result = {}
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device, dtype=dtype) if dtype is not None else v.to(device)
        elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
            result[k] = list_to_device(v, device, dtype)
        else:
            result[k] = v
    return result

def compile_model(transformer, compile_args=None):
    if compile_args is None:
        return transformer
    torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
    try:
        if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'config'):
            torch._dynamo.config.recompile_limit = compile_args["dynamo_recompile_limit"]
    except Exception as e:
        log.warning(f"Could not set recompile_limit: {e}")
    if compile_args["compile_transformer_blocks_only"]:
        for i, block in enumerate(transformer.blocks):
            if hasattr(block, "_orig_mod"):
                block = block._orig_mod
            transformer.blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
        if transformer.vace_layers is not None:
            for i, block in enumerate(transformer.vace_blocks):
                if hasattr(block, "_orig_mod"):
                    block = block._orig_mod
                transformer.vace_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
    else:
        transformer = torch.compile(transformer, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
    return transformer

def tangential_projection(pred_cond: torch.Tensor, pred_uncond: torch.Tensor) -> torch.Tensor:
    cond_dtype = pred_cond.dtype
    preds = torch.stack([pred_cond, pred_uncond], dim=1).float()
    orig_shape = preds.shape[2:]
    preds_flat = preds.flatten(2)
    U, S, Vh = torch.linalg.svd(preds_flat, full_matrices=False)
    Vh_modified = Vh.clone()
    Vh_modified[:, 1] = 0
    recon = U @ torch.diag_embed(S) @ Vh_modified
    return recon[:, 1].view(pred_uncond.shape).to(cond_dtype)
