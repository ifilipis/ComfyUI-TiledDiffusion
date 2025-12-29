import torch
import contextlib
from torch import Tensor
from typing import List, Union, Tuple, Callable, Dict
from weakref import WeakSet
import comfy.utils
import comfy.model_patcher
import comfy.model_management
from nodes import ImageScale
from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher
from comfy.controlnet import ControlNet, T2IAdapter
from comfy.utils import common_upscale
from comfy.model_management import processing_interrupted, loaded_models, load_models_gpu
from math import pi

opt_C = 4
opt_f = 8

def ceildiv(big, small):
    # Correct ceiling division that avoids floating-point errors and importing math.ceil.
    return -(big // -small)

from enum import Enum
class BlendMode(Enum):  # i.e. LayerType
    FOREGROUND = 'Foreground'
    BACKGROUND = 'Background'

class Processing: ...
class Device: ...
devices = Device()
devices.device = comfy.model_management.get_torch_device()

def null_decorator(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

keep_signature = null_decorator
controlnet     = null_decorator
stablesr       = null_decorator
grid_bbox      = null_decorator
custom_bbox    = null_decorator
noise_inverse  = null_decorator

class BBox:
    ''' grid bbox '''

    def __init__(self, x:int, y:int, w:int, h:int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.box = [x, y, x+w, y+h]
        self.slicer = slice(None), slice(None), slice(y, y+h), slice(x, x+w)

    def __getitem__(self, idx:int) -> int:
        return self.box[idx]

    def slicer_for(self, tensor: Tensor):
        if tensor.dim() == 5:
            return slice(None), slice(None), slice(None), slice(self.y, self.y + self.h), slice(self.x, self.x + self.w)
        return self.slicer

def repeat_to_batch_size(tensor, batch_size, dim=0):
    if dim == 0 and tensor.shape[dim] == 1:
        return tensor.expand([batch_size] + [-1] * (len(tensor.shape) - 1))
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    elif tensor.shape[dim] < batch_size:
        return tensor.repeat(dim * [1] + [ceildiv(batch_size, tensor.shape[dim])] + [1] * (len(tensor.shape) - 1 - dim)).narrow(dim, 0, batch_size)
    return tensor

def split_bboxes(w:int, h:int, tile_w:int, tile_h:int, overlap:int=16, init_weight:Union[Tensor, float]=1.0) -> Tuple[List[BBox], Tensor]:
    cols = ceildiv((w - overlap) , (tile_w - overlap))
    rows = ceildiv((h - overlap) , (tile_h - overlap))
    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    bbox_list: List[BBox] = []
    weight = torch.zeros((1, 1, h, w), device=devices.device, dtype=torch.float32)
    for row in range(rows):
        y = min(int(row * dy), h - tile_h)
        for col in range(cols):
            x = min(int(col * dx), w - tile_w)

            bbox = BBox(x, y, tile_w, tile_h)
            bbox_list.append(bbox)
            weight[bbox.slicer] += init_weight

    return bbox_list, weight

class CustomBBox(BBox):
    ''' region control bbox '''
    pass

class AbstractDiffusion:
    def __init__(self):
        self.method = self.__class__.__name__
        self.pbar = None


        self.w: int = 0
        self.h: int = 0
        self.tile_width: int = None
        self.tile_height: int = None
        self.tile_overlap: int = None
        self.tile_batch_size: int = None

        # cache. final result of current sampling step, [B, C=4, H//8, W//8]
        # avoiding overhead of creating new tensors and weight summing
        self.x_buffer: Tensor = None
        # self.w: int = int(self.p.width  // opt_f)       # latent size
        # self.h: int = int(self.p.height // opt_f)
        # weights for background & grid bboxes
        self._weights: Tensor = None
        # self.weights: Tensor = torch.zeros((1, 1, self.h, self.w), device=devices.device, dtype=torch.float32)
        self._init_grid_bbox = None
        self._init_done = None

        # count the step correctly
        self.step_count = 0         
        self.inner_loop_count = 0  
        self.kdiff_step = -1

        # ext. Grid tiling painting (grid bbox)
        self.enable_grid_bbox: bool = False
        self.tile_w: int = None
        self.tile_h: int = None
        self.tile_bs: int = None
        self.num_tiles: int = None
        self.num_batches: int = None
        self.batched_bboxes: List[List[BBox]] = []

        # ext. Region Prompt Control (custom bbox)
        self.enable_custom_bbox: bool = False
        self.custom_bboxes: List[CustomBBox] = []
        # self.cond_basis: Cond = None
        # self.uncond_basis: Uncond = None
        # self.draw_background: bool = True       # by default we draw major prompts in grid tiles
        # self.causal_layers: bool = None

        # ext. ControlNet
        self.enable_controlnet: bool = False
        # self.controlnet_script: ModuleType = None
        self.control_tensor_batch_dict = {}
        self.control_tensor_batch: List[List[Tensor]] = [[]]
        # self.control_params: Dict[str, Tensor] = None # {}
        self.control_params: Dict[Tuple, List[List[Tensor]]] = {}
        self.control_tensor_cpu: bool = None
        self.control_tensor_custom: List[List[Tensor]] = []

        self.draw_background: bool = True       # by default we draw major prompts in grid tiles
        self.control_tensor_cpu = False
        self.weights = None
        self.imagescale = ImageScale()
        self.uniform_distribution = None
        self.sigmas = None

    def _get_rope_patch_sizes(self, model_function) -> Tuple[int, int, int] | None:
        base_model = getattr(model_function, "__self__", None)
        diffusion_model = getattr(base_model, "diffusion_model", None)
        patch_size = getattr(diffusion_model, "patch_size", None)
        if patch_size is None:
            return None
        if isinstance(patch_size, (tuple, list)):
            if len(patch_size) == 3:
                patch_t, patch_h, patch_w = patch_size
            elif len(patch_size) == 2:
                patch_t, patch_h, patch_w = 1, patch_size[0], patch_size[1]
            elif len(patch_size) == 1:
                patch_t, patch_h, patch_w = 1, patch_size[0], patch_size[0]
            else:
                return None
        else:
            patch_t = 1
            patch_h = patch_w = patch_size
        if not isinstance(patch_t, int) or not isinstance(patch_h, int) or not isinstance(patch_w, int):
            return None
        if patch_t <= 0 or patch_h <= 0 or patch_w <= 0:
            return None
        return patch_t, patch_h, patch_w

    def _build_rope_options(self, bbox: BBox, full_t: int, full_h: int, full_w: int, tile_t: int, patch_t: int, patch_h: int, patch_w: int) -> Dict[str, float]:
        global_t_len = (full_t + (patch_t // 2)) // patch_t
        global_h_len = (full_h + (patch_h // 2)) // patch_h
        global_w_len = (full_w + (patch_w // 2)) // patch_w
        tile_t_len = (tile_t + (patch_t // 2)) // patch_t
        tile_h_len = (bbox.h + (patch_h // 2)) // patch_h
        tile_w_len = (bbox.w + (patch_w // 2)) // patch_w
        scale_t = (global_t_len - 1) / max(tile_t_len - 1, 1)
        scale_y = (global_h_len - 1) / max(tile_h_len - 1, 1)
        scale_x = (global_w_len - 1) / max(tile_w_len - 1, 1)
        shift_t = 0
        shift_y = (bbox.y + (patch_h // 2)) // patch_h
        shift_x = (bbox.x + (patch_w // 2)) // patch_w
        return {
            "scale_x": float(scale_x),
            "shift_x": float(shift_x),
            "scale_y": float(scale_y),
            "shift_y": float(shift_y),
            "scale_t": float(scale_t),
            "shift_t": float(shift_t),
        }

    def _tile_transformer_options(self, transformer_options: dict, model_function, bbox: BBox, full_t: int, full_h: int, full_w: int, tile_t: int) -> dict | None:
        patch_sizes = self._get_rope_patch_sizes(model_function)
        if patch_sizes is None:
            return transformer_options
        patch_t, patch_h, patch_w = patch_sizes
        rope_options = self._build_rope_options(bbox, full_t, full_h, full_w, tile_t, patch_t, patch_h, patch_w)
        options = transformer_options.copy() if isinstance(transformer_options, dict) else {}
        rope_base = options.get("rope_options", {})
        if isinstance(rope_base, dict):
            rope_base = rope_base.copy()
        else:
            rope_base = {}
        rope_base.update(rope_options)
        options["rope_options"] = rope_base
        return options

    def _get_diffusion_model(self, model_function):
        base_model = getattr(model_function, "__self__", None)
        return getattr(base_model, "diffusion_model", None)

    def _is_qwen_model(self, model_function) -> bool:
        diffusion_model = self._get_diffusion_model(model_function)
        if diffusion_model is None:
            return False
        return "QwenImage" in diffusion_model.__class__.__name__

    @contextlib.contextmanager
    def _patch_qwen_process_img(self, model_function, bbox: BBox, tile_index: int, control: ControlNet | None):
        patched = []
        diffusion_model = self._get_diffusion_model(model_function)
        if diffusion_model is not None and hasattr(diffusion_model, "process_img"):
            original = diffusion_model.process_img
            def process_img_patched(x, index=0, h_offset=0, w_offset=0, _orig=original):
                return _orig(x, index=tile_index, h_offset=bbox.y, w_offset=bbox.x)
            diffusion_model.process_img = process_img_patched
            patched.append((diffusion_model, original))

        current_control = control
        while current_control is not None:
            if hasattr(current_control, "process_img"):
                original = current_control.process_img
                def process_img_patched(x, index=0, h_offset=0, w_offset=0, _orig=original):
                    return _orig(x, index=tile_index, h_offset=bbox.y, w_offset=bbox.x)
                current_control.process_img = process_img_patched
                patched.append((current_control, original))
            current_control = getattr(current_control, "previous_controlnet", None)

        try:
            yield
        finally:
            for target, original in patched:
                target.process_img = original

    def _slice_control_for_tile(self, control: ControlNet, tile_index: int, batch_size: int) -> list[tuple[ControlNet, Tensor | None]]:
        saved = []
        while control is not None:
            saved.append((control, getattr(control, "cond_hint", None)))
            if control.cond_hint is not None:
                start = tile_index * batch_size
                end = start + batch_size
                control.cond_hint = control.cond_hint[start:end]
            control = control.previous_controlnet
        return saved

    def _restore_control(self, saved: list[tuple[ControlNet, Tensor | None]]) -> None:
        for control, cond_hint in saved:
            control.cond_hint = cond_hint

    def reset(self):
        tile_width = self.tile_width
        tile_height = self.tile_height
        tile_overlap = self.tile_overlap
        tile_batch_size = self.tile_batch_size
        compression = self.compression
        width = self.width
        height  = self.height 
        overlap = self.overlap
        self.__init__()
        self.compression = compression
        self.width = width
        self.height  = height
        self.overlap = overlap
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.tile_overlap = tile_overlap
        self.tile_batch_size = tile_batch_size

    def repeat_tensor(self, x:Tensor, n:int, concat=False, concat_to=0) -> Tensor:
        ''' repeat the tensor on it's first dim '''
        if n == 1: return x
        B = x.shape[0]
        r_dims = len(x.shape) - 1
        if B == 1:      # batch_size = 1 (not `tile_batch_size`)
            shape = [n] + [-1] * r_dims     # [N, -1, ...]
            return x.expand(shape)          # `expand` is much lighter than `tile`
        else:
            if concat:
                return torch.cat([x for _ in range(n)], dim=0)[:concat_to]
            shape = [n] + [1] * r_dims      # [N, 1, ...]
            return x.repeat(shape)
    def update_pbar(self):
        if self.pbar.n >= self.pbar.total:
            self.pbar.close()
        else:
            # self.pbar.update()
            sampling_step = 20
            if self.step_count == sampling_step:
                self.inner_loop_count += 1
                if self.inner_loop_count < self.total_bboxes:
                    self.pbar.update()
            else:
                self.step_count = sampling_step
                self.inner_loop_count = 0
    def reset_buffer(self, x_in:Tensor):
        # Judge if the shape of x_in is the same as the shape of x_buffer
        if self.x_buffer is None or self.x_buffer.shape != x_in.shape:
            self.x_buffer = torch.zeros_like(x_in, device=x_in.device, dtype=x_in.dtype)
        else:
            self.x_buffer.zero_()

    @grid_bbox
    def init_grid_bbox(self, tile_w:int, tile_h:int, overlap:int, tile_bs:int):
        # if self._init_grid_bbox is not None: return
        # self._init_grid_bbox = True
        self.weights = torch.zeros((1, 1, self.h, self.w), device=devices.device, dtype=torch.float32)
        self.enable_grid_bbox = True

        self.tile_w = min(tile_w, self.w)
        self.tile_h = min(tile_h, self.h)
        overlap = max(0, min(overlap, min(tile_w, tile_h) - 4))
        # split the latent into overlapped tiles, then batching
        # weights basically indicate how many times a pixel is painted
        bboxes, weights = split_bboxes(self.w, self.h, self.tile_w, self.tile_h, overlap, self.get_tile_weights())
        self.weights += weights
        self.num_tiles = len(bboxes)
        self.num_batches = ceildiv(self.num_tiles , tile_bs)
        self.tile_bs = ceildiv(len(bboxes) , self.num_batches)          # optimal_batch_size
        self.batched_bboxes = [bboxes[i*self.tile_bs:(i+1)*self.tile_bs] for i in range(self.num_batches)]

    # detached version of above
    @grid_bbox
    def get_grid_bbox(self, tile_w: int, tile_h: int, overlap: int, tile_bs: int, w: int, h: int, 
                    device: torch.device, get_tile_weights: Callable = lambda: 1.0) -> List[List[BBox]]:
        weights = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)
        # enable_grid_bbox = True

        tile_w = min(tile_w, w)
        tile_h = min(tile_h, h)
        overlap = max(0, min(overlap, min(tile_w, tile_h) - 4))
        # split the latent into overlapped tiles, then batching
        # weights basically indicate how many times a pixel is painted
        bboxes, weights_ = split_bboxes(w, h, tile_w, tile_h, overlap, get_tile_weights())
        weights += weights_
        num_tiles = len(bboxes)
        num_batches = ceildiv(num_tiles, tile_bs)
        tile_bs = ceildiv(len(bboxes), num_batches)          # optimal_batch_size
        batched_bboxes = [bboxes[i*tile_bs:(i+1)*tile_bs] for i in range(num_batches)]
        return batched_bboxes

    @grid_bbox
    def get_tile_weights(self) -> Union[Tensor, float]:
        return 1.0

    @noise_inverse
    def init_noise_inverse(self, steps:int, retouch:float, get_cache_callback, set_cache_callback, renoise_strength:float, renoise_kernel:int):
        self.noise_inverse_enabled = True
        self.noise_inverse_steps = steps
        self.noise_inverse_retouch = float(retouch)
        self.noise_inverse_renoise_strength = float(renoise_strength)
        self.noise_inverse_renoise_kernel = int(renoise_kernel)
        self.noise_inverse_set_cache = set_cache_callback
        self.noise_inverse_get_cache = get_cache_callback

    def init_done(self):
        '''
          Call this after all `init_*`, settings are done, now perform:
            - settings sanity check 
            - pre-computations, cache init
            - anything thing needed before denoising starts
        '''

        # if self._init_done is not None: return
        # self._init_done = True
        self.total_bboxes = 0
        if self.enable_grid_bbox:   self.total_bboxes += self.num_batches
        if self.enable_custom_bbox: self.total_bboxes += len(self.custom_bboxes)
        assert self.total_bboxes > 0, "Nothing to paint! No background to draw and no custom bboxes were provided."

        # sampling_steps = _steps
        # self.pbar = tqdm(total=(self.total_bboxes) * sampling_steps, desc=f"{self.method} Sampling: ")

    @controlnet
    def prepare_controlnet_tensors(self, refresh:bool=False, tensor=None):
        ''' Crop the control tensor into tiles and cache them '''
        if not refresh:
            if self.control_tensor_batch is not None or self.control_params is not None: return
        tensors = [tensor]
        self.org_control_tensor_batch = tensors
        self.control_tensor_batch = []
        for i in range(len(tensors)):
            control_tile_list = []
            control_tensor = tensors[i]
            for bboxes in self.batched_bboxes:
                single_batch_tensors = []
                for bbox in bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tensor.unsqueeze_(0)
                    control_tile = control_tensor[:, :, bbox[1]*opt_f:bbox[3]*opt_f, bbox[0]*opt_f:bbox[2]*opt_f]
                    single_batch_tensors.append(control_tile)
                control_tile = torch.cat(single_batch_tensors, dim=0)
                if self.control_tensor_cpu:
                    control_tile = control_tile.cpu()
                control_tile_list.append(control_tile)
            self.control_tensor_batch.append(control_tile_list)

            if len(self.custom_bboxes) > 0:
                custom_control_tile_list = []
                for bbox in self.custom_bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tensor.unsqueeze_(0)
                    control_tile = control_tensor[:, :, bbox[1]*opt_f:bbox[3]*opt_f, bbox[0]*opt_f:bbox[2]*opt_f]
                    if self.control_tensor_cpu:
                        control_tile = control_tile.cpu()
                    custom_control_tile_list.append(control_tile)
                self.control_tensor_custom.append(custom_control_tile_list)

    @controlnet
    def switch_controlnet_tensors(self, batch_id:int, x_batch_size:int, tile_batch_size:int, is_denoise=False):
        # if not self.enable_controlnet: return
        if self.control_tensor_batch is None: return
        # self.control_params = [0]

        # for param_id in range(len(self.control_params)):
        for param_id in range(len(self.control_tensor_batch)):
            # tensor that was concatenated in `prepare_controlnet_tensors`
            control_tile = self.control_tensor_batch[param_id][batch_id]
            # broadcast to latent batch size
            if x_batch_size > 1: # self.is_kdiff:
                all_control_tile = []
                for i in range(tile_batch_size):
                    this_control_tile = [control_tile[i].unsqueeze(0)] * x_batch_size
                    all_control_tile.append(torch.cat(this_control_tile, dim=0))
                control_tile = torch.cat(all_control_tile, dim=0) # [:x_tile.shape[0]]
                self.control_tensor_batch[param_id][batch_id] = control_tile
            # else:
            #     control_tile = control_tile.repeat([x_batch_size if is_denoise else x_batch_size * 2, 1, 1, 1])
            # self.control_params[param_id].hint_cond = control_tile.to(devices.device)

    def process_controlnet(self, x_noisy, c_in: dict, cond_or_uncond: List, bboxes, batch_size: int, batch_id: int, shifts=None, shift_condition=None):
        control: ControlNet = c_in['control']
        param_id = -1 # current controlnet & previous controlnets
        tuple_key = tuple(cond_or_uncond) + tuple(x_noisy.shape)
        while control is not None:
            param_id += 1

            if tuple_key not in self.control_params:
                self.control_params[tuple_key] = [[None]]

            while len(self.control_params[tuple_key]) <= param_id:
                self.control_params[tuple_key].append([None])

            while len(self.control_params[tuple_key][param_id]) <= batch_id:
                self.control_params[tuple_key][param_id].append(None)

            # Below is taken from comfy.controlnet.py, but we need to additionally tile the cnets.
            # if statement: eager eval. first time when cond_hint is None. 
            if self.refresh or control.cond_hint is None or not isinstance(self.control_params[tuple_key][param_id][batch_id], Tensor):
                if control.cond_hint is not None:
                    del control.cond_hint
                control.cond_hint = None
                compression_ratio = control.compression_ratio
                if getattr(control, 'vae', None) is not None:
                    compression_ratio *= control.vae.downscale_ratio
                else:
                    if getattr(control, 'latent_format', None) is not None:
                        raise ValueError("This Controlnet needs a VAE but none was provided, please use a ControlNetApply node with a VAE input and connect it.")
                PH, PW = self.h * compression_ratio, self.w * compression_ratio

                device = getattr(control, 'device', x_noisy.device)
                dtype = getattr(control, 'manual_cast_dtype', None)
                if dtype is None: dtype = getattr(getattr(control, 'control_model', None), 'dtype', None)
                if dtype is None: dtype = x_noisy.dtype

                if isinstance(control, T2IAdapter):
                    width, height = control.scale_image_to(PW, PH)
                    cns = common_upscale(control.cond_hint_original, width, height, control.upscale_algorithm, "center").float().to(device=device)
                    if control.channels_in == 1 and control.cond_hint.shape[1] > 1:
                        cns = torch.mean(control.cond_hint, 1, keepdim=True)
                elif control.__class__.__name__ == 'ControlLLLiteAdvanced':
                    if getattr(control, 'sub_idxs', None) is not None and control.cond_hint_original.shape[0] >= control.full_latent_length:
                        cns = common_upscale(control.cond_hint_original[control.sub_idxs], PW, PH, control.upscale_algorithm, "center").to(dtype=dtype, device=device)
                    else:
                        cns = common_upscale(control.cond_hint_original, PW, PH, control.upscale_algorithm, "center").to(dtype=dtype, device=device)
                else:
                    cns = common_upscale(control.cond_hint_original, PW, PH, control.upscale_algorithm, 'center').to(dtype=dtype, device=device)
                    cns = control.preprocess_image(cns)
                    if getattr(control, 'vae', None) is not None:
                        loaded_models_ = loaded_models(only_currently_used=True)
                        cns = control.vae.encode(cns.movedim(1, -1))
                        load_models_gpu(loaded_models_)
                    if getattr(control, 'latent_format', None) is not None:
                        cns = control.latent_format.process_in(cns)
                    if len(getattr(control, 'extra_concat_orig', ())) > 0:
                        to_concat = []
                        for c in control.extra_concat_orig:
                            c = c.to(device=device)
                            c = common_upscale(c, cns.shape[3], cns.shape[2], control.upscale_algorithm, "center")
                            to_concat.append(repeat_to_batch_size(c, cns.shape[0]))
                        cns = torch.cat([cns] + to_concat, dim=1)

                    cns = cns.to(device=device, dtype=dtype)

                # Tile the ControlNets
                #
                # Below can be in this if clause because self.refresh will trigger on resolution change,
                # e.g. cause of ConditioningSetArea, so that particular case isn't cached atm.
                cf = control.compression_ratio
                if cns.shape[0] != batch_size:
                    cns = repeat_to_batch_size(cns, batch_size)
                if shifts is not None:
                    control.cns = cns
                    # cns = cns.roll(shifts=tuple(x * cf for x in shifts), dims=(-2,-1))
                    sh_h,sh_w=shifts
                    sh_h *= cf
                    sh_w *= cf
                    if (sh_h,sh_w) != (0,0):
                        if sh_h == 0 or sh_w == 0:
                            cns = control.cns.roll(shifts=(sh_h,sh_w), dims=(-2,-1))
                        else:
                            if shift_condition:
                                cns = control.cns.roll(shifts=sh_h, dims=-2)
                            else:
                                cns = control.cns.roll(shifts=sh_w, dims=-1)
                cns_slices = [cns[:, :, bbox[1]*cf:bbox[3]*cf, bbox[0]*cf:bbox[2]*cf] for bbox in bboxes]
                control.cond_hint = torch.cat(cns_slices, dim=0).to(device=cns.device)
                del cns_slices
                del cns
                self.control_params[tuple_key][param_id][batch_id] = control.cond_hint
            else:
                if hasattr(control,'cns') and shifts is not None:
                    cf = control.compression_ratio
                    # cns = control.cns.roll(shifts=tuple(x * cf for x in shifts), dims=(-2,-1))
                    cns = control.cns
                    sh_h,sh_w=shifts
                    sh_h *= cf
                    sh_w *= cf
                    if (sh_h,sh_w) != (0,0):
                        if sh_h == 0 or sh_w == 0:
                            cns = control.cns.roll(shifts=(sh_h,sh_w), dims=(-2,-1))
                        else:
                            if shift_condition:
                                cns = control.cns.roll(shifts=sh_h, dims=-2)
                            else:
                                cns = control.cns.roll(shifts=sh_w, dims=-1)
                    cns_slices = [cns[:, :, bbox[1]*cf:bbox[3]*cf, bbox[0]*cf:bbox[2]*cf] for bbox in bboxes]
                    control.cond_hint = torch.cat(cns_slices, dim=0).to(device=cns.device)
                    del cns_slices
                    del cns
                else:
                    control.cond_hint = self.control_params[tuple_key][param_id][batch_id]
            control = control.previous_controlnet

import numpy as np
from numpy import pi, exp, sqrt
def gaussian_weights(tile_w:int, tile_h:int) -> Tensor:
    '''
    Copy from the original implementation of Mixture of Diffusers
    https://github.com/albarji/mixture-of-diffusers/blob/master/mixdiff/tiling.py
    This generates gaussian weights to smooth the noise of each tile.
    This is critical for this method to work.
    '''
    f = lambda x, midpoint, var=0.01: exp(-(x-midpoint)*(x-midpoint) / (tile_w*tile_w) / (2*var)) / sqrt(2*pi*var)
    x_probs = [f(x, (tile_w - 1) / 2) for x in range(tile_w)]   # -1 because index goes from 0 to latent_width - 1
    y_probs = [f(y,  tile_h      / 2) for y in range(tile_h)]

    w = np.outer(y_probs, x_probs)
    return torch.from_numpy(w).to(devices.device, dtype=torch.float32)

class CondDict: ...

class MultiDiffusion(AbstractDiffusion):
    
    @torch.inference_mode()
    def __call__(self, model_function: BaseModel.apply_model, args: dict):
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: dict = args["c"]
        cond_or_uncond: List = args["cond_or_uncond"]

        N = x_in.shape[0]
        H, W = x_in.shape[-2], x_in.shape[-1]
        rope_enabled = self._get_rope_patch_sizes(model_function) is not None
        rope_enabled = self._get_rope_patch_sizes(model_function) is not None

        # comfyui can feed in a latent that's a different size cause of SetArea, so we'll refresh in that case.
        self.refresh = False
        if self.weights is None or self.h != H or self.w != W:
            self.h, self.w = H, W
            self.refresh = True
            self.init_grid_bbox(self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size)
            # init everything done, perform sanity check & pre-computations
            self.init_done()
        self.h, self.w = H, W
        # clear buffer canvas
        self.reset_buffer(x_in)

        # Background sampling (grid bbox)
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if processing_interrupted(): 
                    # self.pbar.close()
                    return x_in

                if rope_enabled:
                    if 'control' in c_in:
                        x_tile_batch = torch.cat([x_in[bbox.slicer_for(x_in)] for bbox in bboxes], dim=0)
                        self.process_controlnet(x_tile_batch, c_in, cond_or_uncond, bboxes, N, batch_id)
                    bboxes_by_shape = {}
                    for tile_index, bbox in enumerate(bboxes):
                        x_tile = x_in[bbox.slicer_for(x_in)]
                        t_tile = repeat_to_batch_size(t_in, x_tile.shape[0])
                        c_tile = {}
                        for k, v in c_in.items():
                            if isinstance(v, torch.Tensor):
                                if len(v.shape) == len(x_tile.shape):
                                    bbox_ = bbox
                                    if v.shape[-2:] != x_in.shape[-2:]:
                                        key = v.shape[-2:]
                                        if key not in bboxes_by_shape:
                                            cf = x_in.shape[-1] * self.compression // v.shape[-1] # compression factor
                                            bboxes_by_shape[key] = self.get_grid_bbox(
                                                self.width // cf,
                                                self.height // cf,
                                                self.overlap // cf,
                                                self.tile_batch_size,
                                                v.shape[-1],
                                                v.shape[-2],
                                                x_in.device,
                                                self.get_tile_weights,
                                            )
                                        bbox_ = bboxes_by_shape[key][batch_id][tile_index]
                                    v = v[bbox_.slicer_for(v)]
                                if v.shape[0] != x_tile.shape[0]:
                                    v = repeat_to_batch_size(v, x_tile.shape[0])
                            c_tile[k] = v

                        full_t = x_in.shape[-3] if x_in.dim() == 5 else 1
                        tile_t = x_tile.shape[-3] if x_tile.dim() == 5 else 1
                        transformer_options = self._tile_transformer_options(c_in.get("transformer_options", {}), model_function, bbox, full_t, H, W, tile_t)
                        c_tile["transformer_options"] = transformer_options

                        qwen_ctx = contextlib.nullcontext()
                        if self._is_qwen_model(model_function):
                            qwen_ctx = self._patch_qwen_process_img(model_function, bbox, tile_index, c_in.get("control"))
                        with qwen_ctx:
                            if 'control' in c_in:
                                saved = self._slice_control_for_tile(c_in['control'], tile_index, N)
                                c_tile['control'] = c_in['control'].get_control_orig(
                                    x_tile, t_tile, c_tile, len(cond_or_uncond), transformer_options
                                )
                                self._restore_control(saved)

                            x_tile_out = model_function(x_tile, t_tile, **c_tile)
                        self.x_buffer[bbox.slicer_for(self.x_buffer)] += x_tile_out
                        del x_tile_out, x_tile, t_tile, c_tile
                    continue

                # batching & compute tiles
                x_tile = torch.cat([x_in[bbox.slicer_for(x_in)] for bbox in bboxes], dim=0)   # [TB, C, TH, TW]
                t_tile = repeat_to_batch_size(t_in, x_tile.shape[0])
                c_tile = {}
                for k, v in c_in.items():
                    if isinstance(v, torch.Tensor):
                        if len(v.shape) == len(x_tile.shape):
                            bboxes_ = bboxes
                            if v.shape[-2:] != x_in.shape[-2:]:
                                cf = x_in.shape[-1] * self.compression // v.shape[-1] # compression factor
                                bboxes_ = self.get_grid_bbox(
                                    self.width // cf,
                                    self.height // cf,
                                    self.overlap // cf,
                                    self.tile_batch_size,
                                    v.shape[-1],
                                    v.shape[-2],
                                    x_in.device,
                                    self.get_tile_weights,
                                )
                            v = torch.cat([v[bbox_.slicer_for(v)] for bbox_ in bboxes_[batch_id]])
                        if v.shape[0] != x_tile.shape[0]:
                            v = repeat_to_batch_size(v, x_tile.shape[0])
                    c_tile[k] = v

                # controlnet tiling
                # self.switch_controlnet_tensors(batch_id, N, len(bboxes))
                if 'control' in c_in:
                    self.process_controlnet(x_tile, c_in, cond_or_uncond, bboxes, N, batch_id)
                    c_tile['control'] = c_in['control'].get_control_orig(x_tile, t_tile, c_tile, len(cond_or_uncond), c_tile['transformer_options'])

                # stablesr tiling
                # self.switch_stablesr_tensors(batch_id)

                x_tile_out = model_function(x_tile, t_tile, **c_tile)

                for i, bbox in enumerate(bboxes):
                    self.x_buffer[bbox.slicer_for(self.x_buffer)] += x_tile_out[i*N:(i+1)*N]
                del x_tile_out, x_tile, t_tile, c_tile

                # update progress bar
                # self.update_pbar()

        # Averaging background buffer
        x_out = torch.where(self.weights > 1, self.x_buffer / self.weights, self.x_buffer)

        return x_out

from .utils import store

def fibonacci_spacing(x):
    result = torch.zeros_like(x)
    fib = [0, 1]
    while fib[-1] < len(x):
        fib.append(fib[-1] + fib[-2])
    
    used_indices = set()
    for i, val in enumerate(x):
        fib_index = i % len(fib)
        target_index = fib[fib_index] % len(x)
        while target_index in used_indices:
            target_index = (target_index + 1) % len(x)
        result[target_index] = val
        used_indices.add(target_index)
    
    return result

def find_nearest(a,b):
    # Calculate the absolute differences. 
    diff = (a - b).abs()

    # Find the indices of the nearest elements
    nearest_indices = diff.argmin()

    # Get the nearest elements from b
    return b[nearest_indices]

class SpotDiffusion(AbstractDiffusion):
    
    @torch.inference_mode()
    def __call__(self, model_function: BaseModel.apply_model, args: dict):
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: dict = args["c"]
        cond_or_uncond: List = args["cond_or_uncond"]

        N = x_in.shape[0]
        H, W = x_in.shape[-2], x_in.shape[-1]

        # comfyui can feed in a latent that's a different size cause of SetArea, so we'll refresh in that case.
        self.refresh = False
        if self.weights is None or self.h != H or self.w != W:
            self.h, self.w = H, W
            self.refresh = True
            self.init_grid_bbox(self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size)
            # init everything done, perform sanity check & pre-computations
            self.init_done()
        self.h, self.w = H, W
        # clear buffer canvas
        self.reset_buffer(x_in)

        if self.uniform_distribution is None:
            sigmas = self.sigmas = store.sigmas
            shift_method = store.model_options.get('tiled_diffusion_shift_method', 'random')
            seed = store.model_options.get('tiled_diffusion_seed', store.extra_args.get('seed', 0))
            th = self.tile_height
            tw = self.tile_width
            cf = self.compression
            if 'effnet' in c_in:
                cf = x_in.shape[-1] * self.compression // c_in['effnet'].shape[-1] # compression factor
                th = self.height // cf
                tw = self.width // cf
            shift_height = torch.randint(0, th, (len(sigmas)-1,), generator=torch.Generator(device='cpu').manual_seed(seed), device='cpu')
            shift_height = (shift_height * cf / self.compression).round().to(torch.int32)
            shift_width = torch.randint(0, tw, (len(sigmas)-1,), generator=torch.Generator(device='cpu').manual_seed(seed), device='cpu')
            shift_width = (shift_width * cf / self.compression).round().to(torch.int32)
            if shift_method == "sorted":
                shift_height = shift_height.sort().values
                shift_width = shift_width.sort().values
            elif shift_method == "fibonacci":
                shift_height = fibonacci_spacing(shift_height.sort().values)
                shift_width = fibonacci_spacing(shift_width.sort().values)
            self.uniform_distribution = (shift_height, shift_width)

        sigmas = self.sigmas
        ts_in = find_nearest(t_in[0], sigmas)
        cur_i = ss.item() if (ss:=(sigmas == ts_in).nonzero()).shape[0] != 0 else 0

        sh_h = self.uniform_distribution[0][cur_i].item()
        sh_w = self.uniform_distribution[1][cur_i].item()
        if min(self.tile_height, x_in.shape[-2]) == x_in.shape[-2]:
            sh_h=0
        if min(self.tile_width, x_in.shape[-1]) == x_in.shape[-1]:
            sh_w=0
        condition = cur_i % 2 == 0 if self.tile_height > self.tile_width else cur_i % 2 != 0
        if (sh_h,sh_w) != (0,0):
            # x_in = x_in.roll(shifts=(sh_h,sh_w), dims=(-2,-1))
            if sh_h == 0 or sh_w == 0:
                x_in = x_in.roll(shifts=(sh_h,sh_w), dims=(-2,-1))
            else:
                if condition:
                    x_in = x_in.roll(shifts=sh_h, dims=-2)
                else:
                    x_in = x_in.roll(shifts=sh_w, dims=-1)

        # Background sampling (grid bbox)
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if processing_interrupted(): 
                    # self.pbar.close()
                    return x_in

                if rope_enabled:
                    if 'control' in c_in:
                        x_tile_batch = torch.cat([x_in[bbox.slicer_for(x_in)] for bbox in bboxes], dim=0)
                        self.process_controlnet(x_tile_batch, c_in, cond_or_uncond, bboxes, N, batch_id, (sh_h,sh_w), condition)
                    bboxes_by_shape = {}
                    for tile_index, bbox in enumerate(bboxes):
                        x_tile = x_in[bbox.slicer_for(x_in)]
                        t_tile = repeat_to_batch_size(t_in, x_tile.shape[0])
                        c_tile = {}
                        for k, v in c_in.items():
                            if isinstance(v, torch.Tensor):
                                if len(v.shape) == len(x_tile.shape):
                                    bbox_ = bbox
                                    sh_h_new, sh_w_new = sh_h, sh_w
                                    if v.shape[-2:] != x_in.shape[-2:]:
                                        key = v.shape[-2:]
                                        if key not in bboxes_by_shape:
                                            cf = x_in.shape[-1] * self.compression // v.shape[-1] # compression factor
                                            bboxes_by_shape[key] = (
                                                self.get_grid_bbox(
                                                    self.width // cf,
                                                    self.height // cf,
                                                    self.overlap // cf,
                                                    self.tile_batch_size,
                                                    v.shape[-1],
                                                    v.shape[-2],
                                                    x_in.device,
                                                    self.get_tile_weights,
                                                ),
                                                round(sh_h * self.compression / cf),
                                                round(sh_w * self.compression / cf),
                                            )
                                        bboxes_, sh_h_new, sh_w_new = bboxes_by_shape[key]
                                        bbox_ = bboxes_[batch_id][tile_index]
                                    v = v.roll(shifts=(sh_h_new, sh_w_new), dims=(-2,-1))
                                    v = v[bbox_.slicer_for(v)]
                                if v.shape[0] != x_tile.shape[0]:
                                    v = repeat_to_batch_size(v, x_tile.shape[0])
                            c_tile[k] = v

                        full_t = x_in.shape[-3] if x_in.dim() == 5 else 1
                        tile_t = x_tile.shape[-3] if x_tile.dim() == 5 else 1
                        transformer_options = self._tile_transformer_options(c_in.get("transformer_options", {}), model_function, bbox, full_t, H, W, tile_t)
                        c_tile["transformer_options"] = transformer_options

                        qwen_ctx = contextlib.nullcontext()
                        if self._is_qwen_model(model_function):
                            qwen_ctx = self._patch_qwen_process_img(model_function, bbox, tile_index, c_in.get("control"))
                        with qwen_ctx:
                            if 'control' in c_in:
                                saved = self._slice_control_for_tile(c_in['control'], tile_index, N)
                                c_tile['control'] = c_in['control'].get_control_orig(
                                    x_tile, t_tile, c_tile, len(cond_or_uncond), transformer_options
                                )
                                self._restore_control(saved)

                            x_tile_out = model_function(x_tile, t_tile, **c_tile)
                        self.x_buffer[bbox.slicer_for(self.x_buffer)] = x_tile_out
                        del x_tile_out, x_tile, t_tile, c_tile
                    continue

                # batching & compute tiles
                x_tile = torch.cat([x_in[bbox.slicer_for(x_in)] for bbox in bboxes], dim=0)   # [TB, C, TH, TW]
                t_tile = repeat_to_batch_size(t_in, x_tile.shape[0])
                c_tile = {}
                for k, v in c_in.items():
                    if isinstance(v, torch.Tensor):
                        if len(v.shape) == len(x_tile.shape):
                            bboxes_ = bboxes
                            sh_h_new, sh_w_new = sh_h, sh_w
                            if v.shape[-2:] != x_in.shape[-2:]:
                                cf = x_in.shape[-1] * self.compression // v.shape[-1] # compression factor
                                bboxes_ = self.get_grid_bbox(
                                    self.width // cf,
                                    self.height // cf,
                                    self.overlap // cf,
                                    self.tile_batch_size,
                                    v.shape[-1],
                                    v.shape[-2],
                                    x_in.device,
                                    self.get_tile_weights,
                                )
                                sh_h_new, sh_w_new = round(sh_h * self.compression / cf), round(sh_w * self.compression / cf)
                            v = v.roll(shifts=(sh_h_new, sh_w_new), dims=(-2,-1))
                            v = torch.cat([v[bbox_.slicer_for(v)] for bbox_ in bboxes_[batch_id]])
                        if v.shape[0] != x_tile.shape[0]:
                            v = repeat_to_batch_size(v, x_tile.shape[0])
                    c_tile[k] = v

                # controlnet tiling
                # self.switch_controlnet_tensors(batch_id, N, len(bboxes))
                if 'control' in c_in:
                    self.process_controlnet(x_tile, c_in, cond_or_uncond, bboxes, N, batch_id, (sh_h,sh_w), condition)
                    c_tile['control'] = c_in['control'].get_control_orig(x_tile, t_tile, c_tile, len(cond_or_uncond), c_tile['transformer_options'])

                # stablesr tiling
                # self.switch_stablesr_tensors(batch_id)

                x_tile_out = model_function(x_tile, t_tile, **c_tile)

                for i, bbox in enumerate(bboxes):
                    self.x_buffer[bbox.slicer_for(self.x_buffer)] = x_tile_out[i*N:(i+1)*N]

                del x_tile_out, x_tile, t_tile, c_tile

                # update progress bar
                # self.update_pbar()

        if (sh_h,sh_w) != (0,0):
            # self.x_buffer = self.x_buffer.roll(shifts=(-sh_h, -sh_w), dims=(-2, -1))
            if sh_h == 0 or sh_w == 0:
                self.x_buffer = self.x_buffer.roll(shifts=(-sh_h, -sh_w), dims=(-2, -1))
            else:
                if condition:
                    self.x_buffer = self.x_buffer.roll(shifts=-sh_h, dims=-2)
                else:
                    self.x_buffer = self.x_buffer.roll(shifts=-sh_w, dims=-1)

        return self.x_buffer

class MixtureOfDiffusers(AbstractDiffusion):
    """
        Mixture-of-Diffusers Implementation
        https://github.com/albarji/mixture-of-diffusers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # weights for custom bboxes
        self.custom_weights: List[Tensor] = []
        self.get_weight = gaussian_weights

    def init_done(self):
        super().init_done()
        # The original gaussian weights can be extremely small, so we rescale them for numerical stability
        self.rescale_factor = 1 / self.weights
        # Meanwhile, we rescale the custom weights in advance to save time of slicing
        for bbox_id, bbox in enumerate(self.custom_bboxes):
            if bbox.blend_mode == BlendMode.BACKGROUND:
                self.custom_weights[bbox_id] *= self.rescale_factor[bbox.slicer]

    @grid_bbox
    def get_tile_weights(self) -> Tensor:
        # weights for grid bboxes
        # if not hasattr(self, 'tile_weights'):
        # x_in can change sizes cause of ConditioningSetArea, so we have to recalcualte each time
        self.tile_weights = self.get_weight(self.tile_w, self.tile_h)
        return self.tile_weights

    @torch.inference_mode()
    def __call__(self, model_function: BaseModel.apply_model, args: dict):
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: dict = args["c"]
        cond_or_uncond: List= args["cond_or_uncond"]

        N = x_in.shape[0]
        H, W = x_in.shape[-2], x_in.shape[-1]
        rope_enabled = self._get_rope_patch_sizes(model_function) is not None

        self.refresh = False
        # self.refresh = True
        if self.weights is None or self.h != H or self.w != W:
            self.h, self.w = H, W
            self.refresh = True
            self.init_grid_bbox(self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size)
            # init everything done, perform sanity check & pre-computations
            self.init_done()
        self.h, self.w = H, W
        # clear buffer canvas
        self.reset_buffer(x_in)

        # self.pbar = tqdm(total=(self.total_bboxes) * sampling_steps, desc=f"{self.method} Sampling: ")
        # self.pbar = tqdm(total=len(self.batched_bboxes), desc=f"{self.method} Sampling: ")

        # Global sampling
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):     # batch_id is the `Latent tile batch size`
                if processing_interrupted(): 
                    # self.pbar.close()
                    return x_in
                if rope_enabled:
                    if 'control' in c_in:
                        x_tile_batch = torch.cat([x_in[bbox.slicer_for(x_in)] for bbox in bboxes], dim=0)
                        self.process_controlnet(x_tile_batch, c_in, cond_or_uncond, bboxes, N, batch_id)
                    bboxes_by_shape = {}
                    for tile_index, bbox in enumerate(bboxes):
                        x_tile = x_in[bbox.slicer_for(x_in)]
                        t_tile = repeat_to_batch_size(t_in, x_tile.shape[0])
                        c_tile = {}
                        for k, v in c_in.items():
                            if isinstance(v, torch.Tensor):
                                if len(v.shape) == len(x_tile.shape):
                                    bbox_ = bbox
                                    if v.shape[-2:] != x_in.shape[-2:]:
                                        key = v.shape[-2:]
                                        if key not in bboxes_by_shape:
                                            cf = x_in.shape[-1] * self.compression // v.shape[-1] # compression factor
                                            bboxes_by_shape[key] = self.get_grid_bbox(
                                                (tile_w := self.width // cf),
                                                (tile_h := self.height // cf),
                                                self.overlap // cf,
                                                self.tile_batch_size,
                                                v.shape[-1],
                                                v.shape[-2],
                                                x_in.device,
                                                lambda: self.get_weight(tile_w, tile_h),
                                            )
                                        bbox_ = bboxes_by_shape[key][batch_id][tile_index]
                                    v = v[bbox_.slicer_for(v)]
                                if v.shape[0] != x_tile.shape[0]:
                                    v = repeat_to_batch_size(v, x_tile.shape[0])
                            c_tile[k] = v

                        full_t = x_in.shape[-3] if x_in.dim() == 5 else 1
                        tile_t = x_tile.shape[-3] if x_tile.dim() == 5 else 1
                        transformer_options = self._tile_transformer_options(c_in.get("transformer_options", {}), model_function, bbox, full_t, H, W, tile_t)
                        c_tile["transformer_options"] = transformer_options

                        qwen_ctx = contextlib.nullcontext()
                        if self._is_qwen_model(model_function):
                            qwen_ctx = self._patch_qwen_process_img(model_function, bbox, tile_index, c_in.get("control"))
                        with qwen_ctx:
                            if 'control' in c_in:
                                saved = self._slice_control_for_tile(c_in['control'], tile_index, N)
                                c_tile['control'] = c_in['control'].get_control_orig(
                                    x_tile, t_tile, c_tile, len(cond_or_uncond), transformer_options
                                )
                                self._restore_control(saved)

                            x_tile_out = model_function(x_tile, t_tile, **c_tile)
                        w = self.tile_weights * self.rescale_factor[bbox.slicer]
                        self.x_buffer[bbox.slicer_for(self.x_buffer)] += x_tile_out * w
                        del x_tile_out, x_tile, t_tile, c_tile
                    continue

                # batching
                x_tile_list     = []
                for bbox in bboxes:
                    x_tile_list.append(x_in[bbox.slicer_for(x_in)])

                x_tile = torch.cat(x_tile_list, dim=0)                     # differs each
                t_tile = repeat_to_batch_size(t_in, x_tile.shape[0])   # just repeat
                c_tile = {}
                for k, v in c_in.items():
                    if isinstance(v, torch.Tensor):
                        if len(v.shape) == len(x_tile.shape):
                            bboxes_ = bboxes
                            if v.shape[-2:] != x_in.shape[-2:]:
                                cf = x_in.shape[-1] * self.compression // v.shape[-1] # compression factor
                                bboxes_ = self.get_grid_bbox(
                                    (tile_w := self.width // cf),
                                    (tile_h := self.height // cf),
                                    self.overlap // cf,
                                    self.tile_batch_size,
                                    v.shape[-1],
                                    v.shape[-2],
                                    x_in.device,
                                    lambda: self.get_weight(tile_w, tile_h),
                                )
                            v = torch.cat([v[bbox_.slicer_for(v)] for bbox_ in bboxes_[batch_id]])
                        if v.shape[0] != x_tile.shape[0]:
                            v = repeat_to_batch_size(v, x_tile.shape[0])
                    c_tile[k] = v
                
                # controlnet
                # self.switch_controlnet_tensors(batch_id, N, len(bboxes), is_denoise=True)
                if 'control' in c_in:
                    self.process_controlnet(x_tile, c_in, cond_or_uncond, bboxes, N, batch_id)
                    c_tile['control'] = c_in['control'].get_control_orig(x_tile, t_tile, c_tile, len(cond_or_uncond), c_tile['transformer_options'])
                
                # stablesr
                # self.switch_stablesr_tensors(batch_id)

                # denoising: here the x is the noise
                x_tile_out = model_function(x_tile, t_tile, **c_tile)

                # de-batching
                for i, bbox in enumerate(bboxes):
                    # These weights can be calcluated in advance, but will cost a lot of vram 
                    # when you have many tiles. So we calculate it here.
                    w = self.tile_weights * self.rescale_factor[bbox.slicer]
                    self.x_buffer[bbox.slicer_for(self.x_buffer)] += x_tile_out[i*N:(i+1)*N] * w
                del x_tile_out, x_tile, t_tile, c_tile

                # self.update_pbar()
                # self.pbar.update()
        # self.pbar.close()
        x_out = self.x_buffer

        return x_out

MAX_RESOLUTION=8192
class TiledDiffusion():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                                "method": (["MultiDiffusion", "Mixture of Diffusers", "SpotDiffusion"], {"default": "Mixture of Diffusers"}),
                                # "tile_width": ("INT", {"default": 96, "min": 16, "max": 256, "step": 16}),
                                "tile_width": ("INT", {"default": 96*opt_f, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                                # "tile_height": ("INT", {"default": 96, "min": 16, "max": 256, "step": 16}),
                                "tile_height": ("INT", {"default": 96*opt_f, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                                "tile_overlap": ("INT", {"default": 8*opt_f, "min": 0, "max": 256*opt_f, "step": 4*opt_f}),
                                "tile_batch_size": ("INT", {"default": 4, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "_for_testing"
    instances = WeakSet()

    @classmethod
    def IS_CHANGED(s, *args, **kwargs):
        for o in s.instances:
            o.impl.reset()
        return ""
    
    def __init__(self) -> None:
        self.__class__.instances.add(self)

    def apply(self, model: ModelPatcher, method, tile_width, tile_height, tile_overlap, tile_batch_size):
        if method == "Mixture of Diffusers":
            self.impl = MixtureOfDiffusers()
        elif method == "MultiDiffusion":
            self.impl = MultiDiffusion()
        else:
            self.impl = SpotDiffusion()
        
        # if noise_inversion:
        #     get_cache_callback = self.noise_inverse_get_cache
        #     set_cache_callback = None # lambda x0, xt, prompts: self.noise_inverse_set_cache(p, x0, xt, prompts, steps, retouch)
        #     self.impl.init_noise_inverse(steps, retouch, get_cache_callback, set_cache_callback, renoise_strength, renoise_kernel_size)

        compression = 4 if "CASCADE" in str(model.model.model_type) else 8
        self.impl.tile_width = tile_width // compression
        self.impl.tile_height = tile_height // compression
        self.impl.tile_overlap = tile_overlap // compression
        self.impl.tile_batch_size = tile_batch_size
        
        self.impl.compression = compression
        self.impl.width = tile_width
        self.impl.height  = tile_height
        self.impl.overlap = tile_overlap

        # self.impl.init_grid_bbox(tile_width, tile_height, tile_overlap, tile_batch_size)
        # # init everything done, perform sanity check & pre-computations
        # self.impl.init_done()
        # hijack the behaviours
        # self.impl.hook()
        model = model.clone()
        model.set_model_unet_function_wrapper(self.impl)
        model.model_options['tiled_diffusion'] = True
        return (model,)

class SpotDiffusionParams():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                                "shift_method": (["random", "sorted", "fibonacci"], {"default": "random", "tooltip": "Samples a shift size over a uniform distribution to shift tiles."}),
                                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "_for_testing"

    def apply(self, model: ModelPatcher, shift_method, seed):
        model = model.clone()
        model.model_options['tiled_diffusion_seed'] = seed
        model.model_options['tiled_diffusion_shift_method'] = shift_method
        return (model,)

class NoiseInversion():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                                "positive": ("CONDITIONING", ),
                                "negative": ("CONDITIONING", ),
                                "latent_image": ("LATENT", ),
                                "image": ("IMAGE", ),
                                "steps": ("INT", {"default": 10, "min": 1, "max": 208, "step": 1}),
                                "retouch": ("FLOAT", {"default": 1, "min": 1, "max": 100, "step": 0.1}),
                                "renoise_strength": ("FLOAT", {"default": 1, "min": 1, "max": 2, "step": 0.01}),
                                "renoise_kernel_size": ("INT", {"default": 2, "min": 2, "max": 512, "step": 1}),
                            }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    def sample(self, model: ModelPatcher, positive, negative,
                    latent_image, image, steps, retouch, renoise_strength, renoise_kernel_size):
        return (latent_image,)

NODE_CLASS_MAPPINGS = {
    "TiledDiffusion": TiledDiffusion,
    "SpotDiffusionParams_TiledDiffusion": SpotDiffusionParams,
    # "NoiseInversion": NoiseInversion,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledDiffusion": "Tiled Diffusion",
    "SpotDiffusionParams_TiledDiffusion": "SpotDiffusion Parameters",
    # "NoiseInversion": "Noise Inversion",
}
