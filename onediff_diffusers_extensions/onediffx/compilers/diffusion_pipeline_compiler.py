import os
import torch
from onediff.infer_compiler.deployable_module import DeployableModule
from onediff.infer_compiler.utils.log_utils import logger
from .model_compiler import compile_model_part, _recursive_getattr
from multiprocessing import Pool
# set_start_method('spawn')


def _recursive_setattr(obj, attr, value):
    attrs = attr.split(".")
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)


_PARTS = [
    "text_encoder",
    "text_encoder_2",
    "image_encoder",
    "unet",
    "controlnet",
    "fast_unet",  # for deepcache
    "prior",  # for StableCascadePriorPipeline
    "decoder",  # for StableCascadeDecoderPipeline
    "vqgan.down_blocks",  # for StableCascadeDecoderPipeline
    "vqgan.up_blocks",  # for StableCascadeDecoderPipeline
    "vae.decoder",
    "vae.encoder",
]


def _filter_parts(ignores=()):
    filtered_parts = []
    for part in _PARTS:
        skip = False
        for ignore in ignores:
            if part == ignore or part.startswith(ignore + "."):
                skip = True
                break
        if not skip:
            filtered_parts.append(part)

    return filtered_parts


def compile_pipe(
    pipe, *, ignores=(),
):
    if (
        hasattr(pipe, "upcast_vae")
        and pipe.vae.dtype == torch.float16
        and pipe.vae.config.force_upcast
    ):
        pipe.upcast_vae()

    filtered_parts = _filter_parts(ignores=ignores)
    parts_to_compile = []

    for part in filtered_parts:
        obj = _recursive_getattr(pipe, part, None)
        if obj is not None:
            logger.info(f"Compiling {part}")
            obj.share_memory()
            parts_to_compile.append((part, obj))

    # 使用 Pool 来并行编译模型的不同部分
    with Pool(processes=5) as pool:
        compiled_parts = pool.map(compile_model_part, parts_to_compile)

    # 更新模型的编译部分
    for part, compiled_obj in compiled_parts:
        if compiled_obj is not None:
            _recursive_setattr(pipe, part, compiled_obj)

    if hasattr(pipe, "image_processor") and "image_processor" not in ignores:
        logger.info("Patching image_processor")

        from onediffx.utils.patch_image_processor import (
            patch_image_prcessor as patch_image_prcessor_,
        )

        patch_image_prcessor_(pipe.image_processor)

    return pipe


def save_pipe(pipe, dir="cached_pipe", *, ignores=(), overwrite=True):
    if not os.path.exists(dir):
        os.makedirs(dir)
    filtered_parts = _filter_parts(ignores=ignores)
    for part in filtered_parts:
        obj = _recursive_getattr(pipe, part, None)
        if (
            obj is not None
            and isinstance(obj, DeployableModule)
            and obj._deployable_module_dpl_graph is not None
            and obj.get_graph().is_compiled
        ):
            if not overwrite and os.path.isfile(os.path.join(dir, part)):
                logger.info(
                    f"Compiled graph already exists for {part}, not overwriting it."
                )
                continue
            logger.info(f"Saving {part}")
            obj.save_graph(os.path.join(dir, part))


def load_pipe(
    pipe, dir="cached_pipe", *, ignores=(),
):
    if not os.path.exists(dir):
        return
    filtered_parts = _filter_parts(ignores=ignores)
    for part in filtered_parts:
        obj = _recursive_getattr(pipe, part, None)
        if obj is not None and os.path.exists(os.path.join(dir, part)):
            logger.info(f"Loading {part}")
            obj.load_graph(os.path.join(dir, part))

    if "image_processor" not in ignores:
        logger.info("Patching image_processor")

        from onediffx.utils.patch_image_processor import (
            patch_image_prcessor as patch_image_prcessor_,
        )

        patch_image_prcessor_(pipe.image_processor)
