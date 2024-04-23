

from loguru import logger

from onediff.infer_compiler import oneflow_compile


def _recursive_getattr(obj, attr, default=None):
    attrs = attr.split(".")
    for attr in attrs:
        if not hasattr(obj, attr):
            return default
        obj = getattr(obj, attr, default)
    return obj


def compile_part(obj):
    # 在这里调用实际的编译函数
    compiled_obj = oneflow_compile(obj)
    return compiled_obj

# 编译模型的进程函数


def compile_model_part(args):
    part, obj = args
    logger.debug(f"Compiling:{part}")
    if obj is not None:
        compiled_obj = compile_part(obj)
        return part, compiled_obj
    return part, None
