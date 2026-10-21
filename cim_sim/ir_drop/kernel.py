# -*- coding: utf-8 -*-
"""
C/OpenMP 内核的编译与加载。

首次调用时用 gcc 把同目录下的 crossbar_ir.c 编译为共享库，按源码哈希缓存在
$TMPDIR/cim_sim_kernel/（或环境变量 CIM_SIM_KERNEL_DIR 指定的目录）。
编译器可用环境变量 CIM_SIM_CC 覆盖（默认 gcc）。
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_SRC = Path(__file__).with_name("crossbar_ir.c")
_LIB = None


def load_kernel() -> ctypes.CDLL:
    """返回已加载的内核库（含 ir_factor / ir_sor_iter 两个函数）。"""
    global _LIB
    if _LIB is not None:
        return _LIB
    src_text = _SRC.read_text()
    tag = hashlib.sha1(src_text.encode()).hexdigest()[:12]
    d = Path(os.environ.get("CIM_SIM_KERNEL_DIR", Path(tempfile.gettempdir()) / "cim_sim_kernel"))
    d.mkdir(parents=True, exist_ok=True)
    so = d / f"crossbar_ir_{tag}.so"
    if not so.exists():
        cc = os.environ.get("CIM_SIM_CC", "gcc")
        tmp = d / f"crossbar_ir_{tag}.{os.getpid()}.so"
        subprocess.check_call([cc, "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC",
                               str(_SRC), "-o", str(tmp)])
        tmp.replace(so)          # 原子替换，多进程并行时安全
    lib = ctypes.CDLL(str(so))
    P = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    L, D = ctypes.c_long, ctypes.c_double
    lib.ir_factor.argtypes = [P, L, D, P, P]
    lib.ir_factor.restype = None
    lib.ir_sor_iter.argtypes = [P, P, P, P, P, P, P, L, L, D, D, P]
    lib.ir_sor_iter.restype = None
    _LIB = lib
    return lib


# ---------------------------------------------------------------------------
# Rectangular-array kernel (added for physical tiling; the square kernel above is unchanged)
# ---------------------------------------------------------------------------
_RECT_SRC = Path(__file__).with_name("crossbar_ir_rect.c")
_RECT_LIB = None


def load_rect_kernel() -> ctypes.CDLL:
    """
    Return the loaded rectangular-array kernel (ir_factor_rect / ir_sor_iter_rect).

    Compiled on first use with the same compiler flags and cache directory as load_kernel().
    """
    global _RECT_LIB
    if _RECT_LIB is not None:
        return _RECT_LIB
    tag = hashlib.sha1(_RECT_SRC.read_text().encode()).hexdigest()[:12]
    d = Path(os.environ.get("CIM_SIM_KERNEL_DIR", Path(tempfile.gettempdir()) / "cim_sim_kernel"))
    d.mkdir(parents=True, exist_ok=True)
    so = d / f"crossbar_ir_rect_{tag}.so"
    if not so.exists():
        cc = os.environ.get("CIM_SIM_CC", "gcc")
        tmp = d / f"crossbar_ir_rect_{tag}.{os.getpid()}.so"
        subprocess.check_call([cc, "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC",
                               str(_RECT_SRC), "-o", str(tmp)])
        tmp.replace(so)          # atomic replace; safe when several processes compile at once
    lib = ctypes.CDLL(str(so))
    P = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    L, D = ctypes.c_long, ctypes.c_double
    lib.ir_factor_rect.argtypes = [P, L, L, D, P, P]
    lib.ir_factor_rect.restype = None
    lib.ir_sor_iter_rect.argtypes = [P, P, P, P, P, P, P, L, L, L, D, D, P]
    lib.ir_sor_iter_rect.restype = None
    _RECT_LIB = lib
    return lib
