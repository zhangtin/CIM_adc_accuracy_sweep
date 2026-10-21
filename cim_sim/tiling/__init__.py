# -*- coding: utf-8 -*-
"""
Physical array tiling: a logical weight matrix executed on a grid of independent physical crossbars.

    config     TilingConfig, TileADCParams, reduction modes, edge policies
    planner    TileSpec, TilingPlan, plan_tiling            (pure geometry)
    mapping    weight slicing / padding, input routing      (integer levels and codes)
    streams    order-independent per-tile random streams
    reduction  digital partial-sum reduction (analog / hierarchical reserved)
    executor   tile backends and simulate_tiled_crossbar()  (reuses the physical modules)
"""
from .config import (EDGE_POLICIES, IMPLEMENTED_REDUCTION_MODES, REDUCTION_MODES, TileADCParams,
                     TilingConfig)
from .executor import (CPUTileExecutor, TileContext, TiledResult, TileExecutor, TileJob, TileResult,
                       build_jobs, simulate_tiled_crossbar, tile_adc_window, tile_wire_resistance)
from .mapping import output_columns, tile_inputs, tile_weights, validate_operands
from .planner import TileSpec, TilingPlan, plan_tiling
from .reduction import VIEWS, DigitalReducer, make_reducer
from .streams import SequentialStreams, TileStreamProvider, TileStreams, workload_rng

__all__ = ["EDGE_POLICIES", "IMPLEMENTED_REDUCTION_MODES", "REDUCTION_MODES", "TileADCParams", "TilingConfig",
           "CPUTileExecutor", "TileContext", "TiledResult", "TileExecutor", "TileJob", "TileResult", "build_jobs",
           "simulate_tiled_crossbar", "tile_adc_window", "tile_wire_resistance", "output_columns", "tile_inputs",
           "tile_weights", "validate_operands", "TileSpec", "TilingPlan", "plan_tiling", "VIEWS", "DigitalReducer",
           "make_reducer", "SequentialStreams", "TileStreamProvider", "TileStreams", "workload_rng"]
