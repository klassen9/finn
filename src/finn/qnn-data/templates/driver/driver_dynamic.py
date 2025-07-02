import subprocess
import numpy as np
import time
from pynq.ps import Clocks
import numpy as np
import math
import pickle
import os
import h5py
import random
from matplotlib import pyplot as plt

# CovnInpGen

def compute_conv_output_dim(ifm_dim, k, stride, total_pad=0, dilation=1):
    out_dim = int(((ifm_dim + total_pad - dilation * (k - 1) - 1) / stride) + 1)
    return out_dim

def get_buffer_depth(ifm_ch=2, k=(3,1), ifm_dim=(1026,1), stride=(1,1), dilation=(1,1), simd=2):
    k_h, k_w = k
    h, w = ifm_dim
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    mmv_in = 1
    mmv_out = 1
    channel_factor = int(ifm_ch / simd)
    impl_style = "default"
    if impl_style == "default":
        buffer_min_size = (
            (k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w + 1
        ) * channel_factor
        # add additional buffer space in case of stride > 1
        # this minimizes cycle count as it allows an earlier pre-load of inputs
        buffer_depth = (
            buffer_min_size
            + max(
                0,
                ((stride_w - 1) - (int(mmv_out * k_h * k_w / mmv_in))) * channel_factor,
            )
            + max(
                0,
                ((stride_h - 1) * w - (int(mmv_out * k_h * k_w / mmv_in))) * channel_factor,
            )
        )
    elif impl_style == "parallel":
        buffer_min_size = (
            (k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w
        ) * channel_factor + 1
        buffer_depth = buffer_min_size + 1
    return buffer_depth

def prepare_codegen_default(ifm_ch=2, k=(3,1), ifm_dim=(1026,1), stride=(1,1), dilation=(1,1), depthwise=0, simd=2):
    code_gen_dict = {}

    k_h, k_w = k
    h, w = ifm_dim
    pad = [0, 0, 0, 0]  # padding happens in separate padding node for now
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    pad_h = pad[0] + pad[2]
    pad_w = pad[1] + pad[3]
    out_dim_h = compute_conv_output_dim(h, k_h, stride_h, pad_h, dilation_h)
    out_dim_w = compute_conv_output_dim(w, k_w, stride_w, pad_w, dilation_w)
    mmv_in = 1
    mmv_out = 1
    channel_factor = int(ifm_ch / simd)

    # compute minimal buffer length (assuming it holds 1 complete window)
    buffer_min_size = ((k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w + 1) * channel_factor

    buffer_actual_size = get_buffer_depth()
    code_gen_dict["$BUF_ELEM_TOTAL$"] = [str(buffer_actual_size)]

    # compute some intermediate values, e.g., kernel "width" = k_w incl. dilation
    # or cols/rows that are skipped due to imperfect stride<->dim combination
    kernel_width = (k_w - 1) * dilation_w + 1
    kernel_height = (k_h - 1) * dilation_h + 1
    skip_columns = w % (kernel_width + (out_dim_w - 1) * stride_w)
    skip_rows = h % (kernel_height + (out_dim_h - 1) * stride_h)

    # compute address increment values for 5-loop nest
    addr_incr_end_simd = 1
    addr_incr_end_window_elem = (dilation_w - 1) * channel_factor + 1
    addr_incr_end_window_row = (
        ((w - kernel_width) * channel_factor)  # remaining line
        + ((dilation_h - 1) * w * channel_factor)  # skip lines
        + 1  # wrap-around of minimally sized buffer
    )
    addr_incr_end_window = -buffer_min_size + stride_w * channel_factor + 1
    addr_incr_end_row = (
        -buffer_min_size
        + ((skip_columns + kernel_width) * channel_factor)  # remaining line
        + ((stride_h - 1) * w * channel_factor)  # skip lines
        + 1
    )

    # re-use same controller structure -> re-assign address increments
    if depthwise:
        addr_incr_end_window_elem = dilation_w * channel_factor
        addr_incr_end_window_row = (
            channel_factor
            + (w - kernel_width) * channel_factor
            + (dilation_h - 1) * w * channel_factor
        )
        addr_incr_end_simd = -buffer_min_size + (channel_factor + 1)

    # sanity check for wrap logic
    assert not (
        abs(addr_incr_end_window) > buffer_actual_size
    ), "ERROR: W increment > buffer size, try setting parallel_window=1"
    assert not (
        abs(addr_incr_end_row) > buffer_actual_size
    ), "ERROR: H increment > buffer size, try setting parallel_window=1"

    # set certain threshold indices to detect when reading/writing finishes
    code_gen_dict["$LAST_READ_ELEM$"] = [str(h * w * channel_factor - 1)]
    code_gen_dict["$LAST_WRITE_ELEM$"] = [
        str(((h - skip_rows - 1) * w + (w - skip_columns)) * channel_factor - 1)
    ]

    # default controller loop structure: # iterations (counters) map directly
    loop_h_iterations = out_dim_h
    loop_w_iterations = out_dim_w
    loop_kh_iterations = k_h
    loop_kw_iterations = k_w
    loop_simd_iterations = channel_factor

    if depthwise and channel_factor > 1:
        # re-arrange existing controller loop structure for depthwise convolutions
        loop_kh_iterations = channel_factor
        loop_kw_iterations = k_h
        loop_simd_iterations = k_w
        addr_incr_end_simd_ = addr_incr_end_simd
        addr_incr_end_simd = addr_incr_end_window_elem
        addr_incr_end_window_elem = addr_incr_end_window_row
        addr_incr_end_window_row = addr_incr_end_simd_
        elem_per_window = k_h * k_w

        tail_incr_w = addr_incr_end_window + buffer_min_size - channel_factor
        tail_incr_h = addr_incr_end_row + buffer_min_size - channel_factor
        tail_incr_last_window = buffer_min_size - 1
        code_gen_dict["$IS_DEPTHWISE$"] = ["1"]
    else:
        # depthwise output format is equivalent to non-depthwise if SIMD=C
        elem_per_window = k_h * k_w * channel_factor

        tail_incr_w = addr_incr_end_window + buffer_min_size - 1
        tail_incr_h = addr_incr_end_row + buffer_min_size - 1
        tail_incr_last_window = buffer_min_size - 1
        code_gen_dict["$IS_DEPTHWISE$"] = ["0"]

    # support SIMD = IFMChannels and k_w = 1 cases
    # for k = [k_h, k_w] = [1, k_w], no adjustment is needed
    # for k = [k_h, k_w] = [1, 1], do not use this impl. style (mmv_out=K=1)
    # innermost loop is executed at least once -> adjust if needed
    if loop_simd_iterations == 1:
        # skip innermost SIMD loop completely
        if loop_kw_iterations == 1:
            # skip innermost KW loop completely
            code_gen_dict["$INNERMOST_STATE$"] = ["STATE_LOOP_KH"]
            loop_kh_iterations -= 1  # -1 because state is initial state
        else:
            code_gen_dict["$INNERMOST_STATE$"] = ["STATE_LOOP_KW"]
            loop_kw_iterations -= 1  # -1 because state is initial state
    else:
        code_gen_dict["$INNERMOST_STATE$"] = ["STATE_LOOP_SIMD"]
        loop_simd_iterations -= 1  # -1 because state is initial state

    cntr_bitwidth = math.ceil(
        math.log2(
            max(
                loop_h_iterations - 2 + 1,
                loop_w_iterations - 2 + 1,
                loop_kh_iterations - 2 + 1,
                loop_kw_iterations - 2 + 1,
                loop_simd_iterations - 2 + 1,
            )
        )
    )
    code_gen_dict["$CNTR_BITWIDTH$"] = [str(cntr_bitwidth)]
    code_gen_dict["$LOOP_H_ITERATIONS$"] = [str(loop_h_iterations - 2)]
    code_gen_dict["$LOOP_W_ITERATIONS$"] = [str(loop_w_iterations - 2)]
    code_gen_dict["$LOOP_KH_ITERATIONS$"] = [str(loop_kh_iterations - 2)]
    code_gen_dict["$LOOP_KW_ITERATIONS$"] = [str(loop_kw_iterations - 2)]
    code_gen_dict["$LOOP_SIMD_ITERATIONS$"] = [str(loop_simd_iterations - 2)]

    incr_bitwidth = 1 + math.ceil(
        math.log2(
            max(
                abs(addr_incr_end_simd) + 1,
                abs(addr_incr_end_window_elem) + 1,
                abs(addr_incr_end_window_row) + 1,
                abs(addr_incr_end_window) + 1,
                abs(addr_incr_end_row) + 1,
                abs(tail_incr_w) + 1,
                abs(tail_incr_h) + 1,
                abs(tail_incr_last_window) + 1,
            )
        )
    )
    code_gen_dict["$INCR_BITWIDTH$"] = [str(incr_bitwidth)]
    code_gen_dict["$HEAD_INCR_SIMD$"] = [str(addr_incr_end_simd)]
    code_gen_dict["$HEAD_INCR_KW$"] = [str(addr_incr_end_window_elem)]
    code_gen_dict["$HEAD_INCR_KH$"] = [str(addr_incr_end_window_row)]
    code_gen_dict["$HEAD_INCR_W$"] = [str(addr_incr_end_window)]
    code_gen_dict["$HEAD_INCR_H$"] = [str(addr_incr_end_row)]
    code_gen_dict["$TAIL_INCR_W$"] = [str(tail_incr_w)]
    code_gen_dict["$TAIL_INCR_H$"] = [str(tail_incr_h)]
    code_gen_dict["$TAIL_INCR_LAST$"] = [str(tail_incr_last_window)]

    code_gen_dict["$ELEM_PER_WINDOW$"] = [str(elem_per_window)]
    code_gen_dict["$SIMD$"] = [str(simd)]
    code_gen_dict["$MMV_IN$"] = [str(mmv_in)]
    code_gen_dict["$MMV_OUT$"] = [str(mmv_out)]

    return code_gen_dict

def convInpGenConfig(ifm_dim, stride=(1,1), dilation=(1,1), k=(3,1), ifm_ch=2, depthwise=0, simd=2):
    k_h, k_w = k
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    ifm_dim_h, ifm_dim_w = ifm_dim
    ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, 0, dilation_h)
    ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, 0, dilation_w)
    ofm_dim = [ofm_dim_h, ofm_dim_w]
    
    code_gen_dict = prepare_codegen_default(ifm_ch=ifm_ch, k=k, ifm_dim=ifm_dim, stride=stride, dilation=dilation, depthwise=depthwise, simd=simd)

    
    config = {
        #"cfg_wren": (0 * 4, 1),
        "cfg_cntr_simd": (1 * 4, int(code_gen_dict["$LOOP_SIMD_ITERATIONS$"][0])),
        "cfg_cntr_kw": (2 * 4, int(code_gen_dict["$LOOP_KW_ITERATIONS$"][0])),
        "cfg_cntr_kh": (3 * 4, int(code_gen_dict["$LOOP_KH_ITERATIONS$"][0])),
        "cfg_cntr_w": (4 * 4, int(code_gen_dict["$LOOP_W_ITERATIONS$"][0])),
        "cfg_cntr_h": (5 * 4, int(code_gen_dict["$LOOP_H_ITERATIONS$"][0])),
        "cfg_incr_head_simd": (6 * 4, int(code_gen_dict["$HEAD_INCR_SIMD$"][0])),
        "cfg_incr_head_kw": (7 * 4, int(code_gen_dict["$HEAD_INCR_KW$"][0])),
        "cfg_incr_head_kh": (8 * 4, int(code_gen_dict["$HEAD_INCR_KH$"][0])),
        "cfg_incr_head_w": (9 * 4, int(code_gen_dict["$HEAD_INCR_W$"][0])),
        "cfg_incr_head_h": (10 * 4, int(code_gen_dict["$HEAD_INCR_H$"][0])),
        "cfg_incr_tail_w": (11 * 4, int(code_gen_dict["$TAIL_INCR_W$"][0])),
        "cfg_incr_tail_h": (12 * 4, int(code_gen_dict["$TAIL_INCR_H$"][0])),
        "cfg_incr_tail_last": (13 * 4, int(code_gen_dict["$TAIL_INCR_LAST$"][0])),
        "cfg_last_read": (14 * 4, int(code_gen_dict["$LAST_READ_ELEM$"][0])),
        "cfg_last_write": (15 * 4, int(code_gen_dict["$LAST_WRITE_ELEM$"][0])),
    }
    
    return config

# FMPAD

def roundup_to_integer_multiple(x, factor):
    """Round up integer x to the nearest integer multiple of integer factor.
    Returns x if factor is set to -1. Both x and factor must otherwise be
    positive."""
    # ensure integers
    assert int(x) == x, "The input x is not an integer."
    assert int(factor) == factor, "The input factor is not an integer."
    # use -1 to indicate no padding needed
    if factor == -1:
        return x
    # ensure positive values
    assert factor > 0 and x > 0, "Factor and x are <= 0."
    if x < factor:
        return factor
    else:
        if x % factor == 0:
            return x
        else:
            return x + (factor - (x % factor))

def get_template_values(ifm_dims, pads, chans, simd, bitwidth):
    dimY, dimX = ifm_dims
    padT, padL, padB, padR = pads
    y_counter_bits = int(math.ceil(math.log2(padT + dimY + padB + 1)))
    x_counter_bits = int(math.ceil(math.log2(padL + dimX + padR + 1)))
    stream_bits = bitwidth * simd
    stream_bits = int(roundup_to_integer_multiple(stream_bits, 8))
    code_gen_dict = {
        "XCOUNTER_BITS": int(x_counter_bits),
        "YCOUNTER_BITS": int(y_counter_bits),
        "NUM_CHANNELS": int(chans),
        "SIMD": int(simd),
        "ELEM_BITS": bitwidth,
        "INIT_XON": int(padL),
        "INIT_XOFF": int(padL + dimX),
        "INIT_XEND": int(padL + dimX + padR - 1),
        "INIT_YON": int(padT),
        "INIT_YOFF": int(padT + dimY),
        "INIT_YEND": int(padT + dimY + padB - 1),
        "STREAM_BITS": int(stream_bits),
    }
    return code_gen_dict

def fmPadConfig(ifm_dims, pads=(1,0,1,0), chans=2, simd=2, bitwidth=8):
        """Returns a configuration dict to re-configure FM dimension and
        padding amounts during runtime."""

        code_gen_dict = get_template_values(ifm_dims, pads, chans, simd, bitwidth)
        config = {
            "XON": (0 * 4, (code_gen_dict["INIT_XON"])),
            "XOFF": (1 * 4, (code_gen_dict["INIT_XOFF"])),
            "XEND": (2 * 4, (code_gen_dict["INIT_XEND"])),
            "YON": (3 * 4, (code_gen_dict["INIT_YON"])),
            "YOFF": (4 * 4, (code_gen_dict["INIT_YOFF"])),
            "YEND": (5 * 4, (code_gen_dict["INIT_YEND"])),
        }
        return config