# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# Read-credit accounting of the paced axi_dma_rd_u behind fetch_weights. Every AR
# burst reserves BURST_LEN beats of the read-data FIFO, but only the beats it
# delivers are returned, so each weight fetch that is not a whole number of bursts
# loses the remainder for good. With the fetch_weights defaults (N_OUTSTANDING=64,
# BURST_LEN=16) the reader starts with 63 * 16 = 1008 credits and stops issuing
# reads after about 68 fetches that lose 15 each.
#
# A tiled (TH>1) external_mem MVAU fetches the whole weight matrix once per TH
# input vectors (with TH=1 a local weight buffer replays a single fetch instead),
# so a small matrix and enough input vectors run into this in stitched-IP rtlsim,
# where it shows up as an rtlsim timeout.

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.core.rtlsim_exec import rtlsim_exec
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.general import ApplyConfig


@pytest.mark.parametrize(
    "mw, mh, simd, pe, fetch_beats",
    [
        (16, 4, 4, 4, 1),
        (68, 16, 4, 4, 17),
        (64, 16, 4, 4, 16),
    ],
)
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_mvau_external_mem_partial_bursts(mw, mh, simd, pe, fetch_beats):
    # 64 INT4 weights per 256-bit fetch_weights bus word
    assert -(-mw * mh // 64) == fetch_beats

    W = gen_finn_dt_tensor(DataType["INT4"], (mw, mh))
    # 192 / TH = 96 weight fetches, well above the ~68 that exhaust the credit
    ifm = helper.make_tensor_value_info("ifm", TensorProto.FLOAT, [1, 192, mw])
    ofm = helper.make_tensor_value_info("ofm", TensorProto.FLOAT, [1, 192, mh])
    matmul = helper.make_node("MatMul", ["ifm", "weights"], ["ofm"])
    graph = helper.make_graph([matmul], "mvau_external_mem", [ifm], [ofm])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mvau-external-mem-credit"))
    model.set_tensor_datatype("ifm", DataType["INT4"])
    model.set_tensor_datatype("weights", DataType["INT4"])
    model.set_tensor_datatype("ofm", DataType["INT32"])
    model.set_initializer("weights", W)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(SpecializeLayers("xcvc1902-vsva2197-2MP-e-S"))
    model = model.transform(GiveUniqueNodeNames())
    assert model.graph.node[0].op_type == "MVAU_rtl"
    model = model.transform(
        ApplyConfig(
            {
                "Defaults": {},
                "MVAU_rtl_0": {
                    "PE": pe,
                    "SIMD": simd,
                    "TH": 2,
                    "resType": "dsp",
                    "mem_mode": "external_mem",
                },
            }
        )
    )
    model = model.transform(MinimizeAccumulatorWidth())
    model = model.transform(InferDataTypes())

    model = model.transform(PrepareIP("xcvc1902-vsva2197-2MP-e-S", 4.0))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP("xcvc1902-vsva2197-2MP-e-S", 4.0))
    model.set_metadata_prop("exec_mode", "rtlsim")

    A = gen_finn_dt_tensor(DataType["INT4"], (1, 192, mw))
    context = {"global_in": A}
    rtlsim_exec(model, context)
    expected = np.matmul(A.astype(np.int64), W.astype(np.int64))
    produced = context["global_out"].reshape(expected.shape)
    assert (produced == expected).all(), "Stitched-IP rtlsim output differs from MatMul"
