# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2023, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pytest
import os
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
import finn.core.onnx_exec as oxe
from onnx import TensorProto, helper
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.util.basic import make_build_dir
from qonnx.custom_op.registry import getCustomOp
from pyverilator.util.axi_utils import axilite_write, reset_rtlsim, _write_signal, toggle_neg_edge, _read_signal, toggle_pos_edge
from datetime import datetime
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.pyverilator import pyverilate_stitched_ip
import math

tmpdir = os.environ["FINN_BUILD_DIR"]
tmpOnnxDir = make_build_dir("upsample_export_")
rtlsim_trace = True

def createModel(OFMDim, IFMDim, NumChannels, dt):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, IFMDim, 1, NumChannels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, OFMDim, 1, NumChannels])
    upsample_node = helper.make_node(
        "UpsampleNearestNeighbour",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        OFMDim=OFMDim,
        IFMDim=IFMDim,
        NumChannels=NumChannels,
        inputDataType="INT8",
        numInputVectors=1,
        DimMode=1,
        dynamic_mode=1,
        name="UpsampleNearestNeighbour",
        )
    
    graph = helper.make_graph(
        nodes=[upsample_node], name="upsample_graph", inputs=[inp], outputs=[outp]
    )
    model = qonnx_make_model(graph, producer_name="upsample-model")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", dt)
    model.set_tensor_datatype("outp", dt)
    return model

# Helper function that delivers the hook to program the SWG via AXI-Lite
def config_hook(configs):
    if configs is None:
        return None

    def write_swg_config(sim):
        for axi_name, config in configs:
            for config_entry in config.values():
                axilite_write(sim, config_entry[0], config_entry[1], basename=axi_name) 

    return write_swg_config

def prepare_inputs(input_tensor):
    return {"inp": input_tensor}

def rtlsim_multi_io(sim, execution_dict, sname="_V_V_", liveness_threshold=10000):
    for outp in execution_dict[0]["io"]["outputs"]:
        _write_signal(sim, outp + sname + "TREADY", 1)

    total_cycle_count = 0
    signals_to_write = {}

    for execution in execution_dict:
        hook = execution["hook"]
        io_dict = execution["io"]
        num_out_values = execution["outval"]

        output_done = False
        output_count = 0
        old_output_count = 0
        no_change_count = 0

        if hook is not None:
            reset_rtlsim(sim)
            hook(sim)


        while not (output_done):
            toggle_neg_edge(sim)

            # examine signals, decide how to act based on that but don't update yet
            # so only _read_signal access in this block, no _write_signal
            for inp in io_dict["inputs"]:
                inputs = io_dict["inputs"][inp]
                signal_name = inp + sname
                if _read_signal(sim, signal_name + "TREADY") == 1 and _read_signal(sim, signal_name + "TVALID") == 1:
                    inputs = inputs[1:]
                io_dict["inputs"][inp] = inputs

            for outp in io_dict["outputs"]:
                outputs = io_dict["outputs"][outp]
                signal_name = outp + sname
                if _read_signal(sim, signal_name + "TREADY") == 1 and _read_signal(sim, signal_name + "TVALID") == 1:
                    outputs = outputs + [_read_signal(sim, signal_name + "TDATA")]
                    output_count += 1
                io_dict["outputs"][outp] = outputs

            # update signals based on decisions in previous block, but don't examine anything
            # so only _write_signal access in this block, no _read_signal
            for inp in io_dict["inputs"]:
                inputs = io_dict["inputs"][inp]
                signal_name = inp + sname
                signals_to_write[signal_name + "TVALID"] = 1 if len(inputs) > 0 else 0
                signals_to_write[signal_name + "TDATA"] = inputs[0] if len(inputs) > 0 else 0

            # Toggle rising edge to arrive at a delta cycle before the falling edge
            toggle_pos_edge(
                sim, signals_to_write=signals_to_write
            )

            total_cycle_count = total_cycle_count + 1

            if output_count == old_output_count:
                no_change_count = no_change_count + 1
            else:
                no_change_count = 0
                old_output_count = output_count

            if output_count == num_out_values:
                output_done = True

            # end sim on timeout
            if no_change_count == liveness_threshold:
                raise Exception("Error in simulation! Takes too long to produce output.")

        execution["io"] = io_dict 

    return total_cycle_count

def rtlsim_exec(model, execution_dict, dt, NumChannels):    
    trace_file = model.get_metadata_prop("rtlsim_trace")
    sim = pyverilate_stitched_ip(model)
    model.set_metadata_prop("rtlsim_so", sim.lib._name)
    reset_rtlsim(sim)
    sim.start_vcd_trace(trace_file, auto_tracing=False)
    rtlsim_multi_io(sim, execution_dict, sname="_", liveness_threshold=10000)
    sim.flush_vcd_trace()
    sim.stop_vcd_trace()

    ret = []
    axis_width = getCustomOp(model.graph.node[0]).get_outstream_width()

    for batch in execution_dict:
        output = batch["io"]["outputs"]['m_axis_0']
        o_folded_tensor = rtlsim_output_to_npy(
            output, None, dt, (1, batch["outval"], 1, NumChannels), axis_width, dt.bitwidth()
        )
        ret += [o_folded_tensor]

    return ret

@pytest.mark.parametrize("dt", [DataType["INT8"]])
@pytest.mark.parametrize("IFMDim", [5, 10])
@pytest.mark.parametrize("OFMDim", [5*10])
@pytest.mark.parametrize("scales", [[3,5], [1,2,3]])
@pytest.mark.parametrize("NumChannels", [32])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_upsampler(dt, IFMDim, OFMDim, scales, NumChannels):

    model = createModel(OFMDim, IFMDim, NumChannels, dt)
    
    if rtlsim_trace:
        now = datetime.now()
        current_time_string = now.strftime("%m%d_%H%M%S")
        model.set_metadata_prop("rtlsim_trace", f"/home/alex/finnfiles/rtlsim_trace/trace{current_time_string}.vcd")
        os.environ["RTLSIM_TRACE_DEPTH"] = "5"
    
    # Check that all nodes are UpsampleNearestNeighbour_Batch nodes
    for n in model.get_finn_nodes():
        node_check = n.op_type == "UpsampleNearestNeighbour"
        assert node_check, "All nodes should be UpsampleNearestNeighbour nodes."
    input_shape = (1, IFMDim, 1, NumChannels)

    inputs = []
    golden_results = []
    for id, scale in enumerate(scales):
        # Create golden result
        test_input = gen_finn_dt_tensor(dt, input_shape)
        inputs += [test_input]
        input_dict = {model.graph.input[0].name: test_input}

        # Change output to scale
        upsample_node = model.get_nodes_by_op_type("UpsampleNearestNeighbour")[0]
        upsample_inst = getCustomOp(upsample_node)
        shape = model.get_tensor_shape(upsample_node.output[0])
        shape[1] = scale*IFMDim
        model.set_tensor_shape(upsample_node.output[0], shape)
        upsample_inst.set_nodeattr("OFMDim", scale*IFMDim)

        output_dict = oxe.execute_onnx(model, input_dict, True)
        golden_results += [output_dict[model.graph.output[0].name]]

    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xc7z020clg400-1", 10))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP("xc7z020clg400-1", 10))
    model.set_metadata_prop("exec_mode", "rtlsim")

    execution_dict = []
    axis_width = getCustomOp(model.graph.node[0]).get_outstream_width()
    upsample_node = model.get_nodes_by_op_type("UpsampleNearestNeighbour_hls")[0]
    upsample_inst = getCustomOp(upsample_node)

    for id, scale in enumerate(scales):
        config = upsample_inst.get_dynamic_config(scale, IFMDim)
        configs = [("s_axi_control_0_", config)]
        inp_shape = input_shape
        input_tensor = inputs[id]
        packed_input = npy_to_rtlsim_input(input_tensor, dt, axis_width)
        io_dict = {"inputs": {"s_axis_0":packed_input}, "outputs": {"m_axis_0":[]}}
        execution_dict += [{"batch": id,"io": io_dict, "hook":config_hook(configs), "outval": int(math.prod(inp_shape)*scale/(axis_width/8))}]
    
    ret = rtlsim_exec(model, execution_dict, dt, NumChannels)

    for i in range(len(ret)):
        assert (ret[i] == golden_results[i]).all()