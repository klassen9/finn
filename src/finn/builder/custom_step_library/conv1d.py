from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.insert_topk import InsertTopK

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline.absorb as absorb
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.streamline.reorder import MakeScaleResizeNHWC


def step_layout(model: ModelWrapper, cfg: DataflowBuildConfig):
    graph = model.graph
    for n in graph.node:
        if n.op_type == "MultiThreshold":
            n.set_nodeattr("data_layout", "NCW")
    model = model.transform(InferDataLayouts())
    return model

def step_io_surgery(model: ModelWrapper, cfg: DataflowBuildConfig):
    # pre-processing: remove input quantization from model
    first_node = model.graph.node[0]
    if first_node.op_type == "MultiThreshold":
        quantized_input_dtype = model.get_tensor_datatype(first_node.output[0])
        # remove nodes up to first Mul (= MT + Add used for input quant)
        new_input_node = model.get_nodes_by_op_type("Mul")[0]
        new_input_tensor = model.get_tensor_valueinfo(new_input_node.input[0])
        old_input_tensor = model.graph.input[0]
        model.graph.input.remove(old_input_tensor)
        model.graph.input.append(new_input_tensor)
        model.graph.value_info.remove(new_input_tensor) # remove redundant value_info
        new_input_index = model.get_node_index(new_input_node)
        del model.graph.node[0:new_input_index]
        # make sure input datatype is set correctly
        model.set_tensor_datatype(model.graph.input[0].name, quantized_input_dtype)

    # post-processing: remove final softmax node if it remains from training
    final_node = model.graph.node[-1]
    if final_node.op_type in ["LogSoftmax", "Softmax"]:
        softmax_in_tensor = model.get_tensor_valueinfo(final_node.input[0])
        softmax_out_tensor = model.get_tensor_valueinfo(final_node.output[0])
        model.graph.output.remove(softmax_out_tensor)
        model.graph.output.append(softmax_in_tensor)
        model.graph.value_info.remove(softmax_in_tensor) # remove redundant value_info
        model.graph.node.remove(final_node)

    # post-processing: append Top-K node
    final_node = model.graph.node[-1]
    if final_node.op_type != "TopK":
        model = model.transform(InsertTopK(k=1))
    return model

def step_pre_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(Change3DTo4DTensors())
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    return model


def step_convert_final_layers(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(to_hw.InferChannelwiseLinearLayer())
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(MakeScaleResizeNHWC())
    model = model.transform(to_hw.InferUpsample())
    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(GiveUniqueNodeNames())
    return model
