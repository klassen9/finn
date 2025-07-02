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

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.upsampler import UpsampleNearestNeighbour


class UpsampleNearestNeighbour_hls(UpsampleNearestNeighbour, HLSBackend):
    """
    Corresponds to finn-hlslib UpsampleNearestNeighbour function.
    Upsampling is done with the Nearest Neighbour algorithm.
    The layer expects square feature maps for the in and output.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(UpsampleNearestNeighbour.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "upsample.hpp"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = []
        is_dynamic = self.get_nodeattr("dynamic_mode") == 1

        ifm_ch = self.get_nodeattr("NumChannels")
        self.code_gen_dict["$DEFINES$"] += ["#define IFMChannels {}".format(ifm_ch)]

        ibits = self.get_input_datatype().bitwidth()
        self.code_gen_dict["$DEFINES$"] += ["#define Input_precision {}".format(ibits)]

        if (not is_dynamic):
            idim = self.get_nodeattr("IFMDim")
            self.code_gen_dict["$DEFINES$"] += ["#define IFMDim {}".format(idim)]

            odim = self.get_nodeattr("OFMDim")
            self.code_gen_dict["$DEFINES$"] += ["#define OFMDim {}".format(odim)]

            batch_size = self.get_nodeattr("numInputVectors")
            self.code_gen_dict["$DEFINES$"] += ["#define numReps {}".format(batch_size)]
    
    def pragmas(self):
        is_dynamic = self.get_nodeattr("dynamic_mode") == 1
        
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0_V"
        ]
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE axis port=out0_V"
        )

        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

        if is_dynamic:
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE s_axilite port = Scale bundle = control")
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE s_axilite port = IFMDim bundle = control")
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE s_axilite port = return bundle = control")     

    def docompute(self):
        is_2d = self.get_nodeattr("DimMode") == 0
        batch = self.get_nodeattr("numInputVectors")
        is_dynamic = self.get_nodeattr("dynamic_mode") == 1
        if is_2d:
            assert not is_dynamic, "Dynamic mode is only supported for 1D upsampling"
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """UpsampleNearestNeighbour<OFMDim, IFMDim, IFMChannels,
                ap_uint<Input_precision> > (in0_V, out0_V, numReps);"""
            ]
        else:
            assert batch == 1, "1D upsampler currently needs numReps=1"

            if is_dynamic:
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    """UpsampleNearestNeighbourDyn_1D<IFMChannels,
                    ap_uint<Input_precision> > (in0_V, out0_V, Scale, IFMDim);"""
                ]

            else:
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    """UpsampleNearestNeighbour_1D<OFMDim, IFMDim, IFMChannels,
                    ap_uint<Input_precision> > (in0_V, out0_V);"""
                ]

    def blackboxfunction(self):
        is_dynamic = self.get_nodeattr("dynamic_mode") == 1
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits

        if is_dynamic:
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                "void %s(hls::stream<%s > &in0_V, hls::stream<%s > &out0_V, unsigned Scale, unsigned IFMDim)"
                % (
                    self.onnx_node.name,
                    packed_hls_type,
                    packed_hls_type,
                )
            ]
        else:
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                "void %s(hls::stream<%s > &in0_V, hls::stream<%s > &out0_V)"
                % (
                    self.onnx_node.name,
                    packed_hls_type,
                    packed_hls_type,
                )
            ]

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)

    def get_dynamic_config(self, Scales=None, IFMDim=None):
        config = {}
    
        if Scales != None:
            config.update({"cfg_Scales": (0x10, int(Scales))})

        if IFMDim != None:
            config.update({"cfg_IFMDim": (0x18, int(IFMDim))})  

        return config
    
    def get_verilog_top_module_intf_names(self):
        # Overload default HLSCustomOp implementation to add axilite control interface
        is_dynamic = self.get_nodeattr("dynamic_mode") == 1
        intf_names = super().get_verilog_top_module_intf_names()
        if is_dynamic:
            intf_names["axilite"] = ["s_axi_control"]
        return intf_names