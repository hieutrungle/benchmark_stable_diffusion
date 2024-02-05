import diffusers
import transformers
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

try:
    import poptorch
except ImportError:
    print("poptorch not installed")


class PipelinedVAE(diffusers.AutoencoderKL):
    def parallelize(self, ipu_conf):

        self.encoder.conv_in = poptorch.BeginBlock(
            self.encoder.conv_in, "encoder.conv_in", ipu_id=3
        )
        self.encoder.down_blocks = poptorch.BeginBlock(
            self.encoder.down_blocks, "encoder.down_blocks", ipu_id=3
        )
        self.encoder.mid_block = poptorch.BeginBlock(
            self.encoder.mid_block, "encoder.mid_block", ipu_id=3
        )

        self.decoder.conv_in = poptorch.BeginBlock(
            self.decoder.conv_in, "decoder.conv_in", ipu_id=3
        )
        self.decoder.up_blocks = poptorch.BeginBlock(
            self.decoder.up_blocks, "decoder.up_blocks", ipu_id=3
        )
        self.decoder.mid_block = poptorch.BeginBlock(
            self.decoder.mid_block, "decoder.mid_block", ipu_id=3
        )
        self.decoder.conv_norm_out = poptorch.BeginBlock(
            self.decoder.conv_norm_out, "decoder.conv_norm_out", ipu_id=3
        )
        self.decoder.conv_act = poptorch.BeginBlock(
            self.decoder.conv_act, "decoder.conv_act", ipu_id=3
        )
        self.decoder.conv_out = poptorch.BeginBlock(
            self.decoder.conv_out, "decoder.conv_out", ipu_id=3
        )

        self.quant_conv = poptorch.BeginBlock(self.quant_conv, "quant_conv", ipu_id=3)
        self.post_quant_conv = poptorch.BeginBlock(
            self.post_quant_conv, "post_quant_conv", ipu_id=3
        )

    # @poptorch.BlockFunction("CLIPTextModel", ipu_id=3)
    # def forward


class PipelinedCLIPTextModel(transformers.CLIPTextModel):

    def parallelize(self, ipu_conf):

        self.text_model.embeddings = poptorch.BeginBlock(
            self.text_model.embeddings, "text_model.embeddings", ipu_id=0
        )
        self.text_model.encoder = poptorch.BeginBlock(
            self.text_model.encoder, "text_model.encoder", ipu_id=0
        )
        self.text_model.final_layer_norm = poptorch.BeginBlock(
            self.text_model.final_layer_norm, "text_model.final_layer_norm", ipu_id=0
        )


class PipelinedUnet(diffusers.UNet2DConditionModel):
    def parallelize(self, ipu_conf):
        # logger.log("-------------------- Device Allocation --------------------")

        self.conv_in = poptorch.BeginBlock(self.conv_in, "conv_in", ipu_id=1)
        self.time_proj = poptorch.BeginBlock(self.time_proj, "time_proj", ipu_id=1)
        self.time_embedding = poptorch.BeginBlock(
            self.time_embedding, "time_embedding", ipu_id=1
        )
        self.down_blocks = poptorch.BeginBlock(
            self.down_blocks, "down_blocks", ipu_id=1
        )
        self.mid_block = poptorch.BeginBlock(self.mid_block, "mid_block", ipu_id=2)
        self.up_blocks = poptorch.BeginBlock(self.up_blocks, "up_blocks", ipu_id=2)

        self.conv_norm_out = poptorch.BeginBlock(
            self.conv_norm_out, "conv_norm_out", ipu_id=2
        )
        self.conv_act = poptorch.BeginBlock(self.conv_act, "up_blocks", ipu_id=2)
        self.conv_out = poptorch.BeginBlock(self.conv_out, "up_blocks", ipu_id=2)
