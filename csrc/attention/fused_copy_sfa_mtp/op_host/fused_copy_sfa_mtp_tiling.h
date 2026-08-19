#ifndef NANOVLLM_FUSED_COPY_SFA_MTP_TILING_H
#define NANOVLLM_FUSED_COPY_SFA_MTP_TILING_H

#include "../../sparse_tail_attention/op_host/sparse_tail_attention_tiling.h"

namespace optiling {
using namespace sta;

// Keep the production sparse Attention payload as an exact prefix.  The
// kernel reinterprets that prefix as SparseTailAttentionTilingDataMla
// and consumes the suffix for source-aware DRAM gather metadata.
BEGIN_TILING_DATA_DEF(FusedCopySfaMtpTilingData)
TILING_DATA_FIELD_DEF_STRUCT(SparseTailAttentionBaseParamsMla, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(SparseTailAttentionSplitKVParamsMla, splitKVParams);
TILING_DATA_FIELD_DEF_STRUCT(SparseTailAttentionSingleCoreParamsMla, singleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(SparseTailAttentionSingleCoreTensorSizeMla, singleCoreTensorSize);
TILING_DATA_FIELD_DEF_STRUCT(SparseTailAttentionInnerSplitParams, innerSplitParams);
TILING_DATA_FIELD_DEF(uint32_t, copyCap);
TILING_DATA_FIELD_DEF(uint32_t, dramMaxBlockNum);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(
    FusedCopySfaMtp,
    FusedCopySfaMtpTilingData)

struct FusedCopySfaMtpCompileInfo {
};

}  // namespace optiling

#endif
