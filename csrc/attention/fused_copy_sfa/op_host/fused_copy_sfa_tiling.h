#ifndef FUSED_COPY_SFA_TILING_H
#define FUSED_COPY_SFA_TILING_H

#include "../../sparse_tail_attention/op_host/sparse_tail_attention_tiling.h"

namespace optiling {
using namespace sta;

BEGIN_TILING_DATA_DEF(FusedCopySfaTilingData)
TILING_DATA_FIELD_DEF_STRUCT(SparseTailAttentionBaseParamsMla, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(SparseTailAttentionSplitKVParamsMla, splitKVParams);
TILING_DATA_FIELD_DEF_STRUCT(SparseTailAttentionSingleCoreParamsMla, singleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(SparseTailAttentionSingleCoreTensorSizeMla, singleCoreTensorSize);
TILING_DATA_FIELD_DEF_STRUCT(SparseTailAttentionInnerSplitParams, innerSplitParams);
TILING_DATA_FIELD_DEF(uint32_t, copyCap);
TILING_DATA_FIELD_DEF(uint32_t, dramMaxBlockNum);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(
    FusedCopySfa,
    FusedCopySfaTilingData)

struct FusedCopySfaCompileInfo {
};

}  // namespace optiling

#endif
