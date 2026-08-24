#include "kernel_operator.h"
#define C_TEMPLATE 0
#define V_TEMPLATE 1
#define STA_CANONICAL_SOURCE_TILES 1

// OPC generates only this operator's tiling class.  The fused payload has the
// complete production STA payload as its prefix, so expose it under the type
// name expected by the shared STA implementation (the same pattern used by
// the non-MTP fused_copy_sfa kernel).
using SparseTailAttentionTilingDataMla =
    FusedCopySfaMtpTilingData;

#include "../../fused_copy_sfa/op_kernel/sfa_impl/sparse_tail_attention_kernel_mla.h"

using namespace AscendC;
namespace {
template <typename T>
__aicore__ inline void RunFusedMtp(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *cacheTokens,
    __gm__ uint8_t *hbmBlockTable,
    __gm__ uint8_t *actualSeqLengthsQuery,
    __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *queryRope,
    __gm__ uint8_t *hbmKeyRope,
    __gm__ uint8_t *dramKeyRope,
    __gm__ uint8_t *dramKvCache,
    __gm__ uint8_t *dramBlockTable,
    __gm__ uint8_t *topkSourceIds,
    __gm__ uint8_t *missSourceIds,
    __gm__ uint8_t *missDestinationSlots,
    __gm__ uint8_t *missCounts,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *attentionWorkspace,
    const FusedCopySfaMtpTilingData *fusedTiling,
    __gm__ uint8_t *tiling,
    TPipe *pipe)
{
    // The source-aware STA path reuses each query-level DRAM gather for the
    // persistent HBM update.  Compact union metadata remains part of the ABI
    // for metadata/COPYSFA composition but needs no separate pre-copy pass.
    (void)missSourceIds;
    (void)missDestinationSlots;

    using MtpType = STAType<
        T, T, T, false, STA_LAYOUT::TND, STA_LAYOUT::PA_BSND,
        V_TEMPLATE, STA_STAGE_NORMAL, true, true>;
    SparseTailAttentionMla<MtpType> attention;
    const auto *attentionTiling = reinterpret_cast<
        const SparseTailAttentionTilingDataMla *>(fusedTiling);
    attention.Init(
        query, key, value, sparseIndices, cacheTokens, nullptr,
        actualSeqLengthsQuery, actualSeqLengthsKv, hbmBlockTable,
        queryRope, hbmKeyRope, attentionOut,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        attentionWorkspace, attentionTiling, tiling, pipe);
    attention.InitSourceAwareGather(
        dramKeyRope, dramKvCache, dramBlockTable, topkSourceIds,
        missCounts, fusedTiling->copyCap,
        fusedTiling->dramMaxBlockNum);
    attention.Process();
}

}  // namespace

extern "C" __global__ __aicore__ void fused_copy_sfa_mtp(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *cacheTokens,
    __gm__ uint8_t *hbmBlockTable,
    __gm__ uint8_t *actualSeqLengthsQuery,
    __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *queryRope,
    __gm__ uint8_t *hbmKeyRope,
    __gm__ uint8_t *dramKeyRope,
    __gm__ uint8_t *dramKvCache,
    __gm__ uint8_t *dramBlockTable,
    __gm__ uint8_t *topkSourceIds,
    __gm__ uint8_t *missSourceIds,
    __gm__ uint8_t *missDestinationSlots,
    __gm__ uint8_t *missCounts,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(
        FusedCopySfaMtpTilingData,
        fusedTilingIn, tiling);
    const auto *fusedTiling = &fusedTilingIn;

    TPipe pipe;
    __gm__ uint8_t *attentionWorkspace = GetUserWorkspace(workspace);
    if (TILING_KEY_IS(1)) {
        if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16) {
            RunFusedMtp<half>(
                query, key, value, sparseIndices, cacheTokens,
                hbmBlockTable, actualSeqLengthsQuery,
                actualSeqLengthsKv, queryRope, hbmKeyRope,
                dramKeyRope, dramKvCache, dramBlockTable,
                topkSourceIds, missSourceIds, missDestinationSlots,
                missCounts,
                attentionOut, attentionWorkspace, fusedTiling,
                tiling, &pipe);
        } else {
            RunFusedMtp<bfloat16_t>(
                query, key, value, sparseIndices, cacheTokens,
                hbmBlockTable, actualSeqLengthsQuery,
                actualSeqLengthsKv, queryRope, hbmKeyRope,
                dramKeyRope, dramKvCache, dramBlockTable,
                topkSourceIds, missSourceIds, missDestinationSlots,
                missCounts,
                attentionOut, attentionWorkspace, fusedTiling,
                tiling, &pipe);
        }
    }
}
