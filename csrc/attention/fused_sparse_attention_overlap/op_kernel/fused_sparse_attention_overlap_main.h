/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 *
 * Main controller for fused_sparse_attention_overlap with Cube+Vector mode.
 * Coordinates AIC (Matmul) and AIV (Softmax + Gather) cores.
 */
#ifndef FUSED_SPARSE_ATTENTION_OVERLAP_MAIN_H_
#define FUSED_SPARSE_ATTENTION_OVERLAP_MAIN_H_

#include "kernel_operator.h"
#include "fused_sparse_attention_overlap_cube.h"
#include "fused_sparse_attention_overlap_vector.h"

namespace FusedSparseAttentionOverlapNs {
using namespace AscendC;

// CrossCore sync constants
constexpr uint32_t FSA_SYNC_MODE = 2;
constexpr uint16_t FSA_SYNC_AIV_TO_AIC = 6;   // V0 signals AIC: Mm1 request ready
constexpr uint16_t FSA_SYNC_AIC_MM1_DONE = 7;  // AIC signals vector: Mm1 scores ready
constexpr uint16_t FSA_SYNC_AIV_TO_AIC2 = 8;  // vector signals AIC: Mm2 weights ready
constexpr uint16_t FSA_SYNC_AIC_MM2_DONE = 9;  // AIC signals vector: Mm2 output ready
constexpr uint16_t FSA_SYNC_AIC_HELPER_DONE = 4;  // AIC releases sync-only vector lane
constexpr uint16_t FSA_SYNC_AIV_MISS_READY = 5;  // V lanes signal AIC: miss gather ready
constexpr uint16_t FSA_SYNC_AIC_HIT_MM2_DONE = 10;  // AIC signals vector: internal hit Mm2 done
constexpr uint16_t FSA_SYNC_DONE = FSA_SYNC_AIV_TO_AIC;         // AIV signals AIC: all done, exit
constexpr int32_t FSA_COMM_BUF_BYTES = 1088;
constexpr int32_t FSA_COMM_BUF_SLOTS = FSA_COMM_BUF_BYTES / sizeof(int32_t);
constexpr bool FSA_ENABLE_AIC_OVERLAP = true;
constexpr bool FSA_DIAG_AIC_DUMMY = false;
constexpr bool FSA_DIAG_FAKE_AIC = false;
constexpr bool FSA_DIAG_AIV_MARK_MM2 = false;
constexpr bool FSA_DIAG_AIC_MM2_INPUT_MARK = false;
constexpr bool FSA_DIAG_AIC_MARK_MM2_AFTER_COMPUTE = false;
constexpr bool FSA_DIAG_AIC_MM2_MARK_ONLY = false;
constexpr bool FSA_DIAG_AIC_MM2_FORCE_MAIN_SLOT = false;
constexpr bool FSA_DIAG_AIC_MM2_HEARTBEAT_TO_OUTPUT = false;
constexpr bool FSA_DIAG_AIV_FINAL_DIRECT_MARK = false;
constexpr bool FSA_DIAG_STAGE_HEARTBEAT = false;
constexpr bool FSA_ENABLE_PRE_MM2_GATHER_OVERLAP = false;
constexpr bool FSA_ENABLE_SPLIT_HIT_MISS_OVERLAP = false;
constexpr bool FSA_ENABLE_SINGLE_REQUEST_FULL_OVERLAP = false;
constexpr bool FSA_ENABLE_HIT_SCORE_PREFETCH_OVERLAP = false;
constexpr bool FSA_ENABLE_DIRECT_MISS_SOURCE_OVERLAP = true;
constexpr bool FSA_ENABLE_NEXT_CHUNK_GATHER_PREFETCH = true;
constexpr bool FSA_ENABLE_NEXT_CHUNK_HIT_COPY_PREFETCH = true;
constexpr bool FSA_ENABLE_AIV_MM2_ACCUMULATE = true;
constexpr bool FSA_DIAG_SKIP_AIV_MM2_TEMP_ACCUMULATE = false;
constexpr int64_t FSA_MM2_FIXPIPE_M_ALIGN = 16;
constexpr int64_t FSA_MM2_WORKSPACE_SLOT_COUNT = FSA_ENABLE_AIV_MM2_ACCUMULATE ? 2 : 1;
constexpr bool FSA_DIAG_NEXT_CHUNK_PREFETCH = false;
constexpr bool FSA_DIAG_NEXT_CHUNK_PREFETCH_COUNTERS = false;
constexpr int32_t FSA_NEXT_CHUNK_PREFETCH_MM1_LIMIT = 8;
constexpr int32_t FSA_NEXT_CHUNK_PREFETCH_MM2_LIMIT = 24;
constexpr int32_t FSA_NEXT_CHUNK_PREFETCH_DIRECT_MM1_LIMIT = 8;
constexpr int32_t FSA_NEXT_CHUNK_PREFETCH_DIRECT_MM2_LIMIT = 24;
constexpr int32_t FSA_NEXT_CHUNK_PREFETCH_MISS_ONLY_MM1_LIMIT = 8;
constexpr int32_t FSA_NEXT_CHUNK_PREFETCH_MISS_ONLY_MM2_LIMIT = 24;
constexpr bool FSA_ENABLE_DIRECT_PREFETCH_BATCH_WITH_CURRENT_GATHER = true;
constexpr bool FSA_ENABLE_HIT_ONLY_PREFETCH_BATCH_ACROSS_MM1_MM2 = true;
constexpr bool FSA_DIAG_SKIP_FULL_OVERLAP_GATHER = false;
constexpr bool FSA_DIAG_FULL_OVERLAP_NORMAL_PROTOCOL = false;
constexpr bool FSA_DIAG_SIGNAL_MISS_READY_BEFORE_GATHER = false;
constexpr bool FSA_DIAG_COMM_DCCI = false;
constexpr bool FSA_DIAG_ROTATE_CROSSCORE_FLAGS = false;
constexpr bool FSA_ENABLE_AIC_STAGE_POLL = false;
constexpr bool FSA_ENABLE_AIC_DONE_FLAG_ROTATION = false;
constexpr bool FSA_ENABLE_SINGLE_PRODUCER_AIC_REQUEST = true;
constexpr bool FSA_ENABLE_PAIRED_AIV_REQUEST_FLAGS = false;
constexpr bool FSA_ENABLE_PAIRED_AIC_DONE_FLAGS = true;
constexpr int32_t FSA_AIV_TO_AIC_WAIT_COUNT = 1;
constexpr int32_t FSA_AIC_TO_AIV_DONE_SIGNAL_COUNT = 2;
constexpr bool FSA_DIAG_SYNC_ONLY_NO_AIC_SIGNAL = false;
constexpr bool FSA_DIAG_WORKER_NO_AIC_SIGNAL = false;
constexpr bool FSA_DIAG_WAIT_AIC_DONE_BY_ACK = false;
constexpr bool FSA_DIAG_SKIP_NONFIRST_HIT_ONLY_MM2 = false;
constexpr bool FSA_DIAG_SKIP_NONFIRST_HIT_ONLY_MM1_WAIT = false;
constexpr bool FSA_DIAG_STOP_BEFORE_NONFIRST_CHUNK = false;
constexpr bool FSA_DIAG_STOP_AFTER_FIRST_HIT_ONLY_CHUNK = false;
constexpr int32_t FSA_DIRECT_SOURCE_MODE = -2;
constexpr int32_t FSA_PREFETCH_SOURCE_HIT_ONLY_MM1 = 1;
constexpr int32_t FSA_PREFETCH_SOURCE_HIT_ONLY_MM2 = 2;
constexpr int32_t FSA_PREFETCH_SOURCE_DIRECT_MM1 = 3;
constexpr int32_t FSA_PREFETCH_SOURCE_DIRECT_MM2 = 4;
constexpr int32_t FSA_PREFETCH_SOURCE_MISS_ONLY_MM1 = 5;
constexpr int32_t FSA_PREFETCH_SOURCE_MISS_ONLY_MM2 = 6;
constexpr int32_t FSA_DIAG_PREFETCH_TOTAL_COPIED_SLOT = 0;
constexpr int32_t FSA_DIAG_PREFETCH_HIT_ONLY_MM1_COPIED_SLOT = 1;
constexpr int32_t FSA_DIAG_PREFETCH_HIT_ONLY_MM2_COPIED_SLOT = 2;
constexpr int32_t FSA_DIAG_PREFETCH_DIRECT_MM1_COPIED_SLOT = 3;
constexpr int32_t FSA_DIAG_PREFETCH_DIRECT_MM2_COPIED_SLOT = 4;
constexpr int32_t FSA_DIAG_PREFETCH_MISS_ONLY_MM1_COPIED_SLOT = 5;
constexpr int32_t FSA_DIAG_PREFETCH_MISS_ONLY_MM2_COPIED_SLOT = 6;
constexpr int32_t FSA_DIAG_PREFETCH_TOTAL_WINDOWS_SLOT = 7;
constexpr int32_t FSA_DIAG_PREFETCH_HIT_ONLY_MM1_WINDOWS_SLOT = 8;
constexpr int32_t FSA_DIAG_PREFETCH_HIT_ONLY_MM2_WINDOWS_SLOT = 9;
constexpr int32_t FSA_DIAG_PREFETCH_DIRECT_MM1_WINDOWS_SLOT = 10;
constexpr int32_t FSA_DIAG_PREFETCH_DIRECT_MM2_WINDOWS_SLOT = 11;
constexpr int32_t FSA_DIAG_PREFETCH_MISS_ONLY_MM1_WINDOWS_SLOT = 12;
constexpr int32_t FSA_DIAG_PREFETCH_MISS_ONLY_MM2_WINDOWS_SLOT = 13;
constexpr int32_t FSA_DIAG_DIRECT_CURRENT_GATHER_MAX_SLOT = 14;
constexpr int32_t FSA_DIAG_DIRECT_BATCHED_GATHER_MAX_SLOT = 15;

constexpr uint16_t FSA_SYNC_AIV_TO_AIC_ALT = 12;
constexpr uint16_t FSA_SYNC_AIV_TO_AIC2_ALT = 13;
constexpr uint16_t FSA_SYNC_AIC_MM1_DONE_ALT = 14;
constexpr uint16_t FSA_SYNC_AIC_MM2_DONE_ALT = 15;
constexpr uint16_t FSA_AIV1_FLAG_OFFSET = 16;

__aicore__ inline uint16_t FsaCrossCoreFlagPhase(int32_t syncStage)
{
    int32_t safeStage = (syncStage > 0) ? syncStage : 1;
    return static_cast<uint16_t>(((safeStage - 1) >> 1) & 1);
}

__aicore__ inline uint16_t FsaAivToAicFlag(int32_t syncStage)
{
    if constexpr (FSA_DIAG_ROTATE_CROSSCORE_FLAGS) {
        return (FsaCrossCoreFlagPhase(syncStage) == 0) ? FSA_SYNC_AIV_TO_AIC : FSA_SYNC_AIV_TO_AIC_ALT;
    }
    return FSA_SYNC_AIV_TO_AIC;
}

__aicore__ inline uint16_t FsaAicMm1DoneFlag(int32_t syncStage)
{
    if constexpr (FSA_ENABLE_AIC_DONE_FLAG_ROTATION || FSA_DIAG_ROTATE_CROSSCORE_FLAGS) {
        return (FsaCrossCoreFlagPhase(syncStage) == 0) ? FSA_SYNC_AIC_MM1_DONE : FSA_SYNC_AIC_MM1_DONE_ALT;
    }
    return FSA_SYNC_AIC_MM1_DONE;
}

__aicore__ inline uint16_t FsaAivToAic2Flag(int32_t syncStage)
{
    if constexpr (FSA_DIAG_ROTATE_CROSSCORE_FLAGS) {
        return (FsaCrossCoreFlagPhase(syncStage) == 0) ? FSA_SYNC_AIV_TO_AIC2 : FSA_SYNC_AIV_TO_AIC2_ALT;
    }
    return FSA_SYNC_AIV_TO_AIC2;
}

__aicore__ inline uint16_t FsaAicMm2DoneFlag(int32_t syncStage)
{
    if constexpr (FSA_ENABLE_AIC_DONE_FLAG_ROTATION || FSA_DIAG_ROTATE_CROSSCORE_FLAGS) {
        return (FsaCrossCoreFlagPhase(syncStage) == 0) ? FSA_SYNC_AIC_MM2_DONE : FSA_SYNC_AIC_MM2_DONE_ALT;
    }
    return FSA_SYNC_AIC_MM2_DONE;
}

__aicore__ inline uint16_t FsaAicDoneFlagForCurrentAivLane(uint16_t flag)
{
    if constexpr (FSA_ENABLE_PAIRED_AIC_DONE_FLAGS) {
#ifndef ASCENDC_CPU_DEBUG
        return static_cast<uint16_t>(flag + ((GetBlockIdx() % 2 != 0) ? FSA_AIV1_FLAG_OFFSET : 0));
#else
        return flag;
#endif
    }
    return flag;
}

__aicore__ inline void FsaWaitAivToAicFlag(int32_t syncStage)
{
    uint16_t flag = FsaAivToAicFlag(syncStage);
    if constexpr (FSA_ENABLE_PAIRED_AIV_REQUEST_FLAGS) {
        CrossCoreWaitFlag(flag);
        CrossCoreWaitFlag(static_cast<uint16_t>(flag + FSA_AIV1_FLAG_OFFSET));
    } else {
        for (int32_t i = 0; i < FSA_AIV_TO_AIC_WAIT_COUNT; i++) {
            CrossCoreWaitFlag(flag);
        }
    }
}

__aicore__ inline void FsaWaitAivToAic2Flag(int32_t syncStage)
{
    uint16_t flag = FsaAivToAic2Flag(syncStage);
    if constexpr (FSA_ENABLE_PAIRED_AIV_REQUEST_FLAGS) {
        CrossCoreWaitFlag(flag);
        CrossCoreWaitFlag(static_cast<uint16_t>(flag + FSA_AIV1_FLAG_OFFSET));
    } else {
        for (int32_t i = 0; i < FSA_AIV_TO_AIC_WAIT_COUNT; i++) {
            CrossCoreWaitFlag(flag);
        }
    }
}

__aicore__ inline void FsaSignalAicDoneToAiv(uint16_t flag)
{
    if constexpr (FSA_ENABLE_PAIRED_AIC_DONE_FLAGS) {
        CrossCoreSetFlag<FSA_SYNC_MODE, PIPE_FIX>(flag);
        CrossCoreSetFlag<FSA_SYNC_MODE, PIPE_FIX>(static_cast<uint16_t>(flag + FSA_AIV1_FLAG_OFFSET));
    } else {
        for (int32_t i = 0; i < FSA_AIC_TO_AIV_DONE_SIGNAL_COUNT; i++) {
            CrossCoreSetFlag<FSA_SYNC_MODE, PIPE_FIX>(flag);
        }
    }
}

__aicore__ inline void FsaSignalAicMm1Done(int32_t syncStage)
{
    FsaSignalAicDoneToAiv(FsaAicMm1DoneFlag(syncStage));
}

__aicore__ inline void FsaSignalAicMm2Done(int32_t syncStage)
{
    FsaSignalAicDoneToAiv(FsaAicMm2DoneFlag(syncStage));
}

// Max tokens per block for hit/miss classification
constexpr int32_t MAX_BLOCK_TOKENS = 32;

// GM comm buffer layout (int32 slots)
constexpr int32_t COMM_TOKEN_COUNT = 0;      // [0]: nTokens (-1 = DONE)
constexpr int32_t COMM_QUERY_OFF_LO = 1;     // [1]: queryGmOffset low 32bit
constexpr int32_t COMM_QUERY_OFF_HI = 2;     // [2]: queryGmOffset high 32bit
constexpr int32_t COMM_SCORE_OFF_LO = 3;     // [3]: scoreGmOffset low 32bit
constexpr int32_t COMM_SCORE_OFF_HI = 4;     // [4]: scoreGmOffset high 32bit
constexpr int32_t COMM_IS_FIRST = 5;         // [5]: isFirstBlock (0/1)
constexpr int32_t COMM_MM2_OUT_OFF_LO = 6;   // [6]: mm2OutputGmOffset low 32bit
constexpr int32_t COMM_MM2_OUT_OFF_HI = 7;   // [7]: mm2OutputGmOffset high 32bit
constexpr int32_t COMM_KV_ADDRS = 8;         // [8..40): kvAddrs
constexpr int32_t COMM_ROPE_ADDRS = 8 + MAX_BLOCK_TOKENS; // [40..72): ropeAddrs
constexpr int32_t COMM_STAGE = COMM_ROPE_ADDRS + MAX_BLOCK_TOKENS; // [72]: sync stage marker
constexpr int32_t COMM_ACK_STAGE = COMM_STAGE + 1; // AIC completion stage marker
constexpr int32_t COMM_IS_LAST = COMM_ACK_STAGE + 1; // current request is the final AIC request
constexpr int32_t COMM_HIT_COUNT = COMM_IS_LAST + 1; // hit prefix length for in-request overlap
constexpr int32_t COMM_MISS_READY_STAGE = COMM_HIT_COUNT + 1; // AIV finished miss gather for in-request overlap
constexpr int32_t COMM_SOURCE_FLAGS = COMM_MISS_READY_STAGE + 1; // [77..109): 0=selection cache, 1=full cache
constexpr int32_t COMM_FULL_KV_ADDRS = COMM_SOURCE_FLAGS + MAX_BLOCK_TOKENS; // [109..141): full-cache kvAddrs
constexpr int32_t COMM_FULL_ROPE_ADDRS = COMM_FULL_KV_ADDRS + MAX_BLOCK_TOKENS; // [141..173): full-cache ropeAddrs
constexpr int32_t COMM_DIAG_AIC_HEARTBEAT = FSA_COMM_BUF_SLOTS - 1;

// Hit flag constant (aligned with original kernel, defined in kernel.h)
// HIT_FLAG is already defined in fused_sparse_attention_overlap_kernel.h

template <typename T>
class FusedAttentionMainOp {
public:
    using MM_OUT_T = float;

    __aicore__ inline FusedAttentionMainOp() {}

    __aicore__ inline void Init(
        TPipe* pipeIn,
        const FusedSparseAttentionOverlapTilingData* tilingIn,
        GM_ADDR query, GM_ADDR selection_k_rope, GM_ADDR selection_kv_cache,
        GM_ADDR selection_kv_block_table, GM_ADDR selection_kv_block_status,
        GM_ADDR selection_topk_indices,
        GM_ADDR full_k_rope, GM_ADDR full_kv_cache,
        GM_ADDR full_kv_block_table, GM_ADDR full_kv_actual_seq,
        GM_ADDR full_q_actual_seq,
        GM_ADDR hit_mask_out, GM_ADDR miss_indices_out,
        GM_ADDR attention_output, GM_ADDR selection_kv_actual_seq,
        GM_ADDR workspace);

    // AIC-only minimal init (no UB buffer allocation)
    __aicore__ inline void InitAIC(
        TPipe* pipeIn,
        const FusedSparseAttentionOverlapTilingData* tilingIn,
        GM_ADDR query, GM_ADDR selection_k_rope, GM_ADDR selection_kv_cache,
        GM_ADDR selection_kv_block_table, GM_ADDR workspace);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessAIC();

private:
    __aicore__ inline void ProcessDirectTopk();
    __aicore__ inline void ProcessZeroOutput();
    __aicore__ inline float ComputeDirectTokenScore(
        int64_t srcKvAddr, int64_t srcRopeAddr,
        int64_t dstKvAddr, int64_t dstRopeAddr,
        bool srcFromSelection, bool copyToDst,
        LocalTensor<float>& queryFloat, LocalTensor<float>& queryRopeFloat,
        LocalTensor<float>& tmpFloat);

    __aicore__ inline void ProcessOneHead(
        int64_t bsIdx, int64_t seqIdx, int64_t kvHeadIdx, int64_t queryHeadIdx,
        int64_t curFullKvSeqModify, bool updateSelectionCache, bool isLastAicRequestCandidate,
        LocalTensor<int32_t>& selBlockStatLocal, LocalTensor<int32_t>& topkLocal,
        LocalTensor<int32_t>& selKvBlockTableLocal, LocalTensor<int32_t>& selKvActSeqLocal,
        bool syncOnly, int32_t& syncStage);

    // Hit detection (reuse from original kernel)
    __aicore__ inline void DetectHitsForHead(
        int64_t hnIdx, LocalTensor<int32_t>& topkLocal,
        LocalTensor<int32_t>& statLocal, LocalTensor<int32_t>& hitFlagLocal);

    // Gather single miss token (reuse from original kernel)
    __aicore__ inline void IssueSingleGather(
        int64_t fullKvAddr, int64_t fullRopeAddr,
        int64_t selKvAddr, int64_t selRopeAddr,
        int64_t gatherBlockSize);
    __aicore__ inline void InsertStatusHash(
        int32_t topKId, int32_t pos,
        LocalTensor<int32_t>& statusHashKeyLocal,
        LocalTensor<int32_t>& statusHashPosLocal,
        int32_t statusHashSlots);
    __aicore__ inline void IssueSingleSelectionCopy(
        int64_t srcKvAddr, int64_t srcRopeAddr,
        int64_t dstKvAddr, int64_t dstRopeAddr,
        int64_t gatherBlockSize);
    __aicore__ inline int32_t PrefetchNextChunkGathers(
        int64_t nextChunkTopkIdx, int32_t& nextChunkInsertEnd,
        int64_t& scanTopkIdx, int32_t& scanInsertIdx, int64_t curBatch,
        int64_t selBlkTableOff, int32_t maxSelectionId, int64_t lastGatherBlockSize,
        LocalTensor<int32_t>& curStatLocal, LocalTensor<int32_t>& topkLocal,
        LocalTensor<int32_t>& selKvBlockTableLocal, bool updateSelectionCache,
        bool allowStatusHit, LocalTensor<int32_t>& statusHashKeyLocal,
        LocalTensor<int32_t>& statusHashPosLocal, int32_t statusHashSlots,
        int32_t gatherEvtIdValue, bool& gatherSynced, int32_t hitCopyStableSourceEnd,
        int32_t prefetchLimit, int32_t prefetchSource, bool deferFinalDrain = false,
        bool allowHitCopyPrefetch = false);

    // GM comm buffer: write token info and signal AIC
    __aicore__ inline void WriteCommBufAndSignalAIC(
        int64_t queryGmOff, int64_t scoreGmOff, int64_t mm2OutOff,
        int32_t* kvAddrs, int32_t* ropeAddrs,
        int32_t nTokens, bool isFirstBlock, int32_t syncStage, bool isLastRequest,
        int32_t overlapHitCount = 0, int32_t* sourceFlags = nullptr,
        int32_t* fullKvAddrs = nullptr, int32_t* fullRopeAddrs = nullptr);
    __aicore__ inline void WriteMissReadyStage(int32_t syncStage);
    __aicore__ inline void SignalAICStage(int32_t syncStage);
    __aicore__ inline void WaitCommStage(int32_t syncStage);
    __aicore__ inline void WaitCommStageAndSignalAIC(int32_t syncStage);
    __aicore__ inline void WaitCommStageAndSignalAIC2(int32_t syncStage);
    __aicore__ inline void WaitMm1StageOnSyncOnlyLane(int32_t syncStage);
    __aicore__ inline void WaitMm2StageOnWorkerLane(int32_t syncStage);
    __aicore__ inline void SignalAIC2Only(int32_t syncStage);
    __aicore__ inline void SignalAICMm1Stage(int32_t syncStage);
    __aicore__ inline void DiagStageHeartbeat(int32_t slot, int32_t value);
    __aicore__ inline void DiagInitNextChunkPrefetchCounters();
    __aicore__ inline void DiagRecordNextChunkPrefetch(int32_t prefetchSource, int32_t prefetched);
    __aicore__ inline void DiagRecordNextChunkPrefetchCounterMax(int32_t slot, int32_t value);
    __aicore__ inline void WaitAivStage(int32_t syncStage, bool isMm2Stage = false);
    __aicore__ inline void WaitAICStageDone(int32_t syncStage);
    __aicore__ inline void WaitAICAckStage(int32_t syncStage);
    __aicore__ inline void WaitAICMm1Done(int32_t syncStage);
    __aicore__ inline void WaitAICMm2Done(int32_t syncStage);
    __aicore__ inline int64_t AicMm2OutputOffset(int64_t mm2OutOff, bool isFirstBlock);
    __aicore__ inline bool AicMm2UseOverwrite(bool isFirstBlock);
    __aicore__ inline void WriteAicMm2Marker(int64_t outputGmOffset);
    __aicore__ inline int64_t Mm2HeadStride(int64_t kvDimFloatAlign);
    __aicore__ inline int64_t Mm2TempOutputOffset(int64_t mm2OutOff, int64_t kvDimFloatAlign);
    __aicore__ inline void AccumulateAivMm2TempIfNeeded(
        bool syncOnly, bool isFirstBlock, int64_t mm2OutOff, int64_t kvDimFloatAlign);
    __aicore__ inline void WriteDoneAndSignalAIC(int32_t syncStage);


    __aicore__ inline int32_t ReadCommValue(int32_t slot) {
        if constexpr (FSA_DIAG_COMM_DCCI) {
#ifndef ASCENDC_CPU_DEBUG
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                commBufGm_[slot]);
            pipe_barrier(PIPE_ALL);
#endif
        }
        return commBufGm_.GetValue(slot);
    }

    // Helper: write int64 to comm buffer as two int32 slots
    __aicore__ inline void WriteInt64ToComm(int32_t slotLo, int64_t val) {
        commBufGm_.SetValue(slotLo, static_cast<int32_t>(static_cast<uint32_t>(val & 0xFFFFFFFF)));
        commBufGm_.SetValue(slotLo + 1, static_cast<int32_t>(static_cast<uint32_t>((val >> 32) & 0xFFFFFFFF)));
    }

    // Helper: read int64 from comm buffer
    __aicore__ inline int64_t ReadInt64FromComm(int32_t slotLo) {
        uint32_t lo = static_cast<uint32_t>(ReadCommValue(slotLo));
        uint32_t hi = static_cast<uint32_t>(ReadCommValue(slotLo + 1));
        return static_cast<int64_t>(lo) | (static_cast<int64_t>(hi) << 32);
    }
    __aicore__ inline int64_t ReadInt64FromLocalComm(int32_t* commSlots, int32_t slotLo) {
        uint32_t lo = static_cast<uint32_t>(commSlots[slotLo]);
        uint32_t hi = static_cast<uint32_t>(commSlots[slotLo + 1]);
        return static_cast<int64_t>(lo) | (static_cast<int64_t>(hi) << 32);
    }

    template <typename U>
    __aicore__ inline U CeilAlign(U a, U b) { return (b == 0) ? 0 : ((a + b - 1) / b * b); }
    template <typename U>
    __aicore__ inline U CeilDiv(U a, U b) { return (b == 0) ? 0 : ((a + b - 1) / b); }
    __aicore__ inline int64_t QueryHeadNum() const
    {
        return (tiling_->queryHeadNum > 0) ? tiling_->queryHeadNum : tiling_->headnum;
    }
    __aicore__ inline int64_t QueryHeadsPerKvHead() const
    {
        int64_t queryHeadNum = QueryHeadNum();
        return (tiling_->headnum > 0) ? (queryHeadNum / tiling_->headnum) : 1;
    }
    __aicore__ inline int64_t KvHeadForQuery(int64_t queryHeadIdx) const
    {
        int64_t headsPerKv = QueryHeadsPerKvHead();
        if (headsPerKv <= 0) {
            return 0;
        }
        int64_t kvHeadIdx = queryHeadIdx / headsPerKv;
        return (kvHeadIdx < tiling_->headnum) ? kvHeadIdx : (tiling_->headnum - 1);
    }

    TPipe* pipe_ = nullptr;
    const FusedSparseAttentionOverlapTilingData* tiling_ = nullptr;
    int32_t blkIdx_ = -1;
    int64_t bsLoopNum_ = 0;
    int64_t rawSeq_ = 0;

    // Services
    FusedAttentionCubeService<T> cubeService_;
    FusedAttentionVectorService<T> vectorService_;

    // GM tensors
    GlobalTensor<T> queryGm_;
    GlobalTensor<T> selKRopeGm_;
    GlobalTensor<T> selKvCacheGm_;
    GlobalTensor<int32_t> selKvBlockTableGm_;
    GlobalTensor<int32_t> selKvBlockStatusGm_;
    GlobalTensor<int32_t> selTopKIndicesGm_;
    GlobalTensor<T> fullKRopeGm_;
    GlobalTensor<T> fullKvCacheGm_;
    GlobalTensor<int32_t> fullKvBlockTableGm_;
    GlobalTensor<int32_t> fullKvActualSeqGm_;
    GlobalTensor<int32_t> fullQActualSeqGm_;
    GlobalTensor<int32_t> hitMaskOutGm_;
    GlobalTensor<int32_t> missIndicesOutGm_;
    GlobalTensor<T> attentionOutGm_;
    GlobalTensor<int32_t> selKvActualSeqGm_;
    GlobalTensor<MM_OUT_T> workspaceGm_;
    GlobalTensor<T> workspaceTGm_;  // same workspace reinterpreted as T
    GlobalTensor<int32_t> commBufGm_;
    int64_t mm2OutputGmOff_ = 0;
    GM_ADDR workspaceBase_ = nullptr;  // per-core workspace base address

    // UB buffers for hit/miss classification and gather
#ifdef ASCENDC_CPU_DEBUG
    // CPU sim: use TBuf instead of TQue to avoid AllocEventID crashes in tikicpulib
    TBuf<QuePosition::VECIN> selTopKIdxBuf_;
    TBuf<QuePosition::VECIN> gatherBuf_;
#else
    TQue<QuePosition::VECIN, 1> selTopKIdxQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> gatherQue_;
    TQue<QuePosition::VECIN, 1> directIoQue_;
#endif
    TBuf<QuePosition::VECCALC> workBuf_;
    TBuf<QuePosition::VECCALC> commLocalBuf_;
    TBuf<QuePosition::VECCALC> directBuf_;

    int64_t topkAlign_ = 0;
    int64_t topkSortAlign_ = 0;
    int64_t topkOneAlign_ = 0;
    int64_t topkOneSortAlign_ = 0;
    int32_t kRopeUbOffset_ = 0;
    int32_t selTopKIdxUbSize_ = 0;
    int32_t selBlockStatUbSize_ = 0;
    int32_t selKvBlockTableUbSize_ = 0;
    int32_t selKvActSeqUbSize_ = 0;
    LocalTensor<int32_t> hitFlagLocal_;
};

// ============================================================================
// Init
// ============================================================================
template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::Init(
    TPipe* pipeIn,
    const FusedSparseAttentionOverlapTilingData* tilingIn,
    GM_ADDR query, GM_ADDR selection_k_rope, GM_ADDR selection_kv_cache,
    GM_ADDR selection_kv_block_table, GM_ADDR selection_kv_block_status,
    GM_ADDR selection_topk_indices,
    GM_ADDR full_k_rope, GM_ADDR full_kv_cache,
    GM_ADDR full_kv_block_table, GM_ADDR full_kv_actual_seq,
    GM_ADDR full_q_actual_seq,
    GM_ADDR hit_mask_out, GM_ADDR miss_indices_out,
    GM_ADDR attention_output, GM_ADDR selection_kv_actual_seq,
    GM_ADDR workspace)
{
    pipe_ = pipeIn;
    tiling_ = tilingIn;
    if ASCEND_IS_AIV {
        blkIdx_ = GetBlockIdx() / 2;
    } else {
        blkIdx_ = GetBlockIdx();
    }
    if (blkIdx_ >= tiling_->usedCoreNum) return;

    rawSeq_ = tiling_->rawSeq;
    int64_t SH = rawSeq_ * tiling_->headnum;

    topkAlign_ = CeilAlign(static_cast<int64_t>(tiling_->topk),
                           static_cast<int64_t>(BLOCK_BYTES / sizeof(int32_t)));
    topkSortAlign_ = CeilAlign(static_cast<int32_t>(tiling_->topk), ONE_REPEAT_SORT_NUM);
    topkOneAlign_ = CeilAlign(static_cast<int64_t>(tiling_->topk + 1),
                              static_cast<int64_t>(BLOCK_BYTES / sizeof(int32_t)));
    topkOneSortAlign_ = topkSortAlign_ > topkOneAlign_ ? topkSortAlign_ : topkOneAlign_;

    // Precompute values needed by both AIC and AIV
    kRopeUbOffset_ = tiling_->kvCacheUbSize / sizeof(T);

    // NOTE: AIV buffer allocation (gatherQue_, selTopKIdxQue_, workBuf_)
    // is done in Process() inside the AIV path, because ASCEND_IS_AIV
    // is only valid inside the kernel entry function body (where
    // KERNEL_TASK_TYPE_DEFAULT is declared).

    bsLoopNum_ = (blkIdx_ == tiling_->usedCoreNum - 1)
        ? tiling_->tailCoreBsLoopNum : tiling_->mainCoreBsLoopNum;

    // GM tensors (safe on both AIC and AIV)
    queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(query));
    selKRopeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(selection_k_rope));
    selKvCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(selection_kv_cache));
    selKvBlockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(selection_kv_block_table));
    selKvBlockStatusGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(selection_kv_block_status));
    selTopKIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(selection_topk_indices));
    fullKRopeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(full_k_rope));
    fullKvCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(full_kv_cache));
    fullKvBlockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(full_kv_block_table));
    fullKvActualSeqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(full_kv_actual_seq));
    fullQActualSeqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(full_q_actual_seq));
    hitMaskOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(hit_mask_out));
    missIndicesOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(miss_indices_out));
    attentionOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(attention_output));
    selKvActualSeqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(selection_kv_actual_seq));

    // Per-core workspace
    GM_ADDR perCoreBase = workspace + blkIdx_ * tiling_->perCoreWorkspaceSize;
    workspaceBase_ = perCoreBase;
    workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MM_OUT_T*>(perCoreBase));
    workspaceTGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(perCoreBase));
    commBufGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
        perCoreBase + tiling_->commBufGmOffset));
    mm2OutputGmOff_ = tiling_->mm2OutputGmOffset / static_cast<int64_t>(sizeof(MM_OUT_T));
}

// ============================================================================
// Direct fallback path: avoids GM workspace and cross-core synchronization.
// ============================================================================
template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::ProcessZeroOutput()
{
#ifdef ASCENDC_CPU_DEBUG
    return;
#else
    constexpr int32_t DIRECT_BUF_BYTES = 64 * 1024;
    pipe_->InitBuffer(directBuf_, DIRECT_BUF_BYTES);

    int64_t headDim = tiling_->headDim;
    int64_t headDimAlign = CeilAlign(headDim, static_cast<int64_t>(BLOCK_BYTES / sizeof(float)));
    int64_t directIoBytes = CeilAlign(headDim * static_cast<int64_t>(sizeof(T)), static_cast<int64_t>(BLOCK_BYTES));
    pipe_->InitBuffer(directIoQue_, 1, directIoBytes);

    LocalTensor<uint8_t> directLocal = directBuf_.Get<uint8_t>();
    LocalTensor<float> zeroFloat = directLocal.template ReinterpretCast<float>();
    LocalTensor<T> zeroT = directIoQue_.AllocTensor<T>();

    Duplicate(zeroFloat, 0.0f, headDimAlign);
    pipe_barrier(PIPE_V);
    Cast(zeroT, zeroFloat, RoundMode::CAST_ROUND, headDimAlign);
    pipe_barrier(PIPE_ALL);

    DataCopyExtParams outParams{static_cast<uint16_t>(1),
        static_cast<uint32_t>(headDim * sizeof(T)), 0, 0, 0};
    int64_t totalHeads = tiling_->batchsize * QueryHeadNum();
    for (int64_t i = 0; i < totalHeads; i++) {
        DataCopyPad(attentionOutGm_[i * headDim], zeroT, outParams);
    }
    pipe_barrier(PIPE_ALL);
    directIoQue_.FreeTensor(zeroT);
#endif
}

template <typename T>
__aicore__ inline float FusedAttentionMainOp<T>::ComputeDirectTokenScore(
    int64_t srcKvAddr, int64_t srcRopeAddr,
    int64_t dstKvAddr, int64_t dstRopeAddr,
    bool srcFromSelection, bool copyToDst,
    LocalTensor<float>& queryFloat, LocalTensor<float>& queryRopeFloat,
    LocalTensor<float>& tmpFloat)
{
    int64_t headDim = tiling_->headDim;
    int64_t kvDim = tiling_->kvCacheDim;
    int64_t ropeDim = tiling_->kRopeDim;
    int64_t kvDimFloatAlign = CeilAlign(kvDim, static_cast<int64_t>(BLOCK_BYTES / sizeof(float)));
    int64_t ropeDimFloatAlign = CeilAlign(ropeDim, static_cast<int64_t>(BLOCK_BYTES / sizeof(float)));

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    LocalTensor<T> ioT = directIoQue_.AllocTensor<T>();
    DataCopyExtParams kvCopyParams{static_cast<uint16_t>(1),
        static_cast<uint32_t>(kvDim * sizeof(T)), 0, 0, 0};
    if (srcFromSelection) {
        DataCopyPad(ioT, selKvCacheGm_[srcKvAddr], kvCopyParams, padParams);
    } else {
        DataCopyPad(ioT, fullKvCacheGm_[srcKvAddr], kvCopyParams, padParams);
    }
    directIoQue_.EnQue(ioT);
    ioT = directIoQue_.DeQue<T>();
    pipe_barrier(PIPE_ALL);
    Cast(tmpFloat, ioT, RoundMode::CAST_NONE, kvDim);
    pipe_barrier(PIPE_ALL);
    if (copyToDst) {
        DataCopyPad(selKvCacheGm_[dstKvAddr], ioT, kvCopyParams);
        pipe_barrier(PIPE_ALL);
    }
    directIoQue_.FreeTensor(ioT);
    if (kvDim < kvDimFloatAlign) {
        Duplicate(tmpFloat[kvDim], 0.0f, kvDimFloatAlign - kvDim);
        pipe_barrier(PIPE_V);
    }

    float score = 0.0f;
    for (int64_t d = 0; d < kvDim; d++) {
        score += queryFloat.GetValue(d) * tmpFloat.GetValue(d);
    }

    if (ropeDim > 0 && headDim > kvDim) {
        ioT = directIoQue_.AllocTensor<T>();
        DataCopyExtParams ropeCopyParams{static_cast<uint16_t>(1),
            static_cast<uint32_t>(ropeDim * sizeof(T)), 0, 0, 0};
        if (srcFromSelection) {
            DataCopyPad(ioT, selKRopeGm_[srcRopeAddr], ropeCopyParams, padParams);
        } else {
            DataCopyPad(ioT, fullKRopeGm_[srcRopeAddr], ropeCopyParams, padParams);
        }
        directIoQue_.EnQue(ioT);
        ioT = directIoQue_.DeQue<T>();
        pipe_barrier(PIPE_ALL);
        Cast(tmpFloat, ioT, RoundMode::CAST_NONE, ropeDim);
        pipe_barrier(PIPE_ALL);
        if (copyToDst) {
            DataCopyPad(selKRopeGm_[dstRopeAddr], ioT, ropeCopyParams);
            pipe_barrier(PIPE_ALL);
        }
        directIoQue_.FreeTensor(ioT);
        if (ropeDim < ropeDimFloatAlign) {
            Duplicate(tmpFloat[ropeDim], 0.0f, ropeDimFloatAlign - ropeDim);
            pipe_barrier(PIPE_V);
        }
        for (int64_t d = 0; d < ropeDim; d++) {
            score += queryRopeFloat.GetValue(d) * tmpFloat.GetValue(d);
        }
    }

    return score * tiling_->scaleValue;
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::ProcessDirectTopk()
{
#ifdef ASCENDC_CPU_DEBUG
    return;
#else
    constexpr int32_t DIRECT_TOPK_LIMIT = 2048;
    constexpr int32_t DIRECT_BUF_BYTES = 64 * 1024;
    pipe_->InitBuffer(directBuf_, DIRECT_BUF_BYTES);

    int32_t directTopk = static_cast<int32_t>(tiling_->topk);
    if (directTopk <= 0 || directTopk > DIRECT_TOPK_LIMIT) {
        ProcessZeroOutput();
        return;
    }

    int64_t flatBatch = tiling_->batchsize;
    int64_t selectionHeadNum = tiling_->headnum;
    int64_t queryHeadNum = QueryHeadNum();
    int64_t headsPerKvHead = QueryHeadsPerKvHead();
    int64_t headDim = tiling_->headDim;
    int64_t kvDim = tiling_->kvCacheDim;
    int64_t ropeDim = tiling_->kRopeDim;
    int64_t kvDimFloatAlign = CeilAlign(kvDim, static_cast<int64_t>(BLOCK_BYTES / sizeof(float)));
    int64_t ropeDimFloatAlign = CeilAlign(ropeDim, static_cast<int64_t>(BLOCK_BYTES / sizeof(float)));
    int64_t maxDimFloatAlign = kvDimFloatAlign > ropeDimFloatAlign ? kvDimFloatAlign : ropeDimFloatAlign;
    int64_t directIoBytes = CeilAlign(headDim * static_cast<int64_t>(sizeof(T)), static_cast<int64_t>(BLOCK_BYTES));
    pipe_->InitBuffer(directIoQue_, 1, directIoBytes);
    pipe_->InitBuffer(gatherQue_, tiling_->buffNum, tiling_->gatherQueueUbSize);

    int64_t byteOffset = 0;
    LocalTensor<uint8_t> directLocal = directBuf_.Get<uint8_t>();
    LocalTensor<float> queryFloat = directLocal[byteOffset].template ReinterpretCast<float>();
    byteOffset += kvDimFloatAlign * static_cast<int64_t>(sizeof(float));
    LocalTensor<float> queryRopeFloat = directLocal[byteOffset].template ReinterpretCast<float>();
    byteOffset += ropeDimFloatAlign * static_cast<int64_t>(sizeof(float));
    LocalTensor<float> tmpFloat = directLocal[byteOffset].template ReinterpretCast<float>();
    byteOffset += maxDimFloatAlign * static_cast<int64_t>(sizeof(float));
    LocalTensor<float> outFloat = directLocal[byteOffset].template ReinterpretCast<float>();
    byteOffset += kvDimFloatAlign * static_cast<int64_t>(sizeof(float));
    LocalTensor<float> scoreLocal = directLocal[byteOffset].template ReinterpretCast<float>();
    byteOffset += DIRECT_TOPK_LIMIT * static_cast<int64_t>(sizeof(float));
    LocalTensor<int32_t> statLocal = directLocal[byteOffset].template ReinterpretCast<int32_t>();
    int64_t SH = rawSeq_ * selectionHeadNum;
    event_t gatherEvtId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    int64_t aivTaskIdx = 0;
    int64_t aivTaskNum = 1;
#ifndef ASCENDC_CPU_DEBUG
    if ((GetBlockIdx() % 2) != 0) {
        return;
    }
    aivTaskIdx = blkIdx_;
    aivTaskNum = static_cast<int64_t>(tiling_->usedCoreNum);
#endif

    for (int64_t globalBsIdx = aivTaskIdx; globalBsIdx < flatBatch; globalBsIdx += aivTaskNum) {
        int64_t curBatch = globalBsIdx / rawSeq_;
        int64_t curSeq = globalBsIdx % rawSeq_;
        for (int64_t hnIdx = 0; hnIdx < queryHeadNum; hnIdx++) {
            int64_t kvHeadIdx = KvHeadForQuery(hnIdx);
            bool updateSelectionCache = (hnIdx == kvHeadIdx * headsPerKvHead);
            int64_t queryOff = globalBsIdx * queryHeadNum * headDim + hnIdx * headDim;
            int64_t outOff = queryOff;
            int64_t topkBase = curBatch * SH * tiling_->topk +
                curSeq * selectionHeadNum * tiling_->topk + kvHeadIdx * tiling_->topk;
            int64_t blockTableBase = globalBsIdx * selectionHeadNum * tiling_->selMaxBlockNum +
                kvHeadIdx * tiling_->selMaxBlockNum;
            int64_t statusBase = curBatch * SH * (tiling_->topk + 1) +
                curSeq * selectionHeadNum * (tiling_->topk + 1) + kvHeadIdx * (tiling_->topk + 1);
            int64_t curFullKvSeqLen = fullKvActualSeqGm_.GetValue(curBatch);
            int64_t curSeqOffset = (rawSeq_ - 1) - curSeq;
            int64_t curFullKvSeqModify = curFullKvSeqLen - curSeqOffset;
            int32_t maxSelectionId = (curFullKvSeqModify > 0)
                ? static_cast<int32_t>(CeilDiv(curFullKvSeqModify, tiling_->selTopKBlockSize) - 1)
                : static_cast<int32_t>(-1);

            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            LocalTensor<T> ioT = directIoQue_.AllocTensor<T>();
            DataCopyExtParams queryCopyParams{static_cast<uint16_t>(1),
                static_cast<uint32_t>(kvDim * sizeof(T)), 0, 0, 0};
            DataCopyPad(ioT, queryGm_[queryOff], queryCopyParams, padParams);
            directIoQue_.EnQue(ioT);
            ioT = directIoQue_.DeQue<T>();
            pipe_barrier(PIPE_ALL);
            Cast(queryFloat, ioT, RoundMode::CAST_NONE, kvDim);
            pipe_barrier(PIPE_ALL);
            directIoQue_.FreeTensor(ioT);
            if (kvDim < kvDimFloatAlign) {
                Duplicate(queryFloat[kvDim], 0.0f, kvDimFloatAlign - kvDim);
                pipe_barrier(PIPE_V);
            }
            if (ropeDim > 0 && headDim > kvDim) {
                ioT = directIoQue_.AllocTensor<T>();
                DataCopyExtParams queryRopeCopyParams{static_cast<uint16_t>(1),
                    static_cast<uint32_t>(ropeDim * sizeof(T)), 0, 0, 0};
                DataCopyPad(ioT, queryGm_[queryOff + kvDim], queryRopeCopyParams, padParams);
                directIoQue_.EnQue(ioT);
                ioT = directIoQue_.DeQue<T>();
                pipe_barrier(PIPE_ALL);
                Cast(queryRopeFloat, ioT, RoundMode::CAST_NONE, ropeDim);
                pipe_barrier(PIPE_ALL);
                directIoQue_.FreeTensor(ioT);
                if (ropeDim < ropeDimFloatAlign) {
                    Duplicate(queryRopeFloat[ropeDim], 0.0f, ropeDimFloatAlign - ropeDim);
                    pipe_barrier(PIPE_V);
                }
            }

            int32_t kvAddrs[DIRECT_TOPK_LIMIT];
            int32_t validFlags[DIRECT_TOPK_LIMIT];
            float maxScore = -3.4028235e+38f;
            int32_t validCount = 0;
            int32_t selActualSeqLen = 0;
            for (int32_t t = 0; t < directTopk; t++) {
                kvAddrs[t] = 0;
                validFlags[t] = 0;
                statLocal.SetValue(t, selKvBlockStatusGm_.GetValue(statusBase + t));
                scoreLocal.SetValue(t, -3.4028235e+38f);
            }
            bool allowStatusHit = updateSelectionCache;

            int32_t chunkStart = 0;
            while (chunkStart < directTopk) {
                int32_t hitTopIdx[MAX_BLOCK_TOKENS], missTopIdx[MAX_BLOCK_TOKENS];
                int64_t hitSrcKvAddrs[MAX_BLOCK_TOKENS], hitSrcRopeAddrs[MAX_BLOCK_TOKENS];
                int64_t hitDstKvAddrs[MAX_BLOCK_TOKENS], hitDstRopeAddrs[MAX_BLOCK_TOKENS];
                int32_t hitCopyFlags[MAX_BLOCK_TOKENS];
                int64_t missFullKvAddrs[MAX_BLOCK_TOKENS], missFullRopeAddrs[MAX_BLOCK_TOKENS];
                int64_t missDstKvAddrs[MAX_BLOCK_TOKENS], missDstRopeAddrs[MAX_BLOCK_TOKENS];
                int64_t missGatherSizes[MAX_BLOCK_TOKENS];
                int32_t hitCount = 0;
                int32_t missCount = 0;

                int32_t chunkEnd = chunkStart + MAX_BLOCK_TOKENS;
                if (chunkEnd > directTopk) {
                    chunkEnd = directTopk;
                }
                for (int32_t t = chunkStart; t < chunkEnd; t++) {
                    int32_t topkId = selTopKIndicesGm_.GetValue(topkBase + t);
                    if (topkId < 0 || topkId > maxSelectionId) {
                        continue;
                    }

                    int64_t gatherBlockSize = (topkId == maxSelectionId)
                        ? (curFullKvSeqModify - static_cast<int64_t>(maxSelectionId) * tiling_->selTopKBlockSize)
                        : tiling_->selTopKBlockSize;
                    int32_t insertIdx = validCount++;
                    int64_t selKvBlkTableIdx =
                        (static_cast<int64_t>(insertIdx) * tiling_->selTopKBlockSize) / tiling_->selKvBlockSize;
                    int64_t selKvBlkSizeOff =
                        (static_cast<int64_t>(insertIdx) * tiling_->selTopKBlockSize) % tiling_->selKvBlockSize;
                    int32_t selKvBlockNumIdx = selKvBlockTableGm_.GetValue(blockTableBase + selKvBlkTableIdx);
                    if (selKvBlockNumIdx < 0) {
                        continue;
                    }
                    int64_t dstKvAddr = static_cast<int64_t>(selKvBlockNumIdx) * tiling_->selKvBlockSize * kvDim +
                        selKvBlkSizeOff * kvDim;
                    int64_t dstRopeAddr = static_cast<int64_t>(selKvBlockNumIdx) * tiling_->selKvBlockSize * ropeDim +
                        selKvBlkSizeOff * ropeDim;

                    bool isHit = false;
                    int32_t hitPos = -1;
                    if (!updateSelectionCache) {
                        isHit = true;
                        hitPos = insertIdx;
                    } else if (allowStatusHit) {
                        if (statLocal.GetValue(t) == topkId) {
                            isHit = true;
                            hitPos = t;
                        } else {
                            for (int32_t s = 0; s < directTopk; s++) {
                                if (statLocal.GetValue(s) == topkId) {
                                    isHit = true;
                                    hitPos = s;
                                    break;
                                }
                            }
                        }
                    }

                    if (isHit) {
                        int64_t srcSelKvBlkTableIdx =
                            (static_cast<int64_t>(hitPos) * tiling_->selTopKBlockSize) / tiling_->selKvBlockSize;
                        int64_t srcSelKvBlkSizeOff =
                            (static_cast<int64_t>(hitPos) * tiling_->selTopKBlockSize) % tiling_->selKvBlockSize;
                        int32_t srcSelKvBlockNumIdx =
                            selKvBlockTableGm_.GetValue(blockTableBase + srcSelKvBlkTableIdx);
                        if (srcSelKvBlockNumIdx < 0) {
                            continue;
                        }
                        hitTopIdx[hitCount] = t;
                        hitSrcKvAddrs[hitCount] =
                            static_cast<int64_t>(srcSelKvBlockNumIdx) * tiling_->selKvBlockSize * kvDim +
                            srcSelKvBlkSizeOff * kvDim;
                        hitSrcRopeAddrs[hitCount] =
                            static_cast<int64_t>(srcSelKvBlockNumIdx) * tiling_->selKvBlockSize * ropeDim +
                            srcSelKvBlkSizeOff * ropeDim;
                        hitDstKvAddrs[hitCount] = dstKvAddr;
                        hitDstRopeAddrs[hitCount] = dstRopeAddr;
                        hitCopyFlags[hitCount] = (hitPos != insertIdx) ? 1 : 0;
                        hitCount++;
                    } else {
                        int64_t kvBlkTableIdx =
                            (static_cast<int64_t>(topkId) * tiling_->selTopKBlockSize) / tiling_->fullKvBlockSize;
                        int64_t kvBlkSizeOff =
                            (static_cast<int64_t>(topkId) * tiling_->selTopKBlockSize) % tiling_->fullKvBlockSize;
                        int32_t kvBlockNumIdx =
                            fullKvBlockTableGm_.GetValue(curBatch * tiling_->fullMaxBlockNum + kvBlkTableIdx);
                        if (kvBlockNumIdx < 0) {
                            continue;
                        }
                        missTopIdx[missCount] = t;
                        missFullKvAddrs[missCount] =
                            static_cast<int64_t>(kvBlockNumIdx) * tiling_->fullKvBlockSize * kvDim +
                            kvBlkSizeOff * kvDim;
                        missFullRopeAddrs[missCount] =
                            static_cast<int64_t>(kvBlockNumIdx) * tiling_->fullKvBlockSize * ropeDim +
                            kvBlkSizeOff * ropeDim;
                        missDstKvAddrs[missCount] = dstKvAddr;
                        missDstRopeAddrs[missCount] = dstRopeAddr;
                        missGatherSizes[missCount] = gatherBlockSize;
                        missCount++;
                    }
                    if (updateSelectionCache) {
                        selActualSeqLen += static_cast<int32_t>(gatherBlockSize);
                        selKvBlockStatusGm_.SetValue(statusBase + insertIdx, topkId);
                    }
                    kvAddrs[t] = static_cast<int32_t>(dstKvAddr);
                    validFlags[t] = 1;
                }

                int32_t gatherIdx = 0;
                int32_t hitIdx = 0;
                bool gatherInFlight = false;
                if (missCount > 0) {
                    IssueSingleGather(missFullKvAddrs[0], missFullRopeAddrs[0],
                                      missDstKvAddrs[0], missDstRopeAddrs[0],
                                      missGatherSizes[0]);
                    gatherInFlight = true;
                    gatherIdx = 1;
                }
                while (hitIdx < hitCount || gatherIdx < missCount) {
                    if (hitIdx < hitCount) {
                        int32_t t = hitTopIdx[hitIdx];
                        float score = ComputeDirectTokenScore(
                            hitSrcKvAddrs[hitIdx], hitSrcRopeAddrs[hitIdx],
                            hitDstKvAddrs[hitIdx], hitDstRopeAddrs[hitIdx],
                            true, hitCopyFlags[hitIdx] != 0,
                            queryFloat, queryRopeFloat, tmpFloat);
                        scoreLocal.SetValue(t, score);
                        if (score > maxScore) {
                            maxScore = score;
                        }
                        hitIdx++;
                    }
                    if (gatherIdx < missCount) {
                        if (gatherInFlight) {
                            SetFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                            WaitFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                        }
                        IssueSingleGather(missFullKvAddrs[gatherIdx], missFullRopeAddrs[gatherIdx],
                                          missDstKvAddrs[gatherIdx], missDstRopeAddrs[gatherIdx],
                                          missGatherSizes[gatherIdx]);
                        gatherInFlight = true;
                        gatherIdx++;
                    }
                }
                if (gatherInFlight) {
                    SetFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                    WaitFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                }

                for (int32_t missIdx = 0; missIdx < missCount; missIdx++) {
                    int32_t t = missTopIdx[missIdx];
                    float score = ComputeDirectTokenScore(
                        missDstKvAddrs[missIdx], missDstRopeAddrs[missIdx],
                        missDstKvAddrs[missIdx], missDstRopeAddrs[missIdx],
                        true, false,
                        queryFloat, queryRopeFloat, tmpFloat);
                    scoreLocal.SetValue(t, score);
                    if (score > maxScore) {
                        maxScore = score;
                    }
                }
                chunkStart = chunkEnd;
            }

            if (maxScore < -3.0e38f) {
                Duplicate(outFloat, 0.0f, kvDimFloatAlign);
                pipe_barrier(PIPE_V);
                ioT = directIoQue_.AllocTensor<T>();
                Cast(ioT, outFloat, RoundMode::CAST_ROUND, kvDimFloatAlign);
                pipe_barrier(PIPE_ALL);
                DataCopyExtParams zeroKvOutParams{static_cast<uint16_t>(1),
                    static_cast<uint32_t>(kvDim * sizeof(T)), 0, 0, 0};
                DataCopyPad(attentionOutGm_[outOff], ioT, zeroKvOutParams);
                directIoQue_.FreeTensor(ioT);
                pipe_barrier(PIPE_ALL);
                continue;
            }

            for (int32_t t = 0; t < directTopk; t++) {
                float score = scoreLocal.GetValue(t);
                scoreLocal.SetValue(t, (score < -3.0e38f) ? -80.0f : (score - maxScore));
            }
            pipe_barrier(PIPE_ALL);
            Exp(scoreLocal, scoreLocal, directTopk);
            pipe_barrier(PIPE_ALL);

            float sum = 0.0f;
            for (int32_t t = 0; t < directTopk; t++) {
                if (validFlags[t] != 0) {
                    sum += scoreLocal.GetValue(t);
                } else {
                    scoreLocal.SetValue(t, 0.0f);
                }
            }
            if (sum <= 0.0f || sum != sum) {
                sum = 1.0f;
            }
            float invSum = 1.0f / sum;

            Duplicate(outFloat, 0.0f, kvDimFloatAlign);
            pipe_barrier(PIPE_V);
            for (int32_t t = 0; t < directTopk; t++) {
                if (validFlags[t] == 0) {
                    continue;
                }
                float weight = scoreLocal.GetValue(t) * invSum;
                ioT = directIoQue_.AllocTensor<T>();
                DataCopyExtParams vCopyParams{static_cast<uint16_t>(1),
                    static_cast<uint32_t>(kvDim * sizeof(T)), 0, 0, 0};
                DataCopyPad(ioT, selKvCacheGm_[static_cast<int64_t>(kvAddrs[t])], vCopyParams, padParams);
                directIoQue_.EnQue(ioT);
                ioT = directIoQue_.DeQue<T>();
                pipe_barrier(PIPE_ALL);
                Cast(tmpFloat, ioT, RoundMode::CAST_NONE, kvDim);
                pipe_barrier(PIPE_ALL);
                directIoQue_.FreeTensor(ioT);
                if (kvDim < kvDimFloatAlign) {
                    Duplicate(tmpFloat[kvDim], 0.0f, kvDimFloatAlign - kvDim);
                    pipe_barrier(PIPE_V);
                }
                Muls(tmpFloat, tmpFloat, weight, kvDimFloatAlign);
                pipe_barrier(PIPE_V);
                Add(outFloat, outFloat, tmpFloat, kvDimFloatAlign);
                pipe_barrier(PIPE_V);
            }
            pipe_barrier(PIPE_ALL);

            ioT = directIoQue_.AllocTensor<T>();
            Cast(ioT, outFloat, RoundMode::CAST_ROUND, kvDimFloatAlign);
            pipe_barrier(PIPE_ALL);
            DataCopyExtParams outParams{static_cast<uint16_t>(1),
                static_cast<uint32_t>(kvDim * sizeof(T)), 0, 0, 0};
            DataCopyPad(attentionOutGm_[outOff], ioT, outParams);
            directIoQue_.FreeTensor(ioT);
            pipe_barrier(PIPE_ALL);

            if (updateSelectionCache) {
                for (int32_t t = validCount; t < directTopk; t++) {
                    selKvBlockStatusGm_.SetValue(statusBase + t, -1);
                }
                selKvBlockStatusGm_.SetValue(statusBase + directTopk, selActualSeqLen);
                selKvActualSeqGm_.SetValue(globalBsIdx * selectionHeadNum + kvHeadIdx, selActualSeqLen);
                pipe_barrier(PIPE_ALL);
            }
        }
    }
#endif
}

// ============================================================================
// Process: main loop (dual-core)
// AIV: control flow + hit/miss classification + gather + softmax
// AIC: Matmul only (reads token info from GM workspace written by AIV)
// ============================================================================
template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::Process()
{
    if (blkIdx_ >= tiling_->usedCoreNum) return;

#ifndef ASCENDC_CPU_DEBUG
#endif

#ifndef ASCENDC_CPU_DEBUG
#endif

#ifndef ASCENDC_CPU_DEBUG
    bool useCubeOverlap = FSA_ENABLE_AIC_OVERLAP && (tiling_->enableOverlap != 0) &&
        (tiling_->topk > 0) &&
        (tiling_->topk <= 2048) &&
        (tiling_->headDim == tiling_->kvCacheDim + tiling_->kRopeDim) &&
        (tiling_->selTopKBlockSize > 0);
    bool useKernelOverlap = useCubeOverlap && (QueryHeadNum() == tiling_->headnum);

    if ASCEND_IS_AIC {
        if (useKernelOverlap) {
            ProcessAIC();
        }
        return;
    }
    if ASCEND_IS_AIV {
        if (!useKernelOverlap) {
            if (tiling_->topk <= 2048) {
                ProcessDirectTopk();
            } else {
                if (GetBlockIdx() == 0) {
                    ProcessZeroOutput();
                }
            }
            return;
        }
        if (tiling_->usedCoreNum <= 0) {
            if (tiling_->topk <= 2048) {
                ProcessDirectTopk();
            } else {
                ProcessZeroOutput();
            }
            return;
        }
    }

#endif // !ASCENDC_CPU_DEBUG

    // AIV path: full control flow. In MIX_AIC_1_2 both AIV lanes must
    // participate in CrossCore V->C flag posting; lane 0 does real work,
    // lane 1 mirrors synchronization only.
    bool syncOnlyAiv = false;
#ifndef ASCENDC_CPU_DEBUG
    syncOnlyAiv = (GetBlockIdx() % 2 != 0);
#endif

    // Allocate AIV-only buffers (must be inside kernel entry scope where ASCEND_IS_AIV works)
    {
        int64_t SH = rawSeq_ * tiling_->headnum;
#ifdef ASCENDC_CPU_DEBUG
        pipe_->InitBuffer(gatherBuf_, tiling_->gatherQueueUbSize);
        selTopKIdxUbSize_ = SH * topkSortAlign_ * sizeof(int32_t);
        pipe_->InitBuffer(selTopKIdxBuf_, selTopKIdxUbSize_);
#else
        pipe_->InitBuffer(gatherQue_, tiling_->buffNum, tiling_->gatherQueueUbSize);
        selTopKIdxUbSize_ = SH * topkSortAlign_ * sizeof(int32_t);
        pipe_->InitBuffer(selTopKIdxQue_, 1, selTopKIdxUbSize_);
#endif

        selKvBlockTableUbSize_ = CeilAlign(
            static_cast<int64_t>(SH * tiling_->selMaxBlockNum * sizeof(int32_t)),
            static_cast<int64_t>(BLOCK_BYTES));
        selKvActSeqUbSize_ = CeilAlign(
            static_cast<int64_t>(SH * sizeof(int32_t)),
            static_cast<int64_t>(BLOCK_BYTES));
        selBlockStatUbSize_ = SH * topkOneSortAlign_ * sizeof(int32_t);
        int64_t hitFlagSize = topkSortAlign_ * sizeof(int32_t);
        int64_t sortBufSize = topkSortAlign_ * sizeof(int32_t) * 4;
        pipe_->InitBuffer(workBuf_, selKvBlockTableUbSize_ + selKvActSeqUbSize_ +
                          selBlockStatUbSize_ + hitFlagSize + sortBufSize);
        pipe_->InitBuffer(commLocalBuf_, FSA_COMM_BUF_BYTES);
    }

    if (syncOnlyAiv) {
        vectorService_.Init(pipe_, tiling_, workspaceBase_);
    }
#ifdef ASCENDC_CPU_DEBUG
    // CPU sim: also init cube service in AIV path so we can call Matmul directly
    cubeService_.Init(pipe_, tiling_, workspaceBase_);
#endif

    LocalTensor<int32_t> workLocal = workBuf_.Get<int32_t>();
    LocalTensor<int32_t> selKvBlockTableLocal = workLocal;
    LocalTensor<int32_t> selKvActSeqLocal = selKvBlockTableLocal[selKvBlockTableUbSize_ / sizeof(int32_t)];
    LocalTensor<int32_t> selBlockStatLocal = selKvActSeqLocal[selKvActSeqUbSize_ / sizeof(int32_t)];
    hitFlagLocal_ = selBlockStatLocal[selBlockStatUbSize_ / sizeof(int32_t)];

    int32_t syncStage = 1;
    bool aicDoneRequestSent = false;
#ifndef ASCENDC_CPU_DEBUG
    if (!syncOnlyAiv) {
        commBufGm_.SetValue(COMM_STAGE, 0);
        commBufGm_.SetValue(COMM_ACK_STAGE, 0);
        commBufGm_.SetValue(COMM_TOKEN_COUNT, 0);
        DiagInitNextChunkPrefetchCounters();
        pipe_barrier(PIPE_ALL);
    }
#endif

#ifndef ASCENDC_CPU_DEBUG
    if constexpr (FSA_DIAG_AIC_DUMMY) {
        int32_t diagKvAddrs[1] = {0};
        int32_t diagRopeAddrs[1] = {0};
        int64_t diagMm2OutOff = mm2OutputGmOff_;
        int32_t diagMm1Stage = syncStage++;
        if (!syncOnlyAiv) {
            WriteCommBufAndSignalAIC(0, 0, diagMm2OutOff, diagKvAddrs, diagRopeAddrs, 1, true, diagMm1Stage, true);
        } else {
            WaitMm1StageOnSyncOnlyLane(diagMm1Stage);
        }
        WaitAICMm1Done(diagMm1Stage);
        return;
    }
#endif

    for (int64_t bsIdx = 0; bsIdx < bsLoopNum_; bsIdx++) {
        int64_t curBatchSize = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) / rawSeq_;
        int64_t curFullKvSeqLen = fullKvActualSeqGm_.GetValue(curBatchSize);
        if (curFullKvSeqLen <= 0) continue;

        // Copy in topk indices
#ifdef ASCENDC_CPU_DEBUG
        LocalTensor<int32_t> selTopKIdxLocal = selTopKIdxBuf_.Get<int32_t>();
#else
        LocalTensor<int32_t> selTopKIdxLocal = selTopKIdxQue_.AllocTensor<int32_t>();
#endif
        {
            int64_t SH = rawSeq_ * tiling_->headnum;
            int64_t curBatch = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) / rawSeq_;
            int64_t curSeq = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) % rawSeq_;
#ifdef ASCENDC_CPU_DEBUG
            // CPU sim: direct memcpy (DataCopy/DataCopyPad have position checks in tikicpulib)
            int64_t gmOff = curBatch * SH * tiling_->topk + curSeq * tiling_->headnum * tiling_->topk;
            int32_t copyCount = tiling_->headnum * tiling_->topk;
            for (int32_t ii = 0; ii < copyCount; ii++) {
                selTopKIdxLocal.SetValue(ii, selTopKIndicesGm_.GetValue(gmOff + ii));
            }
#else
            DataCopyPadExtParams<int32_t> topkPadParams{false, 0, 0, 0};
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(tiling_->headnum),
                static_cast<uint32_t>(tiling_->topk * sizeof(int32_t)), 0,
                static_cast<uint32_t>((topkSortAlign_ - topkAlign_) / (BLOCK_BYTES / sizeof(int32_t))), 0};
            DataCopyPad(selTopKIdxLocal,
                selTopKIndicesGm_[curBatch * SH * tiling_->topk + curSeq * tiling_->headnum * tiling_->topk],
                copyParams, topkPadParams);
#endif
        }
#ifndef ASCENDC_CPU_DEBUG
        selTopKIdxQue_.EnQue(selTopKIdxLocal);
        selTopKIdxLocal = selTopKIdxQue_.DeQue<int32_t>();
#endif

        // Copy in block status and block table
        {
            int64_t SH = rawSeq_ * tiling_->headnum;
            int64_t curBatch = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) / rawSeq_;
            int64_t curSeq = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) % rawSeq_;

#ifdef ASCENDC_CPU_DEBUG
            // CPU sim: element-wise copy via GetValue/SetValue
            int64_t statGmOff = curBatch * SH * (tiling_->topk + 1) + curSeq * tiling_->headnum * (tiling_->topk + 1);
            for (int32_t ii = 0; ii < tiling_->headnum * (tiling_->topk + 1); ii++) {
                selBlockStatLocal.SetValue(ii, selKvBlockStatusGm_.GetValue(statGmOff + ii));
            }

            int64_t btGmOff = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) * tiling_->headnum * tiling_->selMaxBlockNum;
            for (int32_t ii = 0; ii < tiling_->headnum * tiling_->selMaxBlockNum; ii++) {
                selKvBlockTableLocal.SetValue(ii, selKvBlockTableGm_.GetValue(btGmOff + ii));
            }
#else
            DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

            // Block status
            uint8_t padCnt = topkOneAlign_ - (tiling_->topk + 1);
            DataCopyPadExtParams<int32_t> padParamsSt{true, 0, padCnt, -1};
            uint32_t dstStride = (topkOneSortAlign_ - topkOneAlign_) / (BLOCK_BYTES / sizeof(int32_t));
            DataCopyExtParams statParams{
                static_cast<uint16_t>(tiling_->headnum),
                static_cast<uint32_t>((tiling_->topk + 1) * sizeof(int32_t)), 0, dstStride, 0};
            DataCopyPad(selBlockStatLocal,
                selKvBlockStatusGm_[curBatch * SH * (tiling_->topk + 1) + curSeq * tiling_->headnum * (tiling_->topk + 1)],
                statParams, padParamsSt);

            // Block table
            DataCopyExtParams btParams{
                static_cast<uint16_t>(1),
                static_cast<uint32_t>(tiling_->headnum * tiling_->selMaxBlockNum * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(selKvBlockTableLocal,
                selKvBlockTableGm_[(blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) * tiling_->headnum * tiling_->selMaxBlockNum],
                btParams, padParams);
#endif
        }
        pipe_barrier(PIPE_ALL);

        int64_t curSeq = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) % rawSeq_;
        int64_t offset = (rawSeq_ - 1) - curSeq;
        int64_t curFullKvSeqModify = curFullKvSeqLen - offset;

        if (curFullKvSeqModify > 0) {
            int64_t queryHeadNum = QueryHeadNum();
            int64_t headsPerKvHead = QueryHeadsPerKvHead();
            bool isLastBsLoop = (bsIdx + 1 == bsLoopNum_);
            for (int64_t queryHeadIdx = 0; queryHeadIdx < queryHeadNum; queryHeadIdx++) {
                int64_t kvHeadIdx = KvHeadForQuery(queryHeadIdx);
                bool updateSelectionCache = (queryHeadIdx == kvHeadIdx * headsPerKvHead);
                bool isLastQueryHead = (queryHeadIdx + 1 == queryHeadNum);
                bool isLastAicRequestCandidate = isLastBsLoop && isLastQueryHead;
                int32_t syncStageBeforeHead = syncStage;
                ProcessOneHead(bsIdx, curSeq, kvHeadIdx, queryHeadIdx, curFullKvSeqModify,
                    updateSelectionCache, isLastAicRequestCandidate,
                    selBlockStatLocal, selTopKIdxLocal, selKvBlockTableLocal, selKvActSeqLocal,
                    syncOnlyAiv, syncStage);
                if (isLastAicRequestCandidate && syncStage != syncStageBeforeHead) {
                    aicDoneRequestSent = true;
                }
            }
            if (!syncOnlyAiv) {
                DataCopyExtParams seqOutParams{static_cast<uint16_t>(1),
                    static_cast<uint32_t>(tiling_->headnum * sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(selKvActualSeqGm_[(curBatchSize * rawSeq_ + curSeq) * tiling_->headnum],
                    selKvActSeqLocal, seqOutParams);

                uint32_t srcStride = (topkOneSortAlign_ - topkOneAlign_) / (BLOCK_BYTES / sizeof(int32_t));
                DataCopyExtParams statOutParams{static_cast<uint16_t>(tiling_->headnum),
                    static_cast<uint32_t>((tiling_->topk + 1) * sizeof(int32_t)), srcStride, 0, 0};
                int64_t statOutOff = curBatchSize * rawSeq_ * tiling_->headnum * (tiling_->topk + 1) +
                    curSeq * tiling_->headnum * (tiling_->topk + 1);
                DataCopyPad(selKvBlockStatusGm_[statOutOff], selBlockStatLocal, statOutParams);
            }
        }

#ifndef ASCENDC_CPU_DEBUG
        selTopKIdxQue_.FreeTensor(selTopKIdxLocal);
#endif
    }

#ifndef ASCENDC_CPU_DEBUG
    if (!aicDoneRequestSent) {
        int32_t doneStage = syncStage++;
        if (!syncOnlyAiv) {
            WriteDoneAndSignalAIC(doneStage);
        } else {
            WaitMm1StageOnSyncOnlyLane(doneStage);
        }
    }
#endif
    (void)syncStage;
}

// ============================================================================
// ProcessOneHead: block loop with hit/miss + gather overlap + Matmul + Softmax
// ============================================================================
template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::ProcessOneHead(
    int64_t bsIdx, int64_t seqIdx, int64_t kvHeadIdx, int64_t queryHeadIdx,
    int64_t curFullKvSeqModify, bool updateSelectionCache, bool isLastAicRequestCandidate,
    LocalTensor<int32_t>& selBlockStatLocal, LocalTensor<int32_t>& topkIdxLocal,
    LocalTensor<int32_t>& selKvBlockTableLocal, LocalTensor<int32_t>& selKvActSeqLocal,
    bool syncOnly, int32_t& syncStage)
{
    int64_t selBlkTableOff = kvHeadIdx * tiling_->selMaxBlockNum;
    int32_t maxSelectionId = CeilDiv(curFullKvSeqModify, tiling_->selTopKBlockSize) - 1;
    int64_t lastGatherBlockSize = curFullKvSeqModify - maxSelectionId * tiling_->selTopKBlockSize;
    int64_t globalBsIdx = blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx;
    int64_t curBatch = globalBsIdx / rawSeq_;

    LocalTensor<int32_t> curStatLocal = selBlockStatLocal[kvHeadIdx * topkOneSortAlign_];
    LocalTensor<int32_t> topkLocal = topkIdxLocal[kvHeadIdx * topkSortAlign_];

    // Hit detection is done inline during classification. Build a compact
    // open-addressing map in the workBuf scratch area so large topk does not
    // pay an O(topk^2) status scan.
    bool hasHistoricalStatus = updateSelectionCache && curStatLocal.GetValue(tiling_->topk) > 0;
    bool allowStatusHit = updateSelectionCache;
    int32_t statusHashSlots = static_cast<int32_t>(topkSortAlign_ * 2);
    LocalTensor<int32_t> statusHashKeyLocal = hitFlagLocal_;
    LocalTensor<int32_t> statusHashPosLocal = statusHashKeyLocal[statusHashSlots];
    if (allowStatusHit) {
        Duplicate(statusHashKeyLocal, static_cast<int32_t>(-1), statusHashSlots);
        Duplicate(statusHashPosLocal, static_cast<int32_t>(-1), statusHashSlots);
        PipeBarrier<PIPE_V>();
        if (hasHistoricalStatus) {
            for (int64_t statIdx = 0; statIdx < tiling_->topk; statIdx++) {
                int32_t statVal = curStatLocal.GetValue(statIdx);
                if (statVal < 0) {
                    continue;
                }
                InsertStatusHash(statVal, static_cast<int32_t>(statIdx),
                    statusHashKeyLocal, statusHashPosLocal, statusHashSlots);
            }
        }
        PipeBarrier<PIPE_V>();
    }

#ifdef ASCENDC_CPU_DEBUG
    printf("[CPU] ProcessOneHead: hn=%ld maxSelId=%d curFullKvSeq=%ld\n",
           queryHeadIdx, maxSelectionId, curFullKvSeqModify);
    // Check topk indices
    printf("[CPU] topk[0..3] = %d %d %d %d\n",
           topkLocal.GetValue(0), topkLocal.GetValue(1),
           topkLocal.GetValue(2), topkLocal.GetValue(3));
    // Check block status
    printf("[CPU] blockStat[0..3] = %d %d %d %d\n",
           curStatLocal.GetValue(0), curStatLocal.GetValue(1),
           curStatLocal.GetValue(2), curStatLocal.GetValue(3));
#endif

    // Query GM offset for this head
    int64_t queryHeadNum = QueryHeadNum();
    int64_t queryGmOff = globalBsIdx * queryHeadNum * tiling_->headDim + queryHeadIdx * tiling_->headDim;

    // Block loop: process all topk tokens in blocks
    int32_t insertIdx = 0;
    int32_t selActualSeqLen = 0;
    bool isFirstBlock = true;
    int64_t scoreSlotStride = static_cast<int64_t>(16) *
        CeilAlign(static_cast<int64_t>(MAX_BLOCK_TOKENS), static_cast<int64_t>(16));
    int64_t scoreGmOffset = (bsIdx * queryHeadNum + queryHeadIdx) * scoreSlotStride;

#ifndef ASCENDC_CPU_DEBUG
    event_t gatherEvtId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
#endif

    int64_t topkIdx = 0;
    while (topkIdx < tiling_->topk) {
        // Classify hit/miss for this block (up to MAX_BLOCK_TOKENS)
        int32_t hitKvAddrs[MAX_BLOCK_TOKENS], missKvAddrs[MAX_BLOCK_TOKENS];
        int32_t hitRopeAddrs[MAX_BLOCK_TOKENS], missRopeAddrs[MAX_BLOCK_TOKENS];
        int32_t hitSourcePositions[MAX_BLOCK_TOKENS];
        int64_t hitDstKvAddrs[MAX_BLOCK_TOKENS], hitDstRopeAddrs[MAX_BLOCK_TOKENS];
        int64_t hitGatherSizes[MAX_BLOCK_TOKENS];
        int32_t hitCopyFlags[MAX_BLOCK_TOKENS];
        int64_t missFullKvAddrs[MAX_BLOCK_TOKENS], missFullRopeAddrs[MAX_BLOCK_TOKENS];
        int64_t missSelKvAddrs[MAX_BLOCK_TOKENS], missSelRopeAddrs[MAX_BLOCK_TOKENS];
        int64_t missGatherSizes[MAX_BLOCK_TOKENS];
        int32_t missGatherReady[MAX_BLOCK_TOKENS], missGatherDone[MAX_BLOCK_TOKENS];
        int32_t orderedKvAddrs[MAX_BLOCK_TOKENS], orderedRopeAddrs[MAX_BLOCK_TOKENS];
        int32_t orderedCount = 0;
        int32_t hitCount = 0, missCount = 0;
        int32_t gIdx = 0;
        bool missGatherSynced = true;

        for (; topkIdx < tiling_->topk && (hitCount + missCount) < MAX_BLOCK_TOKENS; topkIdx++) {
            int32_t topKId = topkLocal.GetValue(topkIdx);
            if (topKId < 0 || topKId > maxSelectionId) continue;

            int64_t gatherBlockSize = (topKId == maxSelectionId) ? lastGatherBlockSize : tiling_->selTopKBlockSize;

            // Compute sel_kv_cache address for this token
            int64_t selKvBlkTableIdx = (insertIdx * tiling_->selTopKBlockSize) / tiling_->selKvBlockSize;
            int64_t selKvBlkSizeOff = (insertIdx * tiling_->selTopKBlockSize) % tiling_->selKvBlockSize;
            int32_t selKvBlockNumIdx = selKvBlockTableLocal.GetValue(selBlkTableOff + selKvBlkTableIdx);
            if (selKvBlockNumIdx < 0) {
                continue;
            }
            int64_t selKRopeAddr = selKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kRopeDim
                                   + selKvBlkSizeOff * tiling_->kRopeDim;
            int64_t selKvCacheAddr = selKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kvCacheDim
                                     + selKvBlkSizeOff * tiling_->kvCacheDim;

            bool isHit = false;
            int32_t hitPos = -1;
            if (!updateSelectionCache) {
                isHit = true;
                hitPos = insertIdx;
            } else if (curStatLocal.GetValue(insertIdx) == topKId) {
                isHit = true;
                hitPos = insertIdx;
            } else if (curStatLocal.GetValue(topkIdx) == topKId) {
                isHit = true;
                hitPos = static_cast<int32_t>(topkIdx);
            } else if (allowStatusHit) {
                int32_t slot = topKId % statusHashSlots;
                for (int32_t probe = 0; probe < statusHashSlots; probe++) {
                    int32_t curKey = statusHashKeyLocal.GetValue(slot);
                    if (curKey == topKId) {
                        isHit = true;
                        hitPos = statusHashPosLocal.GetValue(slot);
                        break;
                    }
                    if (curKey < 0) {
                        break;
                    }
                    slot++;
                    if (slot == statusHashSlots) {
                        slot = 0;
                    }
                }
            }

            if (isHit) {
                int64_t srcSelKvBlkTableIdx =
                    (static_cast<int64_t>(hitPos) * tiling_->selTopKBlockSize) / tiling_->selKvBlockSize;
                int64_t srcSelKvBlkSizeOff =
                    (static_cast<int64_t>(hitPos) * tiling_->selTopKBlockSize) % tiling_->selKvBlockSize;
                int32_t srcSelKvBlockNumIdx = selKvBlockTableLocal.GetValue(selBlkTableOff + srcSelKvBlkTableIdx);
                if (srcSelKvBlockNumIdx < 0) {
                    continue;
                }
                int64_t srcKRopeAddr = srcSelKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kRopeDim
                                       + srcSelKvBlkSizeOff * tiling_->kRopeDim;
                int64_t srcKvCacheAddr = srcSelKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kvCacheDim
                                         + srcSelKvBlkSizeOff * tiling_->kvCacheDim;
                hitKvAddrs[hitCount] = static_cast<int32_t>(srcKvCacheAddr);
                hitRopeAddrs[hitCount] = static_cast<int32_t>(srcKRopeAddr);
                hitSourcePositions[hitCount] = hitPos;
                hitDstKvAddrs[hitCount] = selKvCacheAddr;
                hitDstRopeAddrs[hitCount] = selKRopeAddr;
                hitGatherSizes[hitCount] = gatherBlockSize;
                hitCopyFlags[hitCount] = (hitPos != insertIdx) ? 1 : 0;
                orderedKvAddrs[orderedCount] = hitKvAddrs[hitCount];
                orderedRopeAddrs[orderedCount] = hitRopeAddrs[hitCount];
                orderedCount++;
                hitCount++;
            } else {
                missKvAddrs[missCount] = static_cast<int32_t>(selKvCacheAddr);
                missRopeAddrs[missCount] = static_cast<int32_t>(selKRopeAddr);

                // Compute full_kv_cache address for gather
                int64_t kvBlkTableIdx = (topKId * tiling_->selTopKBlockSize) / tiling_->fullKvBlockSize;
                int64_t kvBlkSizeOff = (topKId * tiling_->selTopKBlockSize) % tiling_->fullKvBlockSize;
                int32_t kvBlockNumIdx = fullKvBlockTableGm_.GetValue(curBatch * tiling_->fullMaxBlockNum + kvBlkTableIdx);
                if (kvBlockNumIdx < 0) {
                    continue;
                }
                missFullKvAddrs[missCount] = kvBlockNumIdx * tiling_->fullKvBlockSize * tiling_->kvCacheDim
                                             + kvBlkSizeOff * tiling_->kvCacheDim;
                missFullRopeAddrs[missCount] = kvBlockNumIdx * tiling_->fullKvBlockSize * tiling_->kRopeDim
                                               + kvBlkSizeOff * tiling_->kRopeDim;
                missSelKvAddrs[missCount] = selKvCacheAddr;
                missSelRopeAddrs[missCount] = selKRopeAddr;
                missGatherSizes[missCount] = gatherBlockSize;
                missGatherReady[missCount] = 1;
                missGatherDone[missCount] = 0;
                orderedKvAddrs[orderedCount] = missKvAddrs[missCount];
                orderedRopeAddrs[orderedCount] = missRopeAddrs[missCount];
                orderedCount++;
                missCount++;
            }
            if (updateSelectionCache) {
                selActualSeqLen += static_cast<int32_t>(gatherBlockSize);
                curStatLocal.SetValue(insertIdx, topKId);
                if (allowStatusHit) {
                    InsertStatusHash(topKId, insertIdx,
                        statusHashKeyLocal, statusHashPosLocal, statusHashSlots);
                }
            }
            insertIdx++;
        }

        if (hitCount + missCount == 0) continue;

        const bool isLastHead = isLastAicRequestCandidate;
        const bool isLastBlockForHead = (topkIdx >= tiling_->topk);
        if constexpr (FSA_DIAG_STOP_BEFORE_NONFIRST_CHUNK) {
            if (!isFirstBlock) {
#ifndef ASCENDC_CPU_DEBUG
                int32_t doneStage = syncStage++;
                if (!syncOnly) {
                    WriteDoneAndSignalAIC(doneStage);
                } else {
                    WaitMm1StageOnSyncOnlyLane(doneStage);
                }
#endif
                return;
            }
        }
        bool combineHitMissRequest = (hitCount > 0 && missCount > 0) && !FSA_ENABLE_SPLIT_HIT_MISS_OVERLAP;
        bool currentHitSourcesSafeForPrefetch = true;
        for (int32_t h = 0; h < hitCount; h++) {
            if (hitSourcePositions[h] >= insertIdx) {
                currentHitSourcesSafeForPrefetch = false;
                break;
            }
        }
        bool currentHitCopiesSafeForPrefetch = true;
        for (int32_t h = 0; h < hitCount; h++) {
            if (hitCopyFlags[h] != 0) {
                currentHitCopiesSafeForPrefetch = false;
                break;
            }
        }

        for (int32_t m = 0; m < missCount; m++) {
            bool gatherSafe = true;
            int64_t missKvBegin = missSelKvAddrs[m];
            int64_t missKvEnd = missKvBegin + missGatherSizes[m] * tiling_->kvCacheDim;
            int64_t missRopeBegin = missSelRopeAddrs[m];
            int64_t missRopeEnd = missRopeBegin + missGatherSizes[m] * tiling_->kRopeDim;
            for (int32_t h = 0; h < hitCount; h++) {
                int64_t hitKvBegin = hitKvAddrs[h];
                int64_t hitKvEnd = hitKvBegin + hitGatherSizes[h] * tiling_->kvCacheDim;
                int64_t hitRopeBegin = hitRopeAddrs[h];
                int64_t hitRopeEnd = hitRopeBegin + hitGatherSizes[h] * tiling_->kRopeDim;
                bool kvOverlap = (missKvBegin < hitKvEnd) && (hitKvBegin < missKvEnd);
                bool ropeOverlap = (tiling_->kRopeDim > 0) &&
                    (missRopeBegin < hitRopeEnd) && (hitRopeBegin < missRopeEnd);
                if (kvOverlap || ropeOverlap) {
                    gatherSafe = false;
                    break;
                }
            }
            missGatherReady[m] = gatherSafe ? 1 : 0;
            missGatherDone[m] = 0;
        }

        bool useDirectSourceOverlap = false;
#ifndef ASCENDC_CPU_DEBUG
        useDirectSourceOverlap = FSA_ENABLE_DIRECT_MISS_SOURCE_OVERLAP &&
            (missCount > 0) && !FSA_DIAG_FAKE_AIC;
        if (useDirectSourceOverlap) {
            for (int32_t h = 0; h < hitCount; h++) {
                if (hitCopyFlags[h] != 0) {
                    useDirectSourceOverlap = false;
                    break;
                }
            }
        }
        if (useDirectSourceOverlap) {
            for (int32_t m = 0; m < missCount; m++) {
                if (missGatherReady[m] == 0) {
                    useDirectSourceOverlap = false;
                    break;
                }
            }
        }
        if (useDirectSourceOverlap) {
            int64_t kvDimFloatAlign = CeilAlign(tiling_->kvCacheDim,
                static_cast<int64_t>(BLOCK_BYTES / sizeof(float)));
            int64_t mm2OutOff = mm2OutputGmOff_ +
                (bsIdx * queryHeadNum + queryHeadIdx) * Mm2HeadStride(kvDimFloatAlign);
            int64_t prefetchScanTopkIdx = topkIdx;
            int32_t prefetchScanInsertIdx = insertIdx;
            int32_t prefetchHitCopyStableSourceEnd = prefetchScanInsertIdx;
            int32_t prefetchChunkInsertEnd = insertIdx + MAX_BLOCK_TOKENS;
            if (prefetchChunkInsertEnd > tiling_->topk) {
                prefetchChunkInsertEnd = static_cast<int32_t>(tiling_->topk);
            }
            int32_t combinedKvAddrs[MAX_BLOCK_TOKENS];
            int32_t combinedRopeAddrs[MAX_BLOCK_TOKENS];
            int32_t combinedFullKvAddrs[MAX_BLOCK_TOKENS];
            int32_t combinedFullRopeAddrs[MAX_BLOCK_TOKENS];
            int32_t sourceFlags[MAX_BLOCK_TOKENS];
            int32_t combinedCount = 0;
            for (int32_t h = 0; h < hitCount; h++) {
                combinedKvAddrs[combinedCount] = hitKvAddrs[h];
                combinedRopeAddrs[combinedCount] = hitRopeAddrs[h];
                combinedFullKvAddrs[combinedCount] = 0;
                combinedFullRopeAddrs[combinedCount] = 0;
                sourceFlags[combinedCount] = 0;
                combinedCount++;
            }
            for (int32_t m = 0; m < missCount; m++) {
                combinedKvAddrs[combinedCount] = missKvAddrs[m];
                combinedRopeAddrs[combinedCount] = missRopeAddrs[m];
                combinedFullKvAddrs[combinedCount] = static_cast<int32_t>(missFullKvAddrs[m]);
                combinedFullRopeAddrs[combinedCount] = static_cast<int32_t>(missFullRopeAddrs[m]);
                sourceFlags[combinedCount] = 1;
                combinedCount++;
            }

            int32_t mm1Stage = syncStage++;
            if (!syncOnly) {
                WriteCommBufAndSignalAIC(queryGmOff, scoreGmOffset, mm2OutOff,
                    combinedKvAddrs, combinedRopeAddrs, combinedCount, isFirstBlock, mm1Stage,
                    isLastHead && isLastBlockForHead, FSA_DIRECT_SOURCE_MODE, sourceFlags,
                    combinedFullKvAddrs, combinedFullRopeAddrs);
            } else {
                WaitMm1StageOnSyncOnlyLane(mm1Stage);
            }
            int32_t directOutstandingGatherCount = 0;
            if (!syncOnly) {
                for (int32_t m = 0; m < missCount; m++) {
                    IssueSingleGather(missFullKvAddrs[m], missFullRopeAddrs[m],
                                      missSelKvAddrs[m], missSelRopeAddrs[m],
                                      missGatherSizes[m]);
                    missGatherSynced = false;
                    missGatherDone[m] = 1;
                    directOutstandingGatherCount++;
                }
                DiagRecordNextChunkPrefetchCounterMax(
                    FSA_DIAG_DIRECT_CURRENT_GATHER_MAX_SLOT, directOutstandingGatherCount);
            }
            if (!syncOnly && updateSelectionCache && currentHitSourcesSafeForPrefetch &&
                !FSA_DIAG_FAKE_AIC) {
                int32_t directPrefetched = PrefetchNextChunkGathers(topkIdx, prefetchChunkInsertEnd,
                    prefetchScanTopkIdx, prefetchScanInsertIdx, curBatch,
                    selBlkTableOff, maxSelectionId, lastGatherBlockSize,
                    curStatLocal, topkLocal, selKvBlockTableLocal,
                    updateSelectionCache, allowStatusHit,
                    statusHashKeyLocal, statusHashPosLocal, statusHashSlots,
                    static_cast<int32_t>(gatherEvtId), missGatherSynced,
                    prefetchHitCopyStableSourceEnd,
                    FSA_NEXT_CHUNK_PREFETCH_DIRECT_MM1_LIMIT,
                    FSA_PREFETCH_SOURCE_DIRECT_MM1, true);
                directOutstandingGatherCount += directPrefetched;
                if (missGatherSynced && directOutstandingGatherCount > 0) {
                    DiagRecordNextChunkPrefetchCounterMax(
                        FSA_DIAG_DIRECT_BATCHED_GATHER_MAX_SLOT, directOutstandingGatherCount);
                    directOutstandingGatherCount = 0;
                }
            }
            WaitAICMm1Done(mm1Stage);
            if (syncOnly) {
                if constexpr (!FSA_DIAG_AIC_DUMMY && !FSA_DIAG_FAKE_AIC) {
                    vectorService_.ProcessSoftmaxAndRescale(combinedCount, scoreGmOffset,
                        isFirstBlock, tiling_->scaleValue, mm2OutOff, kvDimFloatAlign);
                    pipe_barrier(PIPE_ALL);
                }
            }
            int32_t mm2Stage = syncStage++;
            if (syncOnly) {
                SignalAICStage(mm2Stage);
            } else {
                WaitMm2StageOnWorkerLane(mm2Stage);
                if (updateSelectionCache && currentHitSourcesSafeForPrefetch &&
                    !FSA_DIAG_FAKE_AIC) {
                    int32_t directPrefetched = PrefetchNextChunkGathers(topkIdx, prefetchChunkInsertEnd,
                        prefetchScanTopkIdx, prefetchScanInsertIdx, curBatch,
                        selBlkTableOff, maxSelectionId, lastGatherBlockSize,
                        curStatLocal, topkLocal, selKvBlockTableLocal,
                        updateSelectionCache, allowStatusHit,
                        statusHashKeyLocal, statusHashPosLocal, statusHashSlots,
                        static_cast<int32_t>(gatherEvtId), missGatherSynced,
                        prefetchHitCopyStableSourceEnd,
                        FSA_NEXT_CHUNK_PREFETCH_DIRECT_MM2_LIMIT,
                        FSA_PREFETCH_SOURCE_DIRECT_MM2);
                    directOutstandingGatherCount += directPrefetched;
                    if (missGatherSynced && directOutstandingGatherCount > 0) {
                        DiagRecordNextChunkPrefetchCounterMax(
                            FSA_DIAG_DIRECT_BATCHED_GATHER_MAX_SLOT, directOutstandingGatherCount);
                        directOutstandingGatherCount = 0;
                    }
                }
            }
            WaitAICStageDone(mm2Stage);
            WaitAICMm2Done(mm2Stage);
            AccumulateAivMm2TempIfNeeded(syncOnly, isFirstBlock, mm2OutOff, kvDimFloatAlign);
            if (!syncOnly && !missGatherSynced) {
                SetFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                WaitFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                missGatherSynced = true;
            }
            if (!syncOnly && directOutstandingGatherCount > 0) {
                DiagRecordNextChunkPrefetchCounterMax(
                    FSA_DIAG_DIRECT_BATCHED_GATHER_MAX_SLOT, directOutstandingGatherCount);
                directOutstandingGatherCount = 0;
            }
            if (!syncOnly) {
                pipe_barrier(PIPE_ALL);
            }
            isFirstBlock = false;
            continue;
        }
#endif

        bool useFullOverlapRequest = false;
#ifndef ASCENDC_CPU_DEBUG
        useFullOverlapRequest = (FSA_ENABLE_SINGLE_REQUEST_FULL_OVERLAP || FSA_ENABLE_HIT_SCORE_PREFETCH_OVERLAP) &&
            combineHitMissRequest && !FSA_DIAG_FAKE_AIC &&
            (hitCount >= static_cast<int32_t>(FSA_N_SPLIT_SIZE));
        if (useFullOverlapRequest) {
            for (int32_t h = 0; h < hitCount; h++) {
                if (hitCopyFlags[h] != 0) {
                    useFullOverlapRequest = false;
                    break;
                }
            }
        }
        if (useFullOverlapRequest) {
            for (int32_t m = 0; m < missCount; m++) {
                if (missGatherReady[m] == 0) {
                    useFullOverlapRequest = false;
                    break;
                }
            }
        }
        if (useFullOverlapRequest) {
            int64_t kvDimFloatAlign = CeilAlign(tiling_->kvCacheDim,
                static_cast<int64_t>(BLOCK_BYTES / sizeof(float)));
            int64_t mm2OutOff = mm2OutputGmOff_ +
                (bsIdx * queryHeadNum + queryHeadIdx) * Mm2HeadStride(kvDimFloatAlign);
            int32_t combinedKvAddrs[MAX_BLOCK_TOKENS];
            int32_t combinedRopeAddrs[MAX_BLOCK_TOKENS];
            int32_t combinedCount = 0;
            for (int32_t h = 0; h < hitCount; h++) {
                combinedKvAddrs[combinedCount] = hitKvAddrs[h];
                combinedRopeAddrs[combinedCount] = hitRopeAddrs[h];
                combinedCount++;
            }
            for (int32_t m = 0; m < missCount; m++) {
                combinedKvAddrs[combinedCount] = missKvAddrs[m];
                combinedRopeAddrs[combinedCount] = missRopeAddrs[m];
                combinedCount++;
            }

            int32_t mm1Stage = syncStage++;
            int32_t missReadyStage = syncStage++;
            if (!syncOnly) {
                WriteCommBufAndSignalAIC(queryGmOff, scoreGmOffset, mm2OutOff,
                    combinedKvAddrs, combinedRopeAddrs, combinedCount, isFirstBlock, mm1Stage,
                    isLastHead && isLastBlockForHead, hitCount);
            } else {
                WaitMm1StageOnSyncOnlyLane(mm1Stage);
            }
            if (!syncOnly && FSA_DIAG_SIGNAL_MISS_READY_BEFORE_GATHER) {
                WriteMissReadyStage(missReadyStage);
            }
            if (!syncOnly && !FSA_DIAG_FAKE_AIC) {
                for (int32_t m = 0; m < missCount; m++) {
                    if (!missGatherSynced) {
                        SetFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                        WaitFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                        missGatherSynced = true;
                    }
                    IssueSingleGather(missFullKvAddrs[m], missFullRopeAddrs[m],
                                      missSelKvAddrs[m], missSelRopeAddrs[m],
                                      missGatherSizes[m]);
                    missGatherSynced = false;
                    missGatherDone[m] = 1;
                }
                if (!missGatherSynced) {
                    SetFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                    WaitFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                    missGatherSynced = true;
                }
            }
            if (!syncOnly) {
                if constexpr (!FSA_DIAG_SIGNAL_MISS_READY_BEFORE_GATHER) {
                    WriteMissReadyStage(missReadyStage);
                }
            } else {
                (void)missReadyStage;
            }
            WaitAICMm1Done(mm1Stage);
            if (syncOnly) {
                if constexpr (!FSA_DIAG_AIC_DUMMY && !FSA_DIAG_FAKE_AIC) {
                    vectorService_.ProcessSoftmaxAndRescale(combinedCount, scoreGmOffset,
                        isFirstBlock, tiling_->scaleValue, mm2OutOff, kvDimFloatAlign);
                    pipe_barrier(PIPE_ALL);
                }
            }
            int32_t mm2Stage = syncStage++;
            if (syncOnly) {
                SignalAICStage(mm2Stage);
            } else {
                WaitMm2StageOnWorkerLane(mm2Stage);
            }
            WaitAICStageDone(mm2Stage);
            WaitAICMm2Done(mm2Stage);
            AccumulateAivMm2TempIfNeeded(syncOnly, isFirstBlock, mm2OutOff, kvDimFloatAlign);
            if (!syncOnly) {
                pipe_barrier(PIPE_ALL);
            }
            isFirstBlock = false;
            continue;
        }
#endif

        // Hit attention: signal AIC for Mm1, wait for scores, do softmax, signal Mm2
        // === Hit attention: AIV writes token info to GM, signals AIC for Matmul ===
        if (hitCount > 0 && !combineHitMissRequest) {
            int64_t kvDimFloatAlign = CeilAlign(tiling_->kvCacheDim,
                static_cast<int64_t>(BLOCK_BYTES / sizeof(float)));
            int64_t mm2OutOff = mm2OutputGmOff_ +
                (bsIdx * queryHeadNum + queryHeadIdx) * Mm2HeadStride(kvDimFloatAlign);
            int64_t prefetchScanTopkIdx = topkIdx;
            int32_t prefetchScanInsertIdx = insertIdx;
            int32_t prefetchHitCopyStableSourceEnd = prefetchScanInsertIdx;
            int32_t prefetchChunkInsertEnd = insertIdx + MAX_BLOCK_TOKENS;
            if (prefetchChunkInsertEnd > tiling_->topk) {
                prefetchChunkInsertEnd = static_cast<int32_t>(tiling_->topk);
            }
            bool skipNonFirstHitOnlyMm2 = FSA_DIAG_SKIP_NONFIRST_HIT_ONLY_MM2 && !isFirstBlock;
            bool skipNonFirstHitOnlyMm1Wait =
                FSA_DIAG_SKIP_NONFIRST_HIT_ONLY_MM1_WAIT && !isFirstBlock;
            bool diagStopAfterFirstHitOnlyChunk =
                FSA_DIAG_STOP_AFTER_FIRST_HIT_ONLY_CHUNK && isFirstBlock &&
                !isLastBlockForHead && (missCount == 0);
            int32_t mm1Stage = syncStage++;
            if (!syncOnly) {
                WriteCommBufAndSignalAIC(queryGmOff, scoreGmOffset, mm2OutOff,
                    hitKvAddrs, hitRopeAddrs, hitCount, isFirstBlock, mm1Stage,
                    isLastHead && (isLastBlockForHead || diagStopAfterFirstHitOnlyChunk) && (missCount == 0),
                    (skipNonFirstHitOnlyMm2 || skipNonFirstHitOnlyMm1Wait) ? -1 : 0);
            } else {
                WaitMm1StageOnSyncOnlyLane(mm1Stage);
            }
            if (skipNonFirstHitOnlyMm1Wait) {
                isFirstBlock = false;
                continue;
            }
#ifndef ASCENDC_CPU_DEBUG
            if (!syncOnly && updateSelectionCache && missCount == 0 && currentHitSourcesSafeForPrefetch &&
                !FSA_DIAG_FAKE_AIC) {
                (void)PrefetchNextChunkGathers(topkIdx, prefetchChunkInsertEnd,
                    prefetchScanTopkIdx, prefetchScanInsertIdx, curBatch,
                    selBlkTableOff, maxSelectionId, lastGatherBlockSize,
                    curStatLocal, topkLocal, selKvBlockTableLocal,
                    updateSelectionCache, allowStatusHit,
                    statusHashKeyLocal, statusHashPosLocal, statusHashSlots,
                    static_cast<int32_t>(gatherEvtId), missGatherSynced,
                    prefetchHitCopyStableSourceEnd,
                    FSA_NEXT_CHUNK_PREFETCH_MM1_LIMIT,
                    FSA_PREFETCH_SOURCE_HIT_ONLY_MM1, true,
                    currentHitCopiesSafeForPrefetch);
            }
#endif
#ifndef ASCENDC_CPU_DEBUG
            WaitAICMm1Done(mm1Stage);
#endif
            if (skipNonFirstHitOnlyMm2) {
                isFirstBlock = false;
                continue;
            }
#ifdef ASCENDC_CPU_DEBUG
            // CPU sim: C++ matmul for Mm1 (Q[1,headDim] 脳 K[nTokens,headDim]^T 鈫?scores[1,nTokens])
            // Cube Mmad is a no-op in tikicpulib, so compute directly
            {
                int64_t hd = tiling_->kvCacheDim;
                int64_t rd = tiling_->kRopeDim;
                int64_t nAlign = CeilAlign(static_cast<int64_t>(hitCount),
                    static_cast<int64_t>(BLOCK_BYTES / sizeof(float)));
                for (int32_t t = 0; t < hitCount; t++) {
                    float dot = 0.0f;
                    for (int64_t d = 0; d < hd; d++) {
                        float q = static_cast<float>(queryGm_.GetValue(queryGmOff + d));
                        float k = static_cast<float>(selKvCacheGm_.GetValue(hitKvAddrs[t] + d));
                        dot += q * k;
                    }
                    for (int64_t d = 0; d < rd; d++) {
                        float q = static_cast<float>(queryGm_.GetValue(queryGmOff + hd + d));
                        float k = static_cast<float>(selKRopeGm_.GetValue(hitRopeAddrs[t] + d));
                        dot += q * k;
                    }
                    workspaceGm_.SetValue(scoreGmOffset + t, dot);
                }
                // zero-pad
                for (int32_t t = hitCount; t < nAlign; t++) {
                    workspaceGm_.SetValue(scoreGmOffset + t, 0.0f);
                }
            }
            {
                float s0 = workspaceGm_.GetValue(scoreGmOffset);
                float s1 = workspaceGm_.GetValue(scoreGmOffset + 1);
                printf("[CPU] Mm1 done: scores[0]=%.4f scores[1]=%.4f\n", s0, s1);
            }
#endif
            if (syncOnly) {
                if constexpr (!FSA_DIAG_AIC_DUMMY && !FSA_DIAG_FAKE_AIC) {
                    vectorService_.ProcessSoftmaxAndRescale(hitCount, scoreGmOffset,
                        isFirstBlock, tiling_->scaleValue, mm2OutOff, kvDimFloatAlign);
                    pipe_barrier(PIPE_ALL);
                }
            }
#ifdef ASCENDC_CPU_DEBUG
            // CPU sim: C++ matmul for Mm2 (weights[1,nTokens] 脳 V[nTokens,kvCacheDim] 鈫?output[1,kvCacheDim])
            {
                int64_t hd = tiling_->kvCacheDim;
                int64_t scoreOffT = scoreGmOffset * static_cast<int64_t>(sizeof(MM_OUT_T) / sizeof(T));
                for (int64_t d = 0; d < hd; d++) {
                    float acc = isFirstBlock ? 0.0f : workspaceGm_.GetValue(mm2OutOff + d);
                    for (int32_t t = 0; t < hitCount; t++) {
                        float w = static_cast<float>(workspaceTGm_.GetValue(scoreOffT + t));
                        float v = static_cast<float>(selKvCacheGm_.GetValue(hitKvAddrs[t] + d));
                        acc += w * v;
                    }
                    workspaceGm_.SetValue(mm2OutOff + d, acc);
                }
            }
#else
            int32_t mm2Stage = syncStage++;
            if (syncOnly) {
                SignalAICStage(mm2Stage);
            } else {
                if (FSA_ENABLE_PRE_MM2_GATHER_OVERLAP && missCount > 0 && !FSA_DIAG_FAKE_AIC) {
                    WaitMm2StageOnWorkerLane(mm2Stage);
                    while (gIdx < missCount) {
                        while (gIdx < missCount &&
                               (missGatherReady[gIdx] == 0 || missGatherDone[gIdx] != 0)) {
                            gIdx++;
                        }
                        if (gIdx >= missCount) {
                            break;
                        }
                        if (!missGatherSynced) {
                            SetFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                            WaitFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                            missGatherSynced = true;
                        }
                        IssueSingleGather(missFullKvAddrs[gIdx], missFullRopeAddrs[gIdx],
                                          missSelKvAddrs[gIdx], missSelRopeAddrs[gIdx],
                                          missGatherSizes[gIdx]);
                        missGatherSynced = false;
                        missGatherDone[gIdx] = 1;
                        gIdx++;
                    }
                } else {
                    WaitMm2StageOnWorkerLane(mm2Stage);
                }
                if (missCount == 0 && updateSelectionCache && currentHitSourcesSafeForPrefetch &&
                    !FSA_DIAG_FAKE_AIC &&
                    !FSA_DIAG_STOP_AFTER_FIRST_HIT_ONLY_CHUNK) {
                    (void)PrefetchNextChunkGathers(topkIdx, prefetchChunkInsertEnd,
                        prefetchScanTopkIdx, prefetchScanInsertIdx, curBatch,
                        selBlkTableOff, maxSelectionId, lastGatherBlockSize,
                        curStatLocal, topkLocal, selKvBlockTableLocal,
                        updateSelectionCache, allowStatusHit,
                        statusHashKeyLocal, statusHashPosLocal, statusHashSlots,
                        static_cast<int32_t>(gatherEvtId), missGatherSynced,
                        prefetchHitCopyStableSourceEnd,
                        FSA_NEXT_CHUNK_PREFETCH_MM2_LIMIT,
                        FSA_PREFETCH_SOURCE_HIT_ONLY_MM2, false,
                        currentHitCopiesSafeForPrefetch);
                }
            }
            if (!syncOnly && missCount > 0 && !missGatherSynced && !FSA_DIAG_FAKE_AIC) {
                SetFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                WaitFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                missGatherSynced = true;
            }
            WaitAICStageDone(mm2Stage);
            WaitAICMm2Done(mm2Stage);
            AccumulateAivMm2TempIfNeeded(syncOnly, isFirstBlock, mm2OutOff, kvDimFloatAlign);
            if (!syncOnly && !missGatherSynced) {
                SetFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                WaitFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                missGatherSynced = true;
            }
            if (diagStopAfterFirstHitOnlyChunk) {
                return;
            }
            if (!syncOnly) {
                pipe_barrier(PIPE_ALL);
            }
#endif
#ifndef ASCENDC_CPU_DEBUG
            if (!syncOnly && missCount > 0 && !missGatherSynced && !FSA_DIAG_FAKE_AIC) {
                SetFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                WaitFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                missGatherSynced = true;
            }
#endif
            if (!syncOnly && updateSelectionCache && !FSA_DIAG_FAKE_AIC) {
                for (int32_t copyIdx = 0; copyIdx < hitCount; copyIdx++) {
                    if (hitCopyFlags[copyIdx] != 0) {
                        IssueSingleSelectionCopy(hitKvAddrs[copyIdx], hitRopeAddrs[copyIdx],
                                                 hitDstKvAddrs[copyIdx], hitDstRopeAddrs[copyIdx],
                                                 hitGatherSizes[copyIdx]);
                    }
                }
                pipe_barrier(PIPE_ALL);
            }
            // Reuse the per-head score slot after Mm2 completion.
            isFirstBlock = false;
        }

        // === Phase B: Finish remaining gathers (AIV only) ===
#ifdef ASCENDC_CPU_DEBUG
        {
#else
        if ASCEND_IS_AIV {
#endif
            for (int32_t m = 0; !syncOnly && m < missCount && !FSA_DIAG_FAKE_AIC; m++) {
                if (missGatherDone[m] != 0) {
                    continue;
                }
#ifndef ASCENDC_CPU_DEBUG
                if (!missGatherSynced) {
                    SetFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                    WaitFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                    missGatherSynced = true;
                }
#endif
                IssueSingleGather(missFullKvAddrs[m], missFullRopeAddrs[m],
                                  missSelKvAddrs[m], missSelRopeAddrs[m],
                                  missGatherSizes[m]);
                missGatherSynced = false;
                missGatherDone[m] = 1;
            }
#ifndef ASCENDC_CPU_DEBUG
            if (!syncOnly && missCount > 0 && !missGatherSynced && !FSA_DIAG_FAKE_AIC) {
                SetFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                WaitFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
                missGatherSynced = true;
            }
#endif
        }

        // === Phase C: Miss attention ===
        // === Phase C: Miss attention ===
        if (missCount > 0) {
            int64_t kvDimFloatAlign = CeilAlign(tiling_->kvCacheDim,
                static_cast<int64_t>(BLOCK_BYTES / sizeof(float)));
            int64_t mm2OutOff = mm2OutputGmOff_ +
                (bsIdx * queryHeadNum + queryHeadIdx) * Mm2HeadStride(kvDimFloatAlign);
            int64_t prefetchScanTopkIdx = topkIdx;
            int32_t prefetchScanInsertIdx = insertIdx;
            int32_t prefetchHitCopyStableSourceEnd = prefetchScanInsertIdx;
            int32_t prefetchChunkInsertEnd = insertIdx + MAX_BLOCK_TOKENS;
            if (prefetchChunkInsertEnd > tiling_->topk) {
                prefetchChunkInsertEnd = static_cast<int32_t>(tiling_->topk);
            }
            int32_t combinedKvAddrs[MAX_BLOCK_TOKENS];
            int32_t combinedRopeAddrs[MAX_BLOCK_TOKENS];
            int32_t combinedCount = 0;
            if (combineHitMissRequest) {
                for (int32_t h = 0; h < hitCount; h++) {
                    combinedKvAddrs[combinedCount] = hitKvAddrs[h];
                    combinedRopeAddrs[combinedCount] = hitRopeAddrs[h];
                    combinedCount++;
                }
                for (int32_t m = 0; m < missCount; m++) {
                    combinedKvAddrs[combinedCount] = missKvAddrs[m];
                    combinedRopeAddrs[combinedCount] = missRopeAddrs[m];
                    combinedCount++;
                }
            }
            int32_t* reqKvAddrs = combineHitMissRequest ? combinedKvAddrs : missKvAddrs;
            int32_t* reqRopeAddrs = combineHitMissRequest ? combinedRopeAddrs : missRopeAddrs;
            int32_t reqCount = combineHitMissRequest ? combinedCount : missCount;
            int32_t mm1Stage = syncStage++;
            if (!syncOnly) {
                WriteCommBufAndSignalAIC(queryGmOff, scoreGmOffset, mm2OutOff,
                    reqKvAddrs, reqRopeAddrs, reqCount, isFirstBlock, mm1Stage,
                    isLastHead && isLastBlockForHead);
            } else {
                WaitMm1StageOnSyncOnlyLane(mm1Stage);
            }
#ifdef ASCENDC_CPU_DEBUG
            // CPU sim: C++ Mm1 for miss tokens
            {
                int64_t hd = tiling_->kvCacheDim;
                int64_t rd = tiling_->kRopeDim;
                int64_t nAlign = CeilAlign(static_cast<int64_t>(reqCount),
                    static_cast<int64_t>(BLOCK_BYTES / sizeof(float)));
                for (int32_t t = 0; t < reqCount; t++) {
                    float dot = 0.0f;
                    for (int64_t d = 0; d < hd; d++) {
                        float q = static_cast<float>(queryGm_.GetValue(queryGmOff + d));
                        float k = static_cast<float>(selKvCacheGm_.GetValue(reqKvAddrs[t] + d));
                        dot += q * k;
                    }
                    for (int64_t d = 0; d < rd; d++) {
                        float q = static_cast<float>(queryGm_.GetValue(queryGmOff + hd + d));
                        float k = static_cast<float>(selKRopeGm_.GetValue(reqRopeAddrs[t] + d));
                        dot += q * k;
                    }
                    workspaceGm_.SetValue(scoreGmOffset + t, dot);
                }
                for (int32_t t = reqCount; t < nAlign; t++)
                    workspaceGm_.SetValue(scoreGmOffset + t, 0.0f);
            }
#else
            WaitAICMm1Done(mm1Stage);
            if (!syncOnly && updateSelectionCache && currentHitSourcesSafeForPrefetch &&
                !FSA_DIAG_FAKE_AIC) {
                (void)PrefetchNextChunkGathers(topkIdx, prefetchChunkInsertEnd,
                    prefetchScanTopkIdx, prefetchScanInsertIdx, curBatch,
                    selBlkTableOff, maxSelectionId, lastGatherBlockSize,
                    curStatLocal, topkLocal, selKvBlockTableLocal,
                    updateSelectionCache, allowStatusHit,
                    statusHashKeyLocal, statusHashPosLocal, statusHashSlots,
                    static_cast<int32_t>(gatherEvtId), missGatherSynced,
                    prefetchHitCopyStableSourceEnd,
                    FSA_NEXT_CHUNK_PREFETCH_MISS_ONLY_MM1_LIMIT,
                    FSA_PREFETCH_SOURCE_MISS_ONLY_MM1);
            }
#endif
            if (syncOnly) {
                if constexpr (!FSA_DIAG_AIC_DUMMY && !FSA_DIAG_FAKE_AIC) {
                    vectorService_.ProcessSoftmaxAndRescale(reqCount, scoreGmOffset,
                        isFirstBlock, tiling_->scaleValue, mm2OutOff, kvDimFloatAlign);
                    pipe_barrier(PIPE_ALL);
                }
            }
#ifdef ASCENDC_CPU_DEBUG
            // CPU sim: C++ Mm2 for miss tokens
            {
                int64_t hd = tiling_->kvCacheDim;
                int64_t scoreOffT = scoreGmOffset * static_cast<int64_t>(sizeof(MM_OUT_T) / sizeof(T));
                for (int64_t d = 0; d < hd; d++) {
                    float acc = isFirstBlock ? 0.0f : workspaceGm_.GetValue(mm2OutOff + d);
                    for (int32_t t = 0; t < reqCount; t++) {
                        float w = static_cast<float>(workspaceTGm_.GetValue(scoreOffT + t));
                        float v = static_cast<float>(selKvCacheGm_.GetValue(reqKvAddrs[t] + d));
                        acc += w * v;
                    }
                    workspaceGm_.SetValue(mm2OutOff + d, acc);
                }
            }
#else
            int32_t mm2Stage = syncStage++;
            if (syncOnly) {
                SignalAICStage(mm2Stage);
            } else {
                WaitMm2StageOnWorkerLane(mm2Stage);
                if (updateSelectionCache && currentHitSourcesSafeForPrefetch &&
                    !FSA_DIAG_FAKE_AIC) {
                    (void)PrefetchNextChunkGathers(topkIdx, prefetchChunkInsertEnd,
                        prefetchScanTopkIdx, prefetchScanInsertIdx, curBatch,
                        selBlkTableOff, maxSelectionId, lastGatherBlockSize,
                        curStatLocal, topkLocal, selKvBlockTableLocal,
                        updateSelectionCache, allowStatusHit,
                        statusHashKeyLocal, statusHashPosLocal, statusHashSlots,
                        static_cast<int32_t>(gatherEvtId), missGatherSynced,
                        prefetchHitCopyStableSourceEnd,
                        FSA_NEXT_CHUNK_PREFETCH_MISS_ONLY_MM2_LIMIT,
                        FSA_PREFETCH_SOURCE_MISS_ONLY_MM2);
                }
            }
            WaitAICStageDone(mm2Stage);
            WaitAICMm2Done(mm2Stage);
            AccumulateAivMm2TempIfNeeded(syncOnly, isFirstBlock, mm2OutOff, kvDimFloatAlign);
            if (!syncOnly) {
                pipe_barrier(PIPE_ALL);
            }
#endif
            if (combineHitMissRequest && updateSelectionCache) {
#ifdef ASCENDC_CPU_DEBUG
                for (int32_t copyIdx = 0; copyIdx < hitCount; copyIdx++) {
                    if (hitCopyFlags[copyIdx] != 0) {
                        IssueSingleSelectionCopy(hitKvAddrs[copyIdx], hitRopeAddrs[copyIdx],
                                                 hitDstKvAddrs[copyIdx], hitDstRopeAddrs[copyIdx],
                                                 hitGatherSizes[copyIdx]);
                    }
                }
#else
                if (!syncOnly && !FSA_DIAG_FAKE_AIC) {
                    bool issuedHitCopies = false;
                    for (int32_t copyIdx = 0; copyIdx < hitCount; copyIdx++) {
                        if (hitCopyFlags[copyIdx] != 0) {
                            IssueSingleSelectionCopy(hitKvAddrs[copyIdx], hitRopeAddrs[copyIdx],
                                                     hitDstKvAddrs[copyIdx], hitDstRopeAddrs[copyIdx],
                                                     hitGatherSizes[copyIdx]);
                            issuedHitCopies = true;
                        }
                    }
                    if (issuedHitCopies) {
                        pipe_barrier(PIPE_ALL);
                    }
                }
#endif
            }
            // Reuse the per-head score slot after Mm2 completion.
            isFirstBlock = false;
        }
    }

    if (updateSelectionCache) {
        for (int32_t j = insertIdx; j < topkOneSortAlign_; j++) {
            curStatLocal.SetValue(j, -1);
        }
        curStatLocal.SetValue(tiling_->topk, selActualSeqLen);
        selKvActSeqLocal.SetValue(kvHeadIdx, selActualSeqLen);
    }

    // === Phase D: Normalize and write output (AIV only) ===
#ifdef ASCENDC_CPU_DEBUG
    // CPU sim: always execute (no AIC/AIV split)
    {
#else
    if ASCEND_IS_AIV {
#endif
        if (syncOnly && !isFirstBlock) {
            int64_t kvDimFloatAlign = CeilAlign(tiling_->kvCacheDim, static_cast<int64_t>(BLOCK_BYTES / sizeof(float)));
            int64_t mm2OutOff = mm2OutputGmOff_ +
                (bsIdx * queryHeadNum + queryHeadIdx) * Mm2HeadStride(kvDimFloatAlign);
#ifdef ASCENDC_CPU_DEBUG
            {
                float m0 = workspaceGm_.GetValue(mm2OutOff);
                float m1 = workspaceGm_.GetValue(mm2OutOff + 1);
                printf("[CPU] Before ProcessVec2: mm2Out[0]=%.4f mm2Out[1]=%.4f\n", m0, m1);
            }
#endif
            int64_t outGmOff = globalBsIdx * queryHeadNum * tiling_->headDim + queryHeadIdx * tiling_->headDim;
            if constexpr (FSA_DIAG_AIC_MM2_HEARTBEAT_TO_OUTPUT) {
                int32_t heartbeat = ReadCommValue(COMM_DIAG_AIC_HEARTBEAT);
                int32_t ackStage = ReadCommValue(COMM_ACK_STAGE);
                T marker = (heartbeat != 0) ? static_cast<T>(5.0f) : static_cast<T>(0.0f);
                if (heartbeat == 0 && ackStage != 0) {
                    marker = static_cast<T>(6.0f);
                }
                for (int64_t d = 0; d < tiling_->kvCacheDim; d++) {
                    attentionOutGm_.SetValue(outGmOff + d, (d == 0) ? marker : static_cast<T>(0.0f));
                }
                pipe_barrier(PIPE_ALL);
            } else if constexpr (FSA_DIAG_AIC_DUMMY) {
                for (int64_t d = 0; d < tiling_->kvCacheDim; d++) {
                    attentionOutGm_.SetValue(outGmOff + d, static_cast<T>(0));
                }
            } else {
                if constexpr (FSA_DIAG_FAKE_AIC) {
                    for (int64_t d = 0; d < tiling_->kvCacheDim; d++) {
                        attentionOutGm_.SetValue(outGmOff + d, static_cast<T>(1.0f));
                    }
                } else if constexpr (FSA_DIAG_AIV_FINAL_DIRECT_MARK) {
                    for (int64_t d = 0; d < tiling_->kvCacheDim; d++) {
                        attentionOutGm_.SetValue(outGmOff + d, static_cast<T>((d == 0) ? 3.0f : 0.0f));
                    }
                    pipe_barrier(PIPE_ALL);
                } else {
                    if constexpr (FSA_DIAG_AIV_MARK_MM2) {
                        LocalTensor<float> marker = commLocalBuf_.Get<float>();
                        Duplicate(marker, 0.0f, kvDimFloatAlign);
                        marker.SetValue(0, 1.0f);
                        pipe_barrier(PIPE_V);
                        DataCopyExtParams markerCopyParams{static_cast<uint16_t>(1),
                            static_cast<uint32_t>(kvDimFloatAlign * sizeof(float)), 0, 0, 0};
                        DataCopyPad(workspaceGm_[mm2OutOff], marker, markerCopyParams);
                        pipe_barrier(PIPE_ALL);
                    }
                    vectorService_.ProcessVec2(attentionOutGm_, outGmOff, mm2OutOff, kvDimFloatAlign);
                }
            }
        }

        // Reset softmax state for next head
        if (syncOnly) {
            vectorService_.ResetSoftmaxState();
        }
    }
}

// ============================================================================
// InitAIC: minimal init for AIC core (no UB buffer allocation)
// ============================================================================
template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::InitAIC(
    TPipe* pipeIn,
    const FusedSparseAttentionOverlapTilingData* tilingIn,
    GM_ADDR query, GM_ADDR selection_k_rope, GM_ADDR selection_kv_cache,
    GM_ADDR selection_kv_block_table, GM_ADDR workspace)
{
    pipe_ = pipeIn;
    tiling_ = tilingIn;
    blkIdx_ = GetBlockIdx();

    queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(query));
    selKRopeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(selection_k_rope));
    selKvCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(selection_kv_cache));
    selKvBlockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(selection_kv_block_table));

    GM_ADDR perCoreBase = workspace + blkIdx_ * tiling_->perCoreWorkspaceSize;
    workspaceBase_ = perCoreBase;
    workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MM_OUT_T*>(perCoreBase));
    workspaceTGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(perCoreBase));
    commBufGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
        perCoreBase + tiling_->commBufGmOffset));

    cubeService_.Init(pipe_, tiling_, workspaceBase_);
}

// ============================================================================
// ProcessAIC: AIC event loop 鈥?wait for AIV signals, execute Matmul
// ============================================================================
template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::ProcessAIC()
{
    if (blkIdx_ >= tiling_->usedCoreNum) return;

    cubeService_.Init(pipe_, tiling_, workspaceBase_);

    int32_t nextMm1Stage = 1;
    while (true) {
        DiagStageHeartbeat(0, 1000 + nextMm1Stage);
        WaitAivStage(nextMm1Stage);
        pipe_barrier(PIPE_ALL);
        DiagStageHeartbeat(1, 1100 + nextMm1Stage);

        if constexpr (FSA_DIAG_AIC_DUMMY) {
            int32_t diagTokens = ReadCommValue(COMM_TOKEN_COUNT);
            int64_t diagQuery = ReadInt64FromComm(COMM_QUERY_OFF_LO);
            int64_t diagScore = ReadInt64FromComm(COMM_SCORE_OFF_LO);
            int64_t diagOut = ReadInt64FromComm(COMM_MM2_OUT_OFF_LO);
            int32_t diagSum = static_cast<int32_t>(diagQuery + diagScore + diagOut);
            for (int32_t i = 0; i < diagTokens && i < MAX_BLOCK_TOKENS; i++) {
                diagSum += ReadCommValue(COMM_KV_ADDRS + i);
                diagSum += ReadCommValue(COMM_ROPE_ADDRS + i);
            }
            workspaceGm_.SetValue(diagScore, 0.0f);
            (void)diagSum;
            pipe_barrier(PIPE_ALL);
            FsaSignalAicMm1Done(nextMm1Stage);
            break;
        }

        int32_t nTokens = ReadCommValue(COMM_TOKEN_COUNT);
        DiagStageHeartbeat(2, 1200 + nTokens);
        while (nTokens == 0) {
            pipe_barrier(PIPE_ALL);
            nTokens = ReadCommValue(COMM_TOKEN_COUNT);
        }
        DiagStageHeartbeat(3, 1300 + nTokens);
        if (nTokens < 0) {
            commBufGm_.SetValue(COMM_TOKEN_COUNT, 0);
            pipe_barrier(PIPE_ALL);
            DiagStageHeartbeat(4, 1400 + nextMm1Stage);
            break;
        }

        int64_t queryGmOff = ReadInt64FromComm(COMM_QUERY_OFF_LO);
        int64_t scoreGmOff = ReadInt64FromComm(COMM_SCORE_OFF_LO);
        int64_t mm2OutOff = ReadInt64FromComm(COMM_MM2_OUT_OFF_LO);
        bool isFirst = (ReadCommValue(COMM_IS_FIRST) != 0);
        bool isLastRequest = (ReadCommValue(COMM_IS_LAST) != 0);
        int32_t reqStage = ReadCommValue(COMM_STAGE);
        nextMm1Stage = reqStage;
        DiagStageHeartbeat(5, 1500 + reqStage);

        int32_t kvAddrs[MAX_BLOCK_TOKENS];
        int32_t ropeAddrs[MAX_BLOCK_TOKENS];
        int32_t sourceFlags[MAX_BLOCK_TOKENS];
        int32_t fullKvAddrs[MAX_BLOCK_TOKENS];
        int32_t fullRopeAddrs[MAX_BLOCK_TOKENS];
        for (int32_t i = 0; i < nTokens; i++) {
            kvAddrs[i] = ReadCommValue(COMM_KV_ADDRS + i);
            ropeAddrs[i] = ReadCommValue(COMM_ROPE_ADDRS + i);
            sourceFlags[i] = ReadCommValue(COMM_SOURCE_FLAGS + i);
            fullKvAddrs[i] = ReadCommValue(COMM_FULL_KV_ADDRS + i);
            fullRopeAddrs[i] = ReadCommValue(COMM_FULL_ROPE_ADDRS + i);
        }

        if constexpr (FSA_DIAG_AIC_DUMMY) {
            (void)queryGmOff;
            (void)scoreGmOff;
            (void)mm2OutOff;
            (void)isFirst;
            commBufGm_.SetValue(COMM_ACK_STAGE, reqStage);
            pipe_barrier(PIPE_ALL);
            FsaSignalAicMm1Done(reqStage);
            break;
        }

        int32_t overlapHitCount = ReadCommValue(COMM_HIT_COUNT);
        if (overlapHitCount == FSA_DIRECT_SOURCE_MODE) {
            int32_t mm1KvAddrs[MAX_BLOCK_TOKENS];
            int32_t mm1RopeAddrs[MAX_BLOCK_TOKENS];
            for (int32_t i = 0; i < nTokens; i++) {
                if (sourceFlags[i] != 0) {
                    mm1KvAddrs[i] = fullKvAddrs[i];
                    mm1RopeAddrs[i] = fullRopeAddrs[i];
                } else {
                    mm1KvAddrs[i] = kvAddrs[i];
                    mm1RopeAddrs[i] = ropeAddrs[i];
                }
            }
            if constexpr (FSA_DIAG_FAKE_AIC) {
                int64_t nAlign = CeilAlign(static_cast<int64_t>(nTokens), static_cast<int64_t>(16));
                for (int64_t i = 0; i < nAlign; i++) {
                    workspaceGm_.SetValue(scoreGmOff + i, 0.0f);
                }
                pipe_barrier(PIPE_ALL);
            } else {
                cubeService_.ComputeMm1MixedSource(queryGm_, queryGmOff,
                    selKvCacheGm_, selKRopeGm_, fullKvCacheGm_, fullKRopeGm_,
                    mm1KvAddrs, mm1RopeAddrs, sourceFlags, nTokens, scoreGmOff);
            }
            commBufGm_.SetValue(COMM_ACK_STAGE, reqStage);
            pipe_barrier(PIPE_ALL);
            FsaSignalAicMm1Done(reqStage);
            DiagStageHeartbeat(6, 1600 + reqStage);

            WaitAivStage(reqStage + 1, true);
            pipe_barrier(PIPE_ALL);
            int32_t mm2Stage = ReadCommValue(COMM_STAGE);
            DiagStageHeartbeat(7, 1700 + mm2Stage);
            if constexpr (FSA_DIAG_FAKE_AIC) {
                int64_t kvDimAlign = CeilAlign(tiling_->kvCacheDim, static_cast<int64_t>(16));
                for (int64_t i = 0; i < kvDimAlign; i++) {
                    workspaceGm_.SetValue(mm2OutOff + i, 1.0f);
                }
                pipe_barrier(PIPE_ALL);
            } else {
                int64_t weightsGmOff = scoreGmOff * static_cast<int64_t>(sizeof(MM_OUT_T) / sizeof(T));
                cubeService_.ResetPhaseEvents();
                if constexpr (FSA_DIAG_AIC_MM2_MARK_ONLY) {
                    (void)weightsGmOff;
                    WriteAicMm2Marker(AicMm2OutputOffset(mm2OutOff, isFirst));
                } else {
                    cubeService_.ComputeMm2MixedSource(selKvCacheGm_, fullKvCacheGm_,
                        mm1KvAddrs, sourceFlags, nTokens,
                        workspaceTGm_, weightsGmOff,
                        workspaceGm_, AicMm2OutputOffset(mm2OutOff, isFirst), AicMm2UseOverwrite(isFirst));
                }
                if constexpr (FSA_DIAG_AIC_MARK_MM2_AFTER_COMPUTE) {
                    workspaceGm_.SetValue(AicMm2OutputOffset(mm2OutOff, isFirst), 7.0f);
                    pipe_barrier(PIPE_ALL);
                }
            }
            if constexpr (FSA_DIAG_AIC_MM2_HEARTBEAT_TO_OUTPUT) {
                commBufGm_.SetValue(COMM_DIAG_AIC_HEARTBEAT, 777000 + mm2Stage);
            }
            commBufGm_.SetValue(COMM_ACK_STAGE, mm2Stage);
            commBufGm_.SetValue(COMM_TOKEN_COUNT, 0);
            pipe_barrier(PIPE_ALL);
            FsaSignalAicMm2Done(mm2Stage);
            DiagStageHeartbeat(8, 1800 + mm2Stage);
            if (isLastRequest) {
                break;
            }
            nextMm1Stage = mm2Stage + 1;
            continue;
        }
        if (overlapHitCount < 0) {
            if constexpr (FSA_DIAG_FAKE_AIC) {
                int64_t nAlign = CeilAlign(static_cast<int64_t>(nTokens), static_cast<int64_t>(16));
                for (int64_t i = 0; i < nAlign; i++) {
                    workspaceGm_.SetValue(scoreGmOff + i, 0.0f);
                }
                pipe_barrier(PIPE_ALL);
            } else {
                cubeService_.ComputeMm1(queryGm_, queryGmOff,
                    selKvCacheGm_, selKRopeGm_, selKvBlockTableGm_,
                    kvAddrs, ropeAddrs, nTokens, scoreGmOff);
            }
            commBufGm_.SetValue(COMM_ACK_STAGE, reqStage);
            commBufGm_.SetValue(COMM_TOKEN_COUNT, 0);
            pipe_barrier(PIPE_ALL);
            FsaSignalAicMm1Done(reqStage);
            DiagStageHeartbeat(9, 1900 + reqStage);
            if (isLastRequest) {
                break;
            }
            nextMm1Stage = reqStage + 1;
            continue;
        }
        bool useInternalOverlap = (overlapHitCount > 0) && (overlapHitCount < nTokens);
        if (useInternalOverlap) {
            int32_t missReadyStage = reqStage + 1;
            if constexpr (FSA_DIAG_FAKE_AIC) {
                while (ReadCommValue(COMM_MISS_READY_STAGE) != missReadyStage) {
                    pipe_barrier(PIPE_ALL);
                }
                int64_t nAlign = CeilAlign(static_cast<int64_t>(nTokens), static_cast<int64_t>(16));
                for (int64_t i = 0; i < nAlign; i++) {
                    workspaceGm_.SetValue(scoreGmOff + i, 0.0f);
                }
                pipe_barrier(PIPE_ALL);
            } else {
                cubeService_.ComputeMm1(queryGm_, queryGmOff,
                    selKvCacheGm_, selKRopeGm_, selKvBlockTableGm_,
                    kvAddrs, ropeAddrs, nTokens, scoreGmOff);
            }
            commBufGm_.SetValue(COMM_ACK_STAGE, reqStage);
            pipe_barrier(PIPE_ALL);
            FsaSignalAicMm1Done(reqStage);
            DiagStageHeartbeat(10, 2000 + reqStage);

            WaitAivStage(reqStage + 2, true);
            pipe_barrier(PIPE_ALL);
            int32_t mm2Stage = reqStage + 2;
            DiagStageHeartbeat(11, 2100 + mm2Stage);
            if constexpr (FSA_DIAG_FAKE_AIC) {
                int64_t kvDimAlign = CeilAlign(tiling_->kvCacheDim, static_cast<int64_t>(16));
                for (int64_t i = 0; i < kvDimAlign; i++) {
                    workspaceGm_.SetValue(mm2OutOff + i, 1.0f);
                }
                pipe_barrier(PIPE_ALL);
            } else {
                int64_t weightsGmOff = scoreGmOff * static_cast<int64_t>(sizeof(MM_OUT_T) / sizeof(T));
                cubeService_.ResetPhaseEvents();
                if constexpr (FSA_DIAG_AIC_MM2_MARK_ONLY) {
                    (void)weightsGmOff;
                    WriteAicMm2Marker(AicMm2OutputOffset(mm2OutOff, isFirst));
                } else {
                    cubeService_.ComputeMm2(selKvCacheGm_, selKvBlockTableGm_,
                        kvAddrs, nTokens,
                        workspaceTGm_, weightsGmOff,
                        workspaceGm_, AicMm2OutputOffset(mm2OutOff, isFirst), AicMm2UseOverwrite(isFirst));
                }
                if constexpr (FSA_DIAG_AIC_MARK_MM2_AFTER_COMPUTE) {
                    workspaceGm_.SetValue(AicMm2OutputOffset(mm2OutOff, isFirst), 7.0f);
                    pipe_barrier(PIPE_ALL);
                }
            }
            commBufGm_.SetValue(COMM_ACK_STAGE, mm2Stage);
            commBufGm_.SetValue(COMM_TOKEN_COUNT, 0);
            pipe_barrier(PIPE_ALL);
            FsaSignalAicMm2Done(mm2Stage);
            DiagStageHeartbeat(12, 2200 + mm2Stage);
            if (isLastRequest) {
                break;
            }
            nextMm1Stage = mm2Stage + 1;
            continue;
        }

        if constexpr (FSA_DIAG_FAKE_AIC) {
            int64_t nAlign = CeilAlign(static_cast<int64_t>(nTokens), static_cast<int64_t>(16));
            for (int64_t i = 0; i < nAlign; i++) {
                workspaceGm_.SetValue(scoreGmOff + i, 0.0f);
            }
            pipe_barrier(PIPE_ALL);
        } else {
            cubeService_.ComputeMm1(queryGm_, queryGmOff,
                selKvCacheGm_, selKRopeGm_, selKvBlockTableGm_,
                kvAddrs, ropeAddrs, nTokens, scoreGmOff);
        }
        commBufGm_.SetValue(COMM_ACK_STAGE, reqStage);
        pipe_barrier(PIPE_ALL);
        FsaSignalAicMm1Done(reqStage);
        DiagStageHeartbeat(13, 2300 + reqStage);

        WaitAivStage(reqStage + 1, true);
        pipe_barrier(PIPE_ALL);
        int32_t mm2Stage = reqStage + 1;
        DiagStageHeartbeat(14, 2400 + mm2Stage);

        if constexpr (FSA_DIAG_FAKE_AIC) {
            int64_t kvDimAlign = CeilAlign(tiling_->kvCacheDim, static_cast<int64_t>(16));
            for (int64_t i = 0; i < kvDimAlign; i++) {
                workspaceGm_.SetValue(mm2OutOff + i, 1.0f);
            }
            pipe_barrier(PIPE_ALL);
        } else if constexpr (FSA_DIAG_AIC_MM2_INPUT_MARK) {
            int64_t weightsGmOff = scoreGmOff * static_cast<int64_t>(sizeof(MM_OUT_T) / sizeof(T));
            int64_t diagOut = AicMm2OutputOffset(mm2OutOff, isFirst);
            workspaceGm_.SetValue(diagOut, static_cast<float>(workspaceTGm_.GetValue(weightsGmOff)));
            workspaceGm_.SetValue(diagOut + 1, static_cast<float>(selKvCacheGm_.GetValue(kvAddrs[0])));
            workspaceGm_.SetValue(diagOut + 2, static_cast<float>(nTokens));
            pipe_barrier(PIPE_ALL);
        } else {
            int64_t weightsGmOff = scoreGmOff * static_cast<int64_t>(sizeof(MM_OUT_T) / sizeof(T));
            cubeService_.ResetPhaseEvents();
            if constexpr (FSA_DIAG_AIC_MM2_MARK_ONLY) {
                (void)weightsGmOff;
                WriteAicMm2Marker(AicMm2OutputOffset(mm2OutOff, isFirst));
            } else {
                cubeService_.ComputeMm2(selKvCacheGm_, selKvBlockTableGm_,
                    kvAddrs, nTokens,
                    workspaceTGm_, weightsGmOff,
                    workspaceGm_, AicMm2OutputOffset(mm2OutOff, isFirst), AicMm2UseOverwrite(isFirst));
            }
            if constexpr (FSA_DIAG_AIC_MARK_MM2_AFTER_COMPUTE) {
                workspaceGm_.SetValue(AicMm2OutputOffset(mm2OutOff, isFirst), 7.0f);
                pipe_barrier(PIPE_ALL);
            }
        }
        if constexpr (FSA_DIAG_AIC_MM2_HEARTBEAT_TO_OUTPUT) {
            commBufGm_.SetValue(COMM_DIAG_AIC_HEARTBEAT, 777000 + mm2Stage);
        }
        commBufGm_.SetValue(COMM_ACK_STAGE, mm2Stage);
        commBufGm_.SetValue(COMM_TOKEN_COUNT, 0);
        pipe_barrier(PIPE_ALL);
        FsaSignalAicMm2Done(mm2Stage);
        DiagStageHeartbeat(15, 2500 + mm2Stage);
        if (isLastRequest) {
            break;
        }
        nextMm1Stage = mm2Stage + 1;
    }
    cubeService_.FreeEventID();
}

// ============================================================================
// WriteCommBufAndSignalAIC: write token info to GM comm buffer and signal AIC
// ============================================================================
template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::WriteCommBufAndSignalAIC(
    int64_t queryGmOff, int64_t scoreGmOff, int64_t mm2OutOff,
    int32_t* kvAddrs, int32_t* ropeAddrs,
    int32_t nTokens, bool isFirstBlock, int32_t syncStage, bool isLastRequest,
    int32_t overlapHitCount, int32_t* sourceFlags, int32_t* fullKvAddrs, int32_t* fullRopeAddrs)
{
    LocalTensor<int32_t> commLocal = commLocalBuf_.Get<int32_t>();
    Duplicate(commLocal, static_cast<int32_t>(0), FSA_COMM_BUF_SLOTS);
    pipe_barrier(PIPE_V);
    commLocal.SetValue(COMM_TOKEN_COUNT, nTokens);
    commLocal.SetValue(COMM_QUERY_OFF_LO, static_cast<int32_t>(queryGmOff & 0xffffffff));
    commLocal.SetValue(COMM_QUERY_OFF_HI, static_cast<int32_t>((static_cast<uint64_t>(queryGmOff) >> 32) & 0xffffffff));
    commLocal.SetValue(COMM_SCORE_OFF_LO, static_cast<int32_t>(scoreGmOff & 0xffffffff));
    commLocal.SetValue(COMM_SCORE_OFF_HI, static_cast<int32_t>((static_cast<uint64_t>(scoreGmOff) >> 32) & 0xffffffff));
    commLocal.SetValue(COMM_MM2_OUT_OFF_LO, static_cast<int32_t>(mm2OutOff & 0xffffffff));
    commLocal.SetValue(COMM_MM2_OUT_OFF_HI, static_cast<int32_t>((static_cast<uint64_t>(mm2OutOff) >> 32) & 0xffffffff));
    commLocal.SetValue(COMM_IS_FIRST, isFirstBlock ? 1 : 0);
    commLocal.SetValue(COMM_IS_LAST, isLastRequest ? 1 : 0);
    commLocal.SetValue(COMM_HIT_COUNT, overlapHitCount);
    commLocal.SetValue(COMM_MISS_READY_STAGE, 0);
    commLocal.SetValue(COMM_ACK_STAGE, syncStage - 1);
    commLocal.SetValue(COMM_STAGE, syncStage - 1);
    for (int32_t i = 0; i < nTokens; i++) {
        commLocal.SetValue(COMM_KV_ADDRS + i, kvAddrs[i]);
        commLocal.SetValue(COMM_ROPE_ADDRS + i, ropeAddrs[i]);
        if (sourceFlags != nullptr) {
            commLocal.SetValue(COMM_SOURCE_FLAGS + i, sourceFlags[i]);
        }
        if (fullKvAddrs != nullptr) {
            commLocal.SetValue(COMM_FULL_KV_ADDRS + i, fullKvAddrs[i]);
        }
        if (fullRopeAddrs != nullptr) {
            commLocal.SetValue(COMM_FULL_ROPE_ADDRS + i, fullRopeAddrs[i]);
        }
    }
    pipe_barrier(PIPE_V);
    DataCopy(commBufGm_, commLocal, FSA_COMM_BUF_SLOTS);
    pipe_barrier(PIPE_ALL);
    commBufGm_.SetValue(COMM_TOKEN_COUNT, nTokens);
    commBufGm_.SetValue(COMM_IS_FIRST, isFirstBlock ? 1 : 0);
    commBufGm_.SetValue(COMM_IS_LAST, isLastRequest ? 1 : 0);
    commBufGm_.SetValue(COMM_HIT_COUNT, overlapHitCount);
    commBufGm_.SetValue(COMM_MISS_READY_STAGE, 0);
    commBufGm_.SetValue(COMM_STAGE, syncStage);
    pipe_barrier(PIPE_ALL);
#ifndef ASCENDC_CPU_DEBUG
    if constexpr (!FSA_DIAG_WORKER_NO_AIC_SIGNAL) {
        if constexpr (FSA_ENABLE_AIC_STAGE_POLL) {
            if (syncStage <= 1) {
                CrossCoreSetFlag<FSA_SYNC_MODE, PIPE_MTE3>(FsaAivToAicFlag(syncStage));
            }
        } else {
            CrossCoreSetFlag<FSA_SYNC_MODE, PIPE_MTE3>(FsaAivToAicFlag(syncStage));
        }
    }
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::WriteMissReadyStage(int32_t syncStage)
{
#ifndef ASCENDC_CPU_DEBUG
    commBufGm_.SetValue(COMM_MISS_READY_STAGE, syncStage);
    pipe_barrier(PIPE_ALL);
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::SignalAICStage(int32_t syncStage)
{
#ifndef ASCENDC_CPU_DEBUG
    commBufGm_.SetValue(COMM_STAGE, syncStage);
    commBufGm_.SetValue(COMM_ACK_STAGE, syncStage - 1);
    pipe_barrier(PIPE_ALL);
    if constexpr (!FSA_DIAG_SYNC_ONLY_NO_AIC_SIGNAL) {
        CrossCoreSetFlag<FSA_SYNC_MODE, PIPE_MTE3>(FsaAivToAic2Flag(syncStage));
    }
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::WaitCommStage(int32_t syncStage)
{
#ifndef ASCENDC_CPU_DEBUG
    while (ReadCommValue(COMM_STAGE) != syncStage) {
    }
    pipe_barrier(PIPE_ALL);
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::WaitCommStageAndSignalAIC(int32_t syncStage)
{
#ifndef ASCENDC_CPU_DEBUG
    WaitCommStage(syncStage);
    if constexpr (!FSA_DIAG_SYNC_ONLY_NO_AIC_SIGNAL) {
        if constexpr (FSA_ENABLE_AIC_STAGE_POLL) {
            if (syncStage <= 1) {
                CrossCoreSetFlag<FSA_SYNC_MODE, PIPE_MTE3>(FsaAivToAicFlag(syncStage));
            }
        } else {
            CrossCoreSetFlag<FSA_SYNC_MODE, PIPE_MTE3>(FsaAivToAicFlag(syncStage));
        }
    }
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::WaitCommStageAndSignalAIC2(int32_t syncStage)
{
#ifndef ASCENDC_CPU_DEBUG
    while (ReadCommValue(COMM_STAGE) != syncStage) {
    }
    pipe_barrier(PIPE_ALL);
    if constexpr (!FSA_DIAG_WORKER_NO_AIC_SIGNAL) {
        CrossCoreSetFlag<FSA_SYNC_MODE, PIPE_MTE3>(FsaAivToAic2Flag(syncStage));
    }
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::WaitMm1StageOnSyncOnlyLane(int32_t syncStage)
{
#ifndef ASCENDC_CPU_DEBUG
    if constexpr (FSA_ENABLE_SINGLE_PRODUCER_AIC_REQUEST) {
        WaitCommStage(syncStage);
    } else {
        WaitCommStageAndSignalAIC(syncStage);
    }
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::WaitMm2StageOnWorkerLane(int32_t syncStage)
{
#ifndef ASCENDC_CPU_DEBUG
    if constexpr (FSA_ENABLE_SINGLE_PRODUCER_AIC_REQUEST) {
        WaitCommStage(syncStage);
    } else {
        WaitCommStageAndSignalAIC2(syncStage);
    }
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::SignalAIC2Only(int32_t syncStage)
{
#ifndef ASCENDC_CPU_DEBUG
    CrossCoreSetFlag<FSA_SYNC_MODE, PIPE_MTE3>(FsaAivToAic2Flag(syncStage));
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::SignalAICMm1Stage(int32_t syncStage)
{
#ifndef ASCENDC_CPU_DEBUG
    commBufGm_.SetValue(COMM_STAGE, syncStage);
    pipe_barrier(PIPE_ALL);
    if constexpr (FSA_ENABLE_AIC_STAGE_POLL) {
        if (syncStage <= 1) {
            CrossCoreSetFlag<FSA_SYNC_MODE, PIPE_MTE3>(FsaAivToAicFlag(syncStage));
        }
    } else {
        CrossCoreSetFlag<FSA_SYNC_MODE, PIPE_MTE3>(FsaAivToAicFlag(syncStage));
    }
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::DiagStageHeartbeat(int32_t slot, int32_t value)
{
#ifndef ASCENDC_CPU_DEBUG
    if constexpr (FSA_DIAG_STAGE_HEARTBEAT) {
        if (blkIdx_ == 0 && slot >= 0 && slot < tiling_->topk) {
            hitMaskOutGm_.SetValue(slot, value);
            pipe_barrier(PIPE_ALL);
        }
    }
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::DiagInitNextChunkPrefetchCounters()
{
#ifndef ASCENDC_CPU_DEBUG
    if constexpr (FSA_DIAG_NEXT_CHUNK_PREFETCH_COUNTERS) {
        if (blkIdx_ == 0 && tiling_->topk > FSA_DIAG_DIRECT_BATCHED_GATHER_MAX_SLOT) {
            for (int32_t slot = FSA_DIAG_PREFETCH_TOTAL_COPIED_SLOT;
                 slot <= FSA_DIAG_DIRECT_BATCHED_GATHER_MAX_SLOT; slot++) {
                hitMaskOutGm_.SetValue(slot, 0);
            }
            pipe_barrier(PIPE_ALL);
        }
    }
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::DiagRecordNextChunkPrefetch(
    int32_t prefetchSource, int32_t prefetched)
{
#ifndef ASCENDC_CPU_DEBUG
    if constexpr (FSA_DIAG_NEXT_CHUNK_PREFETCH_COUNTERS) {
        if (blkIdx_ != 0 || tiling_->topk <= FSA_DIAG_DIRECT_BATCHED_GATHER_MAX_SLOT) {
            return;
        }

        int32_t copiedSlot = -1;
        int32_t windowsSlot = -1;
        if (prefetchSource == FSA_PREFETCH_SOURCE_HIT_ONLY_MM1) {
            copiedSlot = FSA_DIAG_PREFETCH_HIT_ONLY_MM1_COPIED_SLOT;
            windowsSlot = FSA_DIAG_PREFETCH_HIT_ONLY_MM1_WINDOWS_SLOT;
        } else if (prefetchSource == FSA_PREFETCH_SOURCE_HIT_ONLY_MM2) {
            copiedSlot = FSA_DIAG_PREFETCH_HIT_ONLY_MM2_COPIED_SLOT;
            windowsSlot = FSA_DIAG_PREFETCH_HIT_ONLY_MM2_WINDOWS_SLOT;
        } else if (prefetchSource == FSA_PREFETCH_SOURCE_DIRECT_MM1) {
            copiedSlot = FSA_DIAG_PREFETCH_DIRECT_MM1_COPIED_SLOT;
            windowsSlot = FSA_DIAG_PREFETCH_DIRECT_MM1_WINDOWS_SLOT;
        } else if (prefetchSource == FSA_PREFETCH_SOURCE_DIRECT_MM2) {
            copiedSlot = FSA_DIAG_PREFETCH_DIRECT_MM2_COPIED_SLOT;
            windowsSlot = FSA_DIAG_PREFETCH_DIRECT_MM2_WINDOWS_SLOT;
        } else if (prefetchSource == FSA_PREFETCH_SOURCE_MISS_ONLY_MM1) {
            copiedSlot = FSA_DIAG_PREFETCH_MISS_ONLY_MM1_COPIED_SLOT;
            windowsSlot = FSA_DIAG_PREFETCH_MISS_ONLY_MM1_WINDOWS_SLOT;
        } else if (prefetchSource == FSA_PREFETCH_SOURCE_MISS_ONLY_MM2) {
            copiedSlot = FSA_DIAG_PREFETCH_MISS_ONLY_MM2_COPIED_SLOT;
            windowsSlot = FSA_DIAG_PREFETCH_MISS_ONLY_MM2_WINDOWS_SLOT;
        }

        hitMaskOutGm_.SetValue(FSA_DIAG_PREFETCH_TOTAL_COPIED_SLOT,
            hitMaskOutGm_.GetValue(FSA_DIAG_PREFETCH_TOTAL_COPIED_SLOT) + prefetched);
        hitMaskOutGm_.SetValue(FSA_DIAG_PREFETCH_TOTAL_WINDOWS_SLOT,
            hitMaskOutGm_.GetValue(FSA_DIAG_PREFETCH_TOTAL_WINDOWS_SLOT) + 1);
        if (copiedSlot >= 0) {
            hitMaskOutGm_.SetValue(copiedSlot, hitMaskOutGm_.GetValue(copiedSlot) + prefetched);
        }
        if (windowsSlot >= 0) {
            hitMaskOutGm_.SetValue(windowsSlot, hitMaskOutGm_.GetValue(windowsSlot) + 1);
        }
        pipe_barrier(PIPE_ALL);
    }
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::DiagRecordNextChunkPrefetchCounterMax(
    int32_t slot, int32_t value)
{
#ifndef ASCENDC_CPU_DEBUG
    if constexpr (FSA_DIAG_NEXT_CHUNK_PREFETCH_COUNTERS) {
        if (blkIdx_ == 0 && slot >= 0 && tiling_->topk > slot) {
            int32_t current = hitMaskOutGm_.GetValue(slot);
            if (value > current) {
                hitMaskOutGm_.SetValue(slot, value);
                pipe_barrier(PIPE_ALL);
            }
        }
    }
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::WaitAivStage(int32_t syncStage, bool isMm2Stage)
{
#ifndef ASCENDC_CPU_DEBUG
    if constexpr (FSA_ENABLE_AIC_STAGE_POLL) {
        if (isMm2Stage || syncStage <= 1) {
            if (isMm2Stage) {
                FsaWaitAivToAic2Flag(syncStage);
            } else {
                FsaWaitAivToAicFlag(syncStage);
            }
            pipe_barrier(PIPE_ALL);
            return;
        }
        while (ReadCommValue(COMM_STAGE) != syncStage) {
            pipe_barrier(PIPE_ALL);
        }
        pipe_barrier(PIPE_ALL);
    } else {
        if (isMm2Stage) {
            FsaWaitAivToAic2Flag(syncStage);
        } else {
            FsaWaitAivToAicFlag(syncStage);
        }
        pipe_barrier(PIPE_ALL);
    }
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::WaitAICStageDone(int32_t syncStage)
{
#ifndef ASCENDC_CPU_DEBUG
    (void)syncStage;
    pipe_barrier(PIPE_ALL);
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::WaitAICAckStage(int32_t syncStage)
{
#ifndef ASCENDC_CPU_DEBUG
    while (ReadCommValue(COMM_ACK_STAGE) != syncStage) {
    }
    pipe_barrier(PIPE_ALL);
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::WaitAICMm1Done(int32_t syncStage)
{
#ifndef ASCENDC_CPU_DEBUG
    if constexpr (FSA_DIAG_WAIT_AIC_DONE_BY_ACK) {
        WaitAICAckStage(syncStage);
    } else {
        CrossCoreWaitFlag<FSA_SYNC_MODE, PIPE_V>(
            FsaAicDoneFlagForCurrentAivLane(FsaAicMm1DoneFlag(syncStage)));
        pipe_barrier(PIPE_ALL);
    }
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::WaitAICMm2Done(int32_t syncStage)
{
#ifndef ASCENDC_CPU_DEBUG
    if constexpr (FSA_DIAG_WAIT_AIC_DONE_BY_ACK) {
        WaitAICAckStage(syncStage);
    } else {
        CrossCoreWaitFlag<FSA_SYNC_MODE, PIPE_V>(
            FsaAicDoneFlagForCurrentAivLane(FsaAicMm2DoneFlag(syncStage)));
        pipe_barrier(PIPE_ALL);
    }
#endif
}

template <typename T>
__aicore__ inline int64_t FusedAttentionMainOp<T>::AicMm2OutputOffset(int64_t mm2OutOff, bool isFirstBlock)
{
    if constexpr (FSA_DIAG_AIC_MM2_FORCE_MAIN_SLOT) {
        return mm2OutOff;
    }
    if constexpr (FSA_ENABLE_AIV_MM2_ACCUMULATE) {
        if (!isFirstBlock) {
            int64_t kvDimFloatAlign = CeilAlign(tiling_->kvCacheDim,
                static_cast<int64_t>(BLOCK_BYTES / sizeof(float)));
            return Mm2TempOutputOffset(mm2OutOff, kvDimFloatAlign);
        }
    }
    return mm2OutOff;
}

template <typename T>
__aicore__ inline bool FusedAttentionMainOp<T>::AicMm2UseOverwrite(bool isFirstBlock)
{
    if constexpr (FSA_ENABLE_AIV_MM2_ACCUMULATE) {
        return true;
    }
    return isFirstBlock;
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::WriteAicMm2Marker(int64_t outputGmOffset)
{
    int64_t kvDimFloatAlign = CeilAlign(tiling_->kvCacheDim,
        static_cast<int64_t>(BLOCK_BYTES / sizeof(float)));
    for (int64_t d = 0; d < kvDimFloatAlign; d++) {
        float marker = (d == 0) ? 7.0f : 0.0f;
        workspaceGm_.SetValue(outputGmOffset + d, marker);
    }
    pipe_barrier(PIPE_ALL);
}

template <typename T>
__aicore__ inline int64_t FusedAttentionMainOp<T>::Mm2HeadStride(int64_t kvDimFloatAlign)
{
    return FSA_MM2_WORKSPACE_SLOT_COUNT * FSA_MM2_FIXPIPE_M_ALIGN * kvDimFloatAlign;
}

template <typename T>
__aicore__ inline int64_t FusedAttentionMainOp<T>::Mm2TempOutputOffset(
    int64_t mm2OutOff, int64_t kvDimFloatAlign)
{
    return mm2OutOff + FSA_MM2_FIXPIPE_M_ALIGN * kvDimFloatAlign;
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::AccumulateAivMm2TempIfNeeded(
    bool syncOnly, bool isFirstBlock, int64_t mm2OutOff, int64_t kvDimFloatAlign)
{
#ifndef ASCENDC_CPU_DEBUG
    if constexpr (FSA_ENABLE_AIV_MM2_ACCUMULATE) {
        if (syncOnly && !isFirstBlock && !FSA_DIAG_SKIP_AIV_MM2_TEMP_ACCUMULATE) {
            vectorService_.AccumulateMm2Temp(
                mm2OutOff, Mm2TempOutputOffset(mm2OutOff, kvDimFloatAlign), kvDimFloatAlign);
        }
    }
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::WriteDoneAndSignalAIC(int32_t syncStage)
{
    commBufGm_.SetValue(COMM_TOKEN_COUNT, -1);
    commBufGm_.SetValue(COMM_ACK_STAGE, syncStage - 1);
    commBufGm_.SetValue(COMM_STAGE, syncStage);
    pipe_barrier(PIPE_ALL);
#ifndef ASCENDC_CPU_DEBUG
    if constexpr (!FSA_ENABLE_AIC_STAGE_POLL) {
        CrossCoreSetFlag<FSA_SYNC_MODE, PIPE_MTE3>(FsaAivToAicFlag(syncStage));
    }
#endif

}

// ============================================================================
// DetectHitsForHead (same as original kernel)
// ============================================================================
template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::DetectHitsForHead(
    int64_t hnIdx, LocalTensor<int32_t>& topkLocal,
    LocalTensor<int32_t>& statLocal, LocalTensor<int32_t>& hitFlagLocal)
{
    for (int64_t i = 0; i < tiling_->topk; i++) {
        int32_t topkVal = topkLocal.GetValue(i);
        if (topkVal < 0) continue;
        for (int64_t j = 0; j < tiling_->topk; j++) {
            int32_t statVal = statLocal.GetValue(j);
            if (topkVal == statVal) {
                hitFlagLocal.SetValue(i, HIT_FLAG);
                break;
            }
        }
    }
}

// ============================================================================
// IssueSingleGather: async copy full_kv 鈫?sel_kv for one miss token
// ============================================================================
template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::IssueSingleGather(
    int64_t fullKvAddr, int64_t fullRopeAddr,
    int64_t selKvAddr, int64_t selRopeAddr,
    int64_t gatherBlockSize)
{
#ifdef ASCENDC_CPU_DEBUG
    // CPU sim: direct element-wise copy (DataCopyPad doesn't work in tikicpulib)
    int64_t kvCount = gatherBlockSize * tiling_->kvCacheDim;
    for (int64_t i = 0; i < kvCount; i++) {
        selKvCacheGm_.SetValue(selKvAddr + i, fullKvCacheGm_.GetValue(fullKvAddr + i));
    }
    if (tiling_->kRopeDim > 0) {
        int64_t ropeCount = gatherBlockSize * tiling_->kRopeDim;
        for (int64_t i = 0; i < ropeCount; i++) {
            selKRopeGm_.SetValue(selRopeAddr + i, fullKRopeGm_.GetValue(fullRopeAddr + i));
        }
    }
#else
    // NPU path: DMA gather via UB buffer
    LocalTensor<T> gBuf = gatherQue_.AllocTensor<T>();
    DataCopyPadExtParams<T> gPad{false, 0, 0, 0};

    // MTE2: full_kv_cache(GM) 鈫?gBuf(UB)
    DataCopyExtParams kvP{static_cast<uint16_t>(1),
        static_cast<uint32_t>(gatherBlockSize * tiling_->kvCacheDim * sizeof(T)), 0, 0, 0};
    DataCopyPad(gBuf, fullKvCacheGm_[fullKvAddr], kvP, gPad);

    if (tiling_->kRopeDim > 0) {
        DataCopyExtParams rP{static_cast<uint16_t>(1),
            static_cast<uint32_t>(gatherBlockSize * tiling_->kRopeDim * sizeof(T)), 0, 0, 0};
        DataCopyPad(gBuf[kRopeUbOffset_], fullKRopeGm_[fullRopeAddr], rP, gPad);
    }

    gatherQue_.EnQue(gBuf);
    gBuf = gatherQue_.DeQue<T>();

    // MTE3: gBuf(UB) 鈫?sel_kv_cache(GM) 鈥?async
    DataCopyPad(selKvCacheGm_[selKvAddr], gBuf, kvP);
    if (tiling_->kRopeDim > 0) {
        DataCopyExtParams rP{static_cast<uint16_t>(1),
            static_cast<uint32_t>(gatherBlockSize * tiling_->kRopeDim * sizeof(T)), 0, 0, 0};
        DataCopyPad(selKRopeGm_[selRopeAddr], gBuf[kRopeUbOffset_], rP);
    }
    gatherQue_.FreeTensor(gBuf);
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::InsertStatusHash(
    int32_t topKId, int32_t pos,
    LocalTensor<int32_t>& statusHashKeyLocal,
    LocalTensor<int32_t>& statusHashPosLocal,
    int32_t statusHashSlots)
{
    if (topKId < 0 || pos < 0 || statusHashSlots <= 0) {
        return;
    }
    int32_t slot = topKId % statusHashSlots;
    for (int32_t probe = 0; probe < statusHashSlots; probe++) {
        int32_t curKey = statusHashKeyLocal.GetValue(slot);
        if (curKey < 0 || curKey == topKId) {
            statusHashKeyLocal.SetValue(slot, topKId);
            statusHashPosLocal.SetValue(slot, pos);
            break;
        }
        slot++;
        if (slot == statusHashSlots) {
            slot = 0;
        }
    }
}

template <typename T>
__aicore__ inline int32_t FusedAttentionMainOp<T>::PrefetchNextChunkGathers(
    int64_t nextChunkTopkIdx, int32_t& nextChunkInsertEnd,
    int64_t& scanTopkIdx, int32_t& scanInsertIdx, int64_t curBatch,
    int64_t selBlkTableOff, int32_t maxSelectionId, int64_t lastGatherBlockSize,
    LocalTensor<int32_t>& curStatLocal, LocalTensor<int32_t>& topkLocal,
    LocalTensor<int32_t>& selKvBlockTableLocal, bool updateSelectionCache,
    bool allowStatusHit, LocalTensor<int32_t>& statusHashKeyLocal,
    LocalTensor<int32_t>& statusHashPosLocal, int32_t statusHashSlots,
    int32_t gatherEvtIdValue, bool& gatherSynced, int32_t hitCopyStableSourceEnd,
    int32_t prefetchLimit, int32_t prefetchSource, bool deferFinalDrain, bool allowHitCopyPrefetch)
{
#ifdef ASCENDC_CPU_DEBUG
    (void)nextChunkTopkIdx;
    (void)nextChunkInsertEnd;
    (void)scanTopkIdx;
    (void)scanInsertIdx;
    (void)curBatch;
    (void)selBlkTableOff;
    (void)maxSelectionId;
    (void)lastGatherBlockSize;
    (void)curStatLocal;
    (void)topkLocal;
    (void)selKvBlockTableLocal;
    (void)updateSelectionCache;
    (void)allowStatusHit;
    (void)statusHashKeyLocal;
    (void)statusHashPosLocal;
    (void)statusHashSlots;
    (void)gatherEvtIdValue;
    (void)gatherSynced;
    (void)hitCopyStableSourceEnd;
    (void)prefetchLimit;
    (void)prefetchSource;
    (void)deferFinalDrain;
    (void)allowHitCopyPrefetch;
    return 0;
#else
    if constexpr (!FSA_ENABLE_NEXT_CHUNK_GATHER_PREFETCH) {
        (void)nextChunkTopkIdx;
        (void)nextChunkInsertEnd;
        (void)scanTopkIdx;
        (void)scanInsertIdx;
        (void)curBatch;
        (void)selBlkTableOff;
        (void)maxSelectionId;
        (void)lastGatherBlockSize;
        (void)curStatLocal;
        (void)topkLocal;
        (void)selKvBlockTableLocal;
        (void)updateSelectionCache;
        (void)allowStatusHit;
        (void)statusHashKeyLocal;
        (void)statusHashPosLocal;
        (void)statusHashSlots;
        (void)gatherEvtIdValue;
        (void)gatherSynced;
        (void)hitCopyStableSourceEnd;
        (void)prefetchLimit;
        (void)prefetchSource;
        (void)deferFinalDrain;
        (void)allowHitCopyPrefetch;
        return 0;
    }

    if (!updateSelectionCache || nextChunkTopkIdx >= tiling_->topk || scanTopkIdx >= tiling_->topk ||
        scanInsertIdx >= nextChunkInsertEnd || prefetchLimit <= 0) {
        return 0;
    }
    if (scanTopkIdx < nextChunkTopkIdx) {
        scanTopkIdx = nextChunkTopkIdx;
    }
    if (nextChunkInsertEnd > tiling_->topk) {
        nextChunkInsertEnd = static_cast<int32_t>(tiling_->topk);
    }

    bool deferOutstandingGatherDrain = false;
    if constexpr (FSA_ENABLE_DIRECT_PREFETCH_BATCH_WITH_CURRENT_GATHER) {
        deferOutstandingGatherDrain =
            (prefetchSource == FSA_PREFETCH_SOURCE_DIRECT_MM1) ||
            (prefetchSource == FSA_PREFETCH_SOURCE_DIRECT_MM2);
    }
    if constexpr (FSA_ENABLE_HIT_ONLY_PREFETCH_BATCH_ACROSS_MM1_MM2) {
        deferOutstandingGatherDrain = deferOutstandingGatherDrain ||
            (prefetchSource == FSA_PREFETCH_SOURCE_HIT_ONLY_MM1) ||
            (prefetchSource == FSA_PREFETCH_SOURCE_HIT_ONLY_MM2);
    }
    event_t gatherEvtId = static_cast<event_t>(gatherEvtIdValue);
    if (!gatherSynced && !deferOutstandingGatherDrain) {
        SetFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
        WaitFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
        gatherSynced = true;
    }

    bool enableHitCopyPrefetch = allowHitCopyPrefetch;
    if constexpr (!FSA_ENABLE_NEXT_CHUNK_HIT_COPY_PREFETCH) {
        enableHitCopyPrefetch = false;
    }

    int32_t aliasCheckInsertIdx = scanInsertIdx;
    int32_t aliasSafeInsertEnd = nextChunkInsertEnd;
    for (int64_t t = scanTopkIdx; t < tiling_->topk && aliasCheckInsertIdx < nextChunkInsertEnd; t++) {
        int32_t topKId = topkLocal.GetValue(t);
        if (topKId < 0 || topKId > maxSelectionId || aliasCheckInsertIdx < 0 ||
            aliasCheckInsertIdx >= tiling_->topk) {
            continue;
        }

        int64_t selKvBlkTableIdx =
            (static_cast<int64_t>(aliasCheckInsertIdx) * tiling_->selTopKBlockSize) / tiling_->selKvBlockSize;
        int32_t selKvBlockNumIdx = selKvBlockTableLocal.GetValue(selBlkTableOff + selKvBlkTableIdx);
        if (selKvBlockNumIdx < 0) {
            continue;
        }

        bool isHit = (curStatLocal.GetValue(aliasCheckInsertIdx) == topKId);
        int32_t hitPos = isHit ? aliasCheckInsertIdx : static_cast<int32_t>(-1);
        if (!isHit && curStatLocal.GetValue(t) == topKId) {
            isHit = true;
            hitPos = static_cast<int32_t>(t);
        }
        if (!isHit && allowStatusHit) {
            int32_t slot = topKId % statusHashSlots;
            for (int32_t probe = 0; probe < statusHashSlots; probe++) {
                int32_t curKey = statusHashKeyLocal.GetValue(slot);
                if (curKey == topKId) {
                    isHit = true;
                    hitPos = statusHashPosLocal.GetValue(slot);
                    break;
                }
                if (curKey < 0) {
                    break;
                }
                slot++;
                if (slot == statusHashSlots) {
                    slot = 0;
                }
            }
        }

        if (isHit) {
            int64_t srcSelKvBlkTableIdx =
                (static_cast<int64_t>(hitPos) * tiling_->selTopKBlockSize) / tiling_->selKvBlockSize;
            int32_t srcSelKvBlockNumIdx = selKvBlockTableLocal.GetValue(selBlkTableOff + srcSelKvBlkTableIdx);
            if (srcSelKvBlockNumIdx < 0) {
                continue;
            }
            if (hitPos != aliasCheckInsertIdx && hitPos >= scanInsertIdx && hitPos < nextChunkInsertEnd) {
                if (hitPos < aliasSafeInsertEnd) {
                    aliasSafeInsertEnd = hitPos;
                }
            }
            aliasCheckInsertIdx++;
            continue;
        }

        int64_t kvBlkTableIdx = (topKId * tiling_->selTopKBlockSize) / tiling_->fullKvBlockSize;
        int32_t kvBlockNumIdx = fullKvBlockTableGm_.GetValue(curBatch * tiling_->fullMaxBlockNum + kvBlkTableIdx);
        if (kvBlockNumIdx < 0) {
            continue;
        }
        aliasCheckInsertIdx++;
    }
    if (aliasSafeInsertEnd < nextChunkInsertEnd) {
        nextChunkInsertEnd = aliasSafeInsertEnd;
    }
    if (scanInsertIdx >= nextChunkInsertEnd) {
        return 0;
    }

    int32_t prefetched = 0;
    while (scanTopkIdx < tiling_->topk && scanInsertIdx < nextChunkInsertEnd && prefetched < prefetchLimit) {
        int32_t topKId = topkLocal.GetValue(scanTopkIdx);
        if (topKId < 0 || topKId > maxSelectionId || scanInsertIdx < 0 ||
            scanInsertIdx >= tiling_->topk) {
            scanTopkIdx++;
            continue;
        }

        int64_t gatherBlockSize = (topKId == maxSelectionId) ? lastGatherBlockSize : tiling_->selTopKBlockSize;
        int64_t selKvBlkTableIdx =
            (static_cast<int64_t>(scanInsertIdx) * tiling_->selTopKBlockSize) / tiling_->selKvBlockSize;
        int64_t selKvBlkSizeOff =
            (static_cast<int64_t>(scanInsertIdx) * tiling_->selTopKBlockSize) % tiling_->selKvBlockSize;
        int32_t selKvBlockNumIdx = selKvBlockTableLocal.GetValue(selBlkTableOff + selKvBlkTableIdx);
        if (selKvBlockNumIdx < 0) {
            scanTopkIdx++;
            continue;
        }
        int64_t selKRopeAddr = selKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kRopeDim
            + selKvBlkSizeOff * tiling_->kRopeDim;
        int64_t selKvCacheAddr = selKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kvCacheDim
            + selKvBlkSizeOff * tiling_->kvCacheDim;

        bool isHit = (curStatLocal.GetValue(scanInsertIdx) == topKId);
        int32_t hitPos = isHit ? scanInsertIdx : static_cast<int32_t>(-1);
        if (!isHit && curStatLocal.GetValue(scanTopkIdx) == topKId) {
            isHit = true;
            hitPos = static_cast<int32_t>(scanTopkIdx);
        }
        if (!isHit && allowStatusHit) {
            int32_t slot = topKId % statusHashSlots;
            for (int32_t probe = 0; probe < statusHashSlots; probe++) {
                int32_t curKey = statusHashKeyLocal.GetValue(slot);
                if (curKey == topKId) {
                    isHit = true;
                    hitPos = statusHashPosLocal.GetValue(slot);
                    break;
                }
                if (curKey < 0) {
                    break;
                }
                slot++;
                if (slot == statusHashSlots) {
                    slot = 0;
                }
            }
        }
        if (isHit) {
            int64_t srcSelKvBlkTableIdx =
                (static_cast<int64_t>(hitPos) * tiling_->selTopKBlockSize) / tiling_->selKvBlockSize;
            int64_t srcSelKvBlkSizeOff =
                (static_cast<int64_t>(hitPos) * tiling_->selTopKBlockSize) % tiling_->selKvBlockSize;
            int32_t srcSelKvBlockNumIdx = selKvBlockTableLocal.GetValue(selBlkTableOff + srcSelKvBlkTableIdx);
            if (srcSelKvBlockNumIdx < 0) {
                scanTopkIdx++;
                continue;
            }
            if (enableHitCopyPrefetch && hitPos < hitCopyStableSourceEnd) {
                int64_t srcKRopeAddr = srcSelKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kRopeDim
                    + srcSelKvBlkSizeOff * tiling_->kRopeDim;
                int64_t srcKvCacheAddr = srcSelKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kvCacheDim
                    + srcSelKvBlkSizeOff * tiling_->kvCacheDim;
                IssueSingleSelectionCopy(srcKvCacheAddr, srcKRopeAddr, selKvCacheAddr, selKRopeAddr,
                    gatherBlockSize);
                gatherSynced = false;
                curStatLocal.SetValue(scanInsertIdx, topKId);
                prefetched++;
            }
            scanInsertIdx++;
            scanTopkIdx++;
            continue;
        }

        int64_t kvBlkTableIdx = (topKId * tiling_->selTopKBlockSize) / tiling_->fullKvBlockSize;
        int64_t kvBlkSizeOff = (topKId * tiling_->selTopKBlockSize) % tiling_->fullKvBlockSize;
        int32_t kvBlockNumIdx = fullKvBlockTableGm_.GetValue(curBatch * tiling_->fullMaxBlockNum + kvBlkTableIdx);
        if (kvBlockNumIdx < 0) {
            scanTopkIdx++;
            continue;
        }
        int64_t fullKvAddr = kvBlockNumIdx * tiling_->fullKvBlockSize * tiling_->kvCacheDim
            + kvBlkSizeOff * tiling_->kvCacheDim;
        int64_t fullRopeAddr = kvBlockNumIdx * tiling_->fullKvBlockSize * tiling_->kRopeDim
            + kvBlkSizeOff * tiling_->kRopeDim;

        IssueSingleGather(fullKvAddr, fullRopeAddr, selKvCacheAddr, selKRopeAddr, gatherBlockSize);
        gatherSynced = false;

        curStatLocal.SetValue(scanInsertIdx, topKId);
        if (allowStatusHit) {
            InsertStatusHash(topKId, scanInsertIdx,
                statusHashKeyLocal, statusHashPosLocal, statusHashSlots);
        }
        prefetched++;
        scanInsertIdx++;
        scanTopkIdx++;
    }

    if (!gatherSynced && !deferFinalDrain) {
        SetFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
        WaitFlag<HardEvent::MTE3_MTE2>(gatherEvtId);
        gatherSynced = true;
    }
    DiagRecordNextChunkPrefetch(prefetchSource, prefetched);

    #ifdef ASCENDC_CPU_DEBUG
    if constexpr (FSA_DIAG_NEXT_CHUNK_PREFETCH) {
        if (prefetched > 0) {
            printf("[FSA] next_chunk_prefetch start=%ld copied=%d limit=%d\n",
                nextChunkTopkIdx, prefetched, prefetchLimit);
        }
    }
    #endif
    return prefetched;
#endif
}

template <typename T>
__aicore__ inline void FusedAttentionMainOp<T>::IssueSingleSelectionCopy(
    int64_t srcKvAddr, int64_t srcRopeAddr,
    int64_t dstKvAddr, int64_t dstRopeAddr,
    int64_t gatherBlockSize)
{
    if (srcKvAddr == dstKvAddr && srcRopeAddr == dstRopeAddr) {
        return;
    }
#ifdef ASCENDC_CPU_DEBUG
    int64_t kvCount = gatherBlockSize * tiling_->kvCacheDim;
    for (int64_t i = 0; i < kvCount; i++) {
        selKvCacheGm_.SetValue(dstKvAddr + i, selKvCacheGm_.GetValue(srcKvAddr + i));
    }
    if (tiling_->kRopeDim > 0) {
        int64_t ropeCount = gatherBlockSize * tiling_->kRopeDim;
        for (int64_t i = 0; i < ropeCount; i++) {
            selKRopeGm_.SetValue(dstRopeAddr + i, selKRopeGm_.GetValue(srcRopeAddr + i));
        }
    }
#else
    LocalTensor<T> gBuf = gatherQue_.AllocTensor<T>();
    DataCopyPadExtParams<T> gPad{false, 0, 0, 0};

    DataCopyExtParams kvP{static_cast<uint16_t>(1),
        static_cast<uint32_t>(gatherBlockSize * tiling_->kvCacheDim * sizeof(T)), 0, 0, 0};
    DataCopyPad(gBuf, selKvCacheGm_[srcKvAddr], kvP, gPad);

    if (tiling_->kRopeDim > 0) {
        DataCopyExtParams rP{static_cast<uint16_t>(1),
            static_cast<uint32_t>(gatherBlockSize * tiling_->kRopeDim * sizeof(T)), 0, 0, 0};
        DataCopyPad(gBuf[kRopeUbOffset_], selKRopeGm_[srcRopeAddr], rP, gPad);
    }

    gatherQue_.EnQue(gBuf);
    gBuf = gatherQue_.DeQue<T>();

    DataCopyPad(selKvCacheGm_[dstKvAddr], gBuf, kvP);
    if (tiling_->kRopeDim > 0) {
        DataCopyExtParams rP{static_cast<uint16_t>(1),
            static_cast<uint32_t>(gatherBlockSize * tiling_->kRopeDim * sizeof(T)), 0, 0, 0};
        DataCopyPad(selKRopeGm_[dstRopeAddr], gBuf[kRopeUbOffset_], rP);
    }
    gatherQue_.FreeTensor(gBuf);
#endif
}

} // namespace FusedSparseAttentionOverlapNs

#endif // FUSED_SPARSE_ATTENTION_OVERLAP_MAIN_H_


