/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_li_manage_c8_kernel.h
 * \brief C8 variant orchestration class for fused_li_manage.
 */

#ifndef FUSED_LI_MANAGE_C8_KERNEL_H
#define FUSED_LI_MANAGE_C8_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "fused_li_manage_c8_common.h"
#include "fused_li_manage_service_vector.h"
#include "fused_li_manage_c8_service_cube.h"

namespace LIKernel {
using namespace LICommon;
using namespace matmul;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

class LIPreloadC8 {
public:
    using Q_T = int8_t;
    using K_T = int8_t;
    using MM1_OUT_T = float;

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key,
                                __gm__ uint8_t *weights, __gm__ uint8_t *queryScale,
                                __gm__ uint8_t *keyScaleCache,
                                __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *cacheSlots,
                                __gm__ uint8_t *numCandidateTokens, __gm__ uint8_t *numCacheTokens,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *topkIndex,
                                __gm__ uint8_t *topkSlots, __gm__ uint8_t *missCount,
                                __gm__ uint8_t *workspace, const FusedLiManageC8TilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();

private:
    static constexpr uint32_t WS_DOUBLE = 2;

    LIMatmulC8<LITypeC8Cube> matmulService;
    LIVector<LITypeC8Vector, float> vectorService;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<half> queryScaleGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<half> keyScaleGm;
    GlobalTensor<bfloat16_t> weightsGm;
    GlobalTensor<int32_t> cacheSlotsGm;
    GlobalTensor<int32_t> reqPoolEntriesGm;
    GlobalTensor<int32_t> cacheTokensGm;
    GlobalTensor<int32_t> topkIndexGm;
    GlobalTensor<int32_t> topkSlotsGm;
    GlobalTensor<int32_t> missCountGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<uint32_t> actualSeqLengthsGm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<half> weightWorkspaceGm;
    GlobalTensor<float> scoresGm;
    GlobalTensor<float> partialTopkGm;
    GlobalTensor<int32_t> partialMetaGm;

    uint32_t tmpBlockIdx = 0;
    uint32_t aiCoreIdx = 0;
    uint32_t requestStart = 0;
    uint32_t requestCount = 0;
    uint32_t coreNum = 0;
    uint32_t scheduleMode = 0;
    bool balancedSchedule = false;
    uint32_t finalizeRequestIdx = ~0U;
    uint32_t finalizeCacheRowIdx = 0;
    uint32_t finalizeActualSeqLen = 0;
    uint32_t finalizeCacheTokenCount = 0;
    uint32_t finalizeRequestChunkEnd = 0;
    LICommon::ConstInfo constInfo{};

    __aicore__ inline bool ResolveBalancedSchedule(uint32_t requestedCoreNum);
    __aicore__ inline void InitRequestRange(uint32_t requestedCoreNum);
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBalanced();
    __aicore__ inline void ProcessBalancedMain(uint32_t globalChunkStart, uint32_t globalChunkCount);
    __aicore__ inline uint32_t GetChunkCount(uint32_t bIdx);
    __aicore__ inline void ProcessRequestSegment(uint32_t bIdx, uint32_t cacheRowIdx,
                                                 uint32_t actualSeqLen, uint32_t cacheTokenCount,
                                                 uint32_t chunkStart, uint32_t chunkEnd,
                                                 uint32_t partialSlot, uint32_t &loop);
    __aicore__ inline void ProcessChunk(const LICommon::RunInfo &runInfo);
    __aicore__ inline void CleanEmptyRequest(uint32_t bIdx);
};

__aicore__ inline void LIPreloadC8::InitRequestRange(uint32_t requestedCoreNum)
{
    coreNum = requestedCoreNum;
    if (balancedSchedule) {
        requestCount = 0;
        return;
    }
    uint32_t activeCoreNum = Min(requestedCoreNum, static_cast<uint32_t>(constInfo.batchSize));
    if (activeCoreNum == 0 || aiCoreIdx >= activeCoreNum) {
        requestCount = 0;
        return;
    }

    uint32_t requestsPerCore = static_cast<uint32_t>(constInfo.batchSize) / activeCoreNum;
    uint32_t extraRequestCores = static_cast<uint32_t>(constInfo.batchSize) % activeCoreNum;
    requestStart = aiCoreIdx * requestsPerCore + Min(aiCoreIdx, extraRequestCores);
    requestCount = requestsPerCore + (aiCoreIdx < extraRequestCores ? 1U : 0U);
}

__aicore__ inline bool LIPreloadC8::ResolveBalancedSchedule(uint32_t requestedCoreNum)
{
    constexpr uint32_t SCHEDULE_LOCAL = 0;
    constexpr uint32_t SCHEDULE_BALANCED = 1;
    constexpr uint32_t LOCAL_EXTRA_CHUNK_THRESHOLD = 20;
    if (scheduleMode == SCHEDULE_LOCAL) {
        return false;
    }
    if (scheduleMode == SCHEDULE_BALANCED) {
        return true;
    }

    uint32_t batchSize = static_cast<uint32_t>(constInfo.batchSize);
    if (requestedCoreNum == 0U || batchSize == 0U || batchSize % requestedCoreNum != 0U) {
        return true;
    }

    uint32_t requestsPerCore = batchSize / requestedCoreNum;
    uint32_t totalChunks = 0;
    uint32_t maxCoreChunks = 0;
    for (uint32_t coreIdx = 0; coreIdx < requestedCoreNum; ++coreIdx) {
        uint32_t coreChunks = 0;
        uint32_t requestBase = coreIdx * requestsPerCore;
        for (uint32_t requestOffset = 0; requestOffset < requestsPerCore; ++requestOffset) {
            coreChunks += GetChunkCount(requestBase + requestOffset);
        }
        totalChunks += coreChunks;
        maxCoreChunks = Max(maxCoreChunks, coreChunks);
    }

    uint32_t averageChunks = CeilDiv(totalChunks, requestedCoreNum);
    return maxCoreChunks > averageChunks + LOCAL_EXTRA_CHUNK_THRESHOLD;
}

__aicore__ inline void LIPreloadC8::Init(__gm__ uint8_t *query, __gm__ uint8_t *key,
                                         __gm__ uint8_t *weights, __gm__ uint8_t *queryScale,
                                         __gm__ uint8_t *keyScaleCache,
                                         __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *cacheSlots,
                                         __gm__ uint8_t *numCandidateTokens, __gm__ uint8_t *numCacheTokens,
                                         __gm__ uint8_t *blockTable, __gm__ uint8_t *topkIndex,
                                         __gm__ uint8_t *topkSlots, __gm__ uint8_t *missCount,
                                         __gm__ uint8_t *workspace, const FusedLiManageC8TilingData *__restrict tiling,
                                         TPipe *tPipe)
{
    tmpBlockIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        aiCoreIdx = tmpBlockIdx;
    }

    constInfo.batchSize = tiling->bSize;
    constInfo.kSeqSize = tiling->s2Size;
    constInfo.kCacheBlockSize = tiling->blockSize;
    constInfo.maxBlockNumPerBatch = tiling->maxBlockNumPerBatch;
    constInfo.poolSize = tiling->poolSize;
    constInfo.cacheSlotsSize = tiling->cacheSlotsSize;
    constInfo.qHeadNum = tiling->n1Size;
    scheduleMode = tiling->scheduleMode;

    uint64_t singleCoreMm1ResSize =
        WS_DOUBLE * constInfo.s2BaseSize * sizeof(MM1_OUT_T);
    mm1ResGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + aiCoreIdx * singleCoreMm1ResSize));
    // Brcb'd (w*q_scale) fp16 operand for the cube-side head reduction,
    // written by the vector side once per request segment (double buffered).
    uint64_t singleCoreWeightSize =
        WS_DOUBLE * constInfo.qHeadNum * BLOCK_CUBE * sizeof(half);
    uint64_t weightWsOffset = static_cast<uint64_t>(tiling->usedCoreNum) * singleCoreMm1ResSize;
    weightWorkspaceGm.SetGlobalBuffer(
        (__gm__ half *)(workspace + weightWsOffset + aiCoreIdx * singleCoreWeightSize));
    uint64_t scoresOffset = weightWsOffset +
                            static_cast<uint64_t>(tiling->usedCoreNum) * singleCoreWeightSize;
    scoresGm.SetGlobalBuffer((__gm__ float *)(workspace + scoresOffset));
    uint64_t scoreStride = CeilDiv(static_cast<uint64_t>(constInfo.kSeqSize),
                                   static_cast<uint64_t>(constInfo.s2BaseSize)) * constInfo.s2BaseSize;
    uint64_t partialTopkOffset =
        scoresOffset + constInfo.batchSize * scoreStride * sizeof(float);
    partialTopkGm.SetGlobalBuffer((__gm__ float *)(workspace + partialTopkOffset));
    uint64_t partialMetaOffset =
        partialTopkOffset + static_cast<uint64_t>(tiling->usedCoreNum) *
                                PARTIAL_SLOTS_PER_CORE * TOPK_PAIR_FLOATS * sizeof(float);
    partialMetaGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + partialMetaOffset));

    // Per-request sequence metadata arrives as two separate [B] int32 tensors.
    cacheTokensGm.SetGlobalBuffer((__gm__ int32_t *)numCacheTokens, constInfo.batchSize);
    actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)numCandidateTokens, constInfo.batchSize);
    reqPoolEntriesGm.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries, constInfo.batchSize);
    cacheSlotsGm.SetGlobalBuffer((__gm__ int32_t *)cacheSlots);
    balancedSchedule = ResolveBalancedSchedule(tiling->usedCoreNum);
    InitRequestRange(tiling->usedCoreNum);

    if ASCEND_IS_AIV {
        vectorService.InitParams(
            static_cast<uint32_t>(constInfo.kSeqSize),
            static_cast<uint32_t>(constInfo.qHeadNum),
            constInfo.cacheSlotsSize);
        // query_scale is [B, N1] fp16; key_scale_cache is [num_blocks, 128, 1, 1] fp16.
        uint32_t queryScaleCount = constInfo.batchSize * constInfo.qHeadNum;
        queryScaleGm.SetGlobalBuffer((__gm__ half *)queryScale, queryScaleCount);
        keyScaleGm.SetGlobalBuffer((__gm__ half *)keyScaleCache, tiling->keyScaleCount);
        weightsGm.SetGlobalBuffer((__gm__ bfloat16_t *)weights);
        topkIndexGm.SetGlobalBuffer((__gm__ int32_t *)topkIndex);
        topkSlotsGm.SetGlobalBuffer((__gm__ int32_t *)topkSlots);
        missCountGm.SetGlobalBuffer((__gm__ int32_t *)missCount);
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        vectorService.InitVec1GlobalTensor(mm1ResGm, weightsGm, cacheSlotsGm, topkIndexGm,
                                           topkSlotsGm, missCountGm, scoresGm,
                                           partialTopkGm, partialMetaGm);
        vectorService.InitC8GlobalTensor(queryScaleGm, keyScaleGm, blockTableGm, weightWorkspaceGm,
                                         constInfo.maxBlockNumPerBatch, constInfo.kCacheBlockSize);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        keyGm.SetGlobalBuffer((__gm__ K_T *)key);
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, mm1ResGm, weightWorkspaceGm);
    }
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(tPipe);
    } else {
        matmulService.InitBuffers(tPipe);
    }
}

__aicore__ inline void LIPreloadC8::CleanEmptyRequest(uint32_t bIdx)
{
    if ASCEND_IS_AIV {
        if ((tmpBlockIdx & 1U) == 0) {
            vectorService.WriteZeroMissCount(bIdx);
        }
    }
}

__aicore__ inline void LIPreloadC8::Process()
{
    if (balancedSchedule) {
        ProcessBalanced();
        return;
    }
    if (requestCount == 0) {
        return;
    }
    ProcessMain();
}

__aicore__ inline void LIPreloadC8::ProcessMain()
{
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
    }

    uint32_t loop = 0;
    for (uint32_t requestOffset = 0; requestOffset < requestCount; ++requestOffset) {
        uint32_t bIdx = requestStart + requestOffset;
        uint32_t actualSeqLen = actualSeqLengthsGm.GetValue(bIdx);
        if (actualSeqLen == 0 || actualSeqLen > LICommon::ConstInfo::maxActualSeqLen ||
            actualSeqLen > constInfo.kSeqSize ||
            actualSeqLen > constInfo.cacheSlotsSize) {
            CleanEmptyRequest(bIdx);
            continue;
        }

        int32_t cacheRowIdx = reqPoolEntriesGm.GetValue(bIdx);
        if (cacheRowIdx < 0 || static_cast<uint32_t>(cacheRowIdx) >= constInfo.poolSize) {
            CleanEmptyRequest(bIdx);
            continue;
        }
        int32_t cacheMetadata = cacheTokensGm.GetValue(bIdx);
        if (cacheMetadata == 0) {
            CleanEmptyRequest(bIdx);
            continue;
        }
        if (cacheMetadata < 2048) {
            CleanEmptyRequest(bIdx);
            continue;
        }
        uint32_t processSeqLen = actualSeqLen;
        if (static_cast<uint32_t>(cacheMetadata) > processSeqLen) {
            CleanEmptyRequest(bIdx);
            continue;
        }

        uint32_t chunkCount = CeilDiv(processSeqLen, constInfo.s2BaseSize);
        for (uint32_t chunkIdx = 0; chunkIdx < chunkCount; ++chunkIdx) {
            LICommon::RunInfo runInfo{};
            runInfo.loop = loop++;
            runInfo.bIdx = bIdx;
            runInfo.queryRow = bIdx;
            runInfo.queryIdx = 0;
            runInfo.s2Idx = chunkIdx;
            runInfo.segmentChunkIdx = chunkIdx;
            runInfo.actS2Size = processSeqLen;
            runInfo.cacheTokenCount = static_cast<uint32_t>(cacheMetadata);
            runInfo.cacheRowIdx = static_cast<uint32_t>(cacheRowIdx);
            uint32_t chunkStart = chunkIdx * constInfo.s2BaseSize;
            runInfo.actualSingleProcessSInnerSize =
                Min(constInfo.s2BaseSize, processSeqLen - chunkStart);
            runInfo.actualSingleProcessSInnerSizeAlign =
                LICommon::Align(runInfo.actualSingleProcessSInnerSize, LICommon::ConstInfo::BUFFER_SIZE_BYTE_32B);
            runInfo.isFirstS2InnerLoop = chunkIdx == 0;
            runInfo.isLastS2InnerLoop = chunkIdx + 1 == chunkCount;
            ProcessChunk(runInfo);
        }
    }

    if ASCEND_IS_AIC {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncV1C1);
        CrossCoreWaitFlag(constInfo.syncV1C1);
    } else {
        CrossCoreWaitFlag(constInfo.syncC1V0);
        CrossCoreWaitFlag(constInfo.syncC1V0);
    }
}

__aicore__ inline uint32_t LIPreloadC8::GetChunkCount(uint32_t bIdx)
{
    uint32_t actualSeqLen = actualSeqLengthsGm.GetValue(bIdx);
    if (actualSeqLen == 0 || actualSeqLen > LICommon::ConstInfo::maxActualSeqLen ||
        actualSeqLen > constInfo.kSeqSize ||
        actualSeqLen > constInfo.cacheSlotsSize) {
        return 0;
    }
    int32_t cacheRowIdx = reqPoolEntriesGm.GetValue(bIdx);
    if (cacheRowIdx < 0 || static_cast<uint32_t>(cacheRowIdx) >= constInfo.poolSize) {
        return 0;
    }
    int32_t cacheMetadata = cacheTokensGm.GetValue(bIdx);
    if (cacheMetadata < 2048 || static_cast<uint32_t>(cacheMetadata) > actualSeqLen) {
        return 0;
    }
    return CeilDiv(actualSeqLen, constInfo.s2BaseSize);
}

__aicore__ inline void LIPreloadC8::ProcessRequestSegment(
    uint32_t bIdx, uint32_t cacheRowIdx, uint32_t actualSeqLen, uint32_t cacheTokenCount,
    uint32_t chunkStart, uint32_t chunkEnd, uint32_t partialSlot, uint32_t &loop)
{
    uint32_t requestChunkCount = CeilDiv(actualSeqLen, constInfo.s2BaseSize);
    bool isPartialSegment = chunkStart != 0U || chunkEnd != requestChunkCount;
    for (uint32_t chunkIdx = chunkStart; chunkIdx < chunkEnd; ++chunkIdx) {
        LICommon::RunInfo runInfo{};
        runInfo.loop = loop++;
        runInfo.bIdx = bIdx;
        runInfo.queryRow = bIdx;
        runInfo.queryIdx = 0;
        runInfo.cacheRowIdx = cacheRowIdx;
        runInfo.s2Idx = chunkIdx;
        runInfo.segmentChunkIdx = chunkIdx - chunkStart;
        runInfo.actS2Size = actualSeqLen;
        runInfo.cacheTokenCount = cacheTokenCount;
        uint32_t chunkBase = chunkIdx * constInfo.s2BaseSize;
        runInfo.actualSingleProcessSInnerSize =
            Min(constInfo.s2BaseSize, actualSeqLen - chunkBase);
        runInfo.actualSingleProcessSInnerSizeAlign =
            LICommon::Align(runInfo.actualSingleProcessSInnerSize,
                            LICommon::ConstInfo::BUFFER_SIZE_BYTE_32B);
        runInfo.isFirstS2InnerLoop = chunkIdx == chunkStart;
        runInfo.isLastS2InnerLoop = chunkIdx + 1U == chunkEnd;
        runInfo.isPartialSegment = isPartialSegment;
        runInfo.partialSlot = partialSlot;
        ProcessChunk(runInfo);
    }
}

__aicore__ inline void LIPreloadC8::ProcessBalancedMain(uint32_t globalChunkStart,
                                                        uint32_t globalChunkCount)
{
    finalizeRequestIdx = ~0U;
    if (globalChunkCount == 0) {
        return;
    }
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
    }

    uint32_t globalChunkEnd = globalChunkStart + globalChunkCount;
    uint32_t requestChunkBase = 0;
    uint32_t partialSlot = 0;
    uint32_t loop = 0;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize && requestChunkBase < globalChunkEnd; ++bIdx) {
        uint32_t requestChunkCount = GetChunkCount(bIdx);
        uint32_t requestChunkEnd = requestChunkBase + requestChunkCount;
        if (requestChunkEnd > globalChunkStart && requestChunkBase < globalChunkEnd) {
            uint32_t overlapStart = Max(globalChunkStart, requestChunkBase);
            uint32_t overlapEnd = Min(globalChunkEnd, requestChunkEnd);
            uint32_t chunkStart = overlapStart - requestChunkBase;
            uint32_t chunkEnd = overlapEnd - requestChunkBase;
            int32_t cacheRowIdx = reqPoolEntriesGm.GetValue(bIdx);
            uint32_t cacheTokenCount =
                static_cast<uint32_t>(cacheTokensGm.GetValue(bIdx));
            uint32_t actualSeqLen = actualSeqLengthsGm.GetValue(bIdx);
            bool isPartial = chunkStart != 0U || chunkEnd != requestChunkCount;
            if (chunkStart == 0U && chunkEnd < requestChunkCount) {
                finalizeRequestIdx = bIdx;
                finalizeCacheRowIdx = static_cast<uint32_t>(cacheRowIdx);
                finalizeActualSeqLen = actualSeqLen;
                finalizeCacheTokenCount = cacheTokenCount;
                finalizeRequestChunkEnd = requestChunkEnd;
            }
            ProcessRequestSegment(
                bIdx, static_cast<uint32_t>(cacheRowIdx), actualSeqLen,
                cacheTokenCount, chunkStart, chunkEnd, partialSlot, loop);
            if (isPartial) {
                ++partialSlot;
            }
        }
        requestChunkBase = requestChunkEnd;
    }

    if ASCEND_IS_AIC {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncV1C1);
        CrossCoreWaitFlag(constInfo.syncV1C1);
    } else {
        CrossCoreWaitFlag(constInfo.syncC1V0);
        CrossCoreWaitFlag(constInfo.syncC1V0);
    }
}

__aicore__ inline void LIPreloadC8::ProcessBalanced()
{
    if ASCEND_IS_AIV {
        vectorService.InitPartialMetadata(aiCoreIdx);
    }

    uint32_t totalChunkCount = 0;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; ++bIdx) {
        totalChunkCount += GetChunkCount(bIdx);
    }
    uint32_t activeCoreNum = Min(coreNum, totalChunkCount);
    uint32_t chunksPerCore = activeCoreNum == 0 ? 0 : totalChunkCount / activeCoreNum;
    uint32_t extraChunkCores = activeCoreNum == 0 ? 0 : totalChunkCount % activeCoreNum;
    uint32_t globalChunkStart = 0;
    uint32_t globalChunkCount = 0;
    if (activeCoreNum > 0 && aiCoreIdx < activeCoreNum) {
        globalChunkStart = aiCoreIdx * chunksPerCore + Min(aiCoreIdx, extraChunkCores);
        globalChunkCount = chunksPerCore + (aiCoreIdx < extraChunkCores ? 1U : 0U);
    }
    ProcessBalancedMain(globalChunkStart, globalChunkCount);

    if ASCEND_IS_AIV {
        SyncAll();
        if ((tmpBlockIdx & 1U) != 0) {
            return;
        }

        for (uint32_t bIdx = aiCoreIdx; bIdx < constInfo.batchSize; bIdx += coreNum) {
            if (GetChunkCount(bIdx) == 0U) {
                CleanEmptyRequest(bIdx);
            }
        }

        if (finalizeRequestIdx != ~0U) {
            uint32_t requestLastChunk = finalizeRequestChunkEnd - 1U;
            uint32_t largeCoreSpan = extraChunkCores * (chunksPerCore + 1U);
            uint32_t lastOwner = requestLastChunk < largeCoreSpan
                                     ? requestLastChunk / (chunksPerCore + 1U)
                                     : extraChunkCores + (requestLastChunk - largeCoreSpan) / chunksPerCore;
            vectorService.FinalizePartialRequest(finalizeRequestIdx, finalizeCacheRowIdx,
                                                 finalizeActualSeqLen,
                                                 finalizeCacheTokenCount,
                                                 aiCoreIdx, lastOwner);
        }
    }
}

__aicore__ inline void LIPreloadC8::ProcessChunk(const LICommon::RunInfo &runInfo)
{
    if ASCEND_IS_AIC {
        if (runInfo.isFirstS2InnerLoop) {
            CrossCoreWaitFlag(constInfo.syncV0C1);
        }
        CrossCoreWaitFlag(constInfo.syncV1C1);
        matmulService.ComputeMm1(runInfo);
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
        if (runInfo.isLastS2InnerLoop) {
            CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
        }
    } else {
        if (runInfo.isFirstS2InnerLoop) {
            CrossCoreWaitFlag(constInfo.syncC1V0);
            vectorService.ProcessVec0(runInfo);
            CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV0C1);
        }
        CrossCoreWaitFlag(constInfo.syncC1V1);
        vectorService.ProcessVec(runInfo);
        // QLI sets this via PIPE_MTE3: the MTE3 score/topk writes are ordered
        // after the mm1Res MTE2 read through the V pipe, so an MTE3-drained
        // flag transitively guarantees the read has drained. PIPE_MTE2 lets
        // the cube overwrite the parity buffer while the read is in flight.
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C1);
    }
}

} // namespace LIKernel
#endif // FUSED_LI_MANAGE_C8_KERNEL_H
