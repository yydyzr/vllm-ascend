/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 *
 * Cube service for fused_sparse_attention_overlap.
 * Simplified Mm1: no kL1 split, load full 576-dim Q/K, 6 kL0 iterations.
 */
#ifndef FUSED_SPARSE_ATTENTION_OVERLAP_CUBE_H_
#define FUSED_SPARSE_ATTENTION_OVERLAP_CUBE_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

namespace FusedSparseAttentionOverlapNs {
using namespace AscendC;

constexpr uint32_t FSA_N_SPLIT_SIZE = 16;
constexpr uint32_t FSA_MM2_N_SPLIT_SIZE = 128;
constexpr uint32_t FSA_L0A_PP_SIZE = 32 * 1024;
constexpr uint32_t FSA_L0B_PP_SIZE = 32 * 1024;
constexpr uint32_t FSA_L0C_PP_SIZE = 64 * 1024;
// L1 block: 72K bytes (same as SFA)
constexpr uint32_t FSA_L1_BLOCK_SIZE = 64 * (512 + 64) * 2;  // 73728 bytes
constexpr IsResetLoad3dConfig FSA_LOAD3D_CONFIG = {true, true};
constexpr uint32_t FSA_MTE2_MTE1_EVENT = EVENT_ID2;
constexpr uint32_t FSA_L0AB_EVENT0 = EVENT_ID3;
constexpr uint32_t FSA_L0AB_EVENT1 = EVENT_ID4;
constexpr uint32_t FSA_L0C_EVENT0 = EVENT_ID5;
constexpr uint32_t FSA_L0C_EVENT1 = EVENT_ID6;
#ifndef FSA_DIAG_DISABLE_MM2_ATOMIC_ADD
#define FSA_DIAG_DISABLE_MM2_ATOMIC_ADD 0
#endif
constexpr bool FSA_DIAG_MM2_DIRECT_COPY = false;

__aicore__ inline uint32_t FsaL0ABEventId(uint32_t idx)
{
    return (idx == 0) ? FSA_L0AB_EVENT0 : FSA_L0AB_EVENT1;
}

__aicore__ inline uint32_t FsaL0CEventId(uint32_t idx)
{
    return (idx == 0) ? FSA_L0C_EVENT0 : FSA_L0C_EVENT1;
}

template <typename T>
class FusedAttentionCubeService {
public:
    using MM_OUT_T = float;
    __aicore__ inline FusedAttentionCubeService() {}
    __aicore__ inline void Init(TPipe* pipe,
                                const FusedSparseAttentionOverlapTilingData* tiling,
                                GM_ADDR workspace)
    {
        pipe_ = pipe;
        tiling_ = tiling;
        pipe_->InitBuffer(bufL0A_, FSA_L0A_PP_SIZE * 2);
        pipe_->InitBuffer(bufL0B_, FSA_L0B_PP_SIZE * 2);
        pipe_->InitBuffer(bufL0C_, FSA_L0C_PP_SIZE * 2);
        pipe_->InitBuffer(bufQPL1_, FSA_L1_BLOCK_SIZE * 2);  // 2 buffers for Q ping-pong
        pipe_->InitBuffer(bufKVL1_, FSA_L1_BLOCK_SIZE * 3);  // 3 buffers for KV pipeline
        mm1ResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MM_OUT_T*>(workspace));
        abL0BufIter_ = 0;
        cL0BufIter_ = 0;
        SetFlag<HardEvent::M_MTE1>(FSA_L0AB_EVENT0);
        SetFlag<HardEvent::M_MTE1>(FSA_L0AB_EVENT1);
    }

    __aicore__ inline void ComputeMm1(
        GlobalTensor<T>& queryGm, int64_t queryGmOffset,
        GlobalTensor<T>& kvCacheGm, GlobalTensor<T>& kRopeGm,
        GlobalTensor<int32_t>& blockTableGm,
        int32_t* tokenKvAddrs, int32_t* tokenRopeAddrs,
        int32_t nTokens, int64_t scoreGmOffset);
    __aicore__ inline void ComputeMm1MixedSource(
        GlobalTensor<T>& queryGm, int64_t queryGmOffset,
        GlobalTensor<T>& selKvCacheGm, GlobalTensor<T>& selKRopeGm,
        GlobalTensor<T>& fullKvCacheGm, GlobalTensor<T>& fullKRopeGm,
        int32_t* tokenKvAddrs, int32_t* tokenRopeAddrs, int32_t* sourceFlags,
        int32_t nTokens, int64_t scoreGmOffset);

    __aicore__ inline void ComputeMm2(
        GlobalTensor<T>& kvCacheGm, GlobalTensor<int32_t>& blockTableGm,
        int32_t* tokenKvAddrs, int32_t nTokens,
        GlobalTensor<T>& weightsGm, int64_t weightsGmOffset,
        GlobalTensor<MM_OUT_T>& outputGm, int64_t outputGmOffset,
        bool isFirstBlock);
    __aicore__ inline void ComputeMm2MixedSource(
        GlobalTensor<T>& selKvCacheGm, GlobalTensor<T>& fullKvCacheGm,
        int32_t* tokenKvAddrs, int32_t* sourceFlags, int32_t nTokens,
        GlobalTensor<T>& weightsGm, int64_t weightsGmOffset,
        GlobalTensor<MM_OUT_T>& outputGm, int64_t outputGmOffset,
        bool isFirstBlock);

    __aicore__ inline void LoadCommBuffer(
        GlobalTensor<int32_t>& commBufGm, int32_t* commSlots, int32_t slotCount);
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ResetPhaseEvents();

private:
    TPipe* pipe_ = nullptr;
    const FusedSparseAttentionOverlapTilingData* tiling_ = nullptr;
    TBuf<TPosition::A2> bufL0A_;
    TBuf<TPosition::B2> bufL0B_;
    TBuf<TPosition::CO1> bufL0C_;
    TBuf<TPosition::A1> bufQPL1_;
    TBuf<TPosition::B1> bufKVL1_;
    GlobalTensor<MM_OUT_T> mm1ResGm_;
    uint32_t abL0BufIter_ = 0;
    uint32_t cL0BufIter_ = 0;
};

template <typename T>
__aicore__ inline void FusedAttentionCubeService<T>::FreeEventID()
{
    WaitFlag<HardEvent::M_MTE1>(FSA_L0AB_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(FSA_L0AB_EVENT1);
}

template <typename T>
__aicore__ inline void FusedAttentionCubeService<T>::ResetPhaseEvents()
{
    WaitFlag<HardEvent::M_MTE1>(FSA_L0AB_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(FSA_L0AB_EVENT1);
    pipe_barrier(PIPE_ALL);
    SetFlag<HardEvent::M_MTE1>(FSA_L0AB_EVENT0);
    SetFlag<HardEvent::M_MTE1>(FSA_L0AB_EVENT1);
    abL0BufIter_ = 0;
    cL0BufIter_ = 0;
    pipe_barrier(PIPE_ALL);
}

template <typename T>
__aicore__ inline void FusedAttentionCubeService<T>::LoadCommBuffer(
    GlobalTensor<int32_t>& commBufGm, int32_t* commSlots, int32_t slotCount)
{
    if (slotCount <= 0) {
        return;
    }

    LocalTensor<int32_t> commLocal = bufQPL1_.template Get<int32_t>();
    DataCopy(commLocal, commBufGm, static_cast<uint32_t>(slotCount));
    SetFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);
    WaitFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);

    for (int32_t i = 0; i < slotCount; i++) {
        commSlots[i] = commLocal.GetValue(i);
    }
}

// ============================================================================
// ComputeMm1: Q[1, headDim] 脳 K[nTokens, headDim]^T 鈫?scores[1, nTokens]
// headDim = 576 = kvCacheDim(512) + kRopeDim(64)
// No kL1 split: load full 576-dim, use kL0Size=96, kL0Loops=6
// ============================================================================
template <typename T>
__aicore__ inline void FusedAttentionCubeService<T>::ComputeMm1(
    GlobalTensor<T>& queryGm, int64_t queryGmOffset,
    GlobalTensor<T>& kvCacheGm, GlobalTensor<T>& kRopeGm,
    GlobalTensor<int32_t>& blockTableGm,
    int32_t* tokenKvAddrs, int32_t* tokenRopeAddrs,
    int32_t nTokens, int64_t scoreGmOffset)
{
    if (nTokens <= 0) return;

    int64_t kvCacheDim = tiling_->kvCacheDim;   // 512
    int64_t kRopeDim = tiling_->kRopeDim;       // 64
    int64_t headDim = kvCacheDim + kRopeDim;    // 576
    // headDim aligned to 16 for NZ format
    uint32_t headDimAlign = (static_cast<uint32_t>(headDim) + 15) / 16 * 16;  // 576

    constexpr uint32_t mSizeAlign = 16;  // M=1 padded to 16

    uint32_t nSize = static_cast<uint32_t>(nTokens);
    uint32_t nL1Loops = (nSize + FSA_N_SPLIT_SIZE - 1) / FSA_N_SPLIT_SIZE;

    // K direction: full headDim, split into kL0 blocks of 96
    constexpr uint32_t kL0Size = 96;
    uint32_t kL0Loops = (headDimAlign + kL0Size - 1) / kL0Size;  // 576/96 = 6

    LocalTensor<T> l1QPTensor = bufQPL1_.template Get<T>();
    LocalTensor<T> l1KVTensor = bufKVL1_.template Get<T>();

    // Load Q to L1: Q_nope[0:512] then Q_rope[0:64], contiguous in NZ format
    // Q_nope: queryGm[queryGmOffset], 512 dims, srcDValue=headDim
    // Q_rope: queryGm[queryGmOffset + kvCacheDim], 64 dims, srcDValue=kRopeDim
    // In NZ format with dstNzC0Stride=mSizeAlign=16:
    //   Q_nope occupies l1QPTensor[0 .. mSizeAlign*kvCacheDim - 1]
    //   Q_rope occupies l1QPTensor[mSizeAlign*kvCacheDim .. mSizeAlign*headDim - 1]
    {
        Nd2NzParams nd2nz;
        nd2nz.ndNum = 1;
        nd2nz.nValue = 1;  // M=1
        nd2nz.dValue = static_cast<uint32_t>(kvCacheDim);  // 512
        nd2nz.srcDValue = static_cast<uint32_t>(headDim);  // stride in GM
        nd2nz.dstNzC0Stride = mSizeAlign;
        nd2nz.dstNzNStride = 1;
        nd2nz.srcNdMatrixStride = 0;
        nd2nz.dstNzMatrixStride = 0;
        DataCopy(l1QPTensor, queryGm[queryGmOffset], nd2nz);

        // Q_rope right after Q_nope
        nd2nz.dValue = static_cast<uint32_t>(kRopeDim);
        nd2nz.srcDValue = static_cast<uint32_t>(kRopeDim);
        DataCopy(l1QPTensor[mSizeAlign * static_cast<uint32_t>(kvCacheDim)],
                 queryGm[queryGmOffset + kvCacheDim], nd2nz);
    }

    SetFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);
    WaitFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);

    // N loop
    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) {
        uint32_t curNL1Size = (nL1 == nL1Loops - 1) ?
            (nSize - (nL1Loops - 1) * FSA_N_SPLIT_SIZE) : FSA_N_SPLIT_SIZE;
        uint32_t curNL1SizeAlign = (curNL1Size + 15) / 16 * 16;

        // Load K tokens to L1: K_nope[0:512] then K_rope[0:64] per token
        uint32_t kvL1Idx = nL1 % 2;
        uint32_t L1_BLOCK_ELEMS = FSA_L1_BLOCK_SIZE / sizeof(T);
        LocalTensor<T> bL1Tensor = l1KVTensor[kvL1Idx * L1_BLOCK_ELEMS];

        uint32_t startToken = nL1 * FSA_N_SPLIT_SIZE;
        uint32_t blockElemCnt = 32 / sizeof(T);

        for (uint32_t t = 0; t < curNL1Size; t++) {
            int64_t kvAddr = tokenKvAddrs[startToken + t];
            int64_t ropeAddr = tokenRopeAddrs[startToken + t];

            // K_nope: 512 dims from kvCacheGm
            Nd2NzParams nd2nz;
            nd2nz.ndNum = 1;
            nd2nz.nValue = 1;
            nd2nz.dValue = static_cast<uint32_t>(kvCacheDim);
            nd2nz.srcDValue = static_cast<uint32_t>(kvCacheDim);
            nd2nz.dstNzC0Stride = curNL1SizeAlign;
            nd2nz.dstNzNStride = 1;
            nd2nz.srcNdMatrixStride = 0;
            nd2nz.dstNzMatrixStride = 0;
            DataCopy(bL1Tensor[t * blockElemCnt], kvCacheGm[kvAddr], nd2nz);

            // K_rope: 64 dims from kRopeGm, right after K_nope
            nd2nz.dValue = static_cast<uint32_t>(kRopeDim);
            nd2nz.srcDValue = static_cast<uint32_t>(kRopeDim);
            DataCopy(bL1Tensor[curNL1SizeAlign * static_cast<uint32_t>(kvCacheDim) + t * blockElemCnt],
                     kRopeGm[ropeAddr], nd2nz);
        }

        SetFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);

        uint32_t cIdx = cL0BufIter_ % 2;
        uint32_t cEvent = FsaL0CEventId(cIdx);
        LocalTensor<MM_OUT_T> cL0Tensor = bufL0C_.template Get<MM_OUT_T>()[
            cIdx * (FSA_L0C_PP_SIZE / sizeof(MM_OUT_T))];

        // kL0 loop: 6 iterations 脳 96 = 576
        for (uint32_t kL0 = 0; kL0 < kL0Loops; kL0++) {
            uint32_t abIdx = abL0BufIter_ % 2;
            uint32_t abEvent = FsaL0ABEventId(abIdx);
            WaitFlag<HardEvent::M_MTE1>(abEvent);
            uint32_t curKL0Size = (kL0 == kL0Loops - 1) ?
                (headDimAlign - (kL0Loops - 1) * kL0Size) : kL0Size;

            // Load Q sub-block from L1 to L0A
            LocalTensor<T> aL0Tensor = bufL0A_.template Get<T>()[
                abIdx * (FSA_L0A_PP_SIZE / sizeof(T))];
            {
                LocalTensor<T> srcA = l1QPTensor[mSizeAlign * kL0Size * kL0];
                LoadData3DParamsV2<T> params;
                params.l1H = mSizeAlign / 16;
                params.l1W = 16;
                params.mExtension = mSizeAlign;
                params.kExtension = curKL0Size;
                params.enTranspose = 0;
                params.channelSize = curKL0Size;
                params.mStartPt = 0; params.kStartPt = 0;
                params.strideW = 1; params.strideH = 1;
                params.filterW = 1; params.filterH = 1;
                params.filterSizeW = 0; params.filterSizeH = 0;
                params.dilationFilterW = 1; params.dilationFilterH = 1;
                params.fMatrixCtrl = 0;
                params.padList[0] = 0; params.padList[1] = 0;
                params.padList[2] = 0; params.padList[3] = 255;
                LoadData<T, FSA_LOAD3D_CONFIG>(aL0Tensor, srcA, params);
            }

            // Load K sub-block from L1 to L0B
            LocalTensor<T> bL0Tensor = bufL0B_.template Get<T>()[
                abIdx * (FSA_L0B_PP_SIZE / sizeof(T))];
            {
                LocalTensor<T> srcB = bL1Tensor[curNL1SizeAlign * kL0Size * kL0];
                LoadData2DParams loadParams;
                loadParams.startIndex = 0;
                loadParams.repeatTimes = (curNL1SizeAlign + 15) / 16 * curKL0Size / (32 / sizeof(T));
                loadParams.srcStride = 1;
                loadParams.dstGap = 0;
                loadParams.ifTranspose = false;
                LoadData(bL0Tensor, srcB, loadParams);
            }

            SetFlag<HardEvent::MTE1_M>(abEvent);
            WaitFlag<HardEvent::MTE1_M>(abEvent);

            MmadParams mmadParams;
            mmadParams.m = mSizeAlign;
            mmadParams.n = curNL1SizeAlign;
            mmadParams.k = curKL0Size;
            mmadParams.cmatrixInitVal = (kL0 == 0);
            mmadParams.cmatrixSource = false;
            mmadParams.unitFlag = (kL0 == kL0Loops - 1) ? 0b11 : 0b10;
            Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);

            if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
                PipeBarrier<PIPE_M>();
            }
            SetFlag<HardEvent::M_MTE1>(abEvent);
            abL0BufIter_++;
        }

        // Fixpipe: output scores to GM
        uint32_t nSizeAlign = (nSize + 15) / 16 * 16;
        FixpipeParamsV220 fixParams;
        fixParams.nSize = curNL1SizeAlign;
        fixParams.mSize = mSizeAlign;
        fixParams.srcStride = mSizeAlign;
        fixParams.dstStride = nSizeAlign;
        fixParams.unitFlag = 0b11;
        fixParams.ndNum = 1;
        SetFlag<HardEvent::M_FIX>(cEvent);
        WaitFlag<HardEvent::M_FIX>(cEvent);
        Fixpipe(mm1ResGm_[scoreGmOffset + nL1 * FSA_N_SPLIT_SIZE], cL0Tensor, fixParams);
        SetFlag<HardEvent::FIX_M>(cEvent);
        WaitFlag<HardEvent::FIX_M>(cEvent);
        pipe_barrier(PIPE_ALL);
        cL0BufIter_++;
    }
}

template <typename T>
__aicore__ inline void FusedAttentionCubeService<T>::ComputeMm1MixedSource(
    GlobalTensor<T>& queryGm, int64_t queryGmOffset,
    GlobalTensor<T>& selKvCacheGm, GlobalTensor<T>& selKRopeGm,
    GlobalTensor<T>& fullKvCacheGm, GlobalTensor<T>& fullKRopeGm,
    int32_t* tokenKvAddrs, int32_t* tokenRopeAddrs, int32_t* sourceFlags,
    int32_t nTokens, int64_t scoreGmOffset)
{
    if (nTokens <= 0) return;

    int64_t kvCacheDim = tiling_->kvCacheDim;
    int64_t kRopeDim = tiling_->kRopeDim;
    int64_t headDim = kvCacheDim + kRopeDim;
    uint32_t headDimAlign = (static_cast<uint32_t>(headDim) + 15) / 16 * 16;

    constexpr uint32_t mSizeAlign = 16;

    uint32_t nSize = static_cast<uint32_t>(nTokens);
    uint32_t nL1Loops = (nSize + FSA_N_SPLIT_SIZE - 1) / FSA_N_SPLIT_SIZE;

    constexpr uint32_t kL0Size = 96;
    uint32_t kL0Loops = (headDimAlign + kL0Size - 1) / kL0Size;

    LocalTensor<T> l1QPTensor = bufQPL1_.template Get<T>();
    LocalTensor<T> l1KVTensor = bufKVL1_.template Get<T>();

    {
        Nd2NzParams nd2nz;
        nd2nz.ndNum = 1;
        nd2nz.nValue = 1;
        nd2nz.dValue = static_cast<uint32_t>(kvCacheDim);
        nd2nz.srcDValue = static_cast<uint32_t>(headDim);
        nd2nz.dstNzC0Stride = mSizeAlign;
        nd2nz.dstNzNStride = 1;
        nd2nz.srcNdMatrixStride = 0;
        nd2nz.dstNzMatrixStride = 0;
        DataCopy(l1QPTensor, queryGm[queryGmOffset], nd2nz);

        nd2nz.dValue = static_cast<uint32_t>(kRopeDim);
        nd2nz.srcDValue = static_cast<uint32_t>(kRopeDim);
        DataCopy(l1QPTensor[mSizeAlign * static_cast<uint32_t>(kvCacheDim)],
                 queryGm[queryGmOffset + kvCacheDim], nd2nz);
    }

    SetFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);
    WaitFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);

    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) {
        uint32_t curNL1Size = (nL1 == nL1Loops - 1) ?
            (nSize - (nL1Loops - 1) * FSA_N_SPLIT_SIZE) : FSA_N_SPLIT_SIZE;
        uint32_t curNL1SizeAlign = (curNL1Size + 15) / 16 * 16;

        uint32_t kvL1Idx = nL1 % 2;
        uint32_t L1_BLOCK_ELEMS = FSA_L1_BLOCK_SIZE / sizeof(T);
        LocalTensor<T> bL1Tensor = l1KVTensor[kvL1Idx * L1_BLOCK_ELEMS];

        uint32_t startToken = nL1 * FSA_N_SPLIT_SIZE;
        uint32_t blockElemCnt = 32 / sizeof(T);

        for (uint32_t t = 0; t < curNL1Size; t++) {
            int32_t tokenIdx = static_cast<int32_t>(startToken + t);
            int64_t kvAddr = tokenKvAddrs[tokenIdx];
            int64_t ropeAddr = tokenRopeAddrs[tokenIdx];
            bool useFullSource = (sourceFlags[tokenIdx] != 0);

            Nd2NzParams nd2nz;
            nd2nz.ndNum = 1;
            nd2nz.nValue = 1;
            nd2nz.dValue = static_cast<uint32_t>(kvCacheDim);
            nd2nz.srcDValue = static_cast<uint32_t>(kvCacheDim);
            nd2nz.dstNzC0Stride = curNL1SizeAlign;
            nd2nz.dstNzNStride = 1;
            nd2nz.srcNdMatrixStride = 0;
            nd2nz.dstNzMatrixStride = 0;
            if (useFullSource) {
                DataCopy(bL1Tensor[t * blockElemCnt], fullKvCacheGm[kvAddr], nd2nz);
            } else {
                DataCopy(bL1Tensor[t * blockElemCnt], selKvCacheGm[kvAddr], nd2nz);
            }

            nd2nz.dValue = static_cast<uint32_t>(kRopeDim);
            nd2nz.srcDValue = static_cast<uint32_t>(kRopeDim);
            if (useFullSource) {
                DataCopy(bL1Tensor[curNL1SizeAlign * static_cast<uint32_t>(kvCacheDim) + t * blockElemCnt],
                         fullKRopeGm[ropeAddr], nd2nz);
            } else {
                DataCopy(bL1Tensor[curNL1SizeAlign * static_cast<uint32_t>(kvCacheDim) + t * blockElemCnt],
                         selKRopeGm[ropeAddr], nd2nz);
            }
        }

        SetFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);

        uint32_t cIdx = cL0BufIter_ % 2;
        uint32_t cEvent = FsaL0CEventId(cIdx);
        LocalTensor<MM_OUT_T> cL0Tensor = bufL0C_.template Get<MM_OUT_T>()[
            cIdx * (FSA_L0C_PP_SIZE / sizeof(MM_OUT_T))];

        for (uint32_t kL0 = 0; kL0 < kL0Loops; kL0++) {
            uint32_t abIdx = abL0BufIter_ % 2;
            uint32_t abEvent = FsaL0ABEventId(abIdx);
            WaitFlag<HardEvent::M_MTE1>(abEvent);
            uint32_t curKL0Size = (kL0 == kL0Loops - 1) ?
                (headDimAlign - (kL0Loops - 1) * kL0Size) : kL0Size;

            LocalTensor<T> aL0Tensor = bufL0A_.template Get<T>()[
                abIdx * (FSA_L0A_PP_SIZE / sizeof(T))];
            {
                LocalTensor<T> srcA = l1QPTensor[mSizeAlign * kL0Size * kL0];
                LoadData3DParamsV2<T> params;
                params.l1H = mSizeAlign / 16;
                params.l1W = 16;
                params.mExtension = mSizeAlign;
                params.kExtension = curKL0Size;
                params.enTranspose = 0;
                params.channelSize = curKL0Size;
                params.mStartPt = 0; params.kStartPt = 0;
                params.strideW = 1; params.strideH = 1;
                params.filterW = 1; params.filterH = 1;
                params.filterSizeW = 0; params.filterSizeH = 0;
                params.dilationFilterW = 1; params.dilationFilterH = 1;
                params.fMatrixCtrl = 0;
                params.padList[0] = 0; params.padList[1] = 0;
                params.padList[2] = 0; params.padList[3] = 255;
                LoadData<T, FSA_LOAD3D_CONFIG>(aL0Tensor, srcA, params);
            }

            LocalTensor<T> bL0Tensor = bufL0B_.template Get<T>()[
                abIdx * (FSA_L0B_PP_SIZE / sizeof(T))];
            {
                LocalTensor<T> srcB = bL1Tensor[curNL1SizeAlign * kL0Size * kL0];
                LoadData2DParams loadParams;
                loadParams.startIndex = 0;
                loadParams.repeatTimes = (curNL1SizeAlign + 15) / 16 * curKL0Size / (32 / sizeof(T));
                loadParams.srcStride = 1;
                loadParams.dstGap = 0;
                loadParams.ifTranspose = false;
                LoadData(bL0Tensor, srcB, loadParams);
            }

            SetFlag<HardEvent::MTE1_M>(abEvent);
            WaitFlag<HardEvent::MTE1_M>(abEvent);

            MmadParams mmolParams;
            mmolParams.m = mSizeAlign;
            mmolParams.n = curNL1SizeAlign;
            mmolParams.k = curKL0Size;
            mmolParams.cmatrixInitVal = (kL0 == 0);
            mmolParams.cmatrixSource = false;
            mmolParams.unitFlag = (kL0 == kL0Loops - 1) ? 0b11 : 0b10;
            Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmolParams);

            if ((mmolParams.m / 16) * (mmolParams.n / 16) < 10) {
                PipeBarrier<PIPE_M>();
            }
            SetFlag<HardEvent::M_MTE1>(abEvent);
            abL0BufIter_++;
        }

        uint32_t nSizeAlign = (nSize + 15) / 16 * 16;
        FixpipeParamsV220 fixParams;
        fixParams.nSize = curNL1SizeAlign;
        fixParams.mSize = mSizeAlign;
        fixParams.srcStride = mSizeAlign;
        fixParams.dstStride = nSizeAlign;
        fixParams.unitFlag = 0b11;
        fixParams.ndNum = 1;
        SetFlag<HardEvent::M_FIX>(cEvent);
        WaitFlag<HardEvent::M_FIX>(cEvent);
        Fixpipe(mm1ResGm_[scoreGmOffset + nL1 * FSA_N_SPLIT_SIZE], cL0Tensor, fixParams);
        SetFlag<HardEvent::FIX_M>(cEvent);
        WaitFlag<HardEvent::FIX_M>(cEvent);
        pipe_barrier(PIPE_ALL);
        cL0BufIter_++;
    }
}

// ============================================================================
// ComputeMm2: weights[1, nTokens] 脳 V[nTokens, kvCacheDim] 鈫?output[1, kvCacheDim]
// ============================================================================
template <typename T>
__aicore__ inline void FusedAttentionCubeService<T>::ComputeMm2(
    GlobalTensor<T>& kvCacheGm, GlobalTensor<int32_t>& blockTableGm,
    int32_t* tokenKvAddrs, int32_t nTokens,
    GlobalTensor<T>& weightsGm, int64_t weightsGmOffset,
    GlobalTensor<MM_OUT_T>& outputGm, int64_t outputGmOffset,
    bool isFirstBlock)
{
    if (nTokens <= 0) return;

    int64_t kvCacheDim = tiling_->kvCacheDim;
    if constexpr (FSA_DIAG_MM2_DIRECT_COPY) {
        for (int64_t d = 0; d < kvCacheDim; d++) {
            float acc = isFirstBlock ? 0.0f : outputGm.GetValue(outputGmOffset + d);
            for (int32_t t = 0; t < nTokens; t++) {
                float weight = ToFloat(weightsGm.GetValue(weightsGmOffset + t));
                float value = ToFloat(kvCacheGm.GetValue(tokenKvAddrs[t] + d));
                acc += weight * value;
            }
            outputGm.SetValue(outputGmOffset + d, acc);
        }
        pipe_barrier(PIPE_ALL);
        return;
    }
    constexpr uint32_t mSizeAlign = 16;

    uint32_t kSize = static_cast<uint32_t>(nTokens);
    uint32_t kSizeAlign = (kSize + 15) / 16 * 16;
    uint32_t weightsStride = (kSize + 127) / 128 * 128;
    if (weightsStride < 128) {
        weightsStride = 128;
    }

    uint32_t nSize = static_cast<uint32_t>(kvCacheDim);
    uint32_t nL1Loops = (nSize + FSA_MM2_N_SPLIT_SIZE - 1) / FSA_MM2_N_SPLIT_SIZE;

    constexpr uint32_t kL0Size = 128;
    uint32_t kL0Loops = (kSize + kL0Size - 1) / kL0Size;

    LocalTensor<T> l1QPTensor = bufQPL1_.template Get<T>();
    LocalTensor<T> l1KVTensor = bufKVL1_.template Get<T>();

    // Load weights to L1
    {
        Nd2NzParams nd2nz;
        nd2nz.ndNum = 1;
        nd2nz.nValue = 1;
        nd2nz.dValue = kSize;
        nd2nz.srcDValue = weightsStride;
        nd2nz.dstNzC0Stride = mSizeAlign;
        nd2nz.dstNzNStride = 1;
        nd2nz.srcNdMatrixStride = 0;
        nd2nz.dstNzMatrixStride = 0;
        DataCopy(l1QPTensor, weightsGm[weightsGmOffset], nd2nz);
    }

    SetFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);
    WaitFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);


    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) {
        uint32_t curNL1Size = (nL1 == nL1Loops - 1) ?
            (nSize - (nL1Loops - 1) * FSA_MM2_N_SPLIT_SIZE) : FSA_MM2_N_SPLIT_SIZE;
        uint32_t curNL1SizeAlign = (curNL1Size + 15) / 16 * 16;

        uint32_t L1_BLOCK_ELEMS = FSA_L1_BLOCK_SIZE / sizeof(T);
        uint32_t kvL1Idx = nL1 % 2;
        LocalTensor<T> bL1Tensor = l1KVTensor[kvL1Idx * L1_BLOCK_ELEMS];

        for (uint32_t t = 0; t < static_cast<uint32_t>(nTokens); t++) {
            int64_t kvAddr = tokenKvAddrs[t];
            uint32_t blockElemCnt = 32 / sizeof(T);
            Nd2NzParams nd2nz;
            nd2nz.ndNum = 1;
            nd2nz.nValue = 1;
            nd2nz.dValue = curNL1Size;
            nd2nz.srcDValue = static_cast<uint32_t>(kvCacheDim);
            nd2nz.dstNzC0Stride = kSizeAlign;
            nd2nz.dstNzNStride = 1;
            nd2nz.srcNdMatrixStride = 0;
            nd2nz.dstNzMatrixStride = 0;
            DataCopy(bL1Tensor[t * blockElemCnt],
                     kvCacheGm[kvAddr + nL1 * FSA_MM2_N_SPLIT_SIZE], nd2nz);
        }

        SetFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);


        uint32_t cIdx = cL0BufIter_ % 2;
        uint32_t cEvent = FsaL0CEventId(cIdx);
        LocalTensor<MM_OUT_T> cL0Tensor = bufL0C_.template Get<MM_OUT_T>()[
            cIdx * (FSA_L0C_PP_SIZE / sizeof(MM_OUT_T))];

        for (uint32_t kL0 = 0; kL0 < kL0Loops; kL0++) {
            uint32_t abIdx = abL0BufIter_ % 2;
            uint32_t abEvent = FsaL0ABEventId(abIdx);
            WaitFlag<HardEvent::M_MTE1>(abEvent);
            uint32_t curKL0Size = (kL0 == kL0Loops - 1) ?
                (kSize - (kL0Loops - 1) * kL0Size) : kL0Size;
            uint32_t curKL0SizeAlign = (curKL0Size + 15) / 16 * 16;
            LocalTensor<T> aL0Tensor = bufL0A_.template Get<T>()[
                abIdx * (FSA_L0A_PP_SIZE / sizeof(T))];
            {
                LocalTensor<T> srcA = l1QPTensor[mSizeAlign * kL0Size * kL0];
                LoadData3DParamsV2<T> params;
                params.l1H = mSizeAlign / 16;
                params.l1W = 16;
                params.mExtension = mSizeAlign;
                params.kExtension = curKL0SizeAlign;
                params.enTranspose = 0;
                params.channelSize = curKL0SizeAlign;
                params.mStartPt = 0; params.kStartPt = 0;
                params.strideW = 1; params.strideH = 1;
                params.filterW = 1; params.filterH = 1;
                params.filterSizeW = 0; params.filterSizeH = 0;
                params.dilationFilterW = 1; params.dilationFilterH = 1;
                params.fMatrixCtrl = 0;
                params.padList[0] = 0; params.padList[1] = 0;
                params.padList[2] = 0; params.padList[3] = 255;
                LoadData<T, FSA_LOAD3D_CONFIG>(aL0Tensor, srcA, params);
            }

            LocalTensor<T> bL0Tensor = bufL0B_.template Get<T>()[
                abIdx * (FSA_L0B_PP_SIZE / sizeof(T))];
            {
                LocalTensor<T> srcB = bL1Tensor[curNL1SizeAlign * kL0Size * kL0];
                LoadData3DParamsV2<T> params;
                params.l1H = curKL0SizeAlign / 16;
                params.l1W = 16;
                params.mExtension = curKL0SizeAlign;
                params.kExtension = curNL1SizeAlign;
                params.enTranspose = 1;
                params.channelSize = curNL1SizeAlign;
                params.mStartPt = 0; params.kStartPt = 0;
                params.strideW = 1; params.strideH = 1;
                params.filterW = 1; params.filterH = 1;
                params.filterSizeW = 0; params.filterSizeH = 0;
                params.dilationFilterW = 1; params.dilationFilterH = 1;
                params.fMatrixCtrl = 0;
                params.padList[0] = 0; params.padList[1] = 0;
                params.padList[2] = 0; params.padList[3] = 255;
                LoadData<T, FSA_LOAD3D_CONFIG>(bL0Tensor, srcB, params);
            }

            SetFlag<HardEvent::MTE1_M>(abEvent);
            WaitFlag<HardEvent::MTE1_M>(abEvent);


            MmadParams mmadParams;
            mmadParams.m = mSizeAlign;
            mmadParams.n = curNL1SizeAlign;
            mmadParams.k = curKL0Size;
            mmadParams.cmatrixInitVal = (kL0 == 0);
            mmadParams.cmatrixSource = false;
            mmadParams.unitFlag = (kL0 == kL0Loops - 1) ? 0b11 : 0b10;
            Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);

            if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
                PipeBarrier<PIPE_M>();
            }
            SetFlag<HardEvent::M_MTE1>(abEvent);
            abL0BufIter_++;
        }

        uint32_t outNSizeAlign = (nSize + 15) / 16 * 16;
        FixpipeParamsV220 fixParams;
        fixParams.nSize = curNL1SizeAlign;
        fixParams.mSize = mSizeAlign;
        fixParams.srcStride = mSizeAlign;
        fixParams.dstStride = outNSizeAlign;
        fixParams.unitFlag = 0b11;
        fixParams.ndNum = 1;

        SetFlag<HardEvent::M_FIX>(cEvent);
        WaitFlag<HardEvent::M_FIX>(cEvent);
        if (!isFirstBlock && !FSA_DIAG_DISABLE_MM2_ATOMIC_ADD) {
            SetAtomicAdd<MM_OUT_T>();
        }
        Fixpipe(outputGm[outputGmOffset + nL1 * FSA_MM2_N_SPLIT_SIZE], cL0Tensor, fixParams);
        SetFlag<HardEvent::FIX_M>(cEvent);
        WaitFlag<HardEvent::FIX_M>(cEvent);
        if (!isFirstBlock && !FSA_DIAG_DISABLE_MM2_ATOMIC_ADD) {
            SetAtomicNone();
        }
        pipe_barrier(PIPE_ALL);
        cL0BufIter_++;
    }
}

template <typename T>
__aicore__ inline void FusedAttentionCubeService<T>::ComputeMm2MixedSource(
    GlobalTensor<T>& selKvCacheGm, GlobalTensor<T>& fullKvCacheGm,
    int32_t* tokenKvAddrs, int32_t* sourceFlags, int32_t nTokens,
    GlobalTensor<T>& weightsGm, int64_t weightsGmOffset,
    GlobalTensor<MM_OUT_T>& outputGm, int64_t outputGmOffset,
    bool isFirstBlock)
{
    if (nTokens <= 0) return;

    int64_t kvCacheDim = tiling_->kvCacheDim;
    if constexpr (FSA_DIAG_MM2_DIRECT_COPY) {
        for (int64_t d = 0; d < kvCacheDim; d++) {
            float acc = isFirstBlock ? 0.0f : outputGm.GetValue(outputGmOffset + d);
            for (int32_t t = 0; t < nTokens; t++) {
                float weight = ToFloat(weightsGm.GetValue(weightsGmOffset + t));
                float value = (sourceFlags[t] != 0) ?
                    ToFloat(fullKvCacheGm.GetValue(tokenKvAddrs[t] + d)) :
                    ToFloat(selKvCacheGm.GetValue(tokenKvAddrs[t] + d));
                acc += weight * value;
            }
            outputGm.SetValue(outputGmOffset + d, acc);
        }
        pipe_barrier(PIPE_ALL);
        return;
    }
    constexpr uint32_t mSizeAlign = 16;

    uint32_t kSize = static_cast<uint32_t>(nTokens);
    uint32_t kSizeAlign = (kSize + 15) / 16 * 16;
    uint32_t weightsStride = (kSize + 127) / 128 * 128;
    if (weightsStride < 128) {
        weightsStride = 128;
    }

    uint32_t nSize = static_cast<uint32_t>(kvCacheDim);
    uint32_t nL1Loops = (nSize + FSA_MM2_N_SPLIT_SIZE - 1) / FSA_MM2_N_SPLIT_SIZE;

    constexpr uint32_t kL0Size = 128;
    uint32_t kL0Loops = (kSize + kL0Size - 1) / kL0Size;

    LocalTensor<T> l1QPTensor = bufQPL1_.template Get<T>();
    LocalTensor<T> l1KVTensor = bufKVL1_.template Get<T>();

    {
        Nd2NzParams nd2nz;
        nd2nz.ndNum = 1;
        nd2nz.nValue = 1;
        nd2nz.dValue = kSize;
        nd2nz.srcDValue = weightsStride;
        nd2nz.dstNzC0Stride = mSizeAlign;
        nd2nz.dstNzNStride = 1;
        nd2nz.srcNdMatrixStride = 0;
        nd2nz.dstNzMatrixStride = 0;
        DataCopy(l1QPTensor, weightsGm[weightsGmOffset], nd2nz);
    }

    SetFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);
    WaitFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);

    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) {
        uint32_t curNL1Size = (nL1 == nL1Loops - 1) ?
            (nSize - (nL1Loops - 1) * FSA_MM2_N_SPLIT_SIZE) : FSA_MM2_N_SPLIT_SIZE;
        uint32_t curNL1SizeAlign = (curNL1Size + 15) / 16 * 16;

        uint32_t L1_BLOCK_ELEMS = FSA_L1_BLOCK_SIZE / sizeof(T);
        uint32_t kvL1Idx = nL1 % 2;
        LocalTensor<T> bL1Tensor = l1KVTensor[kvL1Idx * L1_BLOCK_ELEMS];

        for (uint32_t t = 0; t < static_cast<uint32_t>(nTokens); t++) {
            int64_t kvAddr = tokenKvAddrs[t];
            uint32_t blockElemCnt = 32 / sizeof(T);
            Nd2NzParams nd2nz;
            nd2nz.ndNum = 1;
            nd2nz.nValue = 1;
            nd2nz.dValue = curNL1Size;
            nd2nz.srcDValue = static_cast<uint32_t>(kvCacheDim);
            nd2nz.dstNzC0Stride = kSizeAlign;
            nd2nz.dstNzNStride = 1;
            nd2nz.srcNdMatrixStride = 0;
            nd2nz.dstNzMatrixStride = 0;
            if (sourceFlags[t] != 0) {
                DataCopy(bL1Tensor[t * blockElemCnt],
                         fullKvCacheGm[kvAddr + nL1 * FSA_MM2_N_SPLIT_SIZE], nd2nz);
            } else {
                DataCopy(bL1Tensor[t * blockElemCnt],
                         selKvCacheGm[kvAddr + nL1 * FSA_MM2_N_SPLIT_SIZE], nd2nz);
            }
        }

        SetFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(FSA_MTE2_MTE1_EVENT);

        uint32_t cIdx = cL0BufIter_ % 2;
        uint32_t cEvent = FsaL0CEventId(cIdx);
        LocalTensor<MM_OUT_T> cL0Tensor = bufL0C_.template Get<MM_OUT_T>()[
            cIdx * (FSA_L0C_PP_SIZE / sizeof(MM_OUT_T))];

        for (uint32_t kL0 = 0; kL0 < kL0Loops; kL0++) {
            uint32_t abIdx = abL0BufIter_ % 2;
            uint32_t abEvent = FsaL0ABEventId(abIdx);
            WaitFlag<HardEvent::M_MTE1>(abEvent);
            uint32_t curKL0Size = (kL0 == kL0Loops - 1) ?
                (kSize - (kL0Loops - 1) * kL0Size) : kL0Size;
            uint32_t curKL0SizeAlign = (curKL0Size + 15) / 16 * 16;
            LocalTensor<T> aL0Tensor = bufL0A_.template Get<T>()[
                abIdx * (FSA_L0A_PP_SIZE / sizeof(T))];
            {
                LocalTensor<T> srcA = l1QPTensor[mSizeAlign * kL0Size * kL0];
                LoadData3DParamsV2<T> params;
                params.l1H = mSizeAlign / 16;
                params.l1W = 16;
                params.mExtension = mSizeAlign;
                params.kExtension = curKL0SizeAlign;
                params.enTranspose = 0;
                params.channelSize = curKL0SizeAlign;
                params.mStartPt = 0; params.kStartPt = 0;
                params.strideW = 1; params.strideH = 1;
                params.filterW = 1; params.filterH = 1;
                params.filterSizeW = 0; params.filterSizeH = 0;
                params.dilationFilterW = 1; params.dilationFilterH = 1;
                params.fMatrixCtrl = 0;
                params.padList[0] = 0; params.padList[1] = 0;
                params.padList[2] = 0; params.padList[3] = 255;
                LoadData<T, FSA_LOAD3D_CONFIG>(aL0Tensor, srcA, params);
            }

            LocalTensor<T> bL0Tensor = bufL0B_.template Get<T>()[
                abIdx * (FSA_L0B_PP_SIZE / sizeof(T))];
            {
                LocalTensor<T> srcB = bL1Tensor[curNL1SizeAlign * kL0Size * kL0];
                LoadData3DParamsV2<T> params;
                params.l1H = curKL0SizeAlign / 16;
                params.l1W = 16;
                params.mExtension = curKL0SizeAlign;
                params.kExtension = curNL1SizeAlign;
                params.enTranspose = 1;
                params.channelSize = curNL1SizeAlign;
                params.mStartPt = 0; params.kStartPt = 0;
                params.strideW = 1; params.strideH = 1;
                params.filterW = 1; params.filterH = 1;
                params.filterSizeW = 0; params.filterSizeH = 0;
                params.dilationFilterW = 1; params.dilationFilterH = 1;
                params.fMatrixCtrl = 0;
                params.padList[0] = 0; params.padList[1] = 0;
                params.padList[2] = 0; params.padList[3] = 255;
                LoadData<T, FSA_LOAD3D_CONFIG>(bL0Tensor, srcB, params);
            }

            SetFlag<HardEvent::MTE1_M>(abEvent);
            WaitFlag<HardEvent::MTE1_M>(abEvent);

            MmadParams mmolParams;
            mmolParams.m = mSizeAlign;
            mmolParams.n = curNL1SizeAlign;
            mmolParams.k = curKL0Size;
            mmolParams.cmatrixInitVal = (kL0 == 0);
            mmolParams.cmatrixSource = false;
            mmolParams.unitFlag = (kL0 == kL0Loops - 1) ? 0b11 : 0b10;
            Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmolParams);

            if ((mmolParams.m / 16) * (mmolParams.n / 16) < 10) {
                PipeBarrier<PIPE_M>();
            }
            SetFlag<HardEvent::M_MTE1>(abEvent);
            abL0BufIter_++;
        }

        uint32_t outNSizeAlign = (nSize + 15) / 16 * 16;
        FixpipeParamsV220 fixParams;
        fixParams.nSize = curNL1SizeAlign;
        fixParams.mSize = mSizeAlign;
        fixParams.srcStride = mSizeAlign;
        fixParams.dstStride = outNSizeAlign;
        fixParams.unitFlag = 0b11;
        fixParams.ndNum = 1;

        SetFlag<HardEvent::M_FIX>(cEvent);
        WaitFlag<HardEvent::M_FIX>(cEvent);
        if (!isFirstBlock && !FSA_DIAG_DISABLE_MM2_ATOMIC_ADD) {
            SetAtomicAdd<MM_OUT_T>();
        }
        Fixpipe(outputGm[outputGmOffset + nL1 * FSA_MM2_N_SPLIT_SIZE], cL0Tensor, fixParams);
        SetFlag<HardEvent::FIX_M>(cEvent);
        WaitFlag<HardEvent::FIX_M>(cEvent);
        if (!isFirstBlock && !FSA_DIAG_DISABLE_MM2_ATOMIC_ADD) {
            SetAtomicNone();
        }
        pipe_barrier(PIPE_ALL);
        cL0BufIter_++;
    }
}

} // namespace FusedSparseAttentionOverlapNs

#endif // FUSED_SPARSE_ATTENTION_OVERLAP_CUBE_H_


