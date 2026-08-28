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
 * \file fused_li_manage_c8_service_cube.h
 * \brief C8 variant of the cube service: int8 QK Mmad followed by an on-chip
 * fp16 head-reduction Mmad (QLI arch22 two-mma structure). Per-head scores
 * stay on chip (L0C -> fixpipe DEQF16 -> L1 -> mma2), only the reduced
 * fp32 score row is written to GM.
 */
#ifndef FUSED_LI_MANAGE_C8_SERVICE_CUBE_H
#define FUSED_LI_MANAGE_C8_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "fused_li_manage_c8_common.h"

namespace LIKernel {
using namespace LICommon;

template <typename LIT>
class LIMatmulC8 {
public:
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    // mma2 output (and the GM score workspace) is fp32.
    using MM1_OUT_T = float;

    __aicore__ inline LIMatmulC8(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                               const GlobalTensor<Q_T> &queryGm, const GlobalTensor<MM1_OUT_T> &mm1ResGm,
                                               const GlobalTensor<half> &weightWorkspaceGm);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const LICommon::RunInfo &runInfo);

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};
    static constexpr uint64_t KEY_BUF_NUM = 3;
    static constexpr uint64_t L0_BUF_NUM = 2;
    static constexpr uint64_t L0AB_BUF_NUM = 4;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QUERY_MTE1_MTE2_EVENT = EVENT_ID5;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;
    // FIX_M guards the L0C double buffer shared by mma1 (int32) and mma2
    // (fp32 reinterpret); M_FIX orders each Mmad before its Fixp drain (no
    // hardware scoreboard for M->FIX on L0C here). FIX_MTE1 publishes the
    // fp16 per-head scores in L1 to the mma2 operand load. Mirrors
    // quant_lightning_indexer (arch22).
    static constexpr uint32_t FIX_M_EVENT = EVENT_ID2;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID0;
    static constexpr uint32_t FIX_MTE1_EVENT = EVENT_ID4;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;

    static constexpr uint64_t D_BASIC_BLOCK = 128;
    static constexpr uint64_t S2_BASIC_BLOCK = 256;

    // int8 NZ fractal: C0 = 32 elements (32B / 1B); the N dim stays 16 (BLOCK_CUBE).
    // Mirrors S8_BLOCK_CUBE in the reference lightning_indexer_quant kernel.
    static constexpr uint64_t S8_BLOCK_CUBE = 32;

    static constexpr uint64_t M_BASIC_BLOCK_L0 = 64;
    static constexpr uint64_t D_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;

    static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK;
    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK * D_BASIC_BLOCK;
    // fp16 per-head score tile in L1: [qHeadNum, S2_BASIC_BLOCK_L0], double buffered.
    static constexpr uint64_t SL1_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;
    // Brcb'd (w*q_scale) operand: qHeadNum blocks of BLOCK_CUBE halfs, single buffer.
    static constexpr uint64_t WEIGHT_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * BLOCK_CUBE;
    // Uniform 16KB L0A/L0B slots (QLI layout): int8 uses 16K elements, fp16 8K halfs.
    static constexpr uint64_t L0AB_BUFFER_OFFSET_S8_16K = 16 * 1024;
    static constexpr uint64_t L0AB_BUFFER_OFFSET_FP16_16K = 16 * 512;
    static constexpr uint64_t L0C_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;

protected:
    __aicore__ inline void ProcessQk(uint64_t s2L1Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                     bool syncKeyMte2, const LICommon::RunInfo &runInfo);
    __aicore__ inline void ProcessWs(uint64_t s2GmOffset, uint64_t s2L0RealSize, uint64_t sL1BufIdx,
                                     const LICommon::RunInfo &runInfo);
    __aicore__ inline void FixpSToL1(uint64_t s2L0RealSize);
    __aicore__ inline void FixpResToGm(uint64_t s2GmOffset, uint64_t s2L0RealSize,
                                       const LICommon::RunInfo &runInfo);
    __aicore__ inline void ComuteL0c(uint64_t s2L0RealSize);
    __aicore__ inline void ComputeWs(uint64_t s2L0RealSize);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                        const LICommon::RunInfo &runInfo);
    __aicore__ inline void LoadQueryToL0a(const LICommon::RunInfo &runInfo);
    __aicore__ inline void LoadSToL0b(uint64_t s2L0RealSize, uint64_t sL1BufIdx);
    __aicore__ inline void LoadWeightToL0a();
    __aicore__ inline void QueryNd2Nz(const LICommon::RunInfo &runInfo);
    __aicore__ inline void WeightDmaCopy(const LICommon::RunInfo &runInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset, const LICommon::RunInfo &runInfo);
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<MM1_OUT_T> mm1ResGm_;
    GlobalTensor<half> weightGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;
    TBuf<TPosition::A1> bufWeightL1_;
    LocalTensor<half> weightL1_;
    TBuf<TPosition::B1> bufSL1_;
    LocalTensor<half> sL1_;

    TBuf<TPosition::A2> bufL0A_;
    LocalTensor<Q_T> l0a_;
    TBuf<TPosition::B2> bufL0B_;
    LocalTensor<K_T> l0b_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<int32_t> cL0_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;
    uint64_t l0cBufIdx_ = 0;
    uint64_t sL1BufIdx_ = 0;

    ConstInfo constInfo_;
};

template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
}

template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufQL1_, QUERY_BUFFER_OFFSET * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * S2_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();
    pipe->InitBuffer(bufWeightL1_, WEIGHT_BUFFER_OFFSET * sizeof(half));
    weightL1_ = bufWeightL1_.Get<half>();
    pipe->InitBuffer(bufSL1_, 2 * SL1_BUFFER_OFFSET * sizeof(half));
    sL1_ = bufSL1_.Get<half>();

    pipe->InitBuffer(bufL0A_, L0AB_BUF_NUM * L0AB_BUFFER_OFFSET_S8_16K);
    l0a_ = bufL0A_.Get<Q_T>();
    pipe->InitBuffer(bufL0B_, L0AB_BUF_NUM * L0AB_BUFFER_OFFSET_S8_16K);
    l0b_ = bufL0B_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * L0C_BUFFER_OFFSET * sizeof(int32_t));
    cL0_ = bufL0C_.Get<int32_t>();
}

template <typename LIT>
__aicore__ inline void
LIMatmulC8<LIT>::InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                     const GlobalTensor<Q_T> &queryGm, const GlobalTensor<MM1_OUT_T> &mm1ResGm,
                                     const GlobalTensor<half> &weightWorkspaceGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
    mm1ResGm_ = mm1ResGm;
    weightGm_ = weightWorkspaceGm;
}

template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::ComputeMm1(const LICommon::RunInfo &runInfo)
{
    uint64_t s2GmBaseOffset = runInfo.s2Idx * constInfo_.s2BaseSize;
    uint64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    uint64_t tileCnt = CeilDiv(s2ProcessSize, S2_BASIC_BLOCK_L0);

    if (runInfo.isFirstS2InnerLoop) {
        WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT);
        // In-order MTE2 pipe: the first even tile's MTE2_MTE1 pair below also
        // covers these query/weight loads (same pattern as QLI arch22).
        QueryNd2Nz(runInfo);
        WeightDmaCopy(runInfo);
    }

    for (uint64_t tileIdx = 0; tileIdx < tileCnt; ++tileIdx) {
        uint64_t s2L1Offset = tileIdx % 2 * S2_BASIC_BLOCK_L0;
        uint64_t s2L1RealSize = 0;
        if (tileIdx % 2 == 0) {
            uint64_t l1TileStart = tileIdx * S2_BASIC_BLOCK_L0;
            s2L1RealSize = l1TileStart + S2_BASIC_BLOCK > s2ProcessSize ? s2ProcessSize - l1TileStart
                                                                        : S2_BASIC_BLOCK;
            WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
            KeyNd2NzForPA(s2L1RealSize, s2GmBaseOffset + l1TileStart, runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        } else {
            uint64_t l1TileStart = (tileIdx - 1) * S2_BASIC_BLOCK_L0;
            s2L1RealSize = l1TileStart + S2_BASIC_BLOCK > s2ProcessSize ? s2ProcessSize - l1TileStart
                                                                        : S2_BASIC_BLOCK;
        }
        uint64_t s2L0RealSize = tileIdx * S2_BASIC_BLOCK_L0 + S2_BASIC_BLOCK_L0 > s2ProcessSize
                                    ? s2ProcessSize - tileIdx * S2_BASIC_BLOCK_L0
                                    : S2_BASIC_BLOCK_L0;

        ProcessQk(s2L1Offset, s2L1RealSize, s2L0RealSize, tileIdx % 2 == 0, runInfo);
        SetFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + sL1BufIdx_ % 2);
        sL1BufIdx_++;
        if (tileIdx > 0) {
            WaitFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + sL1BufIdx_ % 2);
            ProcessWs((tileIdx - 1) * S2_BASIC_BLOCK_L0,
                      (tileIdx - 1) * S2_BASIC_BLOCK_L0 + S2_BASIC_BLOCK_L0 > s2ProcessSize
                          ? s2ProcessSize - (tileIdx - 1) * S2_BASIC_BLOCK_L0
                          : S2_BASIC_BLOCK_L0,
                      sL1BufIdx_, runInfo);
        }
        if (tileIdx % 2 == 1 || tileIdx + 1 == tileCnt) {
            SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
            keyL1BufIdx_++;
        }
    }

    WaitFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + (sL1BufIdx_ + 1) % 2);
    ProcessWs((tileCnt - 1) * S2_BASIC_BLOCK_L0,
              (tileCnt - 1) * S2_BASIC_BLOCK_L0 + S2_BASIC_BLOCK_L0 > s2ProcessSize
                  ? s2ProcessSize - (tileCnt - 1) * S2_BASIC_BLOCK_L0
                  : S2_BASIC_BLOCK_L0,
              sL1BufIdx_ - 1, runInfo);

    if (runInfo.isLastS2InnerLoop) {
        SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT);
    }
}

template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::ProcessQk(uint64_t s2L1Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                                  bool syncKeyMte2, const LICommon::RunInfo &runInfo)
{
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);
    if (syncKeyMte2) {
        // One Set/Wait pair per L1 tile; also covers the segment-start
        // query/weight MTE2 via pipe order.
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
    }
    LoadQueryToL0a(runInfo);
    LoadKeyToL0b(s2L1Offset, s2L1RealSize, s2L0RealSize, runInfo);
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % L0_BUF_NUM);
    ComuteL0c(s2L0RealSize);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);
    FixpSToL1(s2L0RealSize);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % L0_BUF_NUM);
    l0BufIdx_++;
    l0cBufIdx_++;
}

template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::ProcessWs(uint64_t s2GmOffset, uint64_t s2L0RealSize, uint64_t sL1BufIdx,
                                                  const LICommon::RunInfo &runInfo)
{
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % L0_BUF_NUM);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);
    LoadSToL0b(s2L0RealSize, sL1BufIdx);
    LoadWeightToL0a();
    ComputeWs(s2L0RealSize);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);
    FixpResToGm(s2GmOffset, s2L0RealSize, runInfo);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % L0_BUF_NUM);
    l0BufIdx_++;
    l0cBufIdx_++;
}

template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                      const LICommon::RunInfo &runInfo)
{
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / constInfo_.kCacheBlockSize;
        uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % constInfo_.kCacheBlockSize;
        uint64_t keyGmOffset = static_cast<uint64_t>(blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + s2BlkId)) *
                                   constInfo_.kCacheBlockSize * constInfo_.headDim +
                               s2BlkOffset * constInfo_.headDim;
        uint64_t s2Mte2Size = (s2L1RealSize <= S2_BASIC_BLOCK_L0 || s2L1Offset >= S2_BASIC_BLOCK_L0) ?
                                  s2L1RealSize - s2L1Offset :
                                  S2_BASIC_BLOCK_L0 - s2L1Offset;
        s2Mte2Size = s2BlkOffset + s2Mte2Size >= constInfo_.kCacheBlockSize ? constInfo_.kCacheBlockSize - s2BlkOffset :
                                                                              s2Mte2Size;
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size;
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                                      CeilAlign(s2L1RealSize - S2_BASIC_BLOCK_L0, (uint64_t)BLOCK_CUBE) :
                                      (s2L1RealSize > S2_BASIC_BLOCK_L0 ?
                                           S2_BASIC_BLOCK_L0 :
                                           CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE));
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET +
                        (s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                             S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 + (s2L1Offset - S2_BASIC_BLOCK_L0) * S8_BLOCK_CUBE :
                             s2L1Offset * S8_BLOCK_CUBE)],
                 keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::QueryNd2Nz(const LICommon::RunInfo &runInfo)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = constInfo_.qHeadNum;
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(constInfo_.qHeadNum, static_cast<uint64_t>(BLOCK_CUBE));
    nd2nzPara.dstNzNStride = 1;
    uint64_t queryElements = constInfo_.qHeadNum * constInfo_.headDim;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    uint64_t firstQueryRow = static_cast<uint64_t>(runInfo.queryRow);
    DataCopy(queryL1_, queryGm_[firstQueryRow * queryElements], nd2nzPara);
}

// Brcb'd (w * q_scale) fp16 operand prepared by the vector side: qHeadNum
// blocks of BLOCK_CUBE identical halfs. GM -> L1 once per request segment.
template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::WeightDmaCopy(const LICommon::RunInfo &runInfo)
{
    DataCopyParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = constInfo_.qHeadNum;  // qHeadNum * 16 halfs = qHeadNum blocks of 32B
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopy(weightL1_, weightGm_[(runInfo.loop % 2) * WEIGHT_BUFFER_OFFSET], copyInParams);
}

template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::LoadQueryToL0a(const LICommon::RunInfo &runInfo)
{
    LoadData3DParamsV2<Q_T> loadData3DParams;
    loadData3DParams.l1H = CeilDiv(constInfo_.qHeadNum, static_cast<uint64_t>(BLOCK_CUBE));
    loadData3DParams.l1W = BLOCK_CUBE;
    loadData3DParams.channelSize = constInfo_.headDim;

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255;

    loadData3DParams.mExtension = constInfo_.qHeadNum;
    loadData3DParams.kExtension = constInfo_.headDim;
    loadData3DParams.mStartPt = 0;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;

    LoadData<Q_T, LOAD3DV2_CONFIG>(
        l0a_[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_S8_16K],
        queryL1_, loadData3DParams);
}

template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                                     const LICommon::RunInfo &runInfo)
{
    uint64_t keyL1Offset = s2L1Offset >= S2_BASIC_BLOCK_L0 ? S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 : 0;
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(s2L0RealSize, BLOCK_CUBE) * CeilDiv(constInfo_.headDim, S8_BLOCK_CUBE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(l0b_[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_S8_16K],
             keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET + keyL1Offset], loadData2DParams);
}

// fp16 per-head scores [qHeadNum, s2] in L1 -> transposed B operand [k=qHeadNum, n=s2].
// Mirrors QLI arch22 LoadSToL0b with a single head group (mStartPt = 0).
template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::LoadSToL0b(uint64_t s2L0RealSize, uint64_t sL1BufIdx)
{
    LoadData3DParamsV2<half> loadData3DParams;
    loadData3DParams.l1H = M_BASIC_BLOCK_L0 / BLOCK_CUBE;                // 64 head rows in 16-row fractals
    loadData3DParams.l1W = BLOCK_CUBE;
    loadData3DParams.channelSize = CeilAlign(s2L0RealSize, BLOCK_CUBE);  // Cin = s2

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255;  // 尾部数据不影响滑窗的结果

    loadData3DParams.mExtension = constInfo_.qHeadNum;                  // M height = head (k of mma2)
    loadData3DParams.kExtension = CeilAlign(s2L0RealSize, BLOCK_CUBE);  // K width = s2 (n of mma2)
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 1;
    loadData3DParams.fMatrixCtrl = 0;

    loadData3DParams.mStartPt = 0;
    LoadData<half, LOAD3DV2_CONFIG>(
        l0b_.template ReinterpretCast<half>()[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_FP16_16K],
        sL1_[(sL1BufIdx % 2) * SL1_BUFFER_OFFSET], loadData3DParams);
}

// A operand for the head reduction: [m=BLOCK_CUBE, k=qHeadNum] where every m
// row holds the same w*q_scale vector (Brcb'd by the vector side, transposed
// here into NZ fractals). Mirrors QLI arch22 LoadWeightToL0a.
template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::LoadWeightToL0a()
{
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(constInfo_.qHeadNum, static_cast<uint64_t>(BLOCK_CUBE));
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = true;
    LoadData(l0a_.template ReinterpretCast<half>()[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_FP16_16K],
             weightL1_, loadData2DParams);
}

template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::ComuteL0c(uint64_t s2L0RealSize)
{
    MmadParams mmadParams;
    mmadParams.m = constInfo_.qHeadNum;
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    // No unitFlag: the int8 unit-flag accumulate path produced 4x k-dim
    // over-accumulation on A3; the reference quant kernel uses plain Mmad.
    Mmad(cL0_[(l0cBufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET],
         l0a_[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_S8_16K],
         l0b_[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_S8_16K], mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

// Head reduction: [m=BLOCK_CUBE, k=qHeadNum] x [k=qHeadNum, n=s2] -> fp32 L0C.
// Only m row 0 carries the reduced score (all rows are identical by
// construction of the broadcast A operand).
template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::ComputeWs(uint64_t s2L0RealSize)
{
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    MmadParams mmadParams;
    mmadParams.m = BLOCK_CUBE;
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.qHeadNum;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    Mmad(cL0_.template ReinterpretCast<float>()[(l0cBufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET],
         l0a_.template ReinterpretCast<half>()[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_FP16_16K],
         l0b_.template ReinterpretCast<half>()[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_FP16_16K],
         mmadParams);
}

// int32 per-head scores -> fp16 in L1. DEQF16 applies the 2^-10 range
// normalization (SetFixpipePreQuantFlag, float bits 0x3a800000) so the values
// stay inside fp16 range; the uniform positive factor does not change top-k
// order. relu matches the previous NoQuant fixpipe point (pre-reduction).
template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::FixpSToL1(uint64_t s2L0RealSize)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    DataCopyCO12DstParams params;
    params.mSize = CeilAlign(constInfo_.qHeadNum, static_cast<uint64_t>(BLOCK_CUBE));
    params.nSize = CeilAlign(s2L0RealSize, BLOCK_CUBE);
    params.dstStride = M_BASIC_BLOCK_L0;
    params.srcStride = params.mSize;
    params.quantPre = QuantMode_t::DEQF16;
    params.reluPre = 1;
    params.channelSplit = 0;
    params.nz2ndEn = 0;
    SetFixpipePreQuantFlag(0x3a800000);
    DataCopy(sL1_[(sL1BufIdx_ % 2) * SL1_BUFFER_OFFSET],
             cL0_[(l0cBufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], params);
}

// Reduced fp32 score row -> GM: [1, s2] per 128-token tile, one s2BaseSize
// row per chunk buffer (double buffered by loop parity).
template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::FixpResToGm(uint64_t s2GmOffset, uint64_t s2L0RealSize,
                                                    const LICommon::RunInfo &runInfo)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.mSize = 1;
    intriParams.nSize = s2L0RealSize;
    intriParams.dstStride = constInfo_.s2BaseSize;
    intriParams.srcStride = BLOCK_CUBE;
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.nz2ndEn = true;
    intriParams.reluPre = 0;
    AscendC::SetFixpipeNz2ndFlag(1, S2_BASIC_BLOCK_L0 / BLOCK_CUBE, constInfo_.s2BaseSize);
    AscendC::DataCopy(mm1ResGm_[(runInfo.loop % 2) * constInfo_.s2BaseSize + s2GmOffset],
                      cL0_.template ReinterpretCast<float>()[(l0cBufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET],
                      intriParams);
}

template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::AllocEventID()
{
    // No SetMMLayoutTransform: the int8 quant LI kernels (v1/v2) run the MM in
    // the default layout mode; transform mode misreads int8 NZ fractals (C0=32).
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT);

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 2);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 3);

    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}

template <typename LIT>
__aicore__ inline void LIMatmulC8<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 2);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 3);

    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}

} // namespace LIKernel
#endif // FUSED_LI_MANAGE_C8_SERVICE_CUBE_H
