/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 *
 * Vector service for fused_sparse_attention_overlap:
 * Handles SoftmaxFlashV2, online-softmax rescaling, and ProcessVec2 on AIV core.
 *
 * Key insight for M=1 with float atomic-add Mm2:
 *   SoftmaxFlashV2 outputs weights = exp(scores - running_max).
 *   Before each non-first Mm2, we rescale the accumulated result by
 *   exp(old_max - new_max) so all blocks share the same max reference.
 *   Final: output = Mm2_accumulated / softmaxSum.
 */
#ifndef FUSED_SPARSE_ATTENTION_OVERLAP_VECTOR_H_
#define FUSED_SPARSE_ATTENTION_OVERLAP_VECTOR_H_

#include "kernel_operator.h"

namespace FusedSparseAttentionOverlapNs {
using namespace AscendC;

constexpr SoftmaxConfig FSA_SOFTMAX_CFG = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};
constexpr uint32_t FSA_SOFTMAX_TMP_SIZE = 32 * 1024;  // 32KB for SoftmaxFlashV2 temp
constexpr bool FSA_DIAG_PROCESS_VEC2_RAW_MM2 = false;
constexpr bool FSA_DIAG_FORCE_UNIT_MM2_WEIGHTS = false;
constexpr int32_t FSA_DIAG_AIV_MM2_ACCUMULATE_MODE = 0;  // 0: full, 1: read temp[0], 2: read all temp, 3: DMA temp, 4: marker, 5: temp->accum, 6: accum+1

#ifdef ASCENDC_CPU_DEBUG
// ============================================================================
// CPU simulation: pure C++ softmax (SoftmaxFlashV2 crashes in tikicpulib
// due to overly strict buffer overlap checks in Sub/Add vector ops)
// ============================================================================
template <typename T>
class FusedAttentionVectorService {
public:
    using MM_OUT_T = float;
    __aicore__ inline FusedAttentionVectorService() {}

    __aicore__ inline void Init(TPipe* pipe,
                                const FusedSparseAttentionOverlapTilingData* tiling,
                                GM_ADDR workspace)
    {
        pipe_ = pipe;
        tiling_ = tiling;
        mm1ResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MM_OUT_T*>(workspace));
        workspaceTGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace));
        mm2ResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MM_OUT_T*>(workspace));
        runningMax_ = -3.4028235e+38f;
        runningSum_ = 0.0f;
    }

    __aicore__ inline void ProcessSoftmaxAndRescale(
        int32_t nTokens, int64_t scoreGmOffset,
        bool isFirstBlock, float scaleValue,
        int64_t mm2OutGmOff, int64_t kvDimFloatAlign);

    __aicore__ inline void ProcessVec2(
        GlobalTensor<T>& outputGm, int64_t outputGmOffset,
        int64_t mm2OutGmOff, int64_t kvDimFloatAlign);

    __aicore__ inline void AccumulateMm2Temp(
        int64_t mm2OutGmOff, int64_t mm2TempGmOff, int64_t kvDimFloatAlign);

    __aicore__ inline void ResetSoftmaxState() {
        runningMax_ = -3.4028235e+38f;
        runningSum_ = 0.0f;
    }

private:
    TPipe* pipe_ = nullptr;
    const FusedSparseAttentionOverlapTilingData* tiling_ = nullptr;
    float runningMax_;
    float runningSum_;
    GlobalTensor<MM_OUT_T> mm1ResGm_;
    GlobalTensor<T> workspaceTGm_;
    GlobalTensor<MM_OUT_T> mm2ResGm_;
};

// CPU_PLACEHOLDER_FOR_APPEND

template <typename T>
__aicore__ inline void FusedAttentionVectorService<T>::ProcessSoftmaxAndRescale(
    int32_t nTokens, int64_t scoreGmOffset,
    bool isFirstBlock, float scaleValue,
    int64_t mm2OutGmOff, int64_t kvDimFloatAlign)
{
    if (nTokens <= 0) return;

    float oldMax = runningMax_;

    // Step 1: find new max across scores
    float blockMax = -3.4028235e+38f;
    for (int32_t i = 0; i < nTokens; i++) {
        float s = mm1ResGm_.GetValue(scoreGmOffset + i) * scaleValue;
        if (s > blockMax) blockMax = s;
    }
    float newMax = (blockMax > runningMax_) ? blockMax : runningMax_;

    // Step 2: compute weights = exp(score * scale - newMax), and sum
    float blockSum = 0.0f;
    int32_t columnAlign = ((nTokens + 127) / 128) * 128;
    for (int32_t i = 0; i < nTokens; i++) {
        float s = mm1ResGm_.GetValue(scoreGmOffset + i) * scaleValue;
        float w = expf(s - newMax);
        mm1ResGm_.SetValue(scoreGmOffset + i, w);  // reuse as weights (float)
        blockSum += w;
    }
    // zero-pad to columnAlign
    for (int32_t i = nTokens; i < columnAlign; i++) {
        mm1ResGm_.SetValue(scoreGmOffset + i, 0.0f);
    }

    // Step 3: update running sum with correction for max change
    float correction = expf(oldMax - newMax);
    if (!isFirstBlock) {
        runningSum_ = runningSum_ * correction + blockSum;
    } else {
        runningSum_ = blockSum;
    }
    runningMax_ = newMax;

    // Step 4: Cast float鈫扵, write weights to GM for Mm2
    int64_t scoreGmOffsetT = scoreGmOffset * static_cast<int64_t>(sizeof(MM_OUT_T) / sizeof(T));
    for (int32_t i = 0; i < columnAlign; i++) {
        float w = mm1ResGm_.GetValue(scoreGmOffset + i);
        workspaceTGm_.SetValue(scoreGmOffsetT + i, static_cast<T>(w));
    }

    // Step 5: Rescale accumulated Mm2 output for non-first blocks
    if (!isFirstBlock) {
        uint32_t kvSize = static_cast<uint32_t>(kvDimFloatAlign);
        for (uint32_t i = 0; i < kvSize; i++) {
            float v = mm2ResGm_.GetValue(mm2OutGmOff + i);
            mm2ResGm_.SetValue(mm2OutGmOff + i, v * correction);
        }
    }

    printf("[CPU-softmax] nTokens=%d blockMax=%.4f newMax=%.4f sum=%.4f correction=%.4f\n",
           nTokens, blockMax, newMax, runningSum_, correction);
}

template <typename T>
__aicore__ inline void FusedAttentionVectorService<T>::ProcessVec2(
    GlobalTensor<T>& outputGm, int64_t outputGmOffset,
    int64_t mm2OutGmOff, int64_t kvDimFloatAlign)
{
    int64_t kvCacheDim = tiling_->kvCacheDim;
    float invSum = (runningSum_ > 0.0f) ? (1.0f / runningSum_) : 1.0f;

    printf("[CPU-vec2] mm2Out[0]=%.4f mm2Out[1]=%.4f invSum=%.6f\n",
           mm2ResGm_.GetValue(mm2OutGmOff), mm2ResGm_.GetValue(mm2OutGmOff + 1), invSum);

    for (int64_t i = 0; i < kvCacheDim; i++) {
        float v = mm2ResGm_.GetValue(mm2OutGmOff + i) * invSum;
        outputGm.SetValue(outputGmOffset + i, static_cast<T>(v));
    }
}

template <typename T>
__aicore__ inline void FusedAttentionVectorService<T>::AccumulateMm2Temp(
    int64_t mm2OutGmOff, int64_t mm2TempGmOff, int64_t kvDimFloatAlign)
{
    for (int64_t i = 0; i < kvDimFloatAlign; i++) {
        float accum = mm2ResGm_.GetValue(mm2OutGmOff + i);
        float delta = mm2ResGm_.GetValue(mm2TempGmOff + i);
        mm2ResGm_.SetValue(mm2OutGmOff + i, accum + delta);
    }
}

#else // !ASCENDC_CPU_DEBUG 鈥?NPU version below

template <typename T>
class FusedAttentionVectorService {
public:
    using MM_OUT_T = float;

    __aicore__ inline FusedAttentionVectorService() {}

    __aicore__ inline void Init(TPipe* pipe,
                                const FusedSparseAttentionOverlapTilingData* tiling,
                                GM_ADDR workspace)
    {
        pipe_ = pipe;
        tiling_ = tiling;

        // Softmax buffers
        pipe_->InitBuffer(softmaxTmpBuf_, FSA_SOFTMAX_TMP_SIZE);
        pipe_->InitBuffer(scoreInputBuf_, 2 * 4096);
        pipe_->InitBuffer(mm2AccumBuf_, 4096);
        pipe_->InitBuffer(mm2TempBuf_, 4096);
        constexpr int32_t BLOCK_FLOATS = BLOCK_BYTES / sizeof(float);
        int64_t softmaxStateSize = CeilAlign(static_cast<int64_t>(BLOCK_FLOATS * sizeof(float)),
                                              static_cast<int64_t>(BLOCK_BYTES));
        pipe_->InitBuffer(softmaxMaxBuf_, softmaxStateSize * 2);  // ping-pong
        pipe_->InitBuffer(softmaxSumBuf_, softmaxStateSize * 2);  // ping-pong
        pipe_->InitBuffer(softmaxExpBuf_, softmaxStateSize);
        pipe_->InitBuffer(prevMaxBuf_, softmaxStateSize);  // store previous max for rescaling
        pipe_->InitBuffer(rescaleTmpBuf_, softmaxStateSize);  // temp for exp computation

        // GM workspace pointers
        mm1ResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MM_OUT_T*>(workspace));
        workspaceTGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace));
        mm2ResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MM_OUT_T*>(workspace));
    }

    // Process softmax on Mm1 scores, rescale accumulated Mm2 if needed,
    // write T-format weights to GM for Mm2
    __aicore__ inline void ProcessSoftmaxAndRescale(
        int32_t nTokens, int64_t scoreGmOffset,
        bool isFirstBlock, float scaleValue,
        int64_t mm2OutGmOff, int64_t kvDimFloatAlign);

    // ProcessVec2: output = Mm2Output / softmaxSum, cast to T, write to outputGm
    __aicore__ inline void ProcessVec2(
        GlobalTensor<T>& outputGm, int64_t outputGmOffset,
        int64_t mm2OutGmOff, int64_t kvDimFloatAlign);

    __aicore__ inline void AccumulateMm2Temp(
        int64_t mm2OutGmOff, int64_t mm2TempGmOff, int64_t kvDimFloatAlign);

    // Reset softmax state between heads
    __aicore__ inline void ResetSoftmaxState() { prevSlot_ = 0; }

private:
    template <typename U>
    __aicore__ inline U CeilAlign(U a, U b) { return (b == 0) ? 0 : ((a + b - 1) / b * b); }

    TPipe* pipe_ = nullptr;
    const FusedSparseAttentionOverlapTilingData* tiling_ = nullptr;
    int32_t prevSlot_ = 0;

    TBuf<QuePosition::VECCALC> softmaxTmpBuf_;
    TBuf<QuePosition::VECCALC> scoreInputBuf_;
    TBuf<QuePosition::VECCALC> mm2AccumBuf_;
    TBuf<QuePosition::VECCALC> mm2TempBuf_;
    TBuf<QuePosition::VECCALC> softmaxMaxBuf_;
    TBuf<QuePosition::VECCALC> softmaxSumBuf_;
    TBuf<QuePosition::VECCALC> softmaxExpBuf_;
    TBuf<QuePosition::VECCALC> prevMaxBuf_;      // previous block's outMax[0]
    TBuf<QuePosition::VECCALC> rescaleTmpBuf_;   // temp for rescale computation

    GlobalTensor<MM_OUT_T> mm1ResGm_;
    GlobalTensor<T> workspaceTGm_;
    GlobalTensor<MM_OUT_T> mm2ResGm_;
};

// ============================================================================
// ProcessSoftmaxAndRescale
// 1. Load scores from GM, scale, SoftmaxFlashV2
// 2. For non-first blocks: rescale accumulated Mm2 by exp(old_max - new_max)
// 3. Cast float鈫扵, write weights to GM for Mm2
// ============================================================================
template <typename T>
__aicore__ inline void FusedAttentionVectorService<T>::ProcessSoftmaxAndRescale(
    int32_t nTokens, int64_t scoreGmOffset,
    bool isFirstBlock, float scaleValue,
    int64_t mm2OutGmOff, int64_t kvDimFloatAlign)
{
    if (nTokens <= 0) return;

    constexpr int32_t BLOCK_FLOATS = BLOCK_BYTES / sizeof(float);
    uint32_t dealRowCount = 1;  // M=1
    constexpr uint32_t MIN_COLUMN_COUNT = 128;
    uint32_t columnCount = ((nTokens + MIN_COLUMN_COUNT - 1) / MIN_COLUMN_COUNT) * MIN_COLUMN_COUNT;
    uint32_t actualColumnCount = static_cast<uint32_t>(nTokens);
    uint32_t computeSize = dealRowCount * columnCount;

    // --- Step 1: Load scores, scale, SoftmaxFlashV2 ---
    LocalTensor<uint8_t> scoreBytes = scoreInputBuf_.Get<uint8_t>();
    LocalTensor<MM_OUT_T> scoreUb = scoreBytes.template ReinterpretCast<MM_OUT_T>();
    Duplicate(scoreUb, 0.0f, computeSize);
    pipe_barrier(PIPE_V);
    DataCopyExtParams copyParams{static_cast<uint16_t>(1),
        static_cast<uint32_t>(actualColumnCount * sizeof(MM_OUT_T)), 0, 0, 0};
    DataCopyPadExtParams<MM_OUT_T> padParams{false, 0, 0, 0};
    DataCopyPad(scoreUb, mm1ResGm_[scoreGmOffset], copyParams, padParams);
    pipe_barrier(PIPE_ALL);

    Muls(scoreUb, scoreUb, scaleValue, computeSize);
    pipe_barrier(PIPE_V);

    LocalTensor<uint8_t> softmaxTmpUb = softmaxTmpBuf_.Get<uint8_t>();
    LocalTensor<MM_OUT_T> softmaxMaxBuf = softmaxMaxBuf_.Get<MM_OUT_T>();
    LocalTensor<MM_OUT_T> softmaxSumBuf = softmaxSumBuf_.Get<MM_OUT_T>();
    LocalTensor<MM_OUT_T> softmaxExpUb = softmaxExpBuf_.Get<MM_OUT_T>();

    constexpr int32_t STATE_SIZE = BLOCK_FLOATS;
    int32_t outSlot = isFirstBlock ? 0 : (1 - prevSlot_);
    int32_t inSlot = isFirstBlock ? 0 : prevSlot_;
    prevSlot_ = outSlot;

    LocalTensor<MM_OUT_T> outSumUb = softmaxSumBuf[outSlot * STATE_SIZE];
    LocalTensor<MM_OUT_T> outMaxUb = softmaxMaxBuf[outSlot * STATE_SIZE];
    LocalTensor<MM_OUT_T> inSumUb = softmaxSumBuf[inSlot * STATE_SIZE];
    LocalTensor<MM_OUT_T> inMaxUb = softmaxMaxBuf[inSlot * STATE_SIZE];

    if (isFirstBlock) {
        Duplicate(inMaxUb, -3.4028235e+38f, BLOCK_FLOATS);
        Duplicate(inSumUb, 0.0f, BLOCK_FLOATS);
        pipe_barrier(PIPE_V);
    }

    // Save previous max before SoftmaxFlashV2 overwrites it
    LocalTensor<MM_OUT_T> prevMaxUb = prevMaxBuf_.Get<MM_OUT_T>();
    if (!isFirstBlock) {
        Adds(prevMaxUb, inMaxUb, 0.0f, BLOCK_FLOATS);
        pipe_barrier(PIPE_V);
    }

    SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
    SoftMaxTiling newTiling = SoftMaxFlashV2TilingFunc(
        srcShape, sizeof(MM_OUT_T), sizeof(MM_OUT_T),
        softmaxTmpUb.GetSize(), true, false);
    SoftmaxFlashV2<MM_OUT_T, true, true, false, false, FSA_SOFTMAX_CFG>(
        scoreUb, outSumUb, outMaxUb, scoreUb, softmaxExpUb,
        inSumUb, inMaxUb, softmaxTmpUb, newTiling, srcShape);
    pipe_barrier(PIPE_V);

    // --- Step 2: Cast float鈫扵, write weights to GM for Mm2 ---
    // Do this BEFORE rescaling so scoreUb is free for Mm2 read/write
    int64_t scoreGmOffsetT = scoreGmOffset * static_cast<int64_t>(sizeof(MM_OUT_T) / sizeof(T));
    uint32_t scoreTByteOffset = CeilAlign(
        static_cast<uint32_t>(computeSize * sizeof(MM_OUT_T)),
        static_cast<uint32_t>(BLOCK_BYTES));
    LocalTensor<T> scoreT = scoreBytes[scoreTByteOffset].template ReinterpretCast<T>();
    Cast(scoreT, scoreUb, RoundMode::CAST_ROUND, computeSize);
    pipe_barrier(PIPE_V);
    if constexpr (FSA_DIAG_FORCE_UNIT_MM2_WEIGHTS) {
        Duplicate(scoreT, static_cast<T>(1.0f), computeSize);
        pipe_barrier(PIPE_V);
    }
    DataCopyExtParams copyParamsT{static_cast<uint16_t>(1),
        static_cast<uint32_t>(computeSize * sizeof(T)), 0, 0, 0};
    DataCopyPad(workspaceTGm_[scoreGmOffsetT], scoreT, copyParamsT);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    pipe_barrier(PIPE_ALL);

    // --- Step 3: Rescale accumulated Mm2 output for non-first blocks ---
    // correction = exp(old_max - new_max)
    // Now scoreUb is free (weights already written to GM)
    if (!isFirstBlock) {
        LocalTensor<MM_OUT_T> rescaleTmp = rescaleTmpBuf_.Get<MM_OUT_T>();
        float oldMax = prevMaxUb.GetValue(0);
        float newMax = outMaxUb.GetValue(0);
        float correction = oldMax - newMax;
        if (correction < -80.0f) correction = -80.0f;
        Duplicate(rescaleTmp, correction, BLOCK_FLOATS);
        pipe_barrier(PIPE_V);
        Exp(rescaleTmp, rescaleTmp, BLOCK_FLOATS);
        pipe_barrier(PIPE_V);
        float correctionVal = rescaleTmp.GetValue(0);

        // Read accumulated Mm2 from GM, multiply by correction, write back
        uint32_t kvSize = static_cast<uint32_t>(kvDimFloatAlign);
        LocalTensor<MM_OUT_T> mm2Tmp = scoreUb;  // scoreUb is free now
        DataCopyExtParams mm2CopyParams{static_cast<uint16_t>(1),
            static_cast<uint32_t>(kvSize * sizeof(MM_OUT_T)), 0, 0, 0};
        DataCopyPadExtParams<MM_OUT_T> mm2PadParams{false, 0, 0, 0};
        DataCopyPad(mm2Tmp, mm2ResGm_[mm2OutGmOff], mm2CopyParams, mm2PadParams);
        pipe_barrier(PIPE_ALL);

        Muls(mm2Tmp, mm2Tmp, correctionVal, kvSize);
        pipe_barrier(PIPE_V);

        DataCopyPad(mm2ResGm_[mm2OutGmOff], mm2Tmp, mm2CopyParams);
        pipe_barrier(PIPE_ALL);
    }
}

// ============================================================================
// ProcessVec2: output = Mm2Output / softmaxSum, cast to T, write to outputGm
// ============================================================================
template <typename T>
__aicore__ inline void FusedAttentionVectorService<T>::ProcessVec2(
    GlobalTensor<T>& outputGm, int64_t outputGmOffset,
    int64_t mm2OutGmOff, int64_t kvDimFloatAlign)
{
    constexpr int32_t BLOCK_FLOATS = BLOCK_BYTES / sizeof(float);
    int64_t kvCacheDim = tiling_->kvCacheDim;
    uint32_t copySize = static_cast<uint32_t>(kvDimFloatAlign);

    // Read Mm2 output from GM workspace (float)
    LocalTensor<uint8_t> outBytes = scoreInputBuf_.Get<uint8_t>();
    LocalTensor<MM_OUT_T> outFloat = outBytes.template ReinterpretCast<MM_OUT_T>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1),
        static_cast<uint32_t>(copySize * sizeof(MM_OUT_T)), 0, 0, 0};
    DataCopyPadExtParams<MM_OUT_T> padParams{false, 0, 0, 0};
    DataCopyPad(outFloat, mm2ResGm_[mm2OutGmOff], copyParams, padParams);
    pipe_barrier(PIPE_ALL);

    uint32_t outTByteOffset = CeilAlign(
        static_cast<uint32_t>(copySize * sizeof(MM_OUT_T)),
        static_cast<uint32_t>(BLOCK_BYTES));
    LocalTensor<MM_OUT_T> absOut = outBytes[outTByteOffset].template ReinterpretCast<MM_OUT_T>();
    Abs(absOut, outFloat, copySize);
    pipe_barrier(PIPE_V);
    LocalTensor<uint8_t> finiteMask = absOut.template ReinterpretCast<uint8_t>();
    CompareScalar(finiteMask, absOut, static_cast<MM_OUT_T>(1.0e10f), CMPMODE::LE, copySize);
    pipe_barrier(PIPE_V);
    Select(outFloat, finiteMask, outFloat, static_cast<MM_OUT_T>(0.0f),
           SELMODE::VSEL_TENSOR_SCALAR_MODE, copySize);
    pipe_barrier(PIPE_V);

    if constexpr (!FSA_DIAG_PROCESS_VEC2_RAW_MM2) {
        // Get softmaxSum from the last output slot
        LocalTensor<MM_OUT_T> softmaxSumBuf = softmaxSumBuf_.Get<MM_OUT_T>();
        constexpr int32_t STATE_SIZE = BLOCK_FLOATS;
        LocalTensor<MM_OUT_T> lastSumUb = softmaxSumBuf[prevSlot_ * STATE_SIZE];
        float softmaxSum = lastSumUb.GetValue(0);

        if (softmaxSum <= 0.0f || softmaxSum != softmaxSum) softmaxSum = 1.0f;
        float invSum = 1.0f / softmaxSum;

        Muls(outFloat, outFloat, invSum, copySize);
        pipe_barrier(PIPE_V);
    }

    // Cast float 鈫?T
    LocalTensor<T> outT = outBytes[outTByteOffset].template ReinterpretCast<T>();
    Cast(outT, outFloat, RoundMode::CAST_ROUND, copySize);
    pipe_barrier(PIPE_V);

    // Write to output GM (kvCacheDim elements)
    DataCopyExtParams outCopyParams{static_cast<uint16_t>(1),
        static_cast<uint32_t>(kvCacheDim * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGm[outputGmOffset], outT, outCopyParams);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
__aicore__ inline void FusedAttentionVectorService<T>::AccumulateMm2Temp(
    int64_t mm2OutGmOff, int64_t mm2TempGmOff, int64_t kvDimFloatAlign)
{
    if constexpr (FSA_DIAG_AIV_MM2_ACCUMULATE_MODE == 1) {
        float delta = mm2ResGm_.GetValue(mm2TempGmOff);
        (void)delta;
        pipe_barrier(PIPE_ALL);
        return;
    }
    if constexpr (FSA_DIAG_AIV_MM2_ACCUMULATE_MODE == 2) {
        float sum = 0.0f;
        for (int64_t offset = 0; offset < kvDimFloatAlign; offset++) {
            sum += mm2ResGm_.GetValue(mm2TempGmOff + offset);
        }
        (void)sum;
        pipe_barrier(PIPE_ALL);
        return;
    }
    if constexpr (FSA_DIAG_AIV_MM2_ACCUMULATE_MODE == 3) {
        uint32_t kvSize = static_cast<uint32_t>(kvDimFloatAlign);
        uint32_t kvBytes = kvSize * sizeof(MM_OUT_T);
        LocalTensor<uint8_t> bytes = scoreInputBuf_.Get<uint8_t>();
        LocalTensor<MM_OUT_T> temp = bytes.template ReinterpretCast<MM_OUT_T>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), kvBytes, 0, 0, 0};
        DataCopyPadExtParams<MM_OUT_T> padParams{false, 0, 0, 0};
        DataCopyPad(temp, mm2ResGm_[mm2TempGmOff], copyParams, padParams);
        pipe_barrier(PIPE_ALL);
        return;
    }
    if constexpr (FSA_DIAG_AIV_MM2_ACCUMULATE_MODE == 4) {
        LocalTensor<uint8_t> bytes = scoreInputBuf_.Get<uint8_t>();
        LocalTensor<MM_OUT_T> marker = bytes.template ReinterpretCast<MM_OUT_T>();
        Duplicate(marker, 0.0f, static_cast<uint32_t>(BLOCK_BYTES / sizeof(MM_OUT_T)));
        marker.SetValue(0, 123.0f);
        pipe_barrier(PIPE_V);
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(BLOCK_BYTES), 0, 0, 0};
        DataCopyPad(mm2ResGm_[mm2OutGmOff], marker, copyParams);
        pipe_barrier(PIPE_ALL);
        return;
    }
    if constexpr (FSA_DIAG_AIV_MM2_ACCUMULATE_MODE == 5) {
        uint32_t kvSize = static_cast<uint32_t>(kvDimFloatAlign);
        uint32_t kvBytes = kvSize * sizeof(MM_OUT_T);
        LocalTensor<uint8_t> bytes = scoreInputBuf_.Get<uint8_t>();
        LocalTensor<MM_OUT_T> temp = bytes.template ReinterpretCast<MM_OUT_T>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), kvBytes, 0, 0, 0};
        DataCopyPadExtParams<MM_OUT_T> padParams{false, 0, 0, 0};
        DataCopyPad(temp, mm2ResGm_[mm2TempGmOff], copyParams, padParams);
        pipe_barrier(PIPE_ALL);
        DataCopyPad(mm2ResGm_[mm2OutGmOff], temp, copyParams);
        pipe_barrier(PIPE_ALL);
        return;
    }
    if constexpr (FSA_DIAG_AIV_MM2_ACCUMULATE_MODE == 6) {
        uint32_t kvSize = static_cast<uint32_t>(kvDimFloatAlign);
        uint32_t kvBytes = kvSize * sizeof(MM_OUT_T);
        LocalTensor<uint8_t> bytes = scoreInputBuf_.Get<uint8_t>();
        LocalTensor<MM_OUT_T> accum = bytes.template ReinterpretCast<MM_OUT_T>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), kvBytes, 0, 0, 0};
        DataCopyPadExtParams<MM_OUT_T> padParams{false, 0, 0, 0};
        DataCopyPad(accum, mm2ResGm_[mm2OutGmOff], copyParams, padParams);
        pipe_barrier(PIPE_ALL);
        Adds(accum, accum, 1.0f, kvSize);
        pipe_barrier(PIPE_V);
        DataCopyPad(mm2ResGm_[mm2OutGmOff], accum, copyParams);
        pipe_barrier(PIPE_ALL);
        return;
    }
    uint32_t kvSize = static_cast<uint32_t>(kvDimFloatAlign);
    constexpr uint32_t ACCUM_BUF_BYTES = 4096;
    constexpr uint32_t ACCUM_CHUNK_FLOATS = ACCUM_BUF_BYTES / sizeof(MM_OUT_T);
    constexpr uint32_t DELTA_OFFSET = ACCUM_CHUNK_FLOATS;

    (void)DELTA_OFFSET;
    LocalTensor<MM_OUT_T> accum = mm2AccumBuf_.Get<MM_OUT_T>();
    LocalTensor<MM_OUT_T> delta = mm2TempBuf_.Get<MM_OUT_T>();

    DataCopyPadExtParams<MM_OUT_T> padParams{false, 0, 0, 0};
    for (uint32_t offset = 0; offset < kvSize; offset += ACCUM_CHUNK_FLOATS) {
        uint32_t curSize = kvSize - offset;
        if (curSize > ACCUM_CHUNK_FLOATS) {
            curSize = ACCUM_CHUNK_FLOATS;
        }
        uint32_t curBytes = curSize * sizeof(MM_OUT_T);
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), curBytes, 0, 0, 0};
        DataCopyPad(accum, mm2ResGm_[mm2OutGmOff + offset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        DataCopyPad(delta, mm2ResGm_[mm2TempGmOff + offset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        Add(accum, accum, delta, curSize);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        DataCopyPad(mm2ResGm_[mm2OutGmOff + offset], accum, copyParams);
        pipe_barrier(PIPE_ALL);
    }
}

#endif // ASCENDC_CPU_DEBUG

} // namespace FusedSparseAttentionOverlapNs

#endif // FUSED_SPARSE_ATTENTION_OVERLAP_VECTOR_H_
