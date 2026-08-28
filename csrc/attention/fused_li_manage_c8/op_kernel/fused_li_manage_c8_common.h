/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef FUSED_LI_MANAGE_C8_COMMON_H
#define FUSED_LI_MANAGE_C8_COMMON_H

#include "fused_li_manage_common.h"

namespace LICommon {

struct LITypeC8Cube {
    using queryType = int8_t;
    using keyType = int8_t;
};

struct LITypeC8Vector {
    using queryType = bfloat16_t;
    using keyType = bfloat16_t;
    static constexpr bool IS_C8 = true;
};

} // namespace LICommon

#endif // FUSED_LI_MANAGE_C8_COMMON_H
