/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef TEMPLATE_TILING_KEY_LI_MANAGE_C8_H_
#define TEMPLATE_TILING_KEY_LI_MANAGE_C8_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define LI_TPL_C8 42
#define LI_TPL_C8_DUMMY 43

ASCENDC_TPL_ARGS_DECL(FusedLiManageC8,
                      ASCENDC_TPL_DTYPE_DECL(DT, LI_TPL_C8, LI_TPL_C8_DUMMY));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT, LI_TPL_C8)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT, LI_TPL_C8_DUMMY)), );

#endif
