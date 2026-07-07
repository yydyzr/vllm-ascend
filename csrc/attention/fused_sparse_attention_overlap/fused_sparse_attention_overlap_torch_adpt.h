/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FUSED_SPARSE_ATTENTION_OVERLAP_TORCH_ADPT_H
#define FUSED_SPARSE_ATTENTION_OVERLAP_TORCH_ADPT_H

#include <algorithm>
#include <limits>

namespace vllm_ascend {

namespace {

inline int64_t GetInt32Value(const at::Tensor &tensor, int64_t offset)
{
    return static_cast<int64_t>(tensor.data_ptr<int32_t>()[offset]);
}

inline void CopySelectedCpuSourceToSelectionCache(
    const at::Tensor &selection_k_rope,
    const at::Tensor &selection_kv_cache,
    const at::Tensor &selection_kv_block_table,
    const at::Tensor &selection_kv_block_status,
    const at::Tensor &selection_topk_indices,
    const at::Tensor &full_k_rope,
    const at::Tensor &full_kv_cache,
    const at::Tensor &full_kv_block_table,
    int64_t selection_topk_block_size)
{
    TORCH_CHECK(selection_topk_indices.dim() == 3 || selection_topk_indices.dim() == 4,
                "selection_topk_indices must be TND or BSND, but got dim ",
                selection_topk_indices.dim());
    TORCH_CHECK(selection_kv_block_status.dim() == selection_topk_indices.dim(),
                "selection_kv_block_status dim must match selection_topk_indices dim");
    TORCH_CHECK(full_kv_block_table.dim() == 2,
                "full_kv_block_table must be 2-D");

    const int64_t topk = selection_topk_indices.size(selection_topk_indices.dim() - 1);
    const int64_t kv_heads = selection_topk_indices.size(selection_topk_indices.dim() - 2);
    const int64_t row_count = selection_topk_indices.numel() / topk;
    const int64_t query_token_count = row_count / kv_heads;
    const int64_t batch_count = full_kv_block_table.size(0);
    const int64_t full_max_block_num = full_kv_block_table.size(1);
    const int64_t selection_max_block_num = selection_kv_block_table.size(1);
    const int64_t selection_block_size = selection_kv_cache.size(1);
    const int64_t full_block_size = full_kv_cache.size(1);

    auto topk_cpu = selection_topk_indices.to(at::kCPU).contiguous();
    auto selection_block_table_cpu = selection_kv_block_table.to(at::kCPU).contiguous();
    auto full_block_table_cpu = full_kv_block_table.to(at::kCPU).contiguous();
    auto status_cpu = at::full(
        selection_kv_block_status.sizes(),
        -1,
        selection_kv_block_status.options().device(at::kCPU));
    int32_t *status_data = status_cpu.data_ptr<int32_t>();

    using namespace at::indexing;
    for (int64_t row = 0; row < row_count; ++row) {
        const int64_t token_index = row / kv_heads;
        const int64_t batch_index = std::min(token_index, batch_count - 1);
        int64_t actual_selected_tokens = 0;
        for (int64_t topk_index = 0; topk_index < topk; ++topk_index) {
            const int64_t topk_id = GetInt32Value(topk_cpu, row * topk + topk_index);
            if (topk_id < 0) {
                continue;
            }
            status_data[row * (topk + 1) + topk_index] = static_cast<int32_t>(topk_id);
            for (int64_t token_offset = 0; token_offset < selection_topk_block_size; ++token_offset) {
                const int64_t source_token = topk_id * selection_topk_block_size + token_offset;
                const int64_t source_table_index = source_token / full_block_size;
                const int64_t source_block_offset = source_token % full_block_size;
                if (source_table_index >= full_max_block_num) {
                    continue;
                }
                const int64_t source_block = GetInt32Value(
                    full_block_table_cpu,
                    batch_index * full_max_block_num + source_table_index);
                if (source_block < 0) {
                    continue;
                }

                const int64_t selected_token = topk_index * selection_topk_block_size + token_offset;
                const int64_t destination_table_index = selected_token / selection_block_size;
                const int64_t destination_block_offset = selected_token % selection_block_size;
                if (destination_table_index >= selection_max_block_num) {
                    continue;
                }
                const int64_t destination_block = GetInt32Value(
                    selection_block_table_cpu,
                    row * selection_max_block_num + destination_table_index);
                if (destination_block < 0) {
                    continue;
                }

                selection_kv_cache.index({destination_block, destination_block_offset}).copy_(
                    full_kv_cache.index({source_block, source_block_offset}),
                    true);
                if (selection_k_rope.numel() > 0 && full_k_rope.numel() > 0) {
                    selection_k_rope.index({destination_block, destination_block_offset}).copy_(
                        full_k_rope.index({source_block, source_block_offset}),
                        true);
                }
                ++actual_selected_tokens;
            }
        }
        status_data[row * (topk + 1) + topk] = static_cast<int32_t>(
            actual_selected_tokens / selection_topk_block_size);
    }
    selection_kv_block_status.copy_(status_cpu, true);
}

}  // namespace

at::Tensor npu_fused_sparse_attention_overlap(
    const at::Tensor &query,
    const at::Tensor &selection_k_rope,
    const at::Tensor &selection_kv_cache,
    const at::Tensor &selection_kv_block_table,
    const at::Tensor &selection_kv_block_status,
    const at::Tensor &selection_topk_indices,
    const at::Tensor &full_k_rope,
    const at::Tensor &full_kv_cache,
    const at::Tensor &full_kv_block_table,
    const at::Tensor &full_kv_actual_seq,
    const at::Tensor &full_q_actual_seq,
    double scale_value,
    int64_t sparse_block_size,
    int64_t selection_topk_block_size,
    c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode)
{
    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);

    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
                                       "than 0, but shape[", i, "] is ", query.size(i));
    }

    // Construct output tensors
    at::Tensor attention_output = at::empty(query.sizes(), query.options().dtype(query.dtype()));

    const int SIZE = 8;
    c10::SmallVector<int64_t, SIZE> actual_seq_shape = {selection_kv_block_table.size(0)};
    at::Tensor selection_kv_actual_seq = at::empty(actual_seq_shape, selection_kv_block_table.options());

    at::Tensor hit_mask = at::empty(selection_topk_indices.sizes(),
                                    selection_topk_indices.options().dtype(torch::kInt32));
    at::Tensor miss_indices = at::empty(selection_topk_indices.sizes(),
                                        selection_topk_indices.options());

    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_CMD(
        aclnnFusedSparseAttentionOverlap,
        query,
        selection_k_rope, selection_kv_cache, selection_kv_block_table,
        selection_kv_block_status, selection_topk_indices,
        full_k_rope, full_kv_cache, full_kv_block_table,
        full_kv_actual_seq, full_q_actual_seq,
        scale_value, sparse_block_size, selection_topk_block_size,
        layout_query_ptr, layout_kv_ptr, sparse_mode,
        hit_mask, miss_indices,
        attention_output, selection_kv_actual_seq);

    return attention_output;
}

at::Tensor npu_fused_sparse_attention_overlap_cpu_source(
    const at::Tensor &query,
    const at::Tensor &selection_k_rope,
    const at::Tensor &selection_kv_cache,
    const at::Tensor &selection_kv_block_table,
    const at::Tensor &selection_kv_block_status,
    const at::Tensor &selection_topk_indices,
    const at::Tensor &full_k_rope,
    const at::Tensor &full_kv_cache,
    const at::Tensor &full_kv_block_table,
    const at::Tensor &full_kv_actual_seq,
    const at::Tensor &full_q_actual_seq,
    double scale_value,
    int64_t sparse_block_size,
    int64_t selection_topk_block_size,
    c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode)
{
    TORCH_CHECK(query.dim() == 3,
                "cpu-source fused path currently expects TND query, but got dim ",
                query.dim());
    TORCH_CHECK(selection_kv_cache.dim() == 3,
                "selection_kv_cache must be [block, block_size, dim]");

    CopySelectedCpuSourceToSelectionCache(
        selection_k_rope,
        selection_kv_cache,
        selection_kv_block_table,
        selection_kv_block_status,
        selection_topk_indices,
        full_k_rope,
        full_kv_cache,
        full_kv_block_table,
        selection_topk_block_size);

    using namespace at::indexing;
    const int64_t kv_cache_dim = selection_kv_cache.size(2);
    const int64_t k_rope_dim = selection_k_rope.numel() > 0 ? selection_k_rope.size(2) : 0;
    TORCH_CHECK(query.size(2) >= kv_cache_dim + k_rope_dim,
                "query last dim should be at least kv_cache_dim + k_rope_dim, but got ",
                query.size(2), " vs ", kv_cache_dim + k_rope_dim);

    const int64_t topk = selection_topk_indices.size(selection_topk_indices.dim() - 1);
    const int64_t kv_heads = selection_topk_indices.size(selection_topk_indices.dim() - 2);
    const int64_t selected_token_count = topk * selection_topk_block_size;
    const int64_t query_token_count = query.size(0);

    auto query_nope = query.index({Slice(), Slice(), Slice(0, kv_cache_dim)}).contiguous();
    c10::optional<at::Tensor> query_rope = c10::nullopt;
    c10::optional<at::Tensor> key_rope = c10::nullopt;
    if (k_rope_dim > 0) {
        query_rope = query.index({Slice(), Slice(), Slice(kv_cache_dim, kv_cache_dim + k_rope_dim)}).contiguous();
        key_rope = selection_k_rope.unsqueeze(2);
    }

    auto sparse_indices = at::arange(selected_token_count, selection_topk_indices.options())
                              .view({1, 1, selected_token_count})
                              .expand({query_token_count, kv_heads, selected_token_count})
                              .contiguous();
    auto actual_seq_lengths_query = at::ones({query_token_count}, selection_topk_indices.options());
    auto actual_seq_lengths_kv = at::full(
        {query_token_count},
        selected_token_count,
        selection_topk_indices.options());

    auto result = npu_sparse_flash_attention(
        query_nope,
        selection_kv_cache.unsqueeze(2),
        selection_kv_cache.unsqueeze(2),
        sparse_indices,
        scale_value,
        c10::optional<at::Tensor>(selection_kv_block_table),
        c10::optional<at::Tensor>(actual_seq_lengths_query),
        c10::optional<at::Tensor>(actual_seq_lengths_kv),
        query_rope,
        key_rope,
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
        std::numeric_limits<int64_t>::max(),
        std::numeric_limits<int64_t>::max(),
        2,
        false,
        false);
    return std::get<0>(result);
}

}  // namespace vllm_ascend
#endif  // FUSED_SPARSE_ATTENTION_OVERLAP_TORCH_ADPT_H
