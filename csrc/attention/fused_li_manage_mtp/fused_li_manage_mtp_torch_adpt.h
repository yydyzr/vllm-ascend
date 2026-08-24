#ifndef FUSED_LI_MANAGE_MTP_TORCH_ADPT_H_
#define FUSED_LI_MANAGE_MTP_TORCH_ADPT_H_

namespace vllm_ascend {

// MTP3 (query_len=4) fused LightningIndexer + top-2048 union + hit/evict +
// request-pool update. The caller pre-allocates all output buffers; the
// operator writes in place and returns void.
//
// Parameters:
//   query                - bf16/fp16 [B*4, 32, 128], read-only, 4-way MTP3 index queries
//   index_weights        - bf16/fp16 [B*4, 32],      read-only, per-head aggregation weights for 4 queries
//   index_key_cache      - bf16/fp16 [INDEX_BLOCKS, 128, 1, 128], read-only, index key cache
//   index_block_table    - int32 [B, INDEX_MAX_BLOCKS], read-only, index cache block table
//   num_candidate_tokens - int32 [B], read-only, prefill full-block token count per request
//   num_cache_tokens     - int32 [B], read-only, HBM cache token budget C per request
//   req_pool_entries     - int32 [B], read-only, request-pool row index per request
//   cache_slots_pool     - int32 [POOL_SIZE, SOURCE_CAPACITY], read-write, persistent source-to-HBM slot mapping
//   topk_src_ids         - int32 [B*4, 1, 2048], write-only, 4-way top-2048 source token IDs (HBM hit positions are -1)
//   topk_dst_slots       - int32 [B*4, 1, 2048], write-only, 4-way top-2048 HBM logical slots
//   miss_src_ids         - int32 [B, 8192], write-only, unique miss source IDs in the 4-way union (first miss_counts valid)
//   miss_dst_slots       - int32 [B, 8192], write-only, unique miss HBM logical slots (first miss_counts valid)
//   miss_counts          - int32 [B], write-only, unique union miss token count per request
inline void npu_fused_li_manage_mtp(
    const at::Tensor& query,
    const at::Tensor& index_weights,
    const at::Tensor& index_key_cache,
    const at::Tensor& index_block_table,
    const at::Tensor& num_candidate_tokens,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    at::Tensor topk_src_ids,
    at::Tensor topk_dst_slots,
    at::Tensor miss_src_ids,
    at::Tensor miss_dst_slots,
    at::Tensor miss_counts) {
  constexpr int64_t kQueryCount = 4;
  constexpr int64_t kMinQueryHeads = 32;
  constexpr int64_t kMaxQueryHeads = 64;
  constexpr int64_t kTopK = 2048;
  constexpr int64_t kUnionCapacity = 8192;
  TORCH_CHECK(query.dim() == 3 &&
                  (query.size(1) == kMinQueryHeads ||
                   query.size(1) == kMaxQueryHeads) &&
                  query.size(2) == 128,
              "MTP LIM query must be [4B, H, 128], H=32 or 64.");
  TORCH_CHECK(query.device().is_privateuseone(),
              "MTP LIM inputs must be on NPU.");
  TORCH_CHECK(req_pool_entries.dim() == 1,
              "MTP LIM req_pool_entries must be [B].");
  const int64_t batch_size = req_pool_entries.size(0);
  TORCH_CHECK(batch_size > 0 && query.size(0) == batch_size * kQueryCount,
              "MTP LIM requires exactly four packed queries per request.");
  TORCH_CHECK(index_key_cache.dim() == 4 && index_key_cache.size(0) > 0 &&
                  index_key_cache.size(1) == 128 &&
                  index_key_cache.size(2) == 1 &&
                  index_key_cache.size(3) == 128,
              "MTP LIM key must be [blocks, 128, 1, 128].");
  TORCH_CHECK(index_weights.dim() == 2 &&
                  index_weights.size(0) == query.size(0) &&
                  index_weights.size(1) == query.size(1),
              "MTP LIM weights must be [4B, H] and match query heads.");
  TORCH_CHECK(cache_slots_pool.dim() == 2 &&
                  cache_slots_pool.size(0) > 0 &&
                  num_cache_tokens.dim() == 1 &&
                  num_candidate_tokens.dim() == 1 &&
                  index_block_table.dim() == 2 &&
                  index_block_table.size(1) > 0,
              "MTP LIM cache metadata has invalid rank or empty capacity.");
  TORCH_CHECK(num_cache_tokens.size(0) == batch_size &&
                  num_candidate_tokens.size(0) == batch_size &&
                  index_block_table.size(0) == batch_size,
              "MTP LIM request metadata batch dimensions must match.");
  TORCH_CHECK(cache_slots_pool.size(1) == index_block_table.size(1) * 128,
              "MTP LIM pool width must match block-table capacity.");
  TORCH_CHECK(index_block_table.size(1) <= (1 << 11),
              "MTP LIM source capacity must be <= 2^18 tokens.");
  TORCH_CHECK(topk_dst_slots.dim() == 3 &&
                  topk_dst_slots.size(0) == query.size(0) &&
                  topk_dst_slots.size(1) == 1 &&
                  topk_dst_slots.size(2) == kTopK,
              "MTP LIM topk_slots must be [4B, 1, 2048].");
  TORCH_CHECK(topk_src_ids.sizes() == topk_dst_slots.sizes(),
              "MTP LIM topk_source_ids must match topk_slots.");
  TORCH_CHECK(miss_src_ids.dim() == 2 &&
                  miss_src_ids.size(0) == batch_size &&
                  miss_src_ids.size(1) == kUnionCapacity &&
                  miss_dst_slots.sizes() == miss_src_ids.sizes(),
              "MTP LIM miss outputs must be [B, 8192].");
  TORCH_CHECK(miss_counts.dim() == 1 && miss_counts.size(0) == batch_size,
              "MTP LIM miss_counts must be [B].");
  TORCH_CHECK(query.scalar_type() == index_key_cache.scalar_type() &&
                  query.scalar_type() == index_weights.scalar_type() &&
                  (query.scalar_type() == at::kHalf ||
                   query.scalar_type() == at::kBFloat16),
              "MTP LIM query/key/weights must share fp16 or bf16 dtype.");
  TORCH_CHECK(req_pool_entries.scalar_type() == at::kInt &&
                  cache_slots_pool.scalar_type() == at::kInt &&
                  num_cache_tokens.scalar_type() == at::kInt &&
                  num_candidate_tokens.scalar_type() == at::kInt &&
                  index_block_table.scalar_type() == at::kInt &&
                  topk_dst_slots.scalar_type() == at::kInt &&
                  topk_src_ids.scalar_type() == at::kInt &&
                  miss_src_ids.scalar_type() == at::kInt &&
                  miss_dst_slots.scalar_type() == at::kInt &&
                  miss_counts.scalar_type() == at::kInt,
              "MTP LIM metadata and outputs must be int32.");
  TORCH_CHECK(query.is_contiguous() && index_key_cache.is_contiguous() &&
                  index_weights.is_contiguous() &&
                  req_pool_entries.is_contiguous() &&
                  cache_slots_pool.is_contiguous() &&
                  num_cache_tokens.is_contiguous() &&
                  num_candidate_tokens.is_contiguous() &&
                  index_block_table.is_contiguous() &&
                  topk_dst_slots.is_contiguous() &&
                  topk_src_ids.is_contiguous() &&
                  miss_src_ids.is_contiguous() &&
                  miss_dst_slots.is_contiguous() && miss_counts.is_contiguous(),
              "All MTP LIM tensors must be contiguous.");
  const auto device = query.device();
  TORCH_CHECK(index_key_cache.device() == device &&
                  index_weights.device() == device &&
                  req_pool_entries.device() == device &&
                  cache_slots_pool.device() == device &&
                  num_cache_tokens.device() == device &&
                  num_candidate_tokens.device() == device &&
                  index_block_table.device() == device &&
                  topk_dst_slots.device() == device &&
                  topk_src_ids.device() == device &&
                  miss_src_ids.device() == device &&
                  miss_dst_slots.device() == device && miss_counts.device() == device,
              "All MTP LIM tensors must be on the same NPU device.");

  auto keepalive = std::make_tuple(
      query, index_key_cache, index_weights, req_pool_entries,
      cache_slots_pool, num_cache_tokens, num_candidate_tokens,
      index_block_table, topk_dst_slots, topk_src_ids, miss_src_ids,
      miss_dst_slots, miss_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnFusedLiManageMtp,
      keepalive,
      query,
      index_key_cache,
      index_weights,
      req_pool_entries,
      cache_slots_pool,
      num_cache_tokens,
      num_candidate_tokens,
      index_block_table,
      topk_dst_slots,
      topk_src_ids,
      miss_src_ids,
      miss_dst_slots,
      miss_counts,
      cache_slots_pool);
}

} // namespace vllm_ascend
#endif
