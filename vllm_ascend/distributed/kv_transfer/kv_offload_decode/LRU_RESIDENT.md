# KV Offload Decode：LRU Resident 变量与更新逻辑

本文对应实现：

- C++：`kv_offload_decode.cpp`
  - `process_one_lru_resident_row`
  - `lru_resident_compact`
  - `compute_lru_resident_addrs`
- Python：`kv_offload_decode_manager.py`
  - buffer 分配
  - `onload_topk_kv` / `_onload_topk_kv_cpu`

目的：维护 NPU 上 **topk resident buffer** 中，每个 decode 行当前缓存了哪些绝对 token，以及它们的冷热顺序；对本轮 topk 做 hit/miss 判定，仅把 miss 从 CPU KV pool H2D 到 resident。

---

## 1. 角色关系

```text
CPU KV pool (full main K/V)
        │  miss 时 H2D
        ▼
NPU topk resident buffer   ←── LRU 维护的对象
        │
        ▼
sparse attention 用 current_slots 索引读取
```

注意：这和 fused overlap 算子内部的 `selection_kv_block_status` 不是同一套判定。

| | LRU (`onload_topk_kv`) | fused `selection_kv_block_status` |
|--|--|--|
| 键 | 绝对 token 是否已在 resident map | topk 槽位内容是否等于本次 topk token |
| 容量 | `topk_buffer_size`（可 > topk） | 通常按 topk 槽位维护 |
| 驱逐 | 明确冷热顺序，挤掉最冷槽 | 同位覆盖式更新 |

---

## 2. 变量总表

### 2.1 跨 step 持久状态（真正的 LRU 状态）

按 **layer** 各有一份。行数 = `max_num_topk_rows`（decode token 行容量）。

| 变量（Python / C++） | shape | dtype | 含义 |
|------|-------|-------|------|
| `lru_slot_to_token_cpu_list[layer]` / `slot_to_token` | `[rows, capacity]` | int32 | resident slot → 当前缓存的绝对 token id；空槽为 `-1` |
| `lru_slots_cpu_list[layer]` / `lru_slots` | `[rows, capacity]` | int32 | 该行所有 slot 的 LRU 顺序；**靠前更冷/更可驱逐，靠后更热** |
| `lru_last_req_ids_cpu_list[layer]` / `last_req_ids` | `[rows]` | int64 | 该行上次绑定的 request id；变化则整行 reset |

其中：

```text
capacity = topk_buffer_size   # 配置项，默认 4096
rows     = max_num_topk_rows  # 约 min(max_num_batched_tokens, max_num_seqs * decode_width)
```

初始化：

```text
slot_to_token[row, :] = -1
lru_slots[row, :]     = [0, 1, 2, ..., capacity-1]
last_req_ids[row]     = -1
```

### 2.2 每 step 输入

| 变量 | shape | 含义 |
|------|-------|------|
| `req_ids` / `lru_req_ids_cpu` | `[rows]` | 当前行对应的 request id |
| `topk_indices` / `lru_topk_indices_cpu` | `[rows, topk]` | 本轮 indexer 选出的绝对 token id；无效为 `-1` |
| `stable_prefix_lens` / `lru_stable_prefix_lens_cpu` | `[rows]` | 稳定前缀长度。`token >= prefix` 的 resident 内容可能被 MTP/spec 改写，强制失效 |
| `token_to_req`（可选） | `[rows]` | speculative decode 时，把 request 级 block_table 展开到 token 行 |
| `block_table` | `[rows 或 reqs, max_blocks]` | 用于把绝对 token 定位到 CPU pool 物理 block |

### 2.3 每 step 输出

| 变量 | shape | 含义 |
|------|-------|------|
| `lru_current_slots_cpu` / `current_slots` | `[rows, topk]` | 本轮 topk 第 `pos` 个 token 落在哪个 resident slot；未命中且未分配前为 `-1` |
| `lru_miss_count_cpu_list[layer]` / `miss_count` | `[rows]` | 本行需要 H2D 的 miss 数量 |
| `lru_miss_tokens_cpu_list[layer]` / `miss_tokens` | `[rows, topk]` | 第 i 个 miss 的绝对 token id |
| `lru_miss_slots_cpu_list[layer]` / `miss_slots` | `[rows, topk]` | 第 i 个 miss 分配到的 resident slot |

attention 侧最终用 `current_slots`（拷到 NPU 后）去索引 `topk_buffers_k/v`。

### 2.4 临时 workspace（不算长期状态）

| 变量 | shape | 作用 |
|------|-------|------|
| `lru_token_mark_workspace` / `token_mark` | `[threads, max_model_len]` | 配合 epoch，标记“本轮 topk 集合” |
| `lru_token_pos_workspace` / `token_pos` | `[threads, max_model_len]` | 绝对 token → 本轮 topk 中的 pos |
| `lru_epochs` / `epoch` | `[threads]` | 本轮标记世代；过大时重置 mark/pos |
| `lru_slot_workspace` | `[threads, capacity*3]` | 临时拆成 `hit_slots / evictable_slots / assigned_miss_slots` |
| `lru_miss_position_workspace` / `miss_positions` | `[threads, topk]` | miss token 对应的 topk pos |

`epoch` 重置阈值：`EPOCH_RESET_THRESHOLD = 1 << 30`。

---

## 3. 一轮更新判定流程

入口：`lru_resident_compact` → 对每个 row 调 `process_one_lru_resident_row`。

对每一行 `row`：

### Step 1：清空本轮输出

```text
current_slots[row, :] = -1
miss_tokens[row, :]   = -1
miss_slots[row, :]    = -1
miss_count[row]       = 0
```

### Step 2：request 变化则整行 reset

```text
if req_ids[row] != last_req_ids[row]:
    slot_to_token[row, :] = -1
    lru_slots[row, :]     = [0..capacity-1]
    last_req_ids[row]     = req_ids[row]
```

### Step 3：用 epoch 标记本轮需要的 token 集合

```text
base = ++epoch   # 必要时 reset mark/pos

for pos in [0, topk):
    token = topk_indices[row, pos]
    if 0 <= token < max_token and token_mark[token] != base:
        token_mark[token] = base
        token_pos[token]  = pos
```

有效 token 判定：

```text
is_valid(token) := (token >= 0 && token < max_token)
```

### Step 4：扫描现有 resident，分出 hit / 可驱逐

按 `lru_slots` 从冷到热扫描：

```text
for order in [0, capacity):
    slot  = lru_slots[row, order]
    token = slot_to_token[row, slot]

    # 投机后缀保护：token 落在 unstable 区间，作废
    if is_valid(token) and token >= stable_prefix_lens[row]:
        slot_to_token[row, slot] = -1
        token = -1

    if is_valid(token) and token_mark[token] == base:
        # HIT：resident 里已有本轮 topk 需要的 token
        pos = token_pos[token]
        current_slots[row, pos] = slot
        hit_slots.append(slot)
    else:
        # 可驱逐：空槽 / 不在本轮 topk / 已被作废
        evictable_slots.append(slot)
```

### Step 5：找出 miss

```text
for pos in [0, topk):
    token = topk_indices[row, pos]
    if is_valid(token) and current_slots[row, pos] < 0:
        # 本轮需要，但现有 resident 没命中
        miss_tokens.append(token)
        miss_positions.append(pos)
```

### Step 6：给 miss 分配最冷可驱逐槽

```text
assign_count = min(miss_count, evictable_count)

for i in [0, assign_count):
    slot  = evictable_slots[i]     # 最冷优先
    token = miss_tokens[i]
    pos   = miss_positions[i]

    slot_to_token[row, slot] = token
    current_slots[row, pos]  = slot
    miss_slots[i]            = slot
    miss_count[row]          = i + 1
```

### Step 7：重建 LRU 顺序

```text
lru_slots 新顺序 =
    [未使用的可驱逐槽]          # 最冷
  + [刚分配给 miss 的槽]        # 次冷 / 新写入
  + [本轮 hit 的槽]             # 最热
```

对应代码写入顺序：

1. `evictable_slots[assign_count:]`
2. `assigned_miss_slots[0:assign_count]`
3. `hit_slots[0:hit_count]`

---

## 4. Hit / Miss 定义

```text
HIT :
  某 resident slot 已缓存绝对 token T
  且 T 出现在本轮 topk
  且 T < stable_prefix_len

MISS :
  本轮 topk 需要绝对 token T
  但扫完所有 resident 后 current_slots[pos] 仍为 -1
```

不是“和上一次 topk 位置是否相同”，而是：

**这个绝对 token 现在是否已经住在该行的 resident buffer 里。**

---

## 5. LRU 顺序如何表达“最近使用”

不是显式时间戳链表，而是每轮重排 `lru_slots`：

```text
冷 ---------------------------------------------------- 热
未命中且未复用的旧槽 | 本轮新装入的 miss 槽 | 本轮继续命中的槽
```

效果：

- 连续命中的 token 逐渐靠后（更热）
- 长期不在 topk 的 slot 沉到前面，优先被 miss 挤掉
- 新 miss 写入后不会立刻变成最热，排在 hit 前面

---

## 6. miss 之后的 H2D

`compute_lru_resident_addrs` 只处理 miss：

```text
for each miss (token, slot):
  block_id        = token / block_size
  offset_in_block = token % block_size
  block_indice    = block_table[row, block_id]

  gva_k  = cpu_k_base + block_indice * block_bytes_k + offset * token_bytes_k
  gva_v  = cpu_v_base + ...
  addr_k = npu_topk_k_base + (row * capacity + slot) * token_bytes_k
  addr_v = npu_topk_v_base + ...

  写入 sparse_copy 描述符
```

随后：

```text
sparse_copy: CPU pool -> NPU resident[slot]
attention 用 current_slots 读 resident
```

hit 的 token **不拷贝**，直接复用已有 resident 内容。

Python 调用链：

```text
onload_topk_kv
  -> 拷贝 req_ids / topk / stable_prefix / block_table 到 CPU pin memory
  -> _onload_topk_kv_cpu
       -> lru_resident_compact
       -> compute_lru_resident_addrs
  -> sparse_copy (H2D)
  -> current_slots_cpu -> current_slots_npu
```

---

## 7. 小例子

假设 `capacity=4`, `topk=3`：

```text
旧状态:
  slot_to_token = [10, 20, 30, 40]
  lru_slots     = [0, 1, 2, 3]      # 0 最冷, 3 最热

本轮:
  topk = [20, 50, 40]
  stable_prefix_len 足够大
```

判定：

```text
slot1 (token20): HIT -> current_slots[0] = 1
slot3 (token40): HIT -> current_slots[2] = 3
slot0 (token10): 不在 topk -> 可驱逐
slot2 (token30): 不在 topk -> 可驱逐

miss: token50 (pos=1)
分配最冷可驱逐槽 slot0
  slot_to_token[0] = 50
  current_slots[1] = 0
  miss_tokens = [50]
  miss_slots  = [0]
```

更新后：

```text
slot_to_token = [50, 20, 30, 40]
lru_slots     = [2, 0, 1, 3]
                 ^  ^  ^  ^
                 |  |  |  本轮 hit(40)
                 |  |  本轮 hit(20)
                 |  本轮新 miss(50)
                 未使用旧槽(30)，最冷
```

随后仅对 `(token=50, slot=0)` 做 H2D。

---

## 8. 关键边界条件

1. **request 切换**：`req_id` 变化整行 reset，避免串请求污染。
2. **`stable_prefix_len`**：MTP/spec 可能改写后缀 token；resident 中 `token >= prefix` 直接作废，避免读脏。
3. **无效 topk**：`token < 0` 或越界，不参与 hit/miss。
4. **capacity < topk 理论上不够用**：实现按 `min(miss, evictable)` 分配；正常配置要求 `topk_buffer_size >= topk`。
5. **graph mode**：`lru_resident_compact` / `compute_lru_resident_addrs` 通过 host func 在 capture/replay 时于 CPU 侧执行；真正 H2D `sparse_copy` 进 NPU stream。

---

## 9. 快速对照代码位置

| 逻辑 | 位置 |
|------|------|
| 持久 buffer 分配 | `kv_offload_decode_manager.py` 中 `lru_*` 初始化 |
| 每行 hit/miss/驱逐/重排 | `kv_offload_decode.cpp::process_one_lru_resident_row` |
| 并行入口 | `kv_offload_decode.cpp::lru_resident_compact` |
| miss → 地址描述符 | `kv_offload_decode.cpp::compute_lru_resident_addrs` |
| 调度与 H2D | `kv_offload_decode_manager.py::onload_topk_kv` |

---

## 10. 一句话总结

LRU 长期维护每层每行的 `slot_to_token`、`lru_slots`、`last_req_ids`。  
每轮用“本轮 topk 集合 + stable prefix”判定 hit/miss：hit 复用 resident；miss 占用最冷槽并记录 `(miss_token, miss_slot)`，再 H2D；最后把顺序重排为「未用冷槽 → 新 miss → hit」。
