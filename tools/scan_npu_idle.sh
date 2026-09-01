#!/bin/bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

# Scan a range of hosts (or a saved npu-smi dump) and report idle / busy NPUs.
#
# A5 / newer npu-smi info notes:
# - Device table: two rows per NPU (identity row + chip/util row).
# - Idle process table: one line per chip, e.g.
#   "| No running processes found in NPU 0 |"
# - Busy process table may use a combined "NPU Chip" first column
#   ("0  0") and an extra "Process id in container" column.
# Idle HBM is not occupancy: A5 still reports ~4–5 GB driver-reserved HBM
# when no process is running.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

DEFAULT_USER="root"
DEFAULT_PORT=22
DEFAULT_TIMEOUT=10
DEFAULT_CONCURRENT=10

usage() {
    cat <<EOF
用法: $0 -p <prefix> -s <start> -e <end> [选项]
      $0 --parse-file <npu-smi-info.txt>
      $0 --self-test

  -p, --prefix      主机名前缀
  -s, --start       起始编号
  -e, --end         结束编号
  -u, --user        SSH 用户名 (默认: $DEFAULT_USER)
  -P, --port        SSH 端口 (默认: $DEFAULT_PORT)
  -t, --timeout     SSH 超时秒数 (默认: ${DEFAULT_TIMEOUT}s)
  -c, --concurrent  并发数 (默认: $DEFAULT_CONCURRENT)
  -w, --password    直接指定密码
  -f, --parse-file  解析本地 npu-smi info 输出（不 SSH，用于调试 A5/A2 格式）
      --self-test   运行内置解析回归
  -h, --help        显示帮助
EOF
    exit 1
}

# Portable awk (mawk/gawk) parser for `npu-smi info`.
# Prints: total busy_npu_count proc_count idle_count busy_list
PARSE_NPU_SMI_AWK='
function trim(s) {
    gsub(/^[ \t]+|[ \t]+$/, "", s)
    return s
}
function is_device_name(s) {
    s = trim(s)
    if (s == "") return 0
    if (s ~ /:/) return 0
    if (s ~ /^(OK|NA|Warning|Health|Bus-Id|Name)$/) return 0
    if (s ~ /[A-Za-z]/) return 1
    return 0
}
function remember_npu(id) {
    if (id == "" || id !~ /^[0-9]+$/) return
    if (!(id in all_npus)) {
        all_npus[id] = 1
        total++
    }
}
BEGIN {
    busy_count = 0
    proc_count = 0
    idle_count = 0
    total = 0
    in_proc = 0
    all_idle = 0
}
{
    sub(/\r$/, "", $0)
}
/^\+/ { next }
/^[[:space:]]*$/ { next }
/Process[ \t]+id/ { in_proc = 1; next }

!in_proc {
    id_field = trim($2)
    name_field = trim($3)
    if (id_field ~ /^[0-9]+$/ && is_device_name(name_field)) {
        remember_npu(id_field)
        next
    }
    if (id_field ~ /^[0-9]+[ \t]+[A-Za-z0-9]/) {
        split(id_field, parts, /[ \t]+/)
        remember_npu(parts[1])
        next
    }
    next
}

in_proc {
    if ($0 ~ /No running processes found in NPU/) {
        if (match($0, /NPU[ \t]+[0-9]+/)) {
            idle_id = substr($0, RSTART, RLENGTH)
            gsub(/[^0-9]/, "", idle_id)
            if (idle_id ~ /^[0-9]+$/ && !(idle_id in idle_npus)) {
                idle_npus[idle_id] = 1
                idle_count++
            }
            remember_npu(idle_id)
        }
        next
    }
    if ($0 ~ /No running processes found/) {
        all_idle = 1
        next
    }

    npu_field = trim($2)
    pid_field = trim($3)
    container_pid = trim($6)
    npu_id = npu_field
    if (npu_field ~ /^[0-9]+[ \t]+[0-9]+/) {
        split(npu_field, np, /[ \t]+/)
        npu_id = np[1]
    } else {
        gsub(/[ \t].*$/, "", npu_id)
    }

    if (pid_field ~ /^[0-9]+$/ || container_pid ~ /^[0-9]+$/) {
        if (npu_id ~ /^[0-9]+$/) {
            proc_count++
            if (!(npu_id in busy_npus)) {
                busy_npus[npu_id] = 1
                busy_count++
                busy_list = busy_list "NPU" npu_id " "
            }
            remember_npu(npu_id)
        }
    }
}

END {
    if (all_idle && busy_count == 0 && total > 0) {
        idle_count = total
    } else if (total > 0 && idle_count == 0 && proc_count == 0) {
        idle_count = total
    }
    print total + 0, busy_count + 0, proc_count + 0, idle_count + 0, busy_list
}
'

parse_npu_smi_info() {
    awk -F'|' "$PARSE_NPU_SMI_AWK"
}

sort_busy_npus() {
    local raw="$1"
    if [[ -z "${raw// /}" ]]; then
        echo ""
        return
    fi
    printf '%s\n' $raw | sort -V | tr '\n' ' '
}

format_host_status() {
    local label="$1"
    local parse_result="$2"

    local total busy_npu_count proc_count idle_count busy_npus
    total=$(printf '%s\n' "$parse_result" | awk '{print $1}')
    busy_npu_count=$(printf '%s\n' "$parse_result" | awk '{print $2}')
    proc_count=$(printf '%s\n' "$parse_result" | awk '{print $3}')
    idle_count=$(printf '%s\n' "$parse_result" | awk '{print $4}')
    busy_npus=$(printf '%s\n' "$parse_result" | cut -d' ' -f5-)
    busy_npus=$(sort_busy_npus "$busy_npus")

    [[ -z "$total" ]] && total=0
    [[ -z "$busy_npu_count" ]] && busy_npu_count=0
    [[ -z "$proc_count" ]] && proc_count=0
    [[ -z "$idle_count" ]] && idle_count=0

    local total_display="$total"
    [[ "$total" -eq 0 ]] && total_display="?"

    if [[ "$busy_npu_count" -gt 0 || "$proc_count" -gt 0 ]]; then
        printf "${YELLOW}[BUSY]${NC} %-20s %s\n" "$label" \
            "NPU 被占用 (${busy_npu_count} 张卡/${proc_count} 个进程, 共 ${total_display} 个 NPU) ${busy_npus}"
        return 0
    fi

    printf "${GREEN}[OK]${NC}   %-20s %s\n" "$label" \
        "NPU 空闲 (共 ${total_display} 个 NPU，无进程)"
    return 0
}

check_host() {
    local host="$1"
    local ip="${PREFIX}.${host}"

    if ! timeout "$TIMEOUT" bash -c ">/dev/tcp/${ip}/${PORT}" 2>/dev/null; then
        printf "${RED}[FAIL]${NC}  %-20s %s\n" "$ip" "端口不可达"
        return
    fi

    local npu_output
    npu_output=$(sshpass -p "$PASSWORD" ssh -q -o StrictHostKeyChecking=no \
        -o ConnectTimeout="$TIMEOUT" -o BatchMode=no -p "$PORT" "${USER}@${ip}" \
        "npu-smi info 2>/dev/null || echo 'NPU_CMD_NOT_FOUND'" 2>/dev/null) || {
        printf "${RED}[FAIL]${NC}  %-20s %s\n" "$ip" "SSH 失败（密码错误或连接中断）"
        return
    }

    if [[ "$npu_output" == "NPU_CMD_NOT_FOUND" ]] || [[ -z "$npu_output" ]]; then
        printf "${YELLOW}[WARN]${NC}  %-20s %s\n" "$ip" "未找到 npu-smi 命令"
        return
    fi

    local awk_result
    awk_result=$(printf '%s\n' "$npu_output" | parse_npu_smi_info)
    format_host_status "$ip" "$awk_result" || true
}

a5_idle_sample() {
    cat <<'EOF'
+--------+------------------+---------------+----------------------------------------------------------------------+
| NPU ID | Name             | Health        | Power(W)              Temp(C)                  Hugepages-Usage(page) |
|        |                  | Bus-Id        | NPU Util(%)           Memory-Usage(MB)         HBM-Usage(MB)         |
+========+==================+===============+======================================================================+
| 0      | Ascend950DT      | OK            | 377.2                 48                       0     / 0             |
|        |                  | NA            | 0                     0     / 0                4857  / 98304         |
+===========================+===============+======================================================================+
| 1      | Ascend950DT      | OK            | 380.6                 50                       0     / 0             |
|        |                  | NA            | 0                     0     / 0                4855  / 98304         |
+===========================+===============+======================================================================+
| 2      | Ascend950DT      | OK            | 373.1                 49                       0     / 0             |
|        |                  | NA            | 0                     0     / 0                4855  / 98304         |
+===========================+===============+======================================================================+
| 3      | Ascend950DT      | OK            | 395.1                 48                       0     / 0             |
|        |                  | NA            | 0                     0     / 0                4857  / 98304         |
+===========================+===============+======================================================================+
| 4      | Ascend950DT      | OK            | 366.6                 49                       0     / 0             |
|        |                  | NA            | 0                     0     / 0                4856  / 98304         |
+===========================+===============+======================================================================+
| 5      | Ascend950DT      | OK            | 370.7                 49                       0     / 0             |
|        |                  | NA            | 0                     0     / 0                4855  / 98304         |
+===========================+===============+======================================================================+
| 6      | Ascend950DT      | OK            | 373.3                 49                       0     / 0             |
|        |                  | NA            | 0                     0     / 0                4856  / 98304         |
+===========================+===============+======================================================================+
| 7      | Ascend950DT      | OK            | 372.8                 49                       0     / 0             |
|        |                  | NA            | 0                     0     / 0                4856  / 98304         |
+===========================+===============+======================================================================+
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU ID                    | Process id    | Process name       | Process memory(MB)    | Process id in container |
+===========================+===============+======================================================================+
| No running processes found in NPU 0                                                                              |
+===========================+===============+======================================================================+
| No running processes found in NPU 1                                                                              |
+===========================+===============+======================================================================+
| No running processes found in NPU 2                                                                              |
+===========================+===============+======================================================================+
| No running processes found in NPU 3                                                                              |
+===========================+===============+======================================================================+
| No running processes found in NPU 4                                                                              |
+===========================+===============+======================================================================+
| No running processes found in NPU 5                                                                              |
+===========================+===============+======================================================================+
| No running processes found in NPU 6                                                                              |
+===========================+===============+======================================================================+
| No running processes found in NPU 7                                                                              |
+===========================+===============+======================================================================+
EOF
}

a5_partial_busy_sample() {
    cat <<'EOF'
+--------+------------------+---------------+----------------------------------------------------------------------+
| NPU ID | Name             | Health        | Power(W)              Temp(C)                  Hugepages-Usage(page) |
|        |                  | Bus-Id        | NPU Util(%)           Memory-Usage(MB)         HBM-Usage(MB)         |
+========+==================+===============+======================================================================+
| 0      | Ascend950DT      | OK            | 377.2                 48                       0     / 0             |
|        |                  | NA            | 0                     0     / 0                4857  / 98304         |
+===========================+===============+======================================================================+
| 1      | Ascend950DT      | OK            | 380.6                 50                       0     / 0             |
|        |                  | NA            | 0                     0     / 0                4855  / 98304         |
+===========================+===============+======================================================================+
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU ID                    | Process id    | Process name       | Process memory(MB)    | Process id in container |
+===========================+===============+======================================================================+
| 0                         | 12345         | python             | 2048                  | 12345                   |
+===========================+===============+======================================================================+
| No running processes found in NPU 1                                                                              |
+===========================+===============+======================================================================+
EOF
}

legacy_idle_sample() {
    cat <<'EOF'
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B3               | OK            | 92.9        37                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          3379 / 65536         |
+===========================+===============+====================================================+
| 1     910B3               | OK            | 90.1        36                0    / 0             |
| 0                         | 0000:C2:00.0  | 0           0    / 0          3380 / 65536         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found                                                                  |
+===========================+===============+====================================================+
EOF
}

legacy_busy_sample() {
    cat <<'EOF'
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B4               | OK            | 88.2        30                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          26838/ 32768         |
+===========================+===============+====================================================+
| 1     910B4               | OK            | 87.0        31                0    / 0             |
| 0                         | 0000:C2:00.0  | 0           0    / 0          2000 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| 0       0                 | 1406630       | mindie_llm_back          | 21835                   |
| 0       0                 | 60575         | python                   | 2122                    |
+===========================+===============+====================================================+
| No running processes found in NPU 1                                                            |
+===========================+===============+====================================================+
EOF
}

assert_parse() {
    local name="$1"
    local expected="$2"
    local actual
    actual=$(cat | parse_npu_smi_info)
    local actual_core expected_core
    actual_core=$(printf '%s\n' "$actual" | awk '{print $1,$2,$3,$4}')
    expected_core=$(printf '%s\n' "$expected" | awk '{print $1,$2,$3,$4}')
    if [[ "$actual_core" != "$expected_core" ]]; then
        echo -e "${RED}[SELF-TEST FAIL]${NC} $name"
        echo "  expected: $expected"
        echo "  actual:   $actual"
        return 1
    fi
    echo -e "${GREEN}[SELF-TEST OK]${NC}   $name -> $actual"
}

run_self_test() {
    local failed=0
    a5_idle_sample | assert_parse "A5 all idle" "8 0 0 8" || failed=1
    a5_partial_busy_sample | assert_parse "A5 partial busy" "2 1 1 1" || failed=1
    legacy_idle_sample | assert_parse "legacy global idle" "2 0 0 2" || failed=1
    legacy_busy_sample | assert_parse "910B combined NPU Chip" "2 1 2 1" || failed=1
    if [[ "$failed" -ne 0 ]]; then
        echo -e "${RED}self-test failed${NC}"
        exit 1
    fi
    echo "self-test passed"
}

PREFIX=""
START=""
END=""
USER="$DEFAULT_USER"
PORT="$DEFAULT_PORT"
TIMEOUT="$DEFAULT_TIMEOUT"
CONCURRENT="$DEFAULT_CONCURRENT"
PASSWORD=""
PARSE_FILE=""
SELF_TEST=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--prefix) PREFIX="$2"; shift 2 ;;
        -s|--start) START="$2"; shift 2 ;;
        -e|--end) END="$2"; shift 2 ;;
        -u|--user) USER="$2"; shift 2 ;;
        -P|--port) PORT="$2"; shift 2 ;;
        -t|--timeout) TIMEOUT="$2"; shift 2 ;;
        -c|--concurrent) CONCURRENT="$2"; shift 2 ;;
        -w|--password) PASSWORD="$2"; shift 2 ;;
        -f|--parse-file) PARSE_FILE="$2"; shift 2 ;;
        --self-test) SELF_TEST=1; shift ;;
        -h|--help) usage ;;
        *) echo "未知参数: $1"; usage ;;
    esac
done

if [[ "$SELF_TEST" -eq 1 ]]; then
    run_self_test
    exit 0
fi

if [[ -n "$PARSE_FILE" ]]; then
    [[ -f "$PARSE_FILE" ]] || { echo -e "${RED}错误: 文件不存在: $PARSE_FILE${NC}"; exit 1; }
    awk_result=$(parse_npu_smi_info < "$PARSE_FILE")
    format_host_status "$PARSE_FILE" "$awk_result"
    exit 0
fi

[[ -z "$PREFIX" || -z "$START" || -z "$END" ]] && { echo -e "${RED}错误: 必须指定 prefix、start、end${NC}"; usage; }
! [[ "$START" =~ ^[0-9]+$ && "$END" =~ ^[0-9]+$ ]] && { echo -e "${RED}错误: start 和 end 必须是数字${NC}"; exit 1; }
[[ "$START" -gt "$END" ]] && { echo -e "${RED}错误: start 不能大于 end${NC}"; exit 1; }

if [[ -z "$PASSWORD" ]]; then
    echo -n "请输入 SSH 密码: "
    read -s PASSWORD
    echo ""
fi

! command -v sshpass &>/dev/null && { echo -e "${RED}错误: 未找到 sshpass${NC}"; exit 1; }

export -f check_host parse_npu_smi_info format_host_status sort_busy_npus
export PARSE_NPU_SMI_AWK PREFIX PORT USER PASSWORD TIMEOUT RED GREEN YELLOW CYAN NC

echo "=========================================="
echo "  扫描范围: ${PREFIX}.${START} ~ ${PREFIX}.${END}"
echo "  用户名:   $USER | 端口: $PORT | 并发: $CONCURRENT"
echo "=========================================="
echo ""

seq "$START" "$END" | xargs -P "$CONCURRENT" -I{} bash -c 'check_host "$@"' _ {}

echo ""
echo "扫描完成"
