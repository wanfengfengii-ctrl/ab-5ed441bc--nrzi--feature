"""帧模型、CRC-8 校验与联合插入/漏失复原求解器。

信道模型
--------
原始发送序列为若干等长帧 ``同步字 | 载荷 | CRC8`` 的拼接。接收序列相对
校正（发送）序列只可能偶发地：

* 插入一个比特（接收流中多出一个噪声比特）；
* 漏失一个比特（发送的某一位未被收到）。

二者各计一次滑移。复原在整条接收比特串上联合解释所有插入与漏失，先最小化
滑移次数，再在同代价解中取校正串字典序最小者，并判定最优解是否唯一。

直接模式（默认）：收发比特串即逻辑比特串。

NRZI 线电平模式（``line_code="nrzi"``）：收发串均为**物理电平**，逻辑 1
翻转电平、逻辑 0 保持电平，编码状态**跨帧连续**，初始电平（首个发送位之前
的电平）可为 0、1 或 unknown（unknown 时两个候选联合裁决）。逻辑帧约束
（同步字、CRC）施加在由物理串与初始电平解出的逻辑流上；最优解仍按滑移
次数与**物理校正串**字典序裁决，绝不逐帧重置电平状态。

算法
----
在整条接收流上做分层动态规划。直接模式帧内状态为
``(接收游标 i, 已用滑移 cost, CRC 寄存器 reg)``；NRZI 模式额外携带连续
电平 ``lev``（插入不改变连续电平，故插入闭包无需额外参数）。帧内发送
偏移 j 按层推进。转移只允许：

* match     收发各消耗一比特且物理/逻辑位必须相等（同步字区域该发送位
             还须等于同步字位，NRZI 下由同步字逻辑位与当前电平反推物理
             电平并推进连续电平）；
* deletion  发送消耗一比特、接收不动（漏失），该位取 0/1 两个候选，代价 +1；
* insertion 接收消耗一比特、发送不动（插入），代价 +1，校正串不变，NRZI
             下连续电平不变。

到帧末时再移入 8 个零，余数为零则帧校验通过，状态折叠到下一帧帧首；NRZI
模式折叠键包含连续电平，电平状态随帧边界延续。

每个状态只保留一条字典序最小的校正前缀，并以标志位记录到达该状态的
**不同校正串**是 1 个还是多个（不同脚本可能产生同一串，合并时显式去重）。
帧边界同一接收位置只保留最小代价：此后未来可行集只取决于该位置与剩余
预算，高代价路径不可能进入全局最优解。滑移预算 <= 6，同一层接收游标满足
``|i-j| <= cost``，状态空间有界。

CRC-8：多项式 x^8+x^2+x+1（0x07），初值 0，最高位优先。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

# 业务约束
FRAME_COUNT_MIN = 3
FRAME_COUNT_MAX = 8
SYNC_MIN_LEN = 6
SYNC_MAX_LEN = 12
PAYLOAD_MIN_LEN = 16
PAYLOAD_MAX_LEN = 48
SLIPPAGE_MAX_LIMIT = 6
CRC_LEN = 8
CRC_POLY = 0x07  # x^8 + x^2 + x + 1，省略最高项 x^8

# 不同校正串数量的截断表示：1 = 唯一，2 = 多个（唯一性判定只需布尔）。
MULTIPLE = 2


def crc8(bits: str) -> int:
    """对 0/1 比特串计算 CRC-8（MSB 优先，初值零）。

    多项式 x^8+x^2+x+1（0x07）。8 位移位寄存器处理消息位的结果，代数上
    正是"消息多项式补八个零后对生成多项式取余"（CRC-8/SMBus，目录校验
    值 "123456789" -> 0xF4）。
    """
    crc = 0
    for bit in bits:
        crc ^= (ord(bit) - ord("0")) << 7
        if crc & 0x80:
            crc = ((crc << 1) ^ CRC_POLY) & 0xFF
        else:
            crc = (crc << 1) & 0xFF
    return crc


def _crc_step(reg: int, bit: int) -> int:
    v = reg ^ (bit << 7)
    if v & 0x80:
        return ((v << 1) ^ CRC_POLY) & 0xFF
    return (v << 1) & 0xFF


def frame_is_valid(frame: str, sync: str, payload_len: int) -> bool:
    """校验一帧：长度/同步字匹配且 CRC 正确。"""
    if len(frame) != len(sync) + payload_len + CRC_LEN:
        return False
    if not frame.startswith(sync):
        return False
    body = frame[:-CRC_LEN]
    # crc8(body) 即 body 补八个零后对生成多项式的余数
    return crc8(body) == int(frame[-CRC_LEN:], 2)


@dataclass(frozen=True)
class FrameResult:
    index: int
    payload: str
    crc: str
    raw: str

    def to_dict(self) -> dict:
        return {"index": self.index, "payload": self.payload,
                "crc": self.crc, "raw": self.raw}


@dataclass(frozen=True)
class SlipEvent:
    kind: str  # "insertion" | "deletion"
    position: int  # 基于校正串（发送侧）的 0 计位
    frame_index: int | None
    offset: int | None
    bit: str | None
    detail: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "position": self.position,
            "frame_index": self.frame_index,
            "offset": self.offset,
            "bit": self.bit,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ReconstructionResult:
    recoverable: bool
    corrected: str | None
    frames: tuple[FrameResult, ...]
    slippage_count: int
    events: tuple[SlipEvent, ...]
    unique: bool
    alternatives: int  # 0 或 >=1（不同同代价校正串数量的截断标志）
    minimum_slippage_lower_bound: int | None
    budget: int
    line_code: str = "direct"
    # NRZI 模式下最优解推定的首个发送位之前电平："0"/"1"/"unknown"；
    # 直接模式为 None。
    initial_level: str | None = None

    def to_dict(self) -> dict:
        if self.recoverable:
            out = {
                "recoverable": True,
                "corrected": self.corrected,
                "frames": [f.to_dict() for f in self.frames],
                "slippage_count": self.slippage_count,
                "events": [e.to_dict() for e in self.events],
                "unique": self.unique,
                "alternatives": self.alternatives,
                "budget": self.budget,
            }
            if self.line_code == "nrzi":
                out["initial_level"] = self.initial_level
            return out
        return {
            "recoverable": False,
            "reason": "滑移预算内不存在通过同步字与 CRC 校验的完整帧流",
            "minimum_slippage_lower_bound": self.minimum_slippage_lower_bound,
            "verified_up_to": self.budget,
            "budget": self.budget,
        }


def nrzi_decode(physical: str, initial_level: int) -> str:
    """物理电平串 + 首个发送位之前的电平 -> 逻辑比特串。

    NRZI 约定：逻辑 1 翻转电平、逻辑 0 保持电平，故逻辑位等于相邻电平
    （首位与初始电平）之差。
    """
    out: list[str] = []
    lev = initial_level
    for ch in physical:
        b = ord(ch) - ord("0")
        out.append("1" if b != lev else "0")
        lev = b
    return "".join(out)


def reconstruct(received: str, frame_count: int, sync: str, payload_len: int,
                max_slippage: int, line_code: str = "direct",
                initial_level: str | None = None) -> ReconstructionResult:
    """在整条接收流上联合复原 ``frame_count`` 个等长帧。

    滑移预算从 0 逐档放宽：第一档存在完整帧流时，该档代价即为全局最小
    滑移次数（无滑移档只有唯一的匹配路径，求解极快）。

    ``line_code="nrzi"`` 时 ``received`` 为物理电平串，``initial_level`` 为
    "0"/"1"/"unknown"；unknown 时两个候选初始电平同时进入 DP 联合裁决，
    电平状态跨所有帧连续延续。返回的 ``corrected`` 与事件 bit 均为物理
    电平，帧载荷/CRC 为解出的逻辑位。
    """
    sync_len = len(sync)
    frame_len = sync_len + payload_len + CRC_LEN
    total_len = frame_count * frame_len
    n = len(received)
    delta = n - total_len  # 全局 插入数 - 漏失数
    rb = [ord(c) - ord("0") for c in received]
    sb = [ord(c) - ord("0") for c in sync]

    if line_code == "nrzi":
        nrzi = True
        # unknown：两个候选初始电平并行求解、联合裁决
        init_levels = ((int(initial_level),) if initial_level in ("0", "1")
                       else (0, 1))
    else:
        nrzi = False
        init_levels = (-1,)  # 哨兵：直接模式不跟踪电平

    for budget in range(0, max_slippage + 1):
        answer = _run_budget(rb, sb, frame_count, frame_len, n, delta,
                             budget, nrzi, init_levels)
        if answer is not None:
            best_cost, finals = answer
            tied = [(rep, count, init)
                    for key, (c, rep, count, init) in finals.items()
                    if key[0] == n and c == best_cost]
            # 按物理校正串去重（不同初始电平给出同一串时只算一个解）
            best: dict[str, tuple[int, int]] = {}
            for rep, count, init in tied:
                old = best.get(rep)
                if old is None:
                    best[rep] = (count, init)
                else:
                    # 同一物理串对应多个可行初始电平时确定地取较小者
                    best[rep] = (max(old[0], count), min(old[1], init))
            corrected = min(best)
            total_count, initv = best[corrected]
            unique = len(best) == 1 and total_count == 1
            if nrzi:
                logical = nrzi_decode(corrected, initv)
                init_label: str | None = str(initv)
            else:
                logical = corrected
                init_label = None
            frames, events = _split_frames(
                logical, received, corrected, sync, payload_len,
                frame_len, nrzi)
            return ReconstructionResult(
                recoverable=True, corrected=corrected, frames=tuple(frames),
                slippage_count=best_cost, events=tuple(events),
                unique=unique, alternatives=0 if unique else 1,
                minimum_slippage_lower_bound=None, budget=max_slippage,
                line_code="nrzi" if nrzi else "direct",
                initial_level=init_label,
            )

    lower_bound = max(max_slippage + 1, abs(delta))
    return ReconstructionResult(
        recoverable=False, corrected=None, frames=(),
        slippage_count=0, events=(), unique=False, alternatives=0,
        minimum_slippage_lower_bound=lower_bound, budget=max_slippage,
        line_code="nrzi" if nrzi else "direct",
        initial_level=None,
    )


def _run_budget(rb, sb, frame_count, frame_len, n, delta, budget,
                nrzi, init_levels):
    """在固定滑移预算下做分层帧 DP；成功返回 (best_cost, finals)。

    状态键统一为 ``(i, cost, reg, lev)``：直接模式 ``lev`` 恒为哨兵 -1；
    NRZI 模式 ``lev`` 为当前连续物理电平，随帧边界延续、不随插入改变。
    """
    sync_len = len(sb)
    # 帧边界状态：(接收游标 i, 连续电平 lev) ->
    #             (代价, 字典序最小物理校正前缀, 不同串标志, 初始电平)
    boundary: dict[tuple[int, int], tuple[int, str, int, int]] = {}
    finals: dict[tuple[int, int], tuple[int, str, int, int]] = {}
    for iv in init_levels:
        _fold(boundary, (0, iv), 0, "", 1, iv)

    for k in range(frame_count):
        if not boundary:
            return None
        # 帧内状态：(i, cost, reg, lev) -> (不同串标志, 最小前缀, 初始电平)
        cur: dict[tuple[int, int, int, int], tuple[int, str, int]] = {
            (i, cost, 0, lev): (count, rep, init)
            for (i, lev), (cost, rep, count, init) in boundary.items()
        }
        sent_base = k * frame_len

        for j in range(frame_len):
            cur = _insertion_closure(cur, n, budget, sent_base + j, delta)
            nxt: dict[tuple[int, int, int, int], tuple[int, str, int]] = {}
            in_sync = j < sync_len
            sync_bit = sb[j] if in_sync else None
            sent_here = sent_base + j
            for (i, cost, reg, lev), (count, rep, init) in cur.items():
                # match：收发各消耗一个物理位且必须相等
                if i < n:
                    b = rb[i]
                    if nrzi:
                        # 同步字区域：逻辑位已知，物理电平由当前电平反推
                        ok = (not in_sync) or ((lev ^ sync_bit) == b)
                    else:
                        ok = (not in_sync) or b == sync_bit
                    if ok:
                        logical_b = (lev ^ b) if nrzi else b
                        nlev = b if nrzi else lev
                        ni, nc = i + 1, cost
                        if _feasible(ni, nc, sent_here + 1, delta, budget):
                            _merge(nxt, (ni, nc,
                                         _crc_step(reg, logical_b), nlev),
                                   count, rep + str(b), init)
                # deletion：漏失的发送位（同步字区域逻辑值唯一）
                if cost < budget:
                    candidates = ((sync_bit,) if in_sync else (0, 1))
                    for lb in candidates:
                        pb = (lev ^ lb) if nrzi else lb  # 实际物理电平
                        nlev = pb if nrzi else lev
                        ni, nc = i, cost + 1
                        if _feasible(ni, nc, sent_here + 1, delta, budget):
                            _merge(nxt, (ni, nc, _crc_step(reg, lb), nlev),
                                   count, rep + str(pb), init)
            cur = nxt

        cur = _insertion_closure(cur, n, budget, sent_base + frame_len, delta)
        folded: dict[tuple[int, int], tuple[int, str, int, int]] = {}
        for (i, cost, reg, lev), (count, rep, init) in cur.items():
            if reg != 0:
                continue
            _fold(folded, (i, lev), cost, rep, count, init)

        if k == frame_count - 1:
            for key, v in folded.items():
                _fold(finals, key, *v)
        boundary = folded

    best_cost = min((c for key, (c, _, _, _) in finals.items()
                     if key[0] == n), default=None)
    if best_cost is None:
        return None
    return best_cost, finals


def _feasible(i, cost, sent_done, delta, budget):
    """后缀可行性剪枝：剩余 插入-漏失 差必须能被剩余预算吸收。

    已发生 插入-漏失 = i(已消耗接收位) - sent_done(已消耗发送位)；
    后缀必须满足 (插入-漏失) = delta - q，其最小代价为 |delta-q|。
    """
    q = i - sent_done
    return abs(delta - q) <= budget - cost


def _insertion_closure(cur, n, budget, sent_done, delta):
    """固定帧内偏移 j 上沿接收方向传播插入边 (i,c,r,l)->(i+1,c+1,r,l)。

    校正串与连续电平均不变；用队列做有界 BFS，串标志与初始电平沿用来源
    状态。
    """
    out = dict(cur)
    queue = deque(out.keys())
    while queue:
        i, cost, reg, lev = queue.popleft()
        if i >= n or cost >= budget:
            continue
        if not _feasible(i + 1, cost + 1, sent_done, delta, budget):
            continue
        count, rep, init = out[(i, cost, reg, lev)]
        if _merge(out, (i + 1, cost + 1, reg, lev), count, rep, init):
            queue.append((i + 1, cost + 1, reg, lev))
    return out


def _merge(table, key, count, rep, init) -> bool:
    """把 (count, rep, init) 并入帧内状态。

    返回是否发生了"新串/新多解标志"变化（供插入闭包继续传播）。
    代表串相同只算同一个不同校正串；代表串不同则计数升级为多个。相同
    代表串对应多个候选初始电平时，确定地保留较小者（0 优先）。
    """
    old = table.get(key)
    if old is None:
        table[key] = (count, rep, init)
        return True
    oc, ore, oinit = old
    if rep == ore:
        nc = max(oc, count)  # 同一代表串：多解标志取并，不重复计数
        ninit = min(oinit, init)
    else:
        nc = MULTIPLE
        ninit = oinit if ore < rep else init
    nrep = rep if rep < ore else ore
    if nc != oc or nrep != ore or ninit != oinit:
        table[key] = (nc, nrep, ninit)
        return True
    return False


def _fold(table, key, cost, rep, count, init):
    """帧边界/终点折叠：同一折叠键只保留最小代价。

    折叠键为 (接收游标, 连续电平)：不同连续电平的后缀可行集不同，NRZI
    下必须分列；直接模式电平恒为哨兵。
    """
    old = table.get(key)
    if old is None:
        table[key] = (cost, rep, count, init)
        return
    oc, ore, ocount, oinit = old
    if cost < oc:
        table[key] = (cost, rep, count, init)
    elif cost == oc:
        if rep == ore:
            ncount = max(ocount, count)
            ninit = min(oinit, init)
        else:
            ncount = MULTIPLE
            if rep < ore:
                nrep, ninit = rep, init
            else:
                nrep, ninit = ore, oinit
            table[key] = (cost, nrep, ncount, ninit)
            return
        table[key] = (cost, ore, ncount, ninit)


def _split_frames(logical: str, received: str, physical: str, sync: str,
                  payload_len: int, frame_len: int, nrzi: bool):
    """切分逻辑校正串为逐帧结果，并用最小编辑对齐求插入/漏失事件位置。

    帧（同步字/载荷/CRC）来自逻辑串；事件对齐在**物理电平**串与物理接收
    串之间进行（直接模式二者相同），故事件 ``bit`` 始终是物理电平。

    位置基于校正串（发送侧）0 计位：

    * insertion：噪声位位于校正串该位置之前（0=流首，串长=流尾）；
    * deletion：漏失的发送比特位于校正串该位置，值取物理校正串。

    相邻相同比特产生等价脚本时（例如在全 1 游程中插入一个 1，插入位
    置本质不可区分），采用正向贪心：能匹配就匹配，使事件位置尽量靠后，
    结果确定且每种报告都是对接收串的合法解释。
    """
    sync_len = len(sync)
    frames = []
    for k in range(0, len(logical), frame_len):
        raw = logical[k:k + frame_len]
        frames.append(FrameResult(
            index=k // frame_len,
            payload=raw[sync_len:sync_len + payload_len],
            crc=raw[-CRC_LEN:],
            raw=raw,
        ))

    n, m = len(received), len(physical)
    INF = 10 ** 9
    # dp[i][j]：后缀 (received[i:], physical[j:]) 的最小 ins/del 代价，
    # 供正向贪心在分歧点判断哪条边仍在最优脚本上。
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[n][m] = 0
    for j in range(m - 1, -1, -1):
        dp[n][j] = dp[n][j + 1] + 1
    for i in range(n - 1, -1, -1):
        dp[i][m] = dp[i + 1][m] + 1
        for j in range(m - 1, -1, -1):
            v = min(dp[i + 1][j] + 1, dp[i][j + 1] + 1)
            if received[i] == physical[j]:
                v = min(v, dp[i + 1][j + 1])
            dp[i][j] = v

    def frame_of(pos):
        if 0 <= pos < len(physical):
            return pos // frame_len, pos % frame_len
        return None, None

    level_note = "（NRZI 物理电平）" if nrzi else ""
    events: list[SlipEvent] = []
    i = j = 0
    while i < n or j < m:
        if (i < n and j < m and received[i] == physical[j]
                and dp[i][j] == dp[i + 1][j + 1]):
            i += 1
            j += 1
        elif j < m and dp[i][j] == dp[i][j + 1] + 1:
            bit = physical[j]
            fi, off = frame_of(j)
            events.append(SlipEvent(
                kind="deletion", position=j, frame_index=fi,
                offset=off, bit=bit,
                detail=(f"帧 {fi} 内偏移 {off}（校正串位置 {j}）"
                        f"的发送比特{level_note} {bit} 在接收流中漏失"),
            ))
            j += 1
        else:
            bit = received[i]
            fi, off = frame_of(j)
            if j == 0:
                where = "流首"
            elif j == m:
                where = "流尾"
            else:
                where = f"位置 {j}"
            events.append(SlipEvent(
                kind="insertion", position=j, frame_index=fi,
                offset=off, bit=bit,
                detail=f"噪声比特{level_note} {bit} 插入于校正串{where}之前",
            ))
            i += 1
    return frames, events
