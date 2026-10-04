"""NRZI 线电平直接复原测试。

NRZI：逻辑 1 翻转电平、逻辑 0 保持电平，编码状态跨帧连续。求解器联合
裁决初始电平、物理校正串、逻辑帧与插入/漏失，最优解与唯一性按滑移次数
与物理校正串字典序判定。
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import (  # noqa: E402
    crc8,
    frame_is_valid,
    reconstruct,
)


def make_frame(sync: str, payload: str) -> str:
    body = sync + payload
    return body + format(crc8(body), "08b")


def nrzi_encode(logical: str, initial_level: int) -> str:
    """连续 NRZI 编码：逻辑 1 翻转、逻辑 0 保持，状态跨帧延续。"""
    out = []
    lvl = initial_level
    for ch in logical:
        lvl ^= ord(ch) - ord("0")
        out.append(chr(ord("0") + lvl))
    return "".join(out)


def edit_distance_ins_del(a: str, b: str) -> int:
    """仅允许插入/删除的编辑距离。"""
    n, m = len(a), len(b)
    inf = 10 ** 9
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        nxt = [i] + [inf] * m
        for j in range(1, m + 1):
            nxt[j] = min(
                dp[j] + 1,
                nxt[j - 1] + 1,
                dp[j - 1] if a[i - 1] == b[j - 1] else inf,
            )
        dp = nxt
    return dp[m]


class NrziBasicTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(1234)
        self.sync = "11010011"
        self.plen = 18
        self.nf = 4
        self.frames = [
            make_frame(self.sync,
                       "".join(self.rng.choice("01")
                               for _ in range(self.plen)))
            for _ in range(self.nf)
        ]
        self.logical = "".join(self.frames)
        self.init = 1
        self.physical = nrzi_encode(self.logical, self.init)
        self.frame_len = len(self.sync) + self.plen + 8

    def _recover(self, received, initial_level="unknown", budget=6):
        return reconstruct(received, self.nf, self.sync, self.plen, budget,
                           line_code="nrzi", initial_level=initial_level)

    def test_clean_known_initial_level(self):
        for init in (0, 1):
            physical = nrzi_encode(self.logical, init)
            r = self._recover(physical, initial_level=init)
            self.assertTrue(r.recoverable)
            self.assertEqual(r.slippage_count, 0)
            self.assertEqual(r.corrected, physical)
            self.assertEqual(r.initial_level, init)
            self.assertTrue(r.unique)
            self.assertEqual(r.events, ())
            self.assertEqual([f.raw for f in r.frames], self.frames)
            for i, f in enumerate(r.frames):
                self.assertEqual(f.index, i)
                self.assertEqual(
                    f.payload,
                    self.frames[i][len(self.sync):
                                   len(self.sync) + self.plen])
                self.assertEqual(f.crc, self.frames[i][-8:])

    def test_clean_unknown_initial_level(self):
        # 初始电平未知时由求解器联合裁决，两种真实取值都应被正确推定
        for init in (0, 1):
            physical = nrzi_encode(self.logical, init)
            r = self._recover(physical, initial_level="unknown")
            self.assertTrue(r.recoverable)
            self.assertEqual(r.slippage_count, 0)
            self.assertEqual(r.corrected, physical)
            self.assertEqual(r.initial_level, init)
            self.assertEqual([f.raw for f in r.frames], self.frames)

    def test_cross_frame_level_continuity(self):
        # 构造自检：至少一个帧边界处带入电平为 1；若实现逐帧重置 NRZI
        # 状态（如每帧从 0 开始），后续帧同步字/CRC 必然全部译错
        carried = [int(self.physical[k * self.frame_len - 1])
                   for k in range(1, self.nf)]
        self.assertIn(1, carried)
        r = self._recover(self.physical)
        self.assertTrue(r.recoverable)
        self.assertEqual(r.corrected, self.physical)
        self.assertEqual([f.raw for f in r.frames], self.frames)

    def test_wrong_forced_initial_level_is_infeasible(self):
        # 强制错误的初始电平：所需物理流为实际流的逐位取反，
        # 预算内不可复原，且不得返回局部帧或猜测载荷
        r = self._recover(self.physical, initial_level=1 - self.init)
        self.assertFalse(r.recoverable)
        self.assertIsNone(r.corrected)
        self.assertEqual(r.frames, ())
        self.assertGreaterEqual(r.minimum_slippage_lower_bound, 7)

    def test_single_level_deletion(self):
        # 选物理电平与其邻位都不同的位置，漏失位置才可唯一确定
        pos = next(p for p in range(1, len(self.physical) - 1)
                   if (self.physical[p] != self.physical[p - 1]
                       and self.physical[p] != self.physical[p + 1]))
        recv = self.physical[:pos] + self.physical[pos + 1:]
        r = self._recover(recv)
        self.assertTrue(r.recoverable)
        self.assertEqual(r.slippage_count, 1)
        self.assertEqual(r.corrected, self.physical)
        self.assertEqual(r.initial_level, self.init)
        self.assertEqual([e.kind for e in r.events], ["deletion"])
        ev = r.events[0]
        self.assertEqual(ev.position, pos)
        # 事件 bit 表示物理电平
        self.assertEqual(ev.bit, self.physical[pos])
        self.assertEqual([f.raw for f in r.frames], self.frames)

    def test_single_level_insertion(self):
        # 插入位与两侧电平都不同，插入位置才可唯一确定
        pos = next(p for p in range(1, len(self.physical))
                   if (self.physical[p - 1] == "0"
                       and self.physical[p] == "0"))
        recv = self.physical[:pos] + "1" + self.physical[pos:]
        r = self._recover(recv)
        self.assertTrue(r.recoverable)
        self.assertEqual(r.slippage_count, 1)
        self.assertEqual(r.corrected, self.physical)
        self.assertEqual([e.kind for e in r.events], ["insertion"])
        ev = r.events[0]
        self.assertEqual(ev.position, pos)
        self.assertEqual(ev.bit, "1")
        self.assertEqual([f.raw for f in r.frames], self.frames)

    def test_insertion_and_deletion_combined(self):
        fl = self.frame_len
        # 一插一漏分别落在第 2、3 帧，检验跨帧电平延续下的联合复原
        del_pos = next(p for p in range(fl + 1, 2 * fl - 1)
                       if (self.physical[p] != self.physical[p - 1]
                           and self.physical[p] != self.physical[p + 1]))
        ins_pos = next(p for p in range(2 * fl + 1, 3 * fl - 1)
                       if (self.physical[p - 1] == "0"
                           and self.physical[p] == "0"))
        recv = self.physical[:ins_pos] + "1" + self.physical[ins_pos:]
        recv = recv[:del_pos] + recv[del_pos + 1:]
        r = self._recover(recv)
        self.assertTrue(r.recoverable)
        self.assertEqual(r.slippage_count, 2)
        self.assertEqual(r.corrected, self.physical)
        self.assertEqual(r.initial_level, self.init)
        self.assertEqual(sorted(e.kind for e in r.events),
                         ["deletion", "insertion"])
        self.assertEqual([f.raw for f in r.frames], self.frames)

    def test_events_replay_physical_stream(self):
        # 任意破坏后：事件数 == 滑移次数，且按事件回放物理校正串必得接收串
        rng = random.Random(77)
        for _ in range(10):
            stream = self.physical
            for _ in range(rng.randint(1, 4)):
                p = rng.randrange(len(stream))
                if rng.random() < 0.5:
                    stream = stream[:p] + stream[p + 1:]
                else:
                    stream = stream[:p] + rng.choice("01") + stream[p:]
            r = self._recover(stream)
            self.assertTrue(r.recoverable)
            self.assertEqual(len(r.events), r.slippage_count)
            self.assertEqual(
                edit_distance_ins_del(stream, r.corrected),
                r.slippage_count,
            )
            s = r.corrected
            for ev in sorted(r.events, key=lambda e: e.position,
                             reverse=True):
                if ev.kind == "deletion":
                    s = s[:ev.position] + s[ev.position + 1:]
                else:
                    s = s[:ev.position] + ev.bit + s[ev.position:]
            self.assertEqual(s, stream)
            for f in r.frames:
                self.assertTrue(frame_is_valid(f.raw, self.sync, self.plen))

    def test_over_budget_returns_lower_bound(self):
        recv = self.physical
        for p in (120, 100, 88, 70, 44, 20, 5):
            recv = recv[:p] + recv[p + 1:]
        r = self._recover(recv)
        self.assertFalse(r.recoverable)
        self.assertIsNone(r.corrected)
        self.assertEqual(r.frames, ())
        self.assertEqual(r.events, ())
        self.assertGreaterEqual(r.minimum_slippage_lower_bound, 7)
        # 失败结论不得携带局部帧、猜测载荷或初始电平
        d = r.to_dict()
        self.assertNotIn("frames", d)
        self.assertNotIn("corrected", d)
        self.assertNotIn("initial_level", d)

    def test_length_gap_lower_bound(self):
        recv = self.physical + "0101010"
        r = self._recover(recv)
        self.assertFalse(r.recoverable)
        self.assertEqual(r.minimum_slippage_lower_bound, 7)

    def test_legacy_mode_response_shape_unchanged(self):
        # 未提供 line_code 时：响应不含 initial_level，裁决结果与直读一致
        r = reconstruct(self.logical, self.nf, self.sync, self.plen, 6)
        self.assertTrue(r.recoverable)
        self.assertEqual(r.corrected, self.logical)
        self.assertNotIn("initial_level", r.to_dict())


class NrziOptimalityTests(unittest.TestCase):
    """与朴素穷举对拍：最小滑移、物理校正串字典序最小、唯一性。"""

    @staticmethod
    def brute(recv, nf, sync, plen, budget, init_levels):
        sl = len(sync)
        fl = sl + plen + 8

        def step(reg, b):
            v = reg ^ (b << 7)
            return (((v << 1) ^ 0x07) & 0xFF
                    if v & 0x80 else ((v << 1) & 0xFF))

        found = set()  # 物理校正串

        def rec(ri, k, j, reg, lvl, corr, cost):
            if cost > budget:
                return
            if k == nf:
                if ri == len(recv):
                    found.add(corr)
                return
            if j == fl:
                if reg == 0:
                    # 帧边界电平 lvl 延续到下一帧，不重置
                    rec(ri, k + 1, 0, 0, lvl, corr, cost)
                return
            cands = (int(sync[j]),) if j < sl else (0, 1)
            if ri < len(recv):
                rec(ri + 1, k, j, reg, lvl, corr, cost + 1)
            for b in cands:
                v = lvl ^ b  # 逻辑比特 b 对应的发送电平
                if ri < len(recv) and int(recv[ri]) == v:
                    rec(ri + 1, k, j + 1, step(reg, b), v,
                        corr + str(v), cost)
                rec(ri, k, j + 1, step(reg, b), v, corr + str(v), cost + 1)

        for init in init_levels:
            rec(0, 0, 0, 0, init, "", 0)
        return found

    def test_matches_brute_force(self):
        rng = random.Random(555)
        checked = 0
        for trial in range(40):
            slen = rng.randint(6, 7)
            plen = rng.randint(16, 18)
            nf = 3
            sync = "".join(rng.choice("01") for _ in range(slen))
            logical = ""
            for _ in range(nf):
                body = sync + "".join(rng.choice("01") for _ in range(plen))
                logical += body + format(crc8(body), "08b")
            init = rng.randint(0, 1)
            physical = nrzi_encode(logical, init)
            damaged = physical
            for _ in range(rng.randint(0, 2)):
                p = rng.randrange(len(damaged))
                if rng.random() < 0.5:
                    damaged = damaged[:p] + damaged[p + 1:]
                else:
                    damaged = damaged[:p] + rng.choice("01") + damaged[p:]
            budget = rng.randint(1, 2)
            known = rng.random() < 0.5
            il = init if known else "unknown"
            r = reconstruct(damaged, nf, sync, plen, budget,
                            line_code="nrzi", initial_level=il)
            init_levels = (init,) if known else (0, 1)
            opt = self.brute(damaged, nf, sync, plen, budget, init_levels)
            if not opt:
                self.assertFalse(r.recoverable, trial)
                continue
            costs = {x: edit_distance_ins_del(damaged, x) for x in opt}
            best_cost = min(costs.values())
            best = {x for x, c in costs.items() if c == best_cost}
            self.assertTrue(r.recoverable)
            self.assertEqual(r.slippage_count, best_cost)
            self.assertEqual(r.corrected, min(best))
            self.assertEqual(r.unique, len(best) == 1)
            # 推定初始电平与物理校正串自洽（首逻辑比特 = 同步字首比特）
            self.assertEqual(r.initial_level,
                             int(r.corrected[0]) ^ int(sync[0]))
            for f in r.frames:
                self.assertTrue(frame_is_valid(f.raw, sync, plen))
            checked += 1
        self.assertGreater(checked, 15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
