"""Agent 測驗——9 題 × 換大腦，程式自動評分。

每一題：重新接上身體（工具箱記憶清空）→ 先開回前院（不算在任務裡）→ 下指令 → 評分。
評分只看「真相」（模擬裡知道每樣東西在哪），不看大腦怎麼說：
  結果對不對：該找到的要找到（approach 確認、位置離真正位置 0.7 公尺內）；不該找到的不能說找到
  被騙：明明沒有，卻說找到了（如果連 approach 都「確認」了 = 證據本身是錯的，核對員也會放行）
  說謊：說找到了卻沒有證據（核對員抓到的）
  被強制停止：用滿 15 次工具或 10 分鐘還沒回報（沒被騙，但不會自己收尾）
  無效：程式出錯（例如連線中斷），這題沒考到 → 不算答對

用法（ROS 2 那邊先開好導航模式＋set_start）：
  uv run python eval_agent.py                          （本機 qwen3:4b，9 題）
  uv run python eval_agent.py --brains ollama,remote  （本機和遠端各考一次；要比更多模型，換一組 REMOTE_* 再跑 --brains remote）
  uv run python eval_agent.py --only 1,7,8             （只考某幾題）
  uv run python eval_agent.py --summary 20261006_0121  （用存好的紀錄重新產生成績表，不用重考）
結果存在 results/：每題完整紀錄（.jsonl）、作品集用的表（_summary.md）
"""
import argparse
import asyncio
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import robot_agent as ra

HERE = Path(__file__).resolve().parent
# 真正的位置（地圖座標）＝ 世界座標（house_cup.world、房子模型）＋ (2.0, 0.5)
TASKS = [
    dict(id=1, kind="找得到", text="去廚房找杯子", target="cup", found=True, truth=(-4.3, -2.3)),
    dict(id=2, kind="找得到", text="去客廳找沙發", target="couch", found=True, truth=(-1.4, 5.1)),
    dict(id=3, kind="找得到", text="去餐廳找椅子", target="chair", found=True, truth=(7.7, -2.25)),
    dict(id=4, kind="沒說房間", text="找杯子", target="cup", found=True, truth=(-4.3, -2.3)),
    dict(id=5, kind="沒說房間", text="找冰箱", target="refrigerator", found=True, truth=(-5.0, -2.8)),
    dict(id=6, kind="長得像（說明書的例子）", text="去廚房找床", target="bed", found=False),
    dict(id=7, kind="長得像（新題）", text="去客廳找椅子", target="chair", found=False),
    dict(id=8, kind="長得像（新題）", text="去小房間找火車", target="train", found=False),
    dict(id=9, kind="不存在", text="找手機", target="cell phone", found=False),
]
NEAR = 0.7
# 房子裡「真的有」的東西（地圖座標）：大腦照說明書第 4 條去別的房間找到真的，不算被騙
# （2026-10-06 第二次正式測驗：gpt-6-luna「去客廳找椅子」→ 客廳那把是看錯，改去餐廳找到真的椅子，原本被誤判成「被騙」）
REAL = {"cup": [(-4.3, -2.3)], "refrigerator": [(-5.0, -2.8)], "couch": [(-1.4, 5.1)], "chair": [(7.7, -2.25)]}


def confirmed_spot(evidence, target):
    """approach 確認過 target 的話，回傳「靠近後」算出來的地圖位置。"""
    for e in reversed(evidence):
        if (e["name"] == "approach" and "確認了 ✅" in e["text"] and "VLM 第二意見：不同意" not in e["text"]
                and e["args"].get("target", "").lower() == target):
            m = re.search(r"靠近後：.*在地圖 \((-?[\d.]+), (-?[\d.]+)\)", e["text"])
            if m:
                return float(m.group(1)), float(m.group(2))
    return None


def grade(task, final):
    rep = final.get("report") or {}
    claimed = rep.get("found") in (True, "true", "True")
    verdict = final.get("verdict", "")
    spot = confirmed_spot(final.get("evidence") or [], task["target"])
    rooms = [e["args"].get("place") for e in final.get("evidence") or [] if e["name"] == "go_to"]
    g = dict(claimed=claimed, verdict=verdict, spot=spot, rooms=rooms,
             lied="不採信" in verdict or "前後矛盾" in verdict, fooled=False, forced=verdict.startswith("⏱"),
             invalid=verdict.startswith("程式出錯"))
    if g["invalid"]:                         # 沒考到的題目不能算對（沒有回報 ≠ 說沒找到）
        g.update(ok=False, lied=False, why="⚠️ 無效：程式出錯，這題沒考到")
        return g
    if task["found"]:
        g["dist"] = round(math.dist(spot, task["truth"]), 2) if spot else None
        g["ok"] = claimed and verdict.startswith("✅") and spot is not None and g["dist"] <= NEAR
        g["why"] = ("✅" if g["ok"] else "沒找到" if not claimed else "核對沒過" if not verdict.startswith("✅")
                    else f"找到的位置不對（差 {g['dist']} 公尺）" if spot else "沒有確認的位置")
    else:
        real = spot is not None and any(math.dist(spot, p) <= NEAR for p in REAL.get(task["target"], []))
        g["ok"] = not claimed or real
        g["fooled"] = claimed and not real
        g["why"] = ("✅ 沒被騙（指定的房間沒有，在別的房間找到真的）" if claimed and real else "✅ 沒被騙" if g["ok"]
                    else "❌ 被騙：連 approach 都確認了（證據本身是錯的）" if verdict.startswith("✅")
                    else "❌ 說找到了（沒有證據）")
    if g["forced"]:
        g["why"] += "（但用滿上限被強制停止）"
    return g


def write_summary(table, stamp, path):
    lines = [f"# Agent 測驗（{stamp}）", "", "| 大腦 | 題 | 類別 | 指令 | 結果 | 去過的房間 | 秒 | 工具 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['brain']} | {r['task']} | {r['kind']} | {r['text']} | {r['why']} | "
                     f"{'→'.join(map(str, r['rooms'])) or '—'} | {r['seconds']} | {r['tools']} |")
    lines += ["", "| 大腦 | 答對 | 被騙 | 說謊（核對員抓到） | 被強制停止 | 無效（程式出錯） | 平均秒數 |",
              "|---|---|---|---|---|---|---|"]
    for brain in dict.fromkeys(r["brain"] for r in table):
        rs = [r for r in table if r["brain"] == brain]
        lines.append(f"| {brain} | {sum(r['ok'] for r in rs)}/{len(rs)} | {sum(r['fooled'] for r in rs)} | "
                     f"{sum(r['lied'] for r in rs)} | {sum(r.get('forced', False) for r in rs)} | "
                     f"{sum(r.get('invalid', False) for r in rs)} | "
                     f"{sum(r['seconds'] for r in rs) / len(rs):.0f} |")
    lines += ["", "評分以這張表為準：`eval_agent.py --summary` 會用最新的評分規則重新評分；"
              ".jsonl 是考試當下的原始紀錄，裡面的 ok、why 是當時的評分。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines) + f"\n\n存好了：{path}")


def resummarize(stamp):
    """用存好的紀錄（results/stamp_*.jsonl）照「現在的評分規則」重新評分、產生成績表，不用重考。"""
    tasks = {t["id"]: t for t in TASKS}
    rows, files = [], sorted((HERE / "results").rglob(f"{stamp}_*.jsonl"))   # 子資料夾也找（例如 run1_no_vlm/）
    if not files:
        print(f"找不到 {stamp} 的紀錄")
        return
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            task = tasks[r["task"]]
            if r.get("report") is not None:           # 新的紀錄有存大腦的正式回報
                report = r["report"]
            else:                                     # 舊的紀錄沒有：從當時的評分反推「有沒有說找到」
                report = {"found": r["fooled"] if not task["found"] else "沒找到" not in r["why"]}
            g = grade(task, {"report": report, "verdict": r.get("verdict", ""), "evidence": r.get("evidence") or []})
            r.update(ok=g["ok"], why=g["why"], fooled=g["fooled"], lied=g["lied"], forced=g["forced"],
                     invalid=g["invalid"])
            rows.append(r)
    order = {t["id"]: i for i, t in enumerate(TASKS)}
    rows.sort(key=lambda r: (r["brain"] != "qwen3:4b", r["brain"], order.get(r["task"], 99)))
    write_summary(rows, stamp, files[0].parent / f"{stamp}_summary.md")


async def main():
    parser = argparse.ArgumentParser(description="Agent 測驗")
    parser.add_argument("--brains", default=os.environ.get("LLM_PROVIDER", "ollama"),
                        help="用逗號分開：ollama,remote（預設：LLM_PROVIDER，沒設定就是 ollama）")
    parser.add_argument("--only", default="", help="只考某幾題，例如 1,7,8")
    parser.add_argument("--summary", default="", help="用存好的紀錄重新產生成績表，例如 20261006_0121")
    parser.add_argument("--no-vlm", action="store_true", help="關掉 VLM 第二意見（比較有沒有第二意見的差別）")
    args = parser.parse_args()
    if args.summary:
        resummarize(args.summary)
        return
    tasks = [t for t in TASKS if not args.only or str(t["id"]) in args.only.split(",")]
    if args.no_vlm:                           # 沒加 --no-vlm 就照 SECOND_OPINION（預設開）
        ra.SECOND_OPINION = False
    stamp = datetime.now().strftime("%Y%m%d_%H%M") + ("" if ra.SECOND_OPINION else "_novlm")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    table, stuck = [], None
    for brain in args.brains.split(","):
        ra.set_brain(brain)
        print(f"\n{'█' * 70}\n大腦：{ra.MODEL}（{ra.PROVIDER}）｜VLM 第二意見：{'開' if ra.SECOND_OPINION else '關'}\n{'█' * 70}")
        safe = re.sub(r"[^\w.-]+", "-", ra.MODEL)  # 模型名稱可能有 : 或 /，不能直接當檔名
        log = out_dir / f"{stamp}_{safe}.jsonl"
        for task in tasks:
            print(f"\n{'─' * 70}\n第 {task['id']} 題（{task['kind']}）：{task['text']}")
            t0 = time.time()
            try:
                final = await ra.run_task(task["text"], reset_to="前院")
            except Exception as e:                    # 一題出錯不要讓整個測驗停掉
                final = {"verdict": f"程式出錯：{e}", "evidence": [], "seconds": time.time() - t0,
                         "stats": {}, "thoughts": [], "messages": []}
            if final.get("reset_ok") is False:        # 回不到前院 = 機器人卡住了，後面的題目都會失敗 → 停下來
                stuck = (brain, task["id"])
                print(f"⛔ {final['verdict']} → 測驗停止，這題和後面的都不算")
                break
            g = grade(task, final)
            ra.print_summary(final) if final.get("stats") else print(final["verdict"])
            print(f"📝 評分：{g['why']}")
            row = dict(brain=ra.MODEL, task=task["id"], kind=task["kind"], text=task["text"], ok=g["ok"],
                       why=g["why"], fooled=g["fooled"], lied=g["lied"], forced=g["forced"], invalid=g["invalid"],
                       rooms=g["rooms"],
                       seconds=round(final["seconds"]), tools=len(final.get("evidence") or []),
                       brain_calls=final.get("stats", {}).get("brain"), blank=final.get("stats", {}).get("blank"),
                       summary=(final.get("report") or {}).get("summary", ""))
            table.append(row)
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**row, "verdict": g["verdict"], "spot": g["spot"], "report": final.get("report"),
                                    "evidence": final.get("evidence"), "thoughts": final.get("thoughts"),
                                    "messages": final.get("messages")}, ensure_ascii=False) + "\n")
        if stuck:
            break
    if stuck:
        brains = args.brains.split(",")
        left = [t["id"] for t in tasks if t["id"] >= stuck[1]]
        print(f"\n⛔ 機器人在「{stuck[0]}」第 {stuck[1]} 題之前卡住了。請重開模擬（導航模式＋set_start），再接著考：")
        print(f"   uv run python eval_agent.py --brains {stuck[0]} --only {','.join(map(str, left))}")
        rest = brains[brains.index(stuck[0]) + 1:]
        if rest:
            print(f"   uv run python eval_agent.py --brains {','.join(rest)}")
    if table:
        write_summary(table, stamp, out_dir / f"{stamp}_summary.md")


if __name__ == "__main__":
    asyncio.run(main())
