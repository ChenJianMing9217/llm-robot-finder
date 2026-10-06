"""會找東西的機器人 Agent —— 大腦（語言模型）在 Windows 或 Linux，身體（ROS 2 的機器人工具箱）在 WSL 或 Ubuntu。

  指令 →【大腦】想下一步 ─┬─ 寫紙條（呼叫工具）→【身體】照做 → 結果拿給大腦看 → 回到【大腦】
                         └─ 用 report 正式回報 →【核對】程式查證據 ─┬─ 對得上 → 結束
                                                                   └─ 對不上 → 退回【大腦】一次

  大腦：本機 Ollama qwen3:4b（比較過幾個本機小模型，它用工具最穩）；LLM_PROVIDER 可以換成其他 OpenAI 相容的服務，AGENT_MODEL 換本機模型
  身體：機器人工具箱（robot_tools 的 mcp_server.py），用 MCP stdio 接過去（Windows 透過 wsl.exe）
  第二意見：approach 確認後，程式請 VLM（本機 qwen2.5vl:3b）列出看到的東西，VLM 沒提到目標就不算確認
  核對：說「找到了」一定要有 approach 回報「確認了 ✅」的紀錄 —— 不管模型怎麼說（小模型真的會謊報「已完成」）
  安全：最多 15 次工具、約 10 分鐘（每一步做完檢查），超過就停（程式管，不靠模型自律）

用法（ROS 2 那邊要先開好模擬、導航，並做完 set_start）：
  uv run python robot_agent.py "去廚房找杯子"
  uv run python robot_agent.py --think "去廚房找杯子"     （顯示大腦每一步的思考過程）
  測驗（9 題、換大腦、自動評分）：eval_agent.py；程式裡用 set_brain()、run_task()
"""
import argparse
import asyncio
import json
import operator
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Annotated, TypedDict

import openai
from langgraph.graph import END, START, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                   # 同一個資料夾的 llm.py（選大腦）
import llm  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def set_brain(provider=None, model=None):
    """換大腦：provider 交給 llm.py（預設 ollama），model 不給就用預設（ollama 預設 qwen3:4b）。"""
    global PROVIDER, client, MODEL, EXTRA, LIMIT
    llm._load_env()             # .env 裡的 LLM_PROVIDER、AGENT_MODEL 也要算數（要在讀環境變數之前載入）
    PROVIDER = provider or os.environ.get("LLM_PROVIDER", "ollama")
    client, MODEL, EXTRA = llm.get(PROVIDER)
    if not model and PROVIDER == "ollama":     # AGENT_MODEL 只換本機模型，遠端用 llm.py 給的
        model = os.environ.get("AGENT_MODEL") or "qwen3:4b"
    MODEL = model or MODEL
    if MODEL.startswith("qwen3"):
        EXTRA = {}              # 思考模式不要用 temperature=0（官方說容易一直重複），用模型預設的 0.6
    LIMIT = {"max_tokens": 1500} if PROVIDER == "ollama" else {}


set_brain()
MAX_CALLS, MAX_SECONDS, TOOL_TIMEOUT = 15, 600, 360
SHRINK_AT = 2800                # 本機模型的工作記憶只有 4096：對話超過這麼長，就把舊的工具結果縮短

# ───────────── 第二意見 ─────────────
# approach「確認了」之後，由程式把靠近後的畫面交給 VLM（另一雙眼睛）。VLM 沒提到目標 → 不算確認（VLM 出錯時只看 YOLO）。
# ⚠️ 小 VLM 會被問題帶著走：問「有沒有床？」它就說有，還說「杯子放在床上」；提到「相機」它就列出一堆相機
#    → 不提目標、只請它「列出看得到的東西」（英文，方便跟 YOLO 的名稱比對），再由程式比對清單裡有沒有目標
SECOND_OPINION = os.environ.get("SECOND_OPINION", "1") != "0"
VLM, OLLAMA = "qwen2.5vl:3b", "http://127.0.0.1:11434/api"
VLM_PROMPT = "List the objects you can see in this image. Answer with nouns only, separated by commas."
SYNONYMS = {"cup": ["cup", "mug", "glass"], "refrigerator": ["refrigerator", "fridge"], "couch": ["couch", "sofa"],
            "chair": ["chair", "stool"], "dining table": ["table", "desk"], "tv": ["tv", "television", "monitor", "screen"],
            "cell phone": ["phone", "smartphone"], "potted plant": ["plant"], "laptop": ["laptop", "computer"]}


def _ollama(path, body, timeout=300):
    req = urllib.request.Request(f"{OLLAMA}/{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def vlm_list(b64):
    """請 VLM 列出畫面裡看得到的東西。輸出亂碼（qwen2.5vl 偶爾會這樣）→ 卸載模型重來一次；還是不行回傳 None。"""
    body = {"model": VLM, "stream": False, "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 120},
            "messages": [{"role": "user", "content": VLM_PROMPT, "images": [b64]}]}
    for _ in range(2):
        try:
            out = _ollama("chat", body)["message"]["content"].strip()
        except Exception:
            out = ""
        if out and not out.startswith("@@@"):
            return out
        try:
            _ollama("generate", {"model": VLM, "keep_alive": 0}, timeout=60)
        except Exception:
            pass
    return None


def second_opinion(target, b64):
    """回傳 (同不同意, VLM 列出的東西)。同不同意：True／False；VLM 出錯回傳 None（不算反對）。"""
    seen = vlm_list(b64)
    if seen is None:
        return None, "（VLM 輸出亂碼，沒有第二意見）"
    words = SYNONYMS.get(target, [target])
    # 整個字比對（cup 不能算到 cupboard、phone 不能算到 microphone），允許複數（cups、glasses）
    agree = any(re.search(rf"\b{re.escape(w)}(e?s)?\b", seen.lower()) for w in words)
    return agree, seen[:120]

# 身體：機器人工具箱（MCP stdio）。Windows 透過 wsl.exe 叫起 WSL 裡的工具箱；Linux 直接執行。
# ROS_DOMAIN_ID 要和模擬那邊一樣（預設 0）；ROBOT_WS 是 ROS 2 工作區的位置（預設 ~/ros2_ws）
DOMAIN = os.environ.get("ROS_DOMAIN_ID", "0")
_launch = (f"export ROS_DOMAIN_ID={DOMAIN}; source /opt/ros/jazzy/setup.bash; "
           f"source {os.environ.get('ROBOT_WS', '~/ros2_ws')}/install/setup.bash; exec ros2 run robot_tools robot_mcp")
ROBOT = (StdioServerParameters(command="wsl.exe", args=["-e", "bash", "-c", _launch]) if os.name == "nt"
         else StdioServerParameters(command="bash", args=["-c", _launch], env=dict(os.environ)))

RULES = """
你是這台機器人的大腦，用繁體中文回答。使用者會叫你找東西，你用工具一步一步完成。

找東西的步驟：
1. go_to 到指定的房間；沒指定房間，就先去最可能有那樣東西的房間。
2. look 看前面；沒看到要找的東西，就 look_around 轉一圈。
3. 看到了就 approach 靠近確認。approach 說「確認了 ✅」、而且 VLM 第二意見沒有反對，才算找到。
   VLM 第二意見「不同意」= 另一雙眼睛沒看到它，多半是 YOLO 看錯。
4. 這個房間找不到，可以去別的可能房間找；真的找不到就回報沒找到，不要猜。

YOLO 常常看錯，判斷看到的東西時：
- 想想合不合理：這個房間平常會有這樣東西嗎？例如廚房有冰箱很合理；廚房有床多半是把牆看錯。
- 前後一致：真的東西換個角度看，名字和地圖位置都差不多；同一個位置一下叫 A、一下叫 B，多半是看錯。
- 信心高不一定對，信心低也不一定錯。

結束時一定要呼叫 report 回報一次。"""

REPORT = {"type": "function", "function": {
    "name": "report",
    "description": "任務結束時呼叫一次，回報結果。found=true 只能在 approach 對這樣東西回報「確認了 ✅」"
                   "而且 VLM 第二意見沒有反對之後使用。",
    "parameters": {"type": "object", "properties": {
        "target": {"type": "string", "description": "要找的東西，YOLO 英文名稱，例如 cup"},
        "found": {"type": "boolean", "description": "有沒有找到並確認"},
        "summary": {"type": "string", "description": "給使用者看的說明（繁體中文）：在哪個房間、地圖哪裡、怎麼確認的；"
                                                      "哪些東西你判斷是看錯的、為什麼"}},
        "required": ["target", "found", "summary"]}}}

SESSION, SYSTEM, TOOLS, SHOW_THINK = None, "", [], False
STATS, THOUGHTS = {}, []                        # 每個任務重新算（run_task 開頭清空）


def reset_stats():
    STATS.clear()
    STATS.update(brain=0, brain_sec=0.0, tool_sec=0.0, blank=0, last_prompt=0, max_prompt=0, shrunk=False)
    THOUGHTS.clear()


reset_stats()


class State(TypedDict, total=False):
    messages: Annotated[list, operator.add]     # 對話紀錄：每站把新訊息「接在後面」
    evidence: Annotated[list, operator.add]     # 工具真正回傳的結果（程式記的，不是模型說的）
    report: dict | None                         # 大腦用 report 正式回報的內容
    pushback: int                               # 被核對退回的次數
    verdict: str                                # 程式核對的結論
    t0: float


def indent(text, pre="   "):
    return "\n".join(pre + ln for ln in str(text).splitlines())


def view(messages):
    """給大腦看的對話。太長就把比較舊的工具結果縮短（最近 2 個保留完整）—— 大腦的工作記憶有限。"""
    if STATS["last_prompt"] < SHRINK_AT:
        return messages
    if not STATS["shrunk"]:
        STATS["shrunk"] = True
        print(f"   🧹 對話快塞滿大腦的工作記憶（{STATS['last_prompt']} / 4096）→ 舊的工具結果縮短")
    tools = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    old = set(tools[:-2])
    return [{**m, "content": m["content"][:80] + "…（較早的結果，已縮短）"}
            if i in old and len(m["content"]) > 90 else m for i, m in enumerate(messages)]


def chat(msgs, kw):
    for k in range(5):
        try:
            return client.chat.completions.create(model=MODEL, messages=msgs, tools=TOOLS, **LIMIT, **kw)
        except openai.RateLimitError:           # 有些遠端服務限制每分鐘的次數：429 就等一下
            time.sleep(10 * (k + 1))
    return client.chat.completions.create(model=MODEL, messages=msgs, tools=TOOLS, **LIMIT, **kw)


# ───────────── 各站 ─────────────
async def brain(state: State) -> State:
    msgs = [{"role": "system", "content": SYSTEM}] + view(state["messages"])
    t = time.time()
    # ⚠️ 防護：小模型常「交白卷」（沒寫紙條也沒回答）→ 重問，最多 3 次，重問時 temperature 調高
    for attempt in range(3):
        kw = EXTRA if attempt == 0 else {**EXTRA, "temperature": 0.7}
        r = await asyncio.to_thread(chat, msgs, kw)
        m = r.choices[0].message
        if m.tool_calls or (m.content or "").strip():
            break
        STATS["blank"] += 1
        print("   ⚠️ 大腦交白卷" + ("，重問一次" if attempt < 2 else "，重問 2 次仍然空白"))
    sec = time.time() - t
    STATS["brain"] += 1
    STATS["brain_sec"] += sec
    STATS["last_prompt"] = r.usage.prompt_tokens if r.usage else 0
    STATS["max_prompt"] = max(STATS["max_prompt"], STATS["last_prompt"])
    notes = [f"{tc.function.name}({tc.function.arguments})" for tc in m.tool_calls or []]
    print(f"\n🧠 大腦（想了 {sec:.0f} 秒）→ " + ("、".join(notes) if notes else "（沒有寫紙條）"))
    think = (m.model_extra or {}).get("reasoning") or (m.model_extra or {}).get("reasoning_content") or ""
    THOUGHTS.append({"decision": notes, "think": think.strip()[:2000], "content": (m.content or "").strip()})
    if SHOW_THINK and think.strip():
        print(indent(think.strip(), "   💭 "))
    if (m.content or "").strip():
        print(indent(m.content.strip(), "   💬 "))
    return {"messages": [llm.to_history(m)]}


async def act(state: State) -> State:
    msgs, evidence, report = [], [], None
    for tc in state["messages"][-1]["tool_calls"]:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = None
        if not isinstance(args, dict):
            text = f"參數不是正確的 JSON：{tc['function']['arguments']}"
        elif name == "report":
            report, text = args, "收到回報。"
        else:
            t = time.time()
            images = []
            try:
                r = await SESSION.call_tool(name, args, read_timeout_seconds=TOOL_TIMEOUT)
                text, ok = "".join(c.text for c in r.content if c.type == "text"), not r.is_error
                images = [c.data for c in r.content if c.type == "image"]    # 大腦只讀文字，圖片給 VLM
            except Exception as e:              # 例如超過 6 分鐘沒回應
                text, ok = f"工具沒有回應：{e}", False
            if name == "approach" and "確認了 ✅" in text and images and SECOND_OPINION:
                target = str(args.get("target", "")).lower()
                agree, seen = await asyncio.to_thread(second_opinion, target, images[-1])
                text += ("\nVLM 第二意見：同意 ✅（VLM 看到：" + seen + "）" if agree
                         else "\nVLM 第二意見：不同意 ❌（VLM 看到：" + seen + f"；沒有提到 {target}）→ 不算確認"
                         if agree is False else "\nVLM 第二意見：" + seen)
            sec = time.time() - t
            STATS["tool_sec"] += sec
            evidence.append({"name": name, "args": args, "ok": ok, "text": text})
            print(f"🤖 {name}（{sec:.0f} 秒）{'' if ok else '❌'}")
            print(indent(text))
        msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": text})
    out = {"messages": msgs, "evidence": evidence}
    if report is not None:
        out["report"] = report
    return out


def check(state: State) -> State:
    """程式核對：大腦說「找到了」，就要有 approach 回報「確認了 ✅」的紀錄 —— 不管大腦怎麼說。"""
    report, n, evidence = state.get("report"), state.get("pushback", 0), state.get("evidence") or []
    over = len(evidence) >= MAX_CALLS or time.time() - state["t0"] > MAX_SECONDS

    def push_back(msg):
        print(f"\n🔍 程式核對：{msg}\n   → 退回給大腦")
        return {"messages": [{"role": "user", "content": f"（程式核對）{msg}"}], "pushback": n + 1, "report": None}

    if report is None:
        if over:
            return {"verdict": f"⏱ 超過上限（工具 {len(evidence)} 次、{(time.time() - state['t0']) / 60:.1f} 分鐘），強制停止"}
        if n < 1:
            return push_back("你還沒有用 report 回報結果。請繼續用工具找，或呼叫 report 回報。")
        return {"verdict": "⚠️ 大腦沒有用 report 正式回報"}
    target = str(report.get("target", "")).strip()
    found = report.get("found") in (True, "true", "True")
    confirmed = [e for e in evidence if e["name"] == "approach" and "確認了 ✅" in e["text"]
                 and "VLM 第二意見：不同意" not in e["text"]                # 兩雙眼睛都同意才算
                 and (e["args"].get("target", "").lower() == target.lower() or f"{target}（" in e["text"])]
    if found and confirmed:
        proof = [ln for ln in confirmed[-1]["text"].splitlines() if ln.startswith(("靠近後", "結論"))]
        return {"verdict": f"✅ 有證據：approach 確認過 {target}\n" + indent("\n".join(proof))}
    if found:
        if n < 1 and not over:
            return push_back(f"你回報找到了 {target}，但沒有「approach 確認了、VLM 第二意見也沒反對」的紀錄。請再確認，或改成回報沒找到。")
        return {"verdict": f"❌ 大腦說找到了 {target}，但沒有 approach＋VLM 都確認的紀錄 → 不採信"}
    if confirmed:
        return {"verdict": f"⚠️ 大腦說沒找到，但 approach 其實確認過 {target} → 前後矛盾"}
    return {"verdict": f"沒找到 {target}（跟紀錄一致：沒有 approach＋VLM 都確認過它）"}


def after_brain(state: State) -> str:
    return "act" if state["messages"][-1].get("tool_calls") else "check"


def after_act(state: State) -> str:
    if state.get("report") is not None:
        return "check"
    if len(state["evidence"]) >= MAX_CALLS or time.time() - state["t0"] > MAX_SECONDS:
        return "check"                          # 保險：次數、時間上限
    return "brain"


def after_check(state: State) -> str:
    return END if state.get("verdict") else "brain"


builder = StateGraph(State)
builder.add_node("brain", brain)
builder.add_node("act", act)
builder.add_node("check", check)
builder.add_edge(START, "brain")
builder.add_conditional_edges("brain", after_brain, ["act", "check"])
builder.add_conditional_edges("act", after_act, ["brain", "check"])
builder.add_conditional_edges("check", after_check, ["brain", END])
GRAPH = builder.compile()


async def run_task(instruction, reset_to=None):
    """跑一個任務。每次都重新接上身體（新的工具箱 = 記憶清空，題目之間不會互相洩題）；
    reset_to：開始前先開到這個地點（測驗用，不算在任務裡）。回傳最後的狀態＋統計。"""
    global SESSION, SYSTEM, TOOLS
    reset_stats()
    t_connect = time.time()
    async with stdio_client(ROBOT) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            listed = (await session.list_tools()).tools
            SESSION, SYSTEM = session, (init.instructions or "") + "\n" + RULES
            TOOLS = [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                                       "parameters": t.input_schema}} for t in listed] + [REPORT]
            print(f"接上身體了（{time.time() - t_connect:.0f} 秒），工具：{'、'.join(t.name for t in listed)}")
            if reset_to:
                r = await session.call_tool("go_to", {"place": reset_to}, read_timeout_seconds=TOOL_TIMEOUT)
                print(f"（開始前先回到{reset_to}：{'❌ 沒回到' if r.is_error else '✅'}）")
                if r.is_error:                      # 機器人可能卡住了：這題不算，交給呼叫的人決定要不要停
                    return {"reset_ok": False, "verdict": f"無效：開始前沒有回到{reset_to}（機器人可能卡住了）",
                            "evidence": [], "messages": [], "seconds": 0, "stats": dict(STATS), "thoughts": []}
            print(f"\n🗣 指令：{instruction}")
            t0 = time.time()
            final = await GRAPH.ainvoke({"messages": [{"role": "user", "content": instruction}],
                                         "evidence": [], "pushback": 0, "t0": t0},
                                        {"recursion_limit": 100})
    final["seconds"] = time.time() - t0
    final["stats"] = dict(STATS)
    final["thoughts"] = list(THOUGHTS)
    return final


def print_summary(final):
    report = final.get("report")
    print("\n" + "═" * 70)
    if report:
        found = report.get("found") in (True, "true", "True")
        print(f"📋 大腦回報：{'找到了' if found else '沒找到'} {report.get('target', '')}")
        print(indent(report.get("summary", "")))
    print(f"🔍 程式核對：{final['verdict']}")
    st = final["stats"]
    print(f"⏱ 總共 {final['seconds']:.0f} 秒｜大腦想了 {st['brain']} 次（{st['brain_sec']:.0f} 秒"
          + (f"，交白卷 {st['blank']} 次" if st["blank"] else "") + f"）｜工具用了 {len(final['evidence'])} 次"
          f"（{st['tool_sec']:.0f} 秒）｜對話最長 {st['max_prompt']} / 4096")


async def main():
    global SHOW_THINK
    parser = argparse.ArgumentParser(description="會找東西的機器人 Agent")
    parser.add_argument("instruction", nargs="?", default="去廚房找杯子", help="給機器人的指令")
    parser.add_argument("--think", action="store_true", help="顯示大腦每一步的思考過程")
    args = parser.parse_args()
    SHOW_THINK = args.think
    print(f"大腦：{MODEL}（{PROVIDER}）｜身體：{'WSL' if os.name == 'nt' else '本機'}的機器人工具箱（ROS_DOMAIN_ID={DOMAIN}）")
    print_summary(await run_task(args.instruction))


if __name__ == "__main__":
    asyncio.run(main())
