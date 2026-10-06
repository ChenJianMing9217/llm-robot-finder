"""檢查機器人工具箱（MCP）：像 Agent 一樣連上去，列出工具，再依序呼叫看看。
模擬和導航要先開好、set_start 要先做。

用法：ros2 run robot_tools robot_mcp_check            （where_am_i → look）
      ros2 run robot_tools robot_mcp_check --full     （再加上 go_to 廚房 → look → approach cup → look_around）
"""
import argparse
import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run(full):
    # env 一定要整包傳過去：MCP 預設只傳少數幾個環境變數，少了 ROS 的設定，工具箱連 ROS 都找不到
    server = StdioServerParameters(command=sys.executable, args=["-m", "robot_tools.mcp_server"], env=dict(os.environ))
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            print("工具箱裡有：")
            for t in tools:
                print(f"  {t.name}：{(t.description or '').splitlines()[0]}")
            calls = [("where_am_i", {}), ("look", {})]
            if full:
                calls += [("go_to", {"place": "廚房"}), ("look", {}), ("approach", {"target": "cup"}), ("look_around", {})]
            for name, args in calls:
                print(f"\n▶ 呼叫 {name}({args})")
                r = await session.call_tool(name, args)
                print(("❌ " if r.is_error else "") + "".join(c.text for c in r.content if hasattr(c, "text")))


def main():
    parser = argparse.ArgumentParser(description="檢查機器人工具箱（MCP）")
    parser.add_argument("--full", action="store_true", help="也試 go_to、approach、look_around（要花幾分鐘）")
    asyncio.run(run(parser.parse_args().full))
