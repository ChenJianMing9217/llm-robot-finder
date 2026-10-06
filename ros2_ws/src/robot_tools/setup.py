from setuptools import setup

package_name = "robot_tools"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ChenJianMing9217",
    maintainer_email="100664850+ChenJianMing9217@users.noreply.github.com",
    description="給 Agent 用的機器人工具箱（MCP）",
    license="MIT",
    entry_points={
        "console_scripts": [
            "look = robot_tools.look:main",
            "approach = robot_tools.approach:main",
            "robot_mcp = robot_tools.mcp_server:main",
            "robot_mcp_check = robot_tools.mcp_check:main",
        ],
    },
)
