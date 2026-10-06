from setuptools import setup

package_name = "nav_tools"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "places_house.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ChenJianMing9217",
    maintainer_email="100664850+ChenJianMing9217@users.noreply.github.com",
    description="用地點名稱叫 Nav2 導航、設定起點、巡邏畫地圖",
    license="MIT",
    entry_points={
        "console_scripts": [
            "go_to = nav_tools.go_to:main",
            "set_start = nav_tools.set_start:main",
            "explore = nav_tools.explore:main",
        ],
    },
)
