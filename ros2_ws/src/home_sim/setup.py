from glob import glob

from setuptools import setup

package_name = "home_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/worlds", glob("worlds/*.world")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ChenJianMing9217",
    maintainer_email="100664850+ChenJianMing9217@users.noreply.github.com",
    description="房子模擬：TurtleBot3 房子＋家具＋杯子",
    license="MIT",
)
