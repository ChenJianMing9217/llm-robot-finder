from setuptools import setup

package_name = "yolo_vision"

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
    description="把 YOLO 包成 ROS 2 節點",
    license="MIT",
    entry_points={
        "console_scripts": [
            "video_camera = yolo_vision.video_camera:main",
            "yolo_detector = yolo_vision.yolo_detector:main",
        ],
    },
)
