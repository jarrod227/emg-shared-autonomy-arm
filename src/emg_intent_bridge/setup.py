from setuptools import setup


package_name = "emg_intent_bridge"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/python",
            ["../../firmware/tools/emg_protocol.py"],
        ),
    ],
    install_requires=["setuptools", "pyserial"],
    zip_safe=True,
    maintainer="Jiayu Yang",
    maintainer_email="jarrodyang227@gmail.com",
    description=(
        "USB CDC bridge from STM32 EMG INTENT packets to source-independent "
        "ROS 2 intent events."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "emg_intent_bridge = emg_intent_bridge.bridge_node:main",
        ],
    },
)
