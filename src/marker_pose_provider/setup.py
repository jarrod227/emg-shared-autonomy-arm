from setuptools import find_packages, setup

package_name = "marker_pose_provider"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/marker_demo.launch.py"]),
        ("share/" + package_name + "/config", ["config/camera_info.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jiayu Yang",
    maintainer_email="jarrodyang227@gmail.com",
    description="Stateless ArUco marker localization publishing all detected marker poses to /detected_markers.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "marker_node = marker_pose_provider.marker_node:main",
            "accuracy_probe = marker_pose_provider.accuracy_probe:main",
        ],
    },
)
