from setuptools import find_packages, setup

package_name = "target_selector"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/extrinsics.launch.py"]),
        (
            "share/" + package_name + "/config",
            ["config/extrinsics.yaml", "config/grasp_offsets.yaml"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jiayu Yang",
    maintainer_email="jarrodyang227@gmail.com",
    description="Selects a detected marker, applies the grasp offset, transforms it into the planning frame, and publishes /target_object_pose.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            (
                "markerless_candidate_gate = "
                "target_selector.markerless_candidate_gate_node:main"
            ),
            "selector_node = target_selector.selector_node:main",
        ],
    },
)
