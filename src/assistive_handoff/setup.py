from setuptools import find_packages, setup

package_name = "assistive_handoff"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jiayu Yang",
    maintainer_email="jarrodyang227@gmail.com",
    description="Safety-aware handoff state machine (Objective 4.1); M1 skeleton with simulated intent/hand/target publishers.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "handoff_controller = assistive_handoff.handoff_controller:main",
            "sim_intent_publisher = assistive_handoff.sim_intent_publisher:main",
            "sim_hand_publisher = assistive_handoff.sim_hand_publisher:main",
            "sim_target_publisher = assistive_handoff.sim_target_publisher:main",
        ],
    },
)
