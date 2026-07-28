from setuptools import find_packages, setup

package_name = "stereo_hand_observer"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    extras_require={
        "vision": [
            "mediapipe==1.0.0",
            "numpy<2",
            "opencv-contrib-python==4.11.0.86",
        ],
    },
    zip_safe=True,
    maintainer="Jiayu Yang",
    maintainer_email="jarrodyang227@gmail.com",
    description=(
        "Quality-checked stereo hand-keypoint geometry for Objective 4.2."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "hand_detector_demo = "
            "stereo_hand_observer.hand_detector_demo:main",
            "live_observer = "
            "stereo_hand_observer.live_observer:main",
            "synthetic_observer = "
            "stereo_hand_observer.synthetic_observer:main",
        ],
    },
)
