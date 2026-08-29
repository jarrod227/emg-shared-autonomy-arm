from setuptools import find_packages, setup

package_name = "assistive_motion"

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
    description=(
        "Shared MoveIt goal construction for the robot command abstraction."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)
