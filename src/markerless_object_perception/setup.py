from setuptools import find_packages, setup

package_name = 'markerless_object_perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            [
                'package.xml',
                'requirements-yolo.txt',
                'requirements-yolo-gpu.txt',
            ],
        ),
        (
            'share/' + package_name + '/config',
            [
                'config/objective32_dataset.toml',
                'config/source_manifest.example.jsonl',
            ],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jiayu Yang',
    maintainer_email='jarrodyang227@gmail.com',
    description=(
        'Closed-set markerless object perception and mask-filtered '
        'stereo localization for Objective 3.2.'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            (
                'synthetic_candidate_publisher = '
                'markerless_object_perception.'
                'synthetic_candidate_publisher:main'
            ),
            (
                'yolo_webcam_demo = '
                'markerless_object_perception.webcam_segmentation_demo:main'
            ),
            (
                'live_candidate_publisher = '
                'markerless_object_perception.live_candidate_publisher:main'
            ),
            (
                'prepare_yolo_dataset = '
                'markerless_object_perception.prepare_yolo_dataset:main'
            ),
            (
                'train_yolo_segmenter = '
                'markerless_object_perception.yolo_training:main'
            ),
        ],
    },
)
