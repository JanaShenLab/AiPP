from setuptools import setup, find_packages

setup(
    name="aipp",
    version="0.1.0",
    description="",
    author="Guy 'Wayyne' Dayhoff",
    author_email="gdayhoff@rx.umaryland.edu",
    packages=find_packages(where=".", include=["aipp", "aipp.*"]),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=[],
    entry_points={
        "console_scripts": [
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
