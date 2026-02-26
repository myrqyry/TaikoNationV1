from setuptools import setup, find_packages

setup(
    name="taikonation",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'torch>=1.12.0',
        'numpy>=1.21.0',
        'librosa>=0.9.2',
        'fastapi>=0.95.0',
        'uvicorn[standard]>=0.22.0',
        'python-socketio[asyncio]>=5.9.0',
        'pyyaml>=5.4.0',
        'marshmallow>=3.14.0',
    ],
    python_requires='>=3.8',
)
