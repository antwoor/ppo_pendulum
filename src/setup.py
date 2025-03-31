from setuptools import setup, find_packages

setup(
    name="ppo",
    version="0.1.0",
    packages=find_packages(where="ppo"),
    package_dir={"": "src"},
    
    install_requires=[
        "numpy>=1.26.4",
        "torch>=2.5.1",
        "mujoco>=3.3.0",  # Если используете MuJoCo
    ],
    
    # Необязательные зависимости
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
        ],
    },
    
    # Метаданные
    author="Your Name",
    author_email="your.email@example.com",
    description="PPO implementation for inverted pendulum control",
    license="MIT",
    keywords="reinforcement-learning ppo mujoco",
    
    # Включение данных (например, весов моделей)
    #package_data={
    #    "ppo": ["*.pth"],
    #},
)