from setuptools import find_packages, setup

setup(
    name="llm-agent",
    version="0.1.0",
    packages=find_packages(where="src"),  # Specify the source directory
    package_dir={"": "src"},  # Tell setuptools where to find the packages
    install_requires=["litellm", "pydantic", "pyyaml", "pytest", "pytest-asyncio"],
    python_requires=">=3.8",
    description="LLM agent with structured output capabilities",
    author="",
    author_email="",
)
