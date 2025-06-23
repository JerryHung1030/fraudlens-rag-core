from setuptools import setup, find_packages

setup(
    name="scamshield-ai",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.109.2",
        "uvicorn>=0.27.1",
        "redis>=6.1.0",
        "rq>=2.3.3",
        "aiohttp>=3.11.16",
        "pydantic>=2.6.1",
        "python-multipart>=0.0.9",
        "qdrant-client>=1.14.2",
        "openai>=1.74.0",
        "langchain>=0.3.23",
        "langchain-community>=0.3.21",
        "langchain-core>=0.3.59",
        "langchain-openai>=0.3.16",
        "langchain-qdrant>=0.2.0",
        "click>=8.2.0",
        "requests>=2.32.3",
    ],
    entry_points={
        'console_scripts': [
            'ragcore=rag_core.cli:cli',
        ],
    },
) 