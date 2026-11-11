
from pathlib import Path

# Project folders
folders = [
    "app",
    "app/templates",
    "app/static/css",
    "app/static/js",
    "worker",
    "infra",
    "scripts",
    "docs",
]

# Project files
files = [
    # Root files
    "requirements.txt",
    ".env",
    ".env.example",
    ".gitignore",
    "README.md",

    # Backend
    "app/__init__.py",
    "app/main.py",
    "app/config.py",
    "app/db.py",
    "app/aws.py",
    "app/vector.py",
    "app/rag.py",
    "app/memory.py",
    "app/guardrails.py",
    "app/graph.py",
    "app/observability.py",

    # Frontend
    "app/templates/index.html",
    "app/static/css/styles.css",
    "app/static/js/app.js",

    # Document ingestion
    "worker/__init__.py",
    "worker/ingest.py",

    # Database
    "infra/schema.sql",
    "scripts/init_db.py",

    # Documentation
    "docs/SETUP.md",
]


def create_structure():
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

    for file in files:
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    print("Project folders and files created successfully!")
    print(f"Total folders: {len(folders)}")
    print(f"Total files: {len(files)}")


if __name__ == "__main__":
    create_structure()