import json
import shutil
from pathlib import Path

class WorkspaceManager:
    def __init__(self, base_path="storage/workspaces"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def list_workspaces(self):
        return sorted(
            [
                folder.name
                for folder in self.base_path.iterdir()
                if folder.is_dir()
            ]
        )

    def exists(self, workspace):
        return self.get_workspace_path(workspace).exists()

    def create_workspace(self, workspace):
        workspace_path = self.get_workspace_path(workspace)

        (workspace_path / "uploads").mkdir(parents=True, exist_ok=True)
        (workspace_path / "qdrant").mkdir(parents=True, exist_ok=True)

        metadata = workspace_path / "metadata.json"
        chat = workspace_path / "chat.json"

        if not metadata.exists():
            self.save_metadata(
                workspace,
                {
                    "name": workspace,
                    "documents": []
                }
            )

        if not chat.exists():
            self.save_chat(workspace, [])

    def delete_workspace(self, workspace):
        workspace_path = self.get_workspace_path(workspace)

        if workspace_path.exists():
            shutil.rmtree(workspace_path)

    def get_workspace_path(self, workspace):
        return self.base_path / workspace

    def get_upload_path(self, workspace):
        path = self.get_workspace_path(workspace) / "uploads"
        path.mkdir(exist_ok=True)
        return path

    def get_qdrant_path(self, workspace):
        path = self.get_workspace_path(workspace) / "qdrant"
        path.mkdir(exist_ok=True)
        return path

    def load_metadata(self, workspace):
        file = self.get_workspace_path(workspace) / "metadata.json"

        if file.exists():
            with open(file, "r", encoding="utf-8") as f:
                return json.load(f)

        return {
            "name": workspace,
            "documents": []
        }

    def save_metadata(self, workspace, metadata):
        file = self.get_workspace_path(workspace) / "metadata.json"

        with open(file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    def add_document(self, workspace, filename):
        metadata = self.load_metadata(workspace)

        if filename not in metadata["documents"]:
            metadata["documents"].append(filename)

        self.save_metadata(workspace, metadata)

    def load_chat(self, workspace):
        file = self.get_workspace_path(workspace) / "chat.json"

        if file.exists():
            with open(file, "r", encoding="utf-8") as f:
                return json.load(f)

        return []

    def save_chat(self, workspace, history):
        file = self.get_workspace_path(workspace) / "chat.json"

        with open(file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

    def is_processed(self, workspace):
        qdrant = self.get_qdrant_path(workspace)

        return any(qdrant.iterdir())