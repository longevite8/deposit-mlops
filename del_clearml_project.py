from clearml.backend_api.session.client import APIClient
from config import (
    PROJECT_TEMPLATE,
    PROJECT_PIPELINE,
    PROJECT_DATASET,
)


def cleanup_clearml_resources():
    client = APIClient()

    # Danh sách các project prefix cần xóa sạch
    target_project_names = [
        PROJECT_TEMPLATE,
        PROJECT_PIPELINE,
        PROJECT_DATASET,
    ]

    print(f"🚀 Starting cleanup for projects: {target_project_names}")

    # Lấy tất cả projects để tìm IDs
    all_projects = client.projects.get_all()

    # Lọc các project khớp với danh sách mục tiêu (hoặc là con của chúng)
    target_projects = [
        p
        for p in all_projects
        if any(p.name.startswith(name) for name in target_project_names)
    ]

    if not target_projects:
        print("ℹ️ No matching projects found.")
        return

    for project in target_projects:
        print(f"\n--- Processing Project: {project.name} (ID: {project.id}) ---")

        # 1. Xóa tất cả Models trong project này
        models = client.models.get_all(project=[project.id])
        if models:
            print(f"  🗑️ Deleting {len(models)} models...")
            for model in models:
                client.models.delete(model=model.id)

        # 2. Xóa tất cả Tasks (bao gồm Experiments, Pipelines, và Datasets)
        tasks = client.tasks.get_all(project=[project.id])
        if tasks:
            print(f"  🗑️ Deleting {len(tasks)} tasks...")
            for task in tasks:
                # force=True để xóa cả các task đang chạy hoặc ở trạng thái đặc biệt
                client.tasks.delete(task=task.id, force=True)

        # 3. Cuối cùng xóa bản thân Project Folder
        print(f"  🗑️ Deleting project folder: {project.name}")
        # ✅ FIX: Thêm force=True để xóa sạch các ràng buộc Dataset
        client.projects.delete(project=project.id, force=True)

    print("\n✅ Cleanup completed successfully.")


if __name__ == "__main__":
    # Lưu ý: Thao tác này không thể hoàn tác. Hãy cẩn trọng trước khi chạy.
    confirm = input(
        "⚠️ Are you sure you want to delete ALL resources in these projects? (y/N): "
    )
    if confirm.lower() == "y":
        cleanup_clearml_resources()
    else:
        print("❌ Cleanup cancelled.")
