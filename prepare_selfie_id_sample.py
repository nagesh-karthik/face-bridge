import csv
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


REPO_DIR = Path("kaggle_downloads/Selfie-and-ID-Dataset")
BASE_URL = "https://huggingface.co/datasets/ud-biometrics/Selfie-and-ID-Dataset/resolve/main"
IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


def list_repo_files():
    result = subprocess.run(
        ["git", "-C", str(REPO_DIR), "ls-tree", "-r", "--name-only", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def download_file(repo_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return

    url = f"{BASE_URL}/{urllib.parse.quote(repo_path)}?download=true"
    with urllib.request.urlopen(url) as response:
        output_path.write_bytes(response.read())


def main():
    if not REPO_DIR.exists():
        raise SystemExit(f"Expected partial repo at {REPO_DIR}")

    raw_dir = Path("kaggle_downloads/public_sample_raw")
    id_dir = Path("data/id")
    selfie_dir = Path("data/selfie")
    id_dir.mkdir(parents=True, exist_ok=True)
    selfie_dir.mkdir(parents=True, exist_ok=True)

    files = [
        path for path in list_repo_files()
        if path.startswith("files/") and Path(path).suffix.lower() in IMAGE_EXTENSIONS
    ]

    people = {}
    for repo_path in files:
        _, person_id, filename = repo_path.split("/", 2)
        people.setdefault(person_id, {"id": [], "selfie": []})
        if filename.lower().startswith("id_"):
            people[person_id]["id"].append(repo_path)
        elif filename.lower().startswith("selfie_"):
            people[person_id]["selfie"].append(repo_path)

    for repo_path in files:
        download_file(repo_path, raw_dir / repo_path)

    rows = []
    sorted_people = sorted(people, key=lambda value: int(value))
    for person_id in sorted_people:
        person = people[person_id]
        ids = sorted(person["id"])
        selfies = sorted(person["selfie"])
        if not ids or not selfies:
            continue

        id_source = raw_dir / ids[0]
        id_target = id_dir / f"person{int(person_id):03d}_id{Path(ids[0]).suffix.lower()}"
        shutil.copy2(id_source, id_target)

        copied_selfies = []
        for index, selfie_path in enumerate(selfies, start=1):
            selfie_source = raw_dir / selfie_path
            selfie_target = selfie_dir / (
                f"person{int(person_id):03d}_selfie{index:02d}{Path(selfie_path).suffix.lower()}"
            )
            shutil.copy2(selfie_source, selfie_target)
            copied_selfies.append(selfie_target)

        for selfie_target in copied_selfies:
            rows.append([f"./{id_target}", f"./{selfie_target}", 1])

        next_person_id = sorted_people[(sorted_people.index(person_id) + 1) % len(sorted_people)]
        impostor_selfies = sorted(people[next_person_id]["selfie"])
        if impostor_selfies:
            impostor_source = raw_dir / impostor_selfies[0]
            impostor_target = selfie_dir / (
                f"person{int(next_person_id):03d}_impostor_for_{int(person_id):03d}"
                f"{Path(impostor_selfies[0]).suffix.lower()}"
            )
            shutil.copy2(impostor_source, impostor_target)
            rows.append([f"./{id_target}", f"./{impostor_target}", 0])

    with Path("pairs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id_path", "selfie_path", "label"])
        writer.writerows(rows)

    print(f"Downloaded {len(files)} jpg/jpeg files")
    print(f"Prepared {len(rows)} verification pairs in pairs.csv")


if __name__ == "__main__":
    main()
