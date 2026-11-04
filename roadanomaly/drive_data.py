"""Access to the RoadAnomaly dataset stored on Google Drive.

The dataset stays on Google Drive (folder ``config.DRIVE_FOLDER_ID``).  This
module finds it, in order of preference:

1. ``ROADANOMALY_DATA_DIR`` environment variable (explicit path);
2. the local cache (``data/`` in the project, or ``/content/roadanomaly_data`` in Colab),
   filled by a previous download;
3. **Google Drive for Desktop** (Windows/macOS): the Drive is mounted as a disk,
   so the files are read directly from Drive with no copy;
4. **Google Drive API** download into the local cache:
   - in Colab, with ``google.colab.auth`` (same as the original notebook);
   - locally, with an OAuth desktop client (``.secrets/client_secret.json``)
     or an API key (``GOOGLE_API_KEY``) when the folder is shared publicly.

Files already present in the cache with the right size are not downloaded again.
"""

from __future__ import annotations

import concurrent.futures
import html
import io
import json
import os
import re
import string
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import config

FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


@dataclass(frozen=True)
class DatasetLocation:
    data_dir: Path  # folder containing frames/ and frame_list.json
    source: str  # human-readable description of where it was found


# ---------------------------------------------------------------------------
# Local discovery
# ---------------------------------------------------------------------------
def is_valid_data_dir(path: Path | str | None) -> bool:
    if not path:
        return False
    path = Path(path)
    return (path / "frames").is_dir() and (path / "frame_list.json").is_file()


def missing_files(data_dir: Path | str) -> list[str]:
    """Image / semantic-mask files listed in frame_list.json that are absent from ``data_dir``."""
    data_dir = Path(data_dir)
    try:
        names = json.loads((data_dir / "frame_list.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ["frame_list.json"]
    missing = []
    for name in names:
        image = data_dir / "frames" / name
        for path in (image, image.with_suffix(".labels") / config.SEMANTIC_MASK_NAME):
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path.relative_to(data_dir)))
    return missing


def locate_data_dir(root: Path | str | None, require_complete: bool = True) -> Path | None:
    """Return the dataset folder inside ``root`` (tolerates the nested layout).

    With ``require_complete`` a copy lacking images or masks is ignored, so a partial
    download is never mistaken for the dataset.
    """
    if not root:
        return None
    root = Path(root)
    for candidate in (
        root,
        root / "RoadAnomaly",
        root / "RoadAnomaly" / "RoadAnomaly",
        root / config.DRIVE_FOLDER_NAME / "RoadAnomaly",
        root / config.DRIVE_FOLDER_NAME / "RoadAnomaly" / "RoadAnomaly",
    ):
        if is_valid_data_dir(candidate) and (not require_complete or not missing_files(candidate)):
            return candidate
    return None


def _drive_desktop_roots() -> list[Path]:
    """Folders where Google Drive for Desktop may expose the shared Drive folder."""
    roots: list[Path] = []
    env_root = os.environ.get("ROADANOMALY_DRIVE_PATH")
    if env_root:
        roots.append(Path(env_root))

    mount_points: list[Path] = []
    if sys.platform.startswith("win"):
        mount_points = [Path(f"{letter}:\\") for letter in string.ascii_uppercase[3:]]
    elif sys.platform == "darwin":
        cloud = Path.home() / "Library" / "CloudStorage"
        if cloud.is_dir():
            mount_points = sorted(cloud.glob("GoogleDrive-*"))
    mount_points.append(Path.home() / "Google Drive")
    mount_points.append(Path("/content/drive"))  # Colab drive.mount()

    for mount in mount_points:
        try:
            if not mount.is_dir():
                continue
        except OSError:
            continue
        # Folder shared with you and added as a shortcut -> exposed by ID.
        for source in config.DRIVE_SOURCES:
            roots.append(mount / ".shortcut-targets-by-id" / source["id"] / config.DRIVE_FOLDER_NAME)
            roots.append(mount / ".shortcut-targets-by-id" / source["id"])
        for my_drive in ("My Drive", "Mon Drive", "MyDrive", "Mi unidad", "Meine Ablage"):
            roots.append(mount / my_drive / config.DRIVE_FOLDER_NAME)
        roots.append(mount / config.DRIVE_FOLDER_NAME)
    return roots


def source_cache_dir(source: dict) -> Path:
    """Each Drive source has its own cache folder so versions never mix."""
    return config.DATA_CACHE_DIR / f"drive_{source['id'][:8]}"


def find_local_dataset() -> DatasetLocation | None:
    """Look for an already available copy of the dataset (no network access)."""
    if config.DATA_DIR_OVERRIDE:
        found = locate_data_dir(config.DATA_DIR_OVERRIDE)
        if found:
            return DatasetLocation(found, "ROADANOMALY_DATA_DIR")

    for source in config.DRIVE_SOURCES:
        found = locate_data_dir(source_cache_dir(source))
        if found:
            return DatasetLocation(found, f"local cache of Google Drive folder {source['id']} ({found})")
    found = locate_data_dir(config.DATA_CACHE_DIR)
    if found:
        return DatasetLocation(found, f"local cache of the Google Drive folder ({config.DATA_CACHE_DIR})")

    for root in _drive_desktop_roots():
        try:
            found = locate_data_dir(root)
        except OSError:
            found = None
        if found:
            return DatasetLocation(found, f"Google Drive for Desktop ({root})")
    return None


# ---------------------------------------------------------------------------
# Google Drive API
# ---------------------------------------------------------------------------
def build_drive_service(interactive: bool = True):
    """Create an authenticated Drive v3 client for Colab or for a local machine."""
    try:
        from googleapiclient.discovery import build
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Install the Google API client: pip install google-api-python-client "
            "google-auth google-auth-oauthlib"
        ) from error

    if config.IN_COLAB:
        import google.auth
        from google.colab import auth  # type: ignore

        auth.authenticate_user()
        credentials, _ = google.auth.default()
        return build("drive", "v3", credentials=credentials, cache_discovery=False)

    if config.GOOGLE_API_KEY and not config.OAUTH_CLIENT_SECRET.is_file():
        # Works only if the Drive folder is shared as "anyone with the link".
        return build("drive", "v3", developerKey=config.GOOGLE_API_KEY, cache_discovery=False)

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    credentials = None
    if config.OAUTH_TOKEN.is_file():
        credentials = Credentials.from_authorized_user_file(str(config.OAUTH_TOKEN), DRIVE_SCOPES)

    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())

    if not credentials or not credentials.valid:
        if not config.OAUTH_CLIENT_SECRET.is_file():
            raise FileNotFoundError(
                "No Google credentials found. Either:\n"
                "  - install Google Drive for Desktop and add the shared folder to 'My Drive', or\n"
                f"  - put an OAuth *Desktop app* client file at {config.OAUTH_CLIENT_SECRET}, or\n"
                "  - set GOOGLE_API_KEY if the folder is shared with 'anyone with the link', or\n"
                "  - download the folder manually and set ROADANOMALY_DATA_DIR.\n"
                "See README.md, section 'Dataset access'."
            )
        if not interactive:
            raise PermissionError("Google sign-in required: run the notebook data-access cell once.")
        from google_auth_oauthlib.flow import InstalledAppFlow

        flow = InstalledAppFlow.from_client_secrets_file(str(config.OAUTH_CLIENT_SECRET), DRIVE_SCOPES)
        credentials = flow.run_local_server(port=0)
        config.SECRETS_DIR.mkdir(parents=True, exist_ok=True)
        config.OAUTH_TOKEN.write_text(credentials.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _list_children(service, folder_id: str) -> list[dict]:
    items, page_token = [], None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType, size, shortcutDetails)",
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            return items


def _find_child_folder(service, parent_id: str, name: str) -> str | None:
    for item in _list_children(service, parent_id):
        if item["name"] == name and item["mimeType"] == FOLDER_MIME:
            return item["id"]
        if (
            item["name"] == name
            and item["mimeType"] == SHORTCUT_MIME
            and item.get("shortcutDetails", {}).get("targetMimeType") == FOLDER_MIME
        ):
            return item["shortcutDetails"]["targetId"]
    return None


def download_drive_folder(service, folder_id: str, destination: Path, verbose: bool = True) -> dict:
    """Recursively mirror a Drive folder into ``destination`` (incremental)."""
    from googleapiclient.http import MediaIoBaseDownload

    destination.mkdir(parents=True, exist_ok=True)
    stats = {"downloaded": 0, "skipped": 0, "ignored": 0}

    for item in _list_children(service, folder_id):
        name, mime = item["name"], item["mimeType"]
        if name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError(f"Unsupported file name on Drive: {name!r}")
        local_path = destination / name

        if mime == SHORTCUT_MIME:
            details = item.get("shortcutDetails", {})
            if details.get("targetMimeType") == FOLDER_MIME:
                mime, item = FOLDER_MIME, {**item, "id": details["targetId"]}
            else:
                stats["ignored"] += 1
                continue

        if mime == FOLDER_MIME:
            sub = download_drive_folder(service, item["id"], local_path, verbose=False)
            for key in stats:
                stats[key] += sub[key]
            continue

        if mime.startswith("application/vnd.google-apps."):
            stats["ignored"] += 1  # native Google Docs cannot be downloaded as-is
            continue

        expected_size = int(item.get("size", -1))
        if local_path.is_file() and local_path.stat().st_size == expected_size:
            stats["skipped"] += 1
            continue

        buffer = io.BytesIO()
        request = service.files().get_media(fileId=item["id"], supportsAllDrives=True)
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        temporary = local_path.with_suffix(local_path.suffix + ".part")
        temporary.write_bytes(buffer.getvalue())
        temporary.replace(local_path)
        stats["downloaded"] += 1

    if verbose:
        print(
            f"Drive sync -> {destination}: {stats['downloaded']} downloaded, "
            f"{stats['skipped']} already up to date, {stats['ignored']} ignored."
        )
    return stats


# ---------------------------------------------------------------------------
# Keyless download of a publicly shared folder ("anyone with the link")
# ---------------------------------------------------------------------------
_ENTRY_RE = re.compile(r'<div class="flip-entry" id="entry-([^"]+)"(.*?)flip-entry-title">(.*?)</div>', re.S)


def _http_get(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "roadanomaly-dataset-sync"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise ConnectionError(f"Cannot reach {url}: {last_error}")


def list_public_folder(folder_id: str) -> list[dict]:
    """List a public Drive folder through its embedded view (no sign-in required)."""
    page = _http_get(f"https://drive.google.com/embeddedfolderview?id={folder_id}").decode("utf-8", "replace")
    items = []
    for match in _ENTRY_RE.finditer(page):
        entry_id, body, title = match.groups()
        items.append({
            "id": entry_id,
            "name": html.unescape(re.sub(r"<[^>]+>", "", title)).strip(),
            "is_folder": "/drive/folders/" in body,
        })
    return items


def is_public_folder(folder_id: str) -> bool:
    try:
        return bool(list_public_folder(folder_id))
    except (ConnectionError, OSError):
        return False


def _download_public_file(file_id: str, destination: Path) -> None:
    data = _http_get(f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t")
    if data[:15].lower().startswith((b"<!doctype html", b"<html")):
        raise PermissionError(f"Drive returned a web page instead of the file (id {file_id}); the file may not be public.")
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(destination)


def download_public_folder(folder_id: str, destination: Path, verbose: bool = True, workers: int = 8) -> dict:
    """Mirror a publicly shared Drive folder (incremental: existing non-empty files are kept)."""
    destination.mkdir(parents=True, exist_ok=True)
    stats = {"downloaded": 0, "skipped": 0, "ignored": 0}
    jobs: list[tuple[str, Path]] = []

    def collect(current_id: str, target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        for item in list_public_folder(current_id):
            name = item["name"]
            if name in {".", ".."} or "/" in name or "\\" in name:
                raise ValueError(f"Unsupported file name on Drive: {name!r}")
            if item["is_folder"]:
                collect(item["id"], target / name)
            elif (target / name).is_file() and (target / name).stat().st_size > 0:
                stats["skipped"] += 1
            else:
                jobs.append((item["id"], target / name))

    collect(folder_id, destination)
    if verbose:
        print(f"Public Drive sync: {len(jobs)} file(s) to download, {stats['skipped']} already present.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for done, _ in enumerate(pool.map(lambda job: _download_public_file(*job), jobs), start=1):
            stats["downloaded"] = done
            if verbose and done % 50 == 0:
                print(f"  {done}/{len(jobs)} files")
    if verbose:
        print(f"Drive sync -> {destination}: {stats['downloaded']} downloaded, {stats['skipped']} already up to date.")
    return stats


def _resolve_public_subpath(root_id: str, subpath) -> str | None:
    folder_id = root_id
    for part in subpath:
        children = {i["name"]: i for i in list_public_folder(folder_id) if i["is_folder"]}
        if part not in children:
            return None
        folder_id = children[part]["id"]
    return folder_id


def _download_source(source: dict, interactive: bool, verbose: bool) -> Path:
    """Download one Drive source into the cache and return its (possibly incomplete) folder."""
    folder_id, subpath = source["id"], tuple(source["subpath"])
    destination = source_cache_dir(source).joinpath(*subpath)

    if not config.IN_COLAB and not config.OAUTH_CLIENT_SECRET.is_file() and is_public_folder(folder_id):
        target_id = _resolve_public_subpath(folder_id, subpath)
        if target_id is None:
            raise FileNotFoundError(f"Sub-folder {'/'.join(subpath)!r} not found in Drive folder {folder_id}.")
        if verbose:
            print(f"Drive folder {folder_id} is publicly shared: downloading without sign-in.")
        download_public_folder(target_id, destination, verbose=verbose)
    else:
        service = build_drive_service(interactive=interactive)
        info = service.files().get(fileId=folder_id, fields="id, name, mimeType", supportsAllDrives=True).execute()
        if info["mimeType"] != FOLDER_MIME:
            raise ValueError(f"{folder_id} is not a Drive folder.")
        if verbose:
            print(f"Google Drive folder: {info['name']} ({info['id']})")
        current_id = folder_id
        for part in subpath:
            child = _find_child_folder(service, current_id, part)
            if child is None:
                raise FileNotFoundError(f"Sub-folder {part!r} not found in the Drive folder.")
            current_id = child
        download_drive_folder(service, current_id, destination, verbose=verbose)

    found = locate_data_dir(destination, require_complete=False)
    if found is None:
        raise FileNotFoundError(f"Download finished but frames/ or frame_list.json is missing in {destination}.")
    return found


def download_dataset_from_drive(interactive: bool = True, verbose: bool = True) -> Path:
    """Download the RoadAnomaly dataset from the first Drive source that is complete.

    Publicly shared folders are fetched without credentials; otherwise the Drive API
    (Colab sign-in, OAuth desktop client or API key) is used.
    """
    problems = []
    for source in config.DRIVE_SOURCES:
        try:
            data_dir = _download_source(source, interactive, verbose)
        except (OSError, ValueError, ConnectionError, PermissionError) as error:
            problems.append(f"{source['id']}: {error}")
            continue
        missing = missing_files(data_dir)
        if not missing:
            return data_dir
        problems.append(f"{source['id']}: incomplete copy, {len(missing)} file(s) missing (e.g. {missing[0]})")
        if verbose:
            print(f"Drive folder {source['id']} is incomplete ({len(missing)} missing files): trying the next source.")
    raise FileNotFoundError("No complete copy of the dataset found on Google Drive:\n  " + "\n  ".join(problems))


def ensure_dataset(allow_download: bool = True, interactive: bool = True, verbose: bool = True) -> DatasetLocation:
    """Return the dataset location, downloading it from Google Drive if needed."""
    location = find_local_dataset()
    if location is None:
        if not allow_download:
            raise FileNotFoundError(
                "RoadAnomaly dataset not found locally. Run the data-access cell of the notebook "
                "(Google Drive download) or set ROADANOMALY_DATA_DIR."
            )
        data_dir = download_dataset_from_drive(interactive=interactive, verbose=verbose)
        location = DatasetLocation(data_dir, f"downloaded from Google Drive into {config.DATA_CACHE_DIR}")
    if verbose:
        print(f"Dataset folder: {location.data_dir}")
        print(f"Source: {location.source}")
    return location
