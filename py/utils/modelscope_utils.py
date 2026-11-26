"""ModelScope (魔搭社区) metadata helpers.

ModelScope publishes, for every AIGC-trained LoRA repo, a rich detail payload
that maps almost 1:1 onto what the CivitAI model panel already shows:

======================  ==================================================
CivitAI panel tab       ModelScope source
======================  ==================================================
示例 (example images)   ``MuseInfo.versions[].coverImages[].url``
模型描述 (description)  ``ReadMeContent`` (markdown model card)
触发词 (trigger words)  ``TriggerWords`` / ``versions[].modelVersion.triggerWords``
版本 (version list)     ``MuseInfo.versions[]`` (one entry per training step)
======================  ==================================================

This module normalises that payload once, so both the download path
(:mod:`py.routes.handlers.hf_handlers`) and the versions endpoint can consume
an identical shape.

Cover images live on ``resources.modelscope.cn/cover-images/...`` and are
**hot-linkable** — a plain ``GET`` with no ``Referer`` returns the PNG — so the
frontend gallery can render them directly, exactly like CivitAI CDN URLs.

All functions here are pure except :func:`fetch_detail` (network).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

MODELSCOPE_API_BASE = "https://modelscope.cn/api/v1/models"
MODELSCOPE_WEB_BASE = "https://www.modelscope.cn/models"
MODELSCOPE_REFERER = "https://www.modelscope.cn/"

#: Gallery cap. AIGC LoRA repos publish one distinct sample image per training
#: step (typically 6–10); the cap only guards against pathological repos.
MAX_GALLERY_IMAGES = 12

_OWNER_REPO_IN_URL = re.compile(
    r"modelscope\.(?:cn|com)/models/([^/?#]+/[^/?#]+)", re.IGNORECASE
)
_OWNER_REPO_BARE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")


# ---------------------------------------------------------------------------
# Identifier helpers
# ---------------------------------------------------------------------------


def parse_owner_repo(value: str | None) -> str:
    """Extract ``owner/repo`` from a ModelScope URL or a bare ``owner/repo``.

    Returns ``""`` when *value* carries no recognisable identifier.
    """
    if not value or not isinstance(value, str):
        return ""

    text = value.strip()
    if not text:
        return ""

    match = _OWNER_REPO_IN_URL.search(text)
    if match:
        return match.group(1)

    # Bare "owner/repo" (optionally with a ?version= suffix already stripped).
    text = text.split("?", 1)[0].strip("/")
    if _OWNER_REPO_BARE.match(text):
        return text

    return ""


def build_page_url(owner_repo: str, revision: str | None = None) -> str:
    """Human-facing ModelScope model page URL, optionally pinned to a revision."""
    url = f"{MODELSCOPE_WEB_BASE}/{owner_repo}"
    if revision:
        url += f"?version={revision}"
    return url


def _detail_api_url(owner_repo: str) -> str:
    return f"{MODELSCOPE_API_BASE}/{owner_repo}"


# ---------------------------------------------------------------------------
# Per-version field extraction
# ---------------------------------------------------------------------------


def _version_entry(version: Any) -> Dict[str, Any]:
    """The inner ``modelVersion`` dict of a version, or ``{}``."""
    if not isinstance(version, dict):
        return {}
    inner = version.get("modelVersion")
    return inner if isinstance(inner, dict) else {}


def _version_cover_url(version: Any) -> str:
    """First cover image URL of a version (``""`` when absent)."""
    if not isinstance(version, dict):
        return ""
    for cover in version.get("coverImages") or []:
        if isinstance(cover, dict):
            url = (cover.get("url") or "").strip()
            if url:
                return url
    return ""


def _version_files(version: Any) -> List[str]:
    """Model-weight file names published by a version."""
    if not isinstance(version, dict):
        return []
    stats = version.get("stats")
    if not isinstance(stats, dict):
        return []
    files = stats.get("fileList")
    if not isinstance(files, list):
        return []
    return [f for f in files if isinstance(f, str) and f]


def _version_sizes(version: Any) -> List[int]:
    """Per-file sizes, index-aligned with :func:`_version_files`."""
    if not isinstance(version, dict):
        return []
    stats = version.get("stats")
    if not isinstance(stats, dict):
        return []
    sizes = stats.get("fileSizes")
    if not isinstance(sizes, list):
        return []
    out: List[int] = []
    for size in sizes:
        try:
            out.append(int(size or 0))
        except (TypeError, ValueError):
            out.append(0)
    return out


def _trailing_number(text: str) -> Optional[int]:
    """Last run of digits in *text*, as an int (``None`` when none)."""
    if not text:
        return None
    numbers = re.findall(r"\d+", text)
    if not numbers:
        return None
    try:
        return int(numbers[-1])
    except ValueError:
        return None


def _version_step_number(version: Any) -> Optional[int]:
    """Training-step number of a version, used for ordering.

    Prefers ``modelVersion.showName`` (e.g. ``c1-st10000``), falling back to the
    published file name (e.g. ``Nara_c1-st10000.safetensors``).  Returns ``None``
    when a version carries no number at all (a few repos name files by hand).

    ``versionName`` is deliberately **not** used as a source: it is a timestamp
    (``20260807164403``), so its trailing digits would look like a bogus step.
    """
    inner = _version_entry(version)
    number = _trailing_number(str(inner.get("showName") or ""))
    if number is not None:
        return number

    for name in _version_files(version):
        number = _trailing_number(name.rsplit(".", 1)[0])
        if number is not None:
            return number

    return None


def _parse_trigger_words(raw: Any) -> List[str]:
    """Normalise a trigger-word value that may be a list or a JSON string."""
    values: List[str] = []

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, list):
                values = [v for v in parsed if isinstance(v, str)]
            else:
                values = [text]
        else:
            values = [text]
    elif isinstance(raw, (list, tuple)):
        values = [v for v in raw if isinstance(v, str)]

    cleaned: List[str] = []
    seen: set[str] = set()
    for value in values:
        word = value.strip()
        if not word or word.lower() in ("none", "null", "n/a"):
            continue
        if word.lower() in seen:
            continue
        seen.add(word.lower())
        cleaned.append(word)
    return cleaned


def _epoch_ms_to_iso(value: Any) -> str:
    """ModelScope ``gmtCreate`` epoch-milliseconds to an ISO-8601 UTC string."""
    try:
        millis = int(value)
    except (TypeError, ValueError):
        return ""
    if millis <= 0:
        return ""
    try:
        return datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def _to_int(value: Any, default: int = 0) -> int:
    """Best-effort int coercion (ModelScope mixes str and int across fields)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def build_version_entries(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalise every version of a detail payload into a flat dict.

    Newest first, matching ModelScope's own ordering.  Each entry::

        {
          "version_id": 971239,          # numeric ModelScope version id
          "revision": "20260807164403",   # revision string used to download
          "show_name": "c1-st10000",
          "step": 10000,
          "cover_url": "https://resources.modelscope.cn/...",
          "files": ["Nara_c1-st10000.safetensors"],
          "size_bytes": 234680584,
          "released_at": "2026-08-07T...",
          "trigger_words": ["nara"],
        }
    """
    muse = data.get("MuseInfo") or {}
    versions = muse.get("versions") if isinstance(muse, dict) else None
    if not isinstance(versions, list):
        return []

    entries: List[Dict[str, Any]] = []
    for version in versions:
        if not isinstance(version, dict):
            continue

        inner = _version_entry(version)
        files = _version_files(version)
        sizes = _version_sizes(version)
        total = 0
        if isinstance(version.get("stats"), dict):
            try:
                total = int(version["stats"].get("totalSize") or 0)
            except (TypeError, ValueError):
                total = 0
        if not total:
            total = sum(sizes)

        show_name = str(inner.get("showName") or "").strip()
        revision = str(inner.get("versionName") or "").strip()

        entries.append(
            {
                "version_id": inner.get("id"),
                "revision": revision,
                "show_name": show_name or revision,
                "step": _version_step_number(version),
                "cover_url": _version_cover_url(version),
                "files": files,
                "size_bytes": total,
                "released_at": _epoch_ms_to_iso(inner.get("gmtCreate")),
                "trigger_words": _parse_trigger_words(inner.get("triggerWords")),
            }
        )

    return entries


def build_bundle(data: Dict[str, Any], owner_repo: str) -> Dict[str, Any]:
    """ModelScope detail payload to the ``modelscope`` metadata block.

    Stored verbatim in ``.metadata.json`` under a top-level ``modelscope`` key so
    later reads (versions tab, offline fallback) need no network round-trip.
    """
    muse = data.get("MuseInfo") or {}
    model = muse.get("model") if isinstance(muse, dict) else None
    model = model if isinstance(model, dict) else {}

    base_model = data.get("BaseModel")
    if isinstance(base_model, list):
        base_model = base_model[0] if base_model else ""
    base_model = str(base_model or "")

    try:
        version_count = int(model.get("versionCount") or 0)
    except (TypeError, ValueError):
        version_count = 0

    tags = data.get("Tags")
    tags = [t for t in tags if isinstance(t, str)] if isinstance(tags, list) else []

    return {
        "owner_repo": owner_repo,
        "url": build_page_url(owner_repo),
        "chinese_name": str(data.get("ChineseName") or "").strip(),
        "name": str(data.get("Name") or "").strip(),
        "revision": str(data.get("Revision") or "").strip(),
        "base_model": base_model,
        "trigger_words": _parse_trigger_words(data.get("TriggerWords")),
        "tags": tags,
        "license": str(data.get("LicenseName") or data.get("License") or "").strip(),
        "downloads": _to_int(data.get("Downloads")),
        "stars": _to_int(data.get("Stars")),
        "version_count": version_count or len(build_version_entries(data)),
    }


def build_description_html(data: Dict[str, Any]) -> str:
    """``ReadMeContent`` markdown to the HTML stored in ``modelDescription``.

    Delegates to the same converter the HuggingFace enrichment pipeline uses so
    both sources render identically in the 模型描述 tab.
    """
    readme = data.get("ReadMeContent") or ""
    if not isinstance(readme, str) or not readme.strip():
        return ""
    try:
        from ..services.agent.skills.enrich_hf_metadata.readme_processor import (
            convert_readme_to_html,
        )
    except Exception as exc:  # pragma: no cover - import guard only
        logger.warning("README converter unavailable, storing plain text: %s", exc)
        return ""

    try:
        return convert_readme_to_html(readme)
    except Exception as exc:
        logger.warning("Failed to convert ModelScope README to HTML: %s", exc)
        return ""


def build_gallery_images(
    entries: Iterable[Dict[str, Any]], limit: int = MAX_GALLERY_IMAGES
) -> List[Dict[str, Any]]:
    """Version covers as ``civitai.images``-shaped entries for the 示例 gallery.

    Entries carry a non-empty ``url``, which is what makes
    :meth:`MetadataUpdater.prune_stale_example_images` keep them: that pruner
    only drops remote-less entries whose local ``image_<n>.<ext>`` file is gone.
    """
    images: List[Dict[str, Any]] = []
    for entry in entries:
        url = (entry.get("cover_url") or "").strip()
        if not url:
            continue
        images.append(
            {
                "url": url,
                "nsfwLevel": 0,
                "width": 0,
                "height": 0,
                "type": "image",
                "meta": None,
                "hasMeta": False,
                "hasPositivePrompt": False,
                # Non-standard key: lets the showcase label each sample with the
                # training step it came from. Ignored by CivitAI-shaped readers.
                "sourceStep": entry.get("step") or None,
            }
        )
        if len(images) >= limit:
            break
    return images


def build_version_rows(
    entries: Iterable[Dict[str, Any]],
    *,
    owner_repo: str = "",
    base_model: str = "",
    local_by_filename: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Version rows for the 版本 tab, plus the ids present in the library.

    The row shape mirrors ``_serialize_version`` (the CivitAI versions
    serializer) so the existing row renderer, CSS and badges apply unchanged.

    ``local_by_filename`` maps a lower-cased extension-free file name to
    ``{"file_path": ..., "model_name": ...}`` for every model currently in the
    library; a version is "已在库中" when any of its published files matches.
    """
    local_by_filename = local_by_filename or {}
    rows: List[Dict[str, Any]] = []
    in_library_ids: List[int] = []

    for entry in entries:
        version_id = entry.get("version_id")
        try:
            version_id = int(version_id)
        except (TypeError, ValueError):
            continue

        local_path = ""
        local_name_matched = ""
        for name in entry.get("files") or []:
            key = name.rsplit(".", 1)[0].strip().lower()
            hit = local_by_filename.get(key)
            if hit:
                local_path = hit.get("file_path") or ""
                local_name_matched = name
                break

        is_in_library = bool(local_path)
        if is_in_library:
            in_library_ids.append(version_id)

        files = entry.get("files") or []
        rows.append(
            {
                "versionId": version_id,
                "name": entry.get("show_name") or f"#{version_id}",
                "baseModel": base_model,
                "releasedAt": entry.get("released_at") or "",
                "sizeBytes": int(entry.get("size_bytes") or 0),
                "previewUrl": entry.get("cover_url") or "",
                "isInLibrary": is_in_library,
                "hasBeenDownloaded": is_in_library,
                "shouldIgnore": False,
                "filePath": local_path,
                "fileName": local_name_matched or (files[0] if files else ""),
                "fileCount": len(files) if files else None,
                # ModelScope-only extras. The row renderer ignores unknown keys;
                # the ModelScope branch of the tab uses these for its link and
                # its download action.
                "revision": entry.get("revision") or "",
                "externalUrl": build_page_url(owner_repo, entry.get("revision") or "")
                if owner_repo
                else "",
                "step": entry.get("step"),
                "triggerWords": entry.get("trigger_words") or [],
            }
        )

    return rows, in_library_ids


def build_gallery_and_versions(
    data: Dict[str, Any], owner_repo: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """One-shot convenience: ``(bundle, gallery_images, version_entries)``."""
    entries = build_version_entries(data)
    bundle = build_bundle(data, owner_repo)
    return bundle, build_gallery_images(entries), entries


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


async def fetch_detail(
    session: Any, owner_repo: str, timeout: float = 30.0
) -> Optional[Dict[str, Any]]:
    """Fetch the ModelScope detail payload for *owner_repo*.

    *session* is an ``aiohttp.ClientSession`` supplied by the caller (the plugin
    already keeps a shared one for HF/ModelScope traffic).  Returns ``None`` on
    any failure — callers must treat that as non-fatal.
    """
    if not owner_repo:
        return None

    url = _detail_api_url(owner_repo)
    try:
        import aiohttp
    except Exception as exc:  # pragma: no cover - aiohttp always present in ComfyUI
        logger.warning("aiohttp unavailable, cannot query ModelScope: %s", exc)
        return None

    try:
        async with session.get(
            url,
            headers={"Referer": MODELSCOPE_REFERER, "Accept": "application/json"},
            allow_redirects=True,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                logger.warning(
                    "ModelScope detail API returned HTTP %s for %s", resp.status, url
                )
                return None
            payload = await resp.json(content_type=None)
    except Exception as exc:
        logger.warning("Failed to query ModelScope detail API for %s: %s", owner_repo, exc)
        return None

    if not isinstance(payload, dict):
        return None
    data = payload.get("Data")
    if not isinstance(data, dict):
        logger.warning("ModelScope detail API returned no Data for %s", owner_repo)
        return None
    return data
