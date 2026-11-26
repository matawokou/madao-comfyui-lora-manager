"""Handlers for Hugging Face model listing and download.

Minimal MVP implementation — uses direct HTTP to the HF API for file
listing and the project's existing aiohttp-based Downloader for
downloading.  No huggingface_hub dependency required.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import aiohttp
from aiohttp import web

from ...config import config
from ...services.downloader import (
    DownloadProgress,
    get_downloader,
)
from ...services.aria2_downloader import Aria2Downloader
from ...services.settings_manager import get_settings_manager
from ...services.service_registry import ServiceRegistry
from ...services.websocket_manager import ws_manager
from ...utils.constants import MODEL_FILE_EXTENSIONS
from ...utils.file_utils import find_preview_file
from ...utils.lora_metadata import extract_lora_metadata
from ...utils.metadata_manager import MetadataManager
from ...utils.models import LoraMetadata, CheckpointMetadata, EmbeddingMetadata
from ...utils.modelscope_utils import (
    build_description_html,
    build_gallery_and_versions,
    build_page_url,
    build_version_entries,
    build_version_rows,
    fetch_detail,
    parse_owner_repo,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_CLASS = LoraMetadata
_DEFAULT_SCANNER_GETTER = "get_lora_scanner"

# Shared aiohttp session for HF API calls (created on first use)
_hf_api_session: aiohttp.ClientSession | None = None


async def _get_hf_api_session() -> aiohttp.ClientSession:
    """Get or create the shared aiohttp session for HF API calls."""
    global _hf_api_session  # needed because we reassign the module-level name
    if _hf_api_session is None or _hf_api_session.closed:
        _hf_api_session = aiohttp.ClientSession(
            headers={"User-Agent": "ComfyUI-LoRA-Manager/1.0"},
            timeout=aiohttp.ClientTimeout(total=30),
        )
    return _hf_api_session


async def close_hf_api_session() -> None:
    """Close the shared HF API session, if it was ever created."""
    global _hf_api_session
    if _hf_api_session is not None and not _hf_api_session.closed:
        await _hf_api_session.close()
        _hf_api_session = None


def _infer_model_type(model_root: str) -> tuple[Any, str]:
    """Determine model class and scanner by matching ``model_root`` against the
    configured root paths for each model type (from ``Config``).

    The ``model_root`` value comes from the frontend's model-root dropdown,
    which is populated from the current page's scanner roots.  By checking
    which scanner's root list it belongs to, we avoid fragile heuristics
    like substring-matching path names.
    """
    norm = os.path.normpath(model_root).replace(os.sep, "/")

    # LoRA roots
    for p in (config.loras_roots or []) + (config.extra_loras_roots or []):
        if os.path.normpath(p).replace(os.sep, "/") == norm:
            return LoraMetadata, "get_lora_scanner"

    # Checkpoint / UNet roots
    for p in (
        (config.checkpoints_roots or [])
        + (config.extra_checkpoints_roots or [])
        + (config.unet_roots or [])
        + (config.extra_unet_roots or [])
    ):
        if os.path.normpath(p).replace(os.sep, "/") == norm:
            return CheckpointMetadata, "get_checkpoint_scanner"

    # Embedding roots
    for p in (config.embeddings_roots or []) + (config.extra_embeddings_roots or []):
        if os.path.normpath(p).replace(os.sep, "/") == norm:
            return EmbeddingMetadata, "get_embedding_scanner"

    # Fallback — should not happen in normal use
    logger.warning(
        "Could not determine model type for root '%s'; defaulting to LoRA",
        model_root,
    )
    return _DEFAULT_MODEL_CLASS, _DEFAULT_SCANNER_GETTER


async def _save_hf_metadata(dest_path: str, repo: str, model_root: str) -> None:
    """Create a proper .metadata.json and add the model to the scanner cache.

    Uses ``MetadataManager.create_default_metadata()`` which computes the
    SHA256 hash, extracts safetensors header metadata (base_model), and
    produces a fully-populated ``LoraMetadata`` (or ``CheckpointMetadata`` /
    ``EmbeddingMetadata``) object.  We then overlay HF-specific fields and
    register the model in the in-memory scanner cache so it appears
    immediately without a full filesystem walk.
    """
    try:
        hf_url = f"https://huggingface.co/{repo}"
        # Derive the model class from the file's real location (the caller's
        # model_root is often a relative key that matches no configured root).
        matched_root = _find_matching_root(os.path.dirname(dest_path))
        model_class, scanner_getter_name = _infer_model_type(matched_root or model_root)

        # 1. Create proper metadata (computes SHA256, reads safetensors headers)
        metadata = await MetadataManager.create_default_metadata(
            dest_path, model_class=model_class
        )
        if metadata is None:
            logger.warning("create_default_metadata returned None for %s", dest_path)
            return

        # 2. Overlay HF-specific fields
        metadata._unknown_fields["hf_url"] = hf_url
        metadata.from_civitai = False  # HF models are not from CivitAI

        metadata_dict = metadata.to_dict()
        if "trainedWords" in metadata_dict and not metadata_dict["trainedWords"]:
            del metadata_dict["trainedWords"]

        # 3. Save metadata atomically
        await MetadataManager.save_metadata(dest_path, metadata_dict)
        logger.info("Saved HF metadata (with hf_url) for %s", dest_path)

        # 4. Register in the scanner cache with the folder derived from the
        #    file's real location (see _resolve_relative_folder).  Computing the
        #    folder from the caller's `model_root` is unreliable — the frontend
        #    sends a relative key (e.g. "loras"), so the entry would land with
        #    folder="" and the card would never appear in its real folder.
        folder = _resolve_relative_folder(dest_path)
        scanner_getter = getattr(ServiceRegistry, scanner_getter_name, None)
        if scanner_getter is not None:
            scanner = await scanner_getter()
            if scanner is not None:
                metadata_dict = metadata.to_dict()
                metadata_dict["hf_url"] = hf_url
                await scanner.add_model_to_cache(metadata_dict, folder)
                logger.info("Added %s to scanner cache (folder=%s)", dest_path, folder)

    except Exception as exc:
        logger.warning("Failed to save HF metadata for %s: %s", dest_path, exc)


def _find_matching_root(dest_dir: str) -> str | None:
    """Walk up *dest_dir* to find which configured scanner root it belongs to."""
    norm = os.path.normpath(dest_dir).replace(os.sep, "/")
    all_roots = []
    for root_list in (
        config.loras_roots or [],
        config.extra_loras_roots or [],
        config.checkpoints_roots or [],
        config.extra_checkpoints_roots or [],
        config.unet_roots or [],
        config.extra_unet_roots or [],
        config.embeddings_roots or [],
        config.extra_embeddings_roots or [],
    ):
        all_roots.extend([os.path.normpath(p).replace(os.sep, "/") for p in root_list])
    # Find the longest matching prefix
    match: str | None = None
    for root in all_roots:
        if norm.startswith(root):
            if match is None or len(root) > len(match):
                match = root
    return match


def _resolve_relative_folder(dest_path: str, fallback: str = "") -> str:
    """Relative folder of a downloaded file within its configured scanner root.

    The folder is derived from the file's **real** location (via
    :func:`_find_matching_root`, which normalises separators on both sides)
    rather than from the caller-supplied ``model_root``.  The frontend sends
    ``model_root`` as a relative key (e.g. ``loras``) that matches no configured
    absolute root, so computing the folder from it yields ``""`` and the cached
    entry never shows up in its actual folder view.

    Returns a forward-slash relative path (``""`` for the root itself), or
    *fallback* when the file is not within any configured root.
    """
    model_dir = os.path.dirname(dest_path)
    matched_root = _find_matching_root(model_dir)
    if not matched_root:
        return fallback
    rel = os.path.relpath(
        os.path.normpath(model_dir), os.path.normpath(matched_root)
    )
    if rel == ".":
        return ""
    return rel.replace(os.sep, "/")


async def _add_to_scanner_cache(dest_path: str, metadata: dict[str, Any]) -> None:
    model_dir = os.path.dirname(dest_path)
    model_root = _find_matching_root(model_dir)
    if not model_root:
        raise ValueError(f"File path {dest_path} is not within any configured scanner root")
    scanner_getter_name = _infer_model_type(model_root)[1]
    scanner_getter = getattr(ServiceRegistry, scanner_getter_name, None)
    if scanner_getter is None:
        raise RuntimeError(f"Scanner getter '{scanner_getter_name}' not found in ServiceRegistry")
    scanner = await scanner_getter()
    if scanner is None:
        raise RuntimeError(f"Scanner '{scanner_getter_name}' returned None")
    await scanner.update_single_model_cache(dest_path, dest_path, metadata)


def _extract_cover_url(entry: Any) -> str:
    """Normalise a ModelScope ``coverImages`` entry into a plain URL string.

    Entries are either bare URL strings or ``{"url": ...}`` dicts depending on
    the endpoint, so both shapes are accepted.
    """
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        for key in ("url", "Url", "URL", "imageUrl"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _version_file_list(version: dict[str, Any]) -> list[str]:
    """Base names of the weight files a ModelScope version belongs to."""
    stats = version.get("stats") or {}
    if not isinstance(stats, dict):
        stats = {}
    files = stats.get("fileList") or stats.get("FileList") or []
    if isinstance(files, str):
        files = [files]
    if not isinstance(files, list):
        files = []
    return [os.path.basename(str(f)) for f in files]


def _version_cover(version: dict[str, Any]) -> str:
    """First usable cover URL of a ModelScope version (``""`` if none)."""
    for entry in version.get("coverImages") or []:
        url = _extract_cover_url(entry)
        if url:
            return url
    return ""


def _version_number(version: dict[str, Any]) -> int | None:
    """The trailing number of a version's weight file name.

    ModelScope names each version after its training progress
    (``Chingmy_c1-st10000.safetensors`` → ``10000``, the step count), and the
    site's own default tab is the **highest** one.  The *last* digit run is
    used rather than the max of all of them, so a name like
    ``Krea2-..._v10_st6000`` still yields the step count, not the ``2``.
    """
    for name in _version_file_list(version):
        stem = os.path.splitext(name)[0]
        numbers = re.findall(r"\d+", stem)
        if numbers:
            return int(numbers[-1])
    return None


def _pick_modelscope_cover_url(data: dict[str, Any], file_name: str) -> str:
    """Choose the *real* preview image URL for a ModelScope model.

    ModelScope publishes genuine sample images per model **version** under
    ``MuseInfo.versions[].coverImages[].url`` (hosted on
    ``resources.modelscope.cn/cover-images/...``), while the top-level
    ``CoverImages`` list carries the official cover of non-AIGC repos.

    ``cdn.modelscope.cn/social-thumbnails/...`` is deliberately **not** used:
    it is an auto-generated share card (ModelScope logo on a branded
    background) rather than the model's preview, and fetching it made every
    downloaded card show a misleading banner.

    Selection order:

    1. the version with the **largest trailing number** — i.e. the highest
       step count, which is the image ModelScope itself shows as the model's
       default tab.  Each version has a *different* sample image, so this is
       what makes a card look like the model page;
    2. the version whose ``stats.fileList`` contains *file_name* (exact
       checkpoint match — only reachable for repos whose version file names
       carry no number at all);
    3. the first entry of the top-level ``CoverImages``;
    4. the first cover of the first version.

    Returns ``""`` when the model genuinely has no cover image.
    """
    muse = data.get("MuseInfo") or {}
    versions = muse.get("versions") if isinstance(muse, dict) else None
    if not isinstance(versions, list):
        versions = []
    versions = [v for v in versions if isinstance(v, dict)]

    # 1. Highest step count — the model's default showcase image.
    best_number: int | None = None
    best_url = ""
    for version in versions:
        number = _version_number(version)
        if number is None:
            continue
        url = _version_cover(version)
        if not url:
            continue
        if best_number is None or number > best_number:
            best_number, best_url = number, url
    if best_url:
        return best_url

    # 2. Exact match against the downloaded checkpoint's file name.
    if file_name:
        wanted = os.path.basename(file_name)
        for version in versions:
            if wanted in _version_file_list(version):
                url = _version_cover(version)
                if url:
                    return url

    # 3. Official cover list.
    for entry in data.get("CoverImages") or []:
        url = _extract_cover_url(entry)
        if url:
            return url

    # 4. First cover of the first version.
    for version in versions:
        url = _version_cover(version)
        if url:
            return url

    return ""


async def _resolve_modelscope_cover_url(
    owner_repo: str, file_name: str, data: dict[str, Any] | None = None
) -> str:
    """Ask the ModelScope detail API for the real preview URL of *file_name*.

    When *data* is supplied (an already-fetched ``Data`` payload), the network
    round-trip is skipped — the download path fetches the detail once for
    enrichment and reuses it here.
    """
    if not isinstance(data, dict):
        api_url = f"https://modelscope.cn/api/v1/models/{owner_repo}"
        try:
            session = await _get_hf_api_session()
            async with session.get(
                api_url,
                headers={
                    "Referer": "https://www.modelscope.cn/",
                    "Accept": "application/json",
                },
                allow_redirects=True,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        "ModelScope detail API returned HTTP %s for %s",
                        resp.status,
                        api_url,
                    )
                    return ""
                payload = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning("Failed to query ModelScope detail API for %s: %s", owner_repo, exc)
            return ""

        if not isinstance(payload, dict):
            return ""
        data = payload.get("Data") or {}
        if not isinstance(data, dict):
            return ""

    return _pick_modelscope_cover_url(data, file_name)


async def _fetch_modelscope_cover(
    file_path: str,
    owner_repo: str,
    force: bool = False,
    data: dict[str, Any] | None = None,
) -> str:
    """Download the model's real preview image next to the model file.

    Resolves the genuine cover through :func:`_resolve_modelscope_cover_url` —
    by default the **highest-step version's** sample image, i.e. the one
    ModelScope shows as the model's own default tab — and writes it to
    ``<base_name><ext>``.  Any preview file that already exists for that base
    name (``.webp`` / ``.png`` / ``.preview.png`` / …) short-circuits the
    fetch, so a cover the user placed by hand is never overwritten unless
    *force* is set.

    *data* is an optional pre-fetched detail ``Data`` payload, forwarded to
    :func:`_resolve_modelscope_cover_url` to skip a redundant API call.

    Returns the local preview path on success, or an empty string when the
    model has no real cover / the download failed.  Callers must treat an
    empty return as non-fatal — the ModelScope link itself stays valid.
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    directory = os.path.dirname(file_path)

    # An existing preview (any recognised extension) wins unless forced.
    if not force:
        existing_preview = find_preview_file(base_name, directory)
        if existing_preview:
            return existing_preview

    cover_url = await _resolve_modelscope_cover_url(
        owner_repo, os.path.basename(file_path), data=data
    )
    if not cover_url:
        logger.info(
            "ModelScope model %s exposes no real preview image; card stays without cover",
            owner_repo,
        )
        return ""

    try:
        session = await _get_hf_api_session()
        async with session.get(
            cover_url,
            headers={"Referer": "https://www.modelscope.cn/"},
            allow_redirects=True,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                logger.warning(
                    "ModelScope cover fetch got HTTP %s for %s", resp.status, cover_url
                )
                return ""
            data = await resp.read()

        # Sniff the real container so the extension never lies about the bytes.
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            ext = ".png"
        elif data[:3] == b"\xff\xd8\xff":
            ext = ".jpg"
        elif data[:6] in (b"GIF87a", b"GIF89a"):
            ext = ".gif"
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            ext = ".webp"
        else:
            logger.warning(
                "ModelScope cover fetch returned non-image bytes for %s", cover_url
            )
            return ""
        if len(data) < 1024:
            logger.warning(
                "ModelScope cover fetch returned too-small image (%d bytes) for %s",
                len(data),
                cover_url,
            )
            return ""

        target = os.path.join(directory, f"{base_name}{ext}")
        with open(target, "wb") as fh:
            fh.write(data)
        logger.info(
            "Saved ModelScope cover to %s (%d bytes, source=%s)",
            target,
            len(data),
            cover_url,
        )
        return target
    except Exception as exc:
        logger.warning("Failed to fetch ModelScope cover for %s: %s", cover_url, exc)
        return ""


async def _save_modelscope_metadata(
    dest_path: str, owner_repo: str, model_root: str, revision: str | None = None
) -> None:
    """Create a proper .metadata.json for a ModelScope download and register it.

    Mirrors :func:`_save_hf_metadata`: uses
    ``MetadataManager.create_default_metadata()`` (computes SHA256, reads the
    safetensors base_model header) to build a fully-populated metadata object,
    then overlays ModelScope-specific fields and adds the model to the
    in-memory scanner cache so it appears immediately.

    Additionally auto-fetches the ModelScope social-thumbnail cover (via
    :func:`_fetch_modelscope_cover`) and points ``preview_url`` at it so the
    card shows a thumbnail without any manual step.
    """
    try:
        modelscope_url = f"https://www.modelscope.cn/models/{owner_repo}"
        if revision:
            modelscope_url += f"?version={revision}"

        # Derive the model class from the file's *real* location.  The caller's
        # model_root is often a relative key (e.g. "loras") that matches no
        # configured absolute root, which would silently fall back to the
        # default class and mis-type the model.
        matched_root = _find_matching_root(os.path.dirname(dest_path))
        model_class, scanner_getter_name = _infer_model_type(matched_root or model_root)

        # 1. Create proper metadata (computes SHA256, reads safetensors headers)
        metadata = await MetadataManager.create_default_metadata(
            dest_path, model_class=model_class
        )
        if metadata is None:
            logger.warning("create_default_metadata returned None for %s", dest_path)
            return

        # 2. Overlay ModelScope-specific fields
        metadata._unknown_fields["modelscope_url"] = modelscope_url
        metadata.from_civitai = False  # ModelScope models are not from CivitAI

        # 2b. Recover the real base model from the file itself.
        #     ``create_default_metadata`` hard-codes base_model="Unknown" (its
        #     enrichment step is disabled), so a ModelScope card would otherwise
        #     always read "LoRA | UNKN".  MuseAI-trained Krea 2 LoRAs carry
        #     ``base_model = krea/Krea-2-Turbo`` + ``trigger_word`` in their
        #     safetensors header, which is exactly what we want on the card.
        try:
            file_info = await extract_lora_metadata(dest_path)
            recovered = file_info.get("base_model")
            if recovered and recovered != "Unknown":
                metadata.base_model = recovered
            words = file_info.get("trained_words") or []
            if words:
                metadata.trainedWords = words
            logger.info(
                "ModelScope base_model recovery for %s -> base_model=%s words=%s",
                os.path.basename(dest_path), metadata.base_model, words,
            )
        except Exception as exc:
            logger.warning("base_model recovery failed for %s: %s", dest_path, exc)

        # 2c. Enrich from the ModelScope detail API so the modal can show the
        #     same 示例 / 模型描述 / 触发词 / 版本 panels CivitAI models get.
        #
        #     Everything here is best-effort: a ModelScope outage must never fail
        #     a download, so the whole block is wrapped and only logs.
        #
        #     Placement notes:
        #       * ``modelDescription`` is a first-class metadata field and is
        #         exactly what GET /model-description serves the 模型描述 tab.
        #       * Trigger words go into ``metadata.trainedWords`` (the typed,
        #         source-agnostic field) *and* ``civitai.trainedWords``, which is
        #         what the scanner's slim cache payload and the trigger-words
        #         endpoint actually read for non-CivitAI models too. This mirrors
        #         what the HuggingFace enrichment pipeline already does.
        #       * Sample images go into ``civitai.images`` for the same reason:
        #         the 示例 tab reads ``civitai.images`` with no provider guard.
        #         Entries carry a non-empty ``url``, which is what stops
        #         ``prune_stale_example_images`` from discarding them.
        #
        #     ``detail`` is hoisted so step 3's cover resolution can reuse the
        #     same payload instead of issuing a second identical API call.
        detail: dict[str, Any] | None = None
        try:
            session = await _get_hf_api_session()
            detail = await fetch_detail(session, owner_repo)
            if detail:
                bundle, gallery, entries = build_gallery_and_versions(
                    detail, owner_repo
                )

                description_html = build_description_html(detail)
                if description_html:
                    metadata.modelDescription = description_html

                words = bundle.get("trigger_words") or []
                if words:
                    metadata.trainedWords = list(words)

                civitai_block = dict(metadata.civitai or {})
                if words:
                    civitai_block["trainedWords"] = list(words)
                if gallery:
                    civitai_block["images"] = gallery
                if civitai_block:
                    metadata.civitai = civitai_block

                # Keep the version list on disk as an offline fallback for the
                # 版本 tab (the tab itself prefers a live fetch). Per-version
                # trigger words are dropped — the model-level list already
                # covers them and they are identical across versions.
                bundle["versions"] = [
                    {k: v for k, v in entry.items() if k != "trigger_words"}
                    for entry in entries
                ]
                metadata._unknown_fields["modelscope"] = bundle

                logger.info(
                    "ModelScope enrichment for %s: desc=%d chars, triggers=%s, "
                    "gallery=%d, versions=%d",
                    os.path.basename(dest_path), len(description_html), words,
                    len(gallery), len(entries),
                )
        except Exception as exc:
            logger.warning("ModelScope enrichment failed for %s: %s", dest_path, exc)

        # 3. Auto-fetch cover (non-fatal if it fails). The detail payload from
        #    step 2c is reused so this costs no extra API call.
        cover_path = await _fetch_modelscope_cover(
            dest_path, owner_repo, data=detail
        )
        if cover_path:
            metadata.preview_url = cover_path

        metadata_dict = metadata.to_dict()
        if "trainedWords" in metadata_dict and not metadata_dict["trainedWords"]:
            del metadata_dict["trainedWords"]

        # 4. Save metadata atomically
        await MetadataManager.save_metadata(dest_path, metadata_dict)
        logger.info("Saved ModelScope metadata (with modelscope_url) for %s", dest_path)

        # 5. Register in the scanner cache with the folder derived from the
        #    file's real location (see _resolve_relative_folder).  Computing the
        #    folder from the caller's `model_root` is unreliable — the frontend
        #    sends a relative key (e.g. "loras"), so the entry would land with
        #    folder="" and the card would never appear in its real folder.
        folder = _resolve_relative_folder(dest_path)
        scanner_getter = getattr(ServiceRegistry, scanner_getter_name, None)
        if scanner_getter is not None:
            scanner = await scanner_getter()
            if scanner is not None:
                metadata_dict = metadata.to_dict()
                metadata_dict["modelscope_url"] = modelscope_url
                await scanner.add_model_to_cache(metadata_dict, folder)
                logger.info("Added %s to scanner cache (folder=%s)", dest_path, folder)

    except Exception as exc:
        logger.warning("Failed to save ModelScope metadata for %s: %s", dest_path, exc)


class HfHandler:
    """Handle Hugging Face model browsing and download."""

    async def set_hf_url(self, request: web.Request) -> web.Response:
        try:
            payload: dict[str, Any] = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"success": False, "error": "Invalid JSON"}, status=400)

        file_path = (payload.get("file_path") or "").strip()
        hf_url = (payload.get("hf_url") or "").strip()

        if not file_path or not hf_url:
            return web.json_response(
                {"success": False, "error": "Missing required fields: 'file_path' and 'hf_url'"},
                status=400,
            )

        m = re.match(r"^https?://huggingface\.co/([^/]+/[^/]+)/?$", hf_url)
        if not m:
            return web.json_response(
                {
                    "success": False,
                    "error": "Invalid HuggingFace URL. Expected format: https://huggingface.co/user/repo",
                },
                status=400,
            )

        if not os.path.isfile(file_path):
            return web.json_response(
                {"success": False, "error": f"File not found: {file_path}"},
                status=404,
            )

        model_root = _find_matching_root(os.path.dirname(file_path))
        if not model_root:
            return web.json_response(
                {
                    "success": False,
                    "error": "File is not within any configured model directory. Cannot link to HuggingFace.",
                },
                status=400,
            )

        try:
            existing = await MetadataManager.load_metadata_payload(file_path)
            if existing.get("hf_url") == hf_url:
                return web.json_response({
                    "success": True,
                    "message": "hf_url already set",
                    "hf_url": hf_url,
                })

            existing["hf_url"] = hf_url
            existing["from_civitai"] = False
            await MetadataManager.save_metadata(file_path, existing)

            await _add_to_scanner_cache(file_path, existing)

            logger.info("Set hf_url=%s for %s", hf_url, file_path)
            return web.json_response({
                "success": True,
                "message": f"hf_url set to {hf_url}",
                "hf_url": hf_url,
            })
        except Exception as exc:
            logger.error("Failed to set hf_url for %s: %s", file_path, exc)
            return web.json_response(
                {"success": False, "error": str(exc)},
                status=500,
            )

    async def set_modelscope_url(self, request: web.Request) -> web.Response:
        try:
            payload: dict[str, Any] = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"success": False, "error": "Invalid JSON"}, status=400)

        file_path = (payload.get("file_path") or "").strip()
        modelscope_url = (payload.get("modelscope_url") or "").strip()

        if not file_path or not modelscope_url:
            return web.json_response(
                {"success": False, "error": "Missing required fields: 'file_path' and 'modelscope_url'"},
                status=400,
            )

        m = re.match(r"^https?://(www\.)?modelscope\.cn/models/([^/]+/[^/]+)/?$", modelscope_url)
        if not m:
            return web.json_response(
                {
                    "success": False,
                    "error": "Invalid ModelScope URL. Expected format: https://www.modelscope.cn/models/owner/model",
                },
                status=400,
            )

        if not os.path.isfile(file_path):
            return web.json_response(
                {"success": False, "error": f"File not found: {file_path}"},
                status=404,
            )

        model_root = _find_matching_root(os.path.dirname(file_path))
        if not model_root:
            return web.json_response(
                {
                    "success": False,
                    "error": "File is not within any configured model directory. Cannot link to ModelScope.",
                },
                status=400,
            )

        try:
            existing = await MetadataManager.load_metadata_payload(file_path)
            if existing.get("modelscope_url") == modelscope_url:
                return web.json_response({
                    "success": True,
                    "message": "modelscope_url already set",
                    "modelscope_url": modelscope_url,
                })

            existing["modelscope_url"] = modelscope_url
            existing["from_civitai"] = False

            # Auto-fetch the ModelScope cover image and point preview_url at it.
            # Non-fatal: if the fetch fails, the URL link still works.  A
            # preview the user already has (metadata field or a sibling image
            # file) must never be replaced by an auto-grabbed cover.
            owner_repo = m.group(2)  # "owner/repo"
            cover_path = await _fetch_modelscope_cover(file_path, owner_repo)
            if cover_path:
                current_preview = existing.get("preview_url") or ""
                if not current_preview or not os.path.isfile(current_preview):
                    existing["preview_url"] = cover_path

            await MetadataManager.save_metadata(file_path, existing)

            await _add_to_scanner_cache(file_path, existing)

            logger.info("Set modelscope_url=%s for %s (cover=%s)", modelscope_url, file_path, cover_path or "none")
            return web.json_response({
                "success": True,
                "message": f"modelscope_url set to {modelscope_url}",
                "modelscope_url": modelscope_url,
                "preview_url": existing.get("preview_url", ""),
                "cover_fetched": bool(cover_path),
            })
        except Exception as exc:
            logger.error("Failed to set modelscope_url for %s: %s", file_path, exc)
            return web.json_response(
                {"success": False, "error": str(exc)},
                status=500,
            )

    async def get_hf_repo_files(self, request: web.Request) -> web.Response:
        """List model-weight files from a HF repo with real file sizes.

        Uses the HF tree API endpoint which returns accurate file sizes
        (including LFS-tracked files), unlike the model info endpoint.
        """
        repo = request.query.get("repo", "").strip()
        if not repo or "/" not in repo:
            return web.json_response(
                {"error": "Missing or invalid 'repo' parameter (expected user/repo)"},
                status=400,
            )

        url = f"https://huggingface.co/api/models/{repo}/tree/main"

        try:
            session = await _get_hf_api_session()
            async with session.get(url) as resp:
                if resp.status == 404:
                    return web.json_response(
                        {"error": f"Repo '{repo}' not found"}, status=404
                    )
                if resp.status != 200:
                    text = await resp.text()
                    return web.json_response(
                        {"error": f"HF API error {resp.status}: {text[:200]}"},
                        status=resp.status,
                    )
                tree: list[dict[str, Any]] = await resp.json()
        except Exception as exc:
            logger.error("Failed to fetch HF repo files: %s", exc)
            return web.json_response({"error": str(exc)}, status=502)

        files: list[dict[str, Any]] = []
        for entry in tree:
            path: str = entry.get("path", "")
            ext = os.path.splitext(path)[1].lower()
            if ext not in MODEL_FILE_EXTENSIONS:
                continue
            size = entry.get("size", 0) or 0
            if size == 0 and "lfs" in entry:
                size = entry["lfs"].get("size", 0) or 0
            files.append({
                "filename": path,
                "size": size,
            })

        files.sort(key=lambda f: f["size"], reverse=True)
        return web.json_response(files)

    async def download_hf_model(self, request: web.Request) -> web.Response:
        """Download a single file from Hugging Face into the model directory.

        POST JSON body::

            {
              "repo": "dx8152/Flux2-Klein-9B-Consistency",
              "filename": "Flux2-Klein-9B-consistency-V2.safetensors",
              "revision": "main",
              "model_root": "loras",
              "relative_path": "",
              "use_default_paths": false,
              "download_id": "optional-batch-id"
            }

        If ``download_id`` is provided, real-time progress (bytes, speed,
        percentage) is broadcast via the WebSocket progress system, matching
        the CivitAI download experience.

        Respects the ``download_backend`` setting (``aria2`` or ``default``).
        """
        try:
            payload: dict[str, Any] = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        repo = (payload.get("repo") or "").strip()
        filename = (payload.get("filename") or "").strip()
        revision = (payload.get("revision") or "main").strip()
        model_root = (payload.get("model_root") or "").strip()
        relative_path = (payload.get("relative_path") or "").strip()
        use_default_paths = bool(payload.get("use_default_paths", False))
        download_id: str | None = payload.get("download_id")

        logger.info(
            "download_hf_model: repo=%s file=%s root=%s download_id=%s",
            repo, filename, model_root, download_id,
        )

        if not repo or not filename:
            return web.json_response(
                {"error": "Missing required fields: 'repo' and 'filename'"}, status=400
            )

        # Validate repo format — must be user/repo_name
        if repo.count("/") != 1 or not re.match(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$", repo):
            return web.json_response({"error": f"Invalid repo format: {repo}"}, status=400)
        author, repo_name = repo.split("/", 1)
        if ".." in (author, repo_name) or "." in (author, repo_name):
            return web.json_response({"error": f"Invalid repo format: {repo}"}, status=400)

        # Validate filename — must not contain path traversal
        if ".." in filename:
            return web.json_response({"error": "Invalid filename"}, status=400)

        # Validate relative_path — must not be absolute or escape base directory
        if relative_path:
            if os.path.isabs(relative_path):
                return web.json_response({"error": "relative_path must not be absolute"}, status=400)
            if ".." in relative_path.split("/") or "\\" in relative_path:
                return web.json_response({"error": "Invalid relative_path"}, status=400)

        # Use model_root directly as the base directory — same approach as
        # CivitAI's download path (download_manager.py).  No realpath, no
        # allowed-roots validation, no path-traversal check; those are
        # unnecessary when the frontend sends the path from its own dropdown
        # (populated from scanner roots).  Using the "business path" directly
        # keeps dest_path consistent with scanner roots so that later folder
        # derivation (in _save_hf_metadata) works correctly.
        if os.path.isabs(model_root):
            base_dir = os.path.normpath(model_root)
        else:
            base_dir = os.path.normpath(os.path.join(os.getcwd(), "models", model_root))

        if use_default_paths:
            target_dir = os.path.join(base_dir, "huggingface", author, repo_name)
        elif relative_path:
            target_dir = os.path.join(base_dir, relative_path)
        else:
            target_dir = base_dir

        # Strip HF repo subdirectory — "diffusion_models/xxx.safetensors"
        # is an HF repo convention, not meaningful for local storage.
        file_base = os.path.basename(filename)

        os.makedirs(target_dir, exist_ok=True)
        dest_path = os.path.join(target_dir, file_base)

        # Check if already exists (simple skip)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            logger.info("download_hf_model: file already exists, skipping — %s", dest_path)
            return web.json_response({
                "success": True,
                "message": f"File already exists: {dest_path}",
                "path": dest_path,
            })

        # Build HF resolve URL
        resolve_url = (
            f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
        )

        # Set up progress callback if download_id is provided
        progress_callback = None
        if download_id:

            async def _progress_callback(
                progress: float | DownloadProgress,
                snapshot: DownloadProgress | None = None,
            ) -> None:
                percent = 0.0
                metrics = snapshot if isinstance(snapshot, DownloadProgress) else None

                if isinstance(progress, DownloadProgress):
                    percent = progress.percent_complete
                    metrics = progress
                elif isinstance(snapshot, DownloadProgress):
                    percent = snapshot.percent_complete
                else:
                    percent = float(progress)

                broadcast: dict[str, Any] = {
                    "status": "progress",
                    "progress": round(percent),
                }
                if metrics:
                    broadcast["bytes_downloaded"] = metrics.bytes_downloaded
                    broadcast["total_bytes"] = metrics.total_bytes
                    broadcast["bytes_per_second"] = metrics.bytes_per_second

                await ws_manager.broadcast_download_progress(download_id, broadcast)

            progress_callback = _progress_callback

        # Respect download backend setting (aria2 vs default)
        download_backend = (
            get_settings_manager().get("download_backend", "default")
        )

        if download_backend == "aria2":
            aria2 = await Aria2Downloader.get_instance()
            aid = download_id or f"hf_{repo}_{filename}"
            try:
                hf_success, hf_result = await aria2.download_file(
                    url=resolve_url,
                    save_path=dest_path,
                    download_id=aid,
                    progress_callback=progress_callback,
                )
                if hf_success:
                    await _save_hf_metadata(dest_path, repo, model_root)
                    return web.json_response({
                        "success": True,
                        "message": f"Downloaded to {dest_path}",
                        "path": dest_path,
                    })
                else:
                    return web.json_response(
                        {"success": False, "error": hf_result or "aria2 download failed"},
                        status=500,
                    )
            except Exception as exc:
                logger.error("HF download (aria2) failed: %s", exc)
                return web.json_response(
                    {"success": False, "error": str(exc)}, status=500
                )

        # Default: use built-in aiohttp Downloader
        downloader = await get_downloader()
        try:
            success, result = await downloader.download_file(
                url=resolve_url,
                save_path=dest_path,
                use_auth=False,
                allow_resume=True,
                progress_callback=progress_callback,
            )
            if success:
                await _save_hf_metadata(dest_path, repo, model_root)
                return web.json_response({
                    "success": True,
                    "message": f"Downloaded to {result}",
                    "path": result,
                })
            else:
                return web.json_response(
                    {"success": False, "error": result or "Download failed"},
                    status=500,
                )
        except Exception as exc:
            logger.error("HF download failed: %s", exc)
            return web.json_response(
                {"success": False, "error": str(exc)}, status=500
            )

    async def get_modelscope_repo_files(self, request: web.Request) -> web.Response:
        """List model-weight files from a ModelScope repo with real file sizes.

        Mirrors :meth:`get_hf_repo_files` but queries the ModelScope files API
        (``/api/v1/models/{owner}/{repo}/repo/files``).  Returns the same
        ``[{filename, size}]`` shape the frontend already consumes for HF.
        """
        owner = (request.query.get("owner") or "").strip()
        repo = (request.query.get("repo") or "").strip()
        revision = (request.query.get("revision") or "master").strip()
        if not owner or not repo:
            return web.json_response(
                {
                    "error": "Missing required 'owner' and 'repo' parameters "
                    "(expected owner/repo)"
                },
                status=400,
            )

        url = (
            f"https://modelscope.cn/api/v1/models/{owner}/{repo}"
            f"/repo/files?Revision={revision}"
        )

        try:
            session = await _get_hf_api_session()
            async with session.get(url) as resp:
                if resp.status == 404:
                    return web.json_response(
                        {"error": f"Repo '{owner}/{repo}' not found"}, status=404
                    )
                if resp.status != 200:
                    text = await resp.text()
                    return web.json_response(
                        {
                            "error": f"ModelScope API error {resp.status}: "
                            f"{text[:200]}"
                        },
                        status=resp.status,
                    )
                payload: dict[str, Any] = await resp.json()
        except Exception as exc:
            logger.error("Failed to fetch ModelScope repo files: %s", exc)
            return web.json_response({"error": str(exc)}, status=502)

        data = payload.get("Data") or {}
        raw_files = data.get("Files") or []
        files: list[dict[str, Any]] = []
        for entry in raw_files:
            if entry.get("Type") != "blob":
                continue
            path = entry.get("Path") or entry.get("Name") or ""
            if not path:
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext not in MODEL_FILE_EXTENSIONS:
                continue
            files.append({"filename": path, "size": entry.get("Size") or 0})

        files.sort(key=lambda f: f["size"], reverse=True)
        return web.json_response(files)

    async def download_modelscope_model(self, request: web.Request) -> web.Response:
        """Download a single file from ModelScope into the model directory.

        POST JSON body::

            {
              "owner": "amdati112233",
              "repo": "stockings_Krea2_lora",
              "filename": "stockings_Krea2_lora_c1-st6000.safetensors",
              "revision": "20260721141238",
              "model_root": "loras",
              "relative_path": "",
              "use_default_paths": false,
              "download_id": "optional-batch-id"
            }

        The file is streamed directly from the ModelScope file-content API
        (no SDK required).  If ``download_id`` is provided, real-time progress
        (bytes, speed, percentage) is broadcast via the WebSocket progress
        system, matching the CivitAI/HF download experience.

        After a successful download, a full ``.metadata.json`` sidecar is
        written (with a ``modelscope_url`` link + auto-fetched cover) and the
        model is registered in the in-memory scanner cache.

        Respects the ``download_backend`` setting (``aria2`` or ``default``).
        """
        try:
            payload: dict[str, Any] = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        owner = (payload.get("owner") or "").strip()
        repo = (payload.get("repo") or "").strip()
        filename = (payload.get("filename") or "").strip()
        revision = (payload.get("revision") or "master").strip()
        model_root = (payload.get("model_root") or "").strip()
        relative_path = (payload.get("relative_path") or "").strip()
        use_default_paths = bool(payload.get("use_default_paths", False))
        download_id: str | None = payload.get("download_id")

        logger.info(
            "download_modelscope_model: owner=%s repo=%s file=%s rev=%s root=%s download_id=%s",
            owner, repo, filename, revision, model_root, download_id,
        )

        if not owner or not repo or not filename:
            return web.json_response(
                {
                    "error": "Missing required fields: 'owner', 'repo' and 'filename'"
                },
                status=400,
            )

        if owner.count("/") != 0 or repo.count("/") != 0:
            return web.json_response(
                {"error": f"Invalid owner/repo: {owner}/{repo}"}, status=400
            )
        if ".." in (owner, repo) or "." in (owner, repo):
            return web.json_response(
                {"error": f"Invalid owner/repo: {owner}/{repo}"}, status=400
            )
        if ".." in filename:
            return web.json_response({"error": "Invalid filename"}, status=400)
        if relative_path:
            if os.path.isabs(relative_path):
                return web.json_response(
                    {"error": "relative_path must not be absolute"}, status=400
                )
            if ".." in relative_path.split("/") or "\\" in relative_path:
                return web.json_response(
                    {"error": "Invalid relative_path"}, status=400
                )

        # Use model_root directly as the base directory (same approach as HF).
        if os.path.isabs(model_root):
            base_dir = os.path.normpath(model_root)
        else:
            base_dir = os.path.normpath(os.path.join(os.getcwd(), "models", model_root))

        if use_default_paths:
            target_dir = os.path.join(base_dir, "modelscope", owner, repo)
        elif relative_path:
            target_dir = os.path.join(base_dir, relative_path)
        else:
            target_dir = base_dir

        file_base = os.path.basename(filename)
        os.makedirs(target_dir, exist_ok=True)
        dest_path = os.path.join(target_dir, file_base)

        # Check if already exists (simple skip)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            logger.info(
                "download_modelscope_model: file already exists, skipping — %s", dest_path
            )
            return web.json_response(
                {
                    "success": True,
                    "message": f"File already exists: {dest_path}",
                    "path": dest_path,
                }
            )

        # ModelScope file-content API returns the raw bytes directly (no SDK).
        file_url = (
            f"https://modelscope.cn/api/v1/models/{owner}/{repo}/repo"
            f"?Revision={revision}&FilePath={filename}"
        )

        # Set up progress callback if download_id is provided
        progress_callback = None
        if download_id:

            async def _progress_callback(
                progress: float | DownloadProgress,
                snapshot: DownloadProgress | None = None,
            ) -> None:
                percent = 0.0
                metrics = snapshot if isinstance(snapshot, DownloadProgress) else None

                if isinstance(progress, DownloadProgress):
                    percent = progress.percent_complete
                    metrics = progress
                elif isinstance(snapshot, DownloadProgress):
                    percent = snapshot.percent_complete
                else:
                    percent = float(progress)

                broadcast: dict[str, Any] = {
                    "status": "progress",
                    "progress": round(percent),
                }
                if metrics:
                    broadcast["bytes_downloaded"] = metrics.bytes_downloaded
                    broadcast["total_bytes"] = metrics.total_bytes
                    broadcast["bytes_per_second"] = metrics.bytes_per_second

                await ws_manager.broadcast_download_progress(download_id, broadcast)

            progress_callback = _progress_callback

        # Respect download backend setting (aria2 vs default)
        download_backend = get_settings_manager().get("download_backend", "default")

        if download_backend == "aria2":
            aria2 = await Aria2Downloader.get_instance()
            aid = download_id or f"ms_{owner}_{repo}_{filename}"
            try:
                ms_success, ms_result = await aria2.download_file(
                    url=file_url,
                    save_path=dest_path,
                    download_id=aid,
                    progress_callback=progress_callback,
                )
                if ms_success:
                    await _save_modelscope_metadata(
                        dest_path, f"{owner}/{repo}", model_root, revision
                    )
                    return web.json_response(
                        {
                            "success": True,
                            "message": f"Downloaded to {dest_path}",
                            "path": dest_path,
                        }
                    )
                else:
                    return web.json_response(
                        {"success": False, "error": ms_result or "aria2 download failed"},
                        status=500,
                    )
            except Exception as exc:
                logger.error("ModelScope download (aria2) failed: %s", exc)
                return web.json_response(
                    {"success": False, "error": str(exc)}, status=500
                )

        # Default: use built-in aiohttp Downloader
        downloader = await get_downloader()
        try:
            success, result = await downloader.download_file(
                url=file_url,
                save_path=dest_path,
                use_auth=False,
                allow_resume=True,
                progress_callback=progress_callback,
            )
            if success:
                await _save_modelscope_metadata(
                    dest_path, f"{owner}/{repo}", model_root, revision
                )
                return web.json_response(
                    {
                        "success": True,
                        "message": f"Downloaded to {result}",
                        "path": result,
                    }
                )
            else:
                return web.json_response(
                    {"success": False, "error": result or "Download failed"},
                    status=500,
                )
        except Exception as exc:
            logger.error("ModelScope download failed: %s", exc)
            return web.json_response(
                {"success": False, "error": str(exc)}, status=500
            )

    # ------------------------------------------------------------------
    # ModelScope version list (the 版本 tab for ModelScope models)
    # ------------------------------------------------------------------

    async def _read_modelscope_metadata(self, file_path: str):
        """Load ``(metadata, model_class, scanner_getter_name)`` for a model file."""
        matched_root = _find_matching_root(os.path.dirname(file_path))
        model_class, scanner_getter_name = _infer_model_type(matched_root or "")
        metadata, _should_skip = await MetadataManager.load_metadata(
            file_path, model_class
        )
        return metadata, model_class, scanner_getter_name

    @staticmethod
    async def _build_local_name_index(scanner) -> dict[str, list[dict[str, Any]]]:
        """Map extension-free, lower-cased file name to candidate cache entries.

        A list (not a single entry) is kept so callers can prefer the candidate
        that belongs to the same upstream repo when two libraries hold a
        same-named file.
        """
        index: dict[str, list[dict[str, Any]]] = {}
        try:
            cache = await scanner.get_cached_data()
        except Exception as exc:
            logger.warning("Could not read scanner cache for ModelScope versions: %s", exc)
            return index

        for item in getattr(cache, "raw_data", None) or []:
            if not isinstance(item, dict):
                continue
            file_path = item.get("file_path") or ""
            name = item.get("file_name") or ""
            if not name and file_path:
                name = os.path.splitext(os.path.basename(file_path))[0]
            if not name:
                continue
            index.setdefault(name.strip().lower(), []).append(item)
        return index

    @staticmethod
    def _prefer_same_repo(
        candidates: list[dict[str, Any]], owner_repo: str
    ) -> dict[str, Any]:
        """Pick the candidate already linked to *owner_repo*, else the first."""
        lowered = owner_repo.lower()
        for item in candidates:
            url = (item.get("modelscope_url") or "").lower()
            if lowered and lowered in url:
                return item
        return candidates[0]

    async def get_modelscope_versions(self, request: web.Request) -> web.Response:
        """Version history for a ModelScope-linked model.

        Serves the same ``record`` envelope the CivitAI versions endpoint uses
        (``versions`` + ``inLibraryVersionIds``) so the shared renderer, badges
        and styling apply unchanged.  Extra ``source`` / ``ownerRepo`` /
        ``triggerWords`` keys let the frontend distinguish a ModelScope record.

        Accepts either ``?file_path=<abs path to the model>`` (the frontend's
        normal call — the repo is read from the model's own metadata) or an
        explicit ``?owner_repo=owner/repo``.

        The live ModelScope API is preferred so newly published training steps
        show up; the version list stored in ``.metadata.json`` at download time
        is used as an offline fallback.
        """
        file_path = (request.query.get("file_path") or "").strip()
        owner_repo = parse_owner_repo(request.query.get("owner_repo") or "")

        metadata = None
        scanner_getter_name = _DEFAULT_SCANNER_GETTER
        base_model = ""

        if file_path:
            if not os.path.isfile(file_path):
                return web.json_response(
                    {"success": False, "error": f"Model file not found: {file_path}"},
                    status=404,
                )
            try:
                metadata, _model_class, scanner_getter_name = (
                    await self._read_modelscope_metadata(file_path)
                )
            except Exception as exc:
                logger.warning("Failed to read metadata for %s: %s", file_path, exc)

        stored_bundle: dict[str, Any] = {}
        if metadata is not None:
            stored_bundle = (metadata._unknown_fields or {}).get("modelscope") or {}
            base_model = getattr(metadata, "base_model", "") or ""

        if not owner_repo:
            owner_repo = parse_owner_repo(
                (metadata._unknown_fields or {}).get("modelscope_url", "")
                if metadata is not None
                else ""
            ) or stored_bundle.get("owner_repo", "")

        if not owner_repo:
            return web.json_response(
                {
                    "success": False,
                    "error": "This model has no ModelScope repository link.",
                },
                status=400,
            )

        # ---- Live fetch (preferred) -------------------------------------
        detail = None
        offline = False
        try:
            session = await _get_hf_api_session()
            detail = await fetch_detail(session, owner_repo)
        except Exception as exc:
            logger.warning("ModelScope detail fetch failed for %s: %s", owner_repo, exc)

        if detail is not None:
            entries = build_version_entries(detail)
            bundle = stored_bundle or {}
        else:
            # ---- Offline fallback: the bundle written at download time ----
            offline = True
            entries = []
            for raw in stored_bundle.get("versions") or []:
                if isinstance(raw, dict):
                    entry = dict(raw)
                    entry.setdefault("trigger_words", [])
                    entries.append(entry)
            bundle = stored_bundle
            if not entries:
                return web.json_response(
                    {
                        "success": False,
                        "error": (
                            f"Could not reach ModelScope for {owner_repo}, and no "
                            "cached version list is stored for this model."
                        ),
                    },
                    status=502,
                )

        # ---- Which versions are already on disk? ------------------------
        local_index: dict[str, list[dict[str, Any]]] = {}
        scanner_getter = getattr(ServiceRegistry, scanner_getter_name, None)
        if scanner_getter is not None:
            try:
                scanner = await scanner_getter()
                if scanner is not None:
                    local_index = await self._build_local_name_index(scanner)
            except Exception as exc:
                logger.warning("Local library index build failed: %s", exc)

        local_by_filename: dict[str, dict[str, Any]] = {
            name: self._prefer_same_repo(candidates, owner_repo)
            for name, candidates in local_index.items()
            if candidates
        }

        rows, in_library_ids = build_version_rows(
            entries,
            owner_repo=owner_repo,
            base_model=base_model,
            local_by_filename=local_by_filename,
        )

        trigger_words = (bundle.get("trigger_words") or []) if bundle else []
        record = {
            "source": "modelscope",
            "ownerRepo": owner_repo,
            "url": build_page_url(owner_repo),
            "chineseName": bundle.get("chinese_name") or "",
            "triggerWords": trigger_words,
            "license": bundle.get("license") or "",
            "baseModel": base_model,
            "offline": offline,
            "versions": rows,
            "inLibraryVersionIds": in_library_ids,
            "shouldIgnore": False,
            "hasUpdate": False,
        }
        return web.json_response({"success": True, "record": record})
