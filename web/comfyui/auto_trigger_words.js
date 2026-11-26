/**
 * LOCAL FORK ADDITION
 * ---------------------------------------------------------------------------
 * Auto-append a LoRA's CivitAI `trainedWords` to the workflow's prompt node
 * when the LoRA is switched ON in the added-LoRA list, and remove exactly the
 * text we inserted when it is switched OFF again.
 *
 * Reads trigger words through the plugin's existing endpoint
 * (`GET /api/lm/loras/get-trigger-words?name=...`). The widget write mirrors
 * workflow_registry.applyWidgetUpdate locally and synchronously, so the
 * journal and the widget can never get out of sync. No Python changes.
 *
 * Safety: for each insertion we remember the exact before/after text in
 * `sourceNode.properties.lm_auto_trigger_words` (serialized with the workflow).
 * Removal restores `before` verbatim when nothing else changed, otherwise it
 * removes only our own snippet once, and gives up (with a warning) if the user
 * has edited that part by hand. Text the user typed is never deleted.
 *
 * All files here are plain ES modules under WEB_DIRECTORY, so a browser
 * hard-refresh is enough; no npm build and no ComfyUI restart is required.
 */
import { app } from "../../scripts/app.js";
import {
    getAllGraphNodes,
    getNodeReference,
    getNodeFromGraph,
} from "./utils.js";
import { showToast } from "./loras_widget_utils.js";

const STORAGE_KEY = "lm.auto_trigger_words.enabled";
const STATE_KEY = "lm_auto_trigger_words";

// A toggle can be switched OFF while the trigger-word lookup is still in
// flight. Remember that intent so the pending append is discarded instead of
// leaving stray words behind.
const cancelledLookups = new Set();
const lookupKey = (sourceNode, rawName) => `${sourceNode?.id ?? "?"}:${rawName}`;

// Same classes the plugin's workflow registry treats as prompt text nodes.
const PROMPT_NODE_CLASSES = new Set([
    "CLIPTextEncode",
    "Prompt (LoraManager)",
    "Text (LoraManager)",
    "Text Multiline", // WAS Node Suite
]);

const BOLT_ICON = `
<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" focusable="false">
  <path d="M13.1 2.1 5.2 13.1h5.4l-1.0 8.8 8.2-11.2h-5.5l.8-8.6z" fill="currentColor"/>
</svg>`;

// ---------------------------------------------------------------------------
// On/off switch (persisted in localStorage, shared by all LoRA list widgets)
// ---------------------------------------------------------------------------

export function isAutoTriggerWordsEnabled() {
    try {
        const value = localStorage.getItem(STORAGE_KEY);
        return value === null ? true : value === "1";
    } catch (_error) {
        return true;
    }
}

function syncButton(button) {
    const enabled = isAutoTriggerWordsEnabled();
    button.dataset.on = enabled ? "true" : "false";
    button.setAttribute("aria-pressed", enabled ? "true" : "false");
    button.title = enabled
        ? "自动触发词：已开启 — 勾选写入、取消勾选移除（点击关闭）"
        : "自动触发词：已关闭 — 勾选/取消都不改动提示词（点击开启）";
}

function syncAllButtons() {
    try {
        document.querySelectorAll(".lm-auto-trigger-btn").forEach(syncButton);
    } catch (_error) {
        /* ignore */
    }
}

export function setAutoTriggerWordsEnabled(enabled) {
    try {
        localStorage.setItem(STORAGE_KEY, enabled ? "1" : "0");
    } catch (_error) {
        /* ignore */
    }
    syncAllButtons();
}

/** Build the labelled toolbar toggle. Returned node is appended by loras_widget.js. */
export function buildAutoTriggerWordsButton() {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "lm-auto-trigger-btn";
    button.innerHTML = BOLT_ICON;
    const label = document.createElement("span");
    label.className = "lm-auto-trigger-label";
    label.textContent = "自动触发词";
    button.appendChild(label);
    syncButton(button);

    button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        const next = !isAutoTriggerWordsEnabled();
        setAutoTriggerWordsEnabled(next);
        showToast(
            next
                ? "已开启：勾选 LoRA 写入触发词，取消勾选自动移除"
                : "已关闭：勾选/取消都不再改动提示词",
            "info",
        );
    });
    button.addEventListener("keydown", (event) => event.stopPropagation());
    return button;
}

// ---------------------------------------------------------------------------
// Per-LoRA insertion journal (persisted on the LoRA Loader node)
// ---------------------------------------------------------------------------

function readState(sourceNode) {
    const state = sourceNode?.properties?.[STATE_KEY];
    return state && typeof state === "object" && !Array.isArray(state) ? state : {};
}

function writeState(sourceNode, state) {
    if (!sourceNode) return;
    if (!sourceNode.properties || typeof sourceNode.properties !== "object") {
        sourceNode.properties = {};
    }
    sourceNode.properties[STATE_KEY] = state;
}

function rememberInsertion(sourceNode, rawName, record) {
    if (!sourceNode) return;
    const state = readState(sourceNode);
    state[rawName] = record;
    writeState(sourceNode, state);
}

// ---------------------------------------------------------------------------
// Trigger word lookup
// ---------------------------------------------------------------------------

/** LoRA stack names may be a bare file name, a "folder/name" path, or carry an extension. */
function nameCandidates(rawName) {
    const original = String(rawName || "").trim();
    if (!original) return [];
    const withoutExt = original.replace(/\.(safetensors|ckpt|pt|gguf)$/i, "");
    const basename = withoutExt.split(/[\\/]/).pop();
    return [...new Set([original, withoutExt, basename].filter(Boolean))];
}

async function fetchTriggerWords(rawName) {
    for (const candidate of nameCandidates(rawName)) {
        try {
            const response = await fetch(
                `/api/lm/loras/get-trigger-words?name=${encodeURIComponent(candidate)}`,
            );
            if (!response.ok) continue;
            const data = await response.json();
            const words = Array.isArray(data?.trigger_words) ? data.trigger_words : [];
            const cleaned = words
                .map((word) => String(word ?? "").trim())
                .filter(Boolean);
            if (cleaned.length) return cleaned;
        } catch (error) {
            console.debug("[LM:AutoTriggerWords] lookup failed", candidate, error);
        }
    }
    return [];
}

// ---------------------------------------------------------------------------
// Prompt node discovery
// ---------------------------------------------------------------------------

function textWidgetOf(node) {
    if (!Array.isArray(node?.widgets)) return null;
    return (
        node.widgets.find((widget) => {
            const type = typeof widget?.type === "string" ? widget.type.toLowerCase() : "";
            return (
                type === "string" ||
                type === "customtext" ||
                type.includes("string") ||
                widget?.name === "text"
            );
        }) || null
    );
}

/** A widget backed by a connected input cannot be changed by value writes. */
function isTextWidgetConnected(node) {
    if (!Array.isArray(node?.inputs)) return false;
    return node.inputs.some(
        (input) =>
            (input?.name === "text" || input?.widget?.name === "text") &&
            input.link != null,
    );
}

function linkById(graph, linkId) {
    const links = graph?.links;
    if (!links) return null;
    if (links instanceof Map) return links.get(linkId) || null;
    if (Array.isArray(links)) {
        return links.find((link) => link && link.id === linkId) || links[linkId] || null;
    }
    return links[linkId] || null;
}

function collectPromptNodes() {
    const entries = [];
    for (const { graph, node } of getAllGraphNodes(app.graph)) {
        const cls = node?.comfyClass || node?.type;
        // Any node explicitly marked "Send Prompt Target" qualifies, whatever
        // its class; otherwise fall back to the known prompt node classes.
        const marked =
            node?.properties?.lm_marker_role === "send_prompt_target";
        if (!PROMPT_NODE_CLASSES.has(cls) && !marked) continue;
        const widget = textWidgetOf(node);
        if (!widget || isTextWidgetConnected(node)) continue;
        entries.push({ graph, node, widget });
    }
    return entries;
}

/**
 * Priority:
 *   1. node marked "Send Prompt Target" (plugin's existing marker role)
 *   2. node wired into a sampler's `positive` input (avoids the negative prompt)
 *   3. the first remaining prompt node, with a hint about marking a target
 */
function pickTargets(entries) {
    const marked = entries.filter(
        (entry) => entry.node.properties?.lm_marker_role === "send_prompt_target",
    );
    if (marked.length) return { targets: marked, how: "已标记的提示词节点" };

    const positiveIds = new Set();
    for (const { graph, node } of getAllGraphNodes(app.graph)) {
        if (!Array.isArray(node?.inputs)) continue;
        for (const input of node.inputs) {
            if (input?.name !== "positive" || input.link == null) continue;
            const link = linkById(graph, input.link);
            if (link && link.origin_id != null) positiveIds.add(Number(link.origin_id));
        }
    }
    const positive = entries.filter((entry) => positiveIds.has(Number(entry.node.id)));
    if (positive.length) return { targets: positive, how: "正向提示词节点" };

    if (entries.length) {
        return {
            targets: [entries[0]],
            how: `提示词节点「${nodeLabel(entries[0].node)}」（可右键标记 Send Prompt Target 指定）`,
        };
    }
    return { targets: [], how: "" };
}

function nodeLabel(node) {
    return node?.title || node?.comfyClass || node?.type || `节点 ${node?.id}`;
}

/** Write a widget value the same way workflow_registry.applyWidgetUpdate does. */
function setWidgetValue(node, widget, value) {
    widget.value = value;
    const index = Array.isArray(node.widgets) ? node.widgets.indexOf(widget) : -1;
    if (
        index >= 0 &&
        Array.isArray(node.widgets_values) &&
        node.widgets_values.length > index
    ) {
        node.widgets_values[index] = value;
    }
    if (typeof widget.callback === "function") {
        try {
            widget.callback(value);
        } catch (error) {
            console.warn("[LM:AutoTriggerWords] widget callback failed", error);
        }
    }
    if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true);
    if (typeof app.graph?.setDirtyCanvas === "function") {
        app.graph.setDirtyCanvas(true, true);
    }
}

// ---------------------------------------------------------------------------
// ON: append + journal
// ---------------------------------------------------------------------------

export async function appendTriggerWordsForLora(rawName, sourceNode) {
    if (!isAutoTriggerWordsEnabled()) return;

    const key = lookupKey(sourceNode, rawName);
    cancelledLookups.delete(key);

    const words = await fetchTriggerWords(rawName);
    // The toggle was switched OFF while the lookup was in flight.
    if (cancelledLookups.has(key)) {
        cancelledLookups.delete(key);
        return;
    }
    // Most LoRAs have no trainedWords. Stay silent for those.
    if (!words.length) {
        console.debug("[LM:AutoTriggerWords] no trigger words for", rawName);
        return;
    }

    const entries = collectPromptNodes();
    if (!entries.length) {
        showToast(
            "该 LoRA 有触发词，但画布上没有可写入的提示词节点（CLIP Text Encode / Prompt (LoraManager)）",
            "warning",
        );
        return;
    }

    const { targets, how } = pickTargets(entries);
    if (!targets.length) {
        showToast("未找到可用的提示词节点，触发词未写入", "warning");
        return;
    }

    const written = [];
    const journal = [];
    let skipped = 0;

    for (const target of targets) {
        const current = String(target.widget.value ?? "");
        const lower = current.toLowerCase();
        // Idempotent: never append a word the prompt already contains.
        const missing = words.filter((word) => !lower.includes(word.toLowerCase()));
        if (!missing.length) {
            skipped += 1;
            continue;
        }

        const inserted = missing.join(", ");
        // Prepend mode: trigger words go to the FRONT of the prompt, followed
        // by an English comma before the existing text.
        const after = current ? `${inserted}, ${current}` : inserted;

        // Synchronous write keeps the journal and the widget in lockstep, so
        // switching the toggle OFF immediately afterwards always succeeds.
        // Mirrors workflow_registry.applyWidgetUpdate (value + widgets_values
        // + callback + dirty canvas); connected widgets are already excluded.
        setWidgetValue(target.node, target.widget, after);

        journal.push({
            ref: getNodeReference(target.node),
            widgetName: target.widget.name || "text",
            inserted,
            before: current,
            after,
        });
        written.push(nodeLabel(target.node));
    }

    if (journal.length) {
        rememberInsertion(sourceNode, rawName, { targets: journal });
    }

    if (written.length) {
        const preview = words.join(", ");
        const short = preview.length > 90 ? `${preview.slice(0, 90)}…` : preview;
        showToast(
            `已追加 ${words.length} 个触发词到「${written.join("、")}」：${short}`,
            "success",
        );
    } else if (skipped) {
        showToast(`触发词已存在于「${how}」，未重复添加（原有文字不会被动）`, "info");
    }
}

// ---------------------------------------------------------------------------
// OFF: undo exactly what we inserted
// ---------------------------------------------------------------------------

export function removeTriggerWordsForLora(rawName, sourceNode) {
    // Mark an in-flight lookup as cancelled so it cannot append afterwards.
    const key = lookupKey(sourceNode, rawName);
    cancelledLookups.add(key);
    setTimeout(() => cancelledLookups.delete(key), 5000);

    const state = readState(sourceNode);
    const record = state[rawName];
    if (!record || !Array.isArray(record.targets) || !record.targets.length) {
        // Nothing was inserted by us (no trigger words, or the word was already
        // in the prompt and we skipped it). Never touch user-authored text.
        return;
    }
    delete state[rawName];
    writeState(sourceNode, state);

    let removed = 0;
    let kept = 0;
    const labels = [];

    for (const entry of record.targets) {
        const node = getNodeFromGraph(entry.ref?.graph_id, entry.ref?.node_id);
        const widget = node?.widgets?.find((w) => w?.name === entry.widgetName);
        if (!node || !widget || isTextWidgetConnected(node)) {
            kept += 1;
            continue;
        }

        const current = String(widget.value ?? "");
        let next = null;

        // Exact replay: nothing else touched the prompt since we appended.
        if (current === entry.after) {
            next = entry.before;
        } else {
            // Something changed afterwards (another LoRA, or manual edits).
            // Remove only our own snippet, once, together with the separator
            // we added (leading for append mode, trailing for prepend mode).
            const index = current.indexOf(entry.inserted);
            if (index >= 0) {
                let start = index;
                let end = index + entry.inserted.length;
                if (start > 0 && current[start - 1] === " ") start -= 1;
                else if (start > 1 && current.slice(start - 2, start) === ", ") start -= 2;
                // Consume the separator we added after the snippet (prepend
                // mode adds ", ", append mode added " ").
                if (current.slice(end, end + 2) === ", ") end += 2;
                else if (end < current.length && current[end] === " ") end += 1;
                next = (current.slice(0, start) + current.slice(end)).trim();
            }
        }

        if (next === null) {
            // The user edited our snippet by hand. Keep it rather than risk
            // deleting something they typed.
            kept += 1;
            continue;
        }

        setWidgetValue(node, widget, next);
        removed += 1;
        labels.push(nodeLabel(node));
    }

    if (removed && !kept) {
        showToast(`已移除「${rawName}」自动追加的触发词（${labels.join("、")}）`, "info");
    } else if (removed && kept) {
        showToast(
            `已移除部分触发词；其余提示词被手动修改过，保留未动以免误删`,
            "warning",
        );
    } else if (kept) {
        showToast(
            `「${rawName}」的触发词未自动移除：提示词已被手动修改或节点已变化，保留原样`,
            "warning",
        );
    }
}
