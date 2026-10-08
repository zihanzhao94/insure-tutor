import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

// Compile the actual helpers with the existing TypeScript installation.
const frontend = dirname(dirname(fileURLToPath(import.meta.url)));
const output = mkdtempSync(join(tmpdir(), "insure-tutor-frontend-tests-"));
after(() => rmSync(output, { recursive: true, force: true }));
writeFileSync(join(output, "package.json"), '{"type":"module"}');
execFileSync(join(frontend, "node_modules", ".bin", "tsc"), [
  "--target", "ES2022", "--module", "ES2022", "--moduleResolution", "Bundler",
  "--lib", "ES2022,DOM", "--skipLibCheck", "--strict", "--outDir", output,
  join(frontend, "src", "stream.ts"), join(frontend, "src", "citationParts.ts"), join(frontend, "src", "api.ts"), join(frontend, "src", "progressiveText.ts"),
]);
// The browser's bundler resolves extensionless imports; native Node ESM needs .js.
const compiledApi = join(output, "api.js");
writeFileSync(compiledApi, readFileSync(compiledApi, "utf8").replace('from "./stream"', 'from "./stream.js"'));
const { SseParser, readChatStream } = await import(pathToFileURL(join(output, "stream.js")));
const { citationParts, sourceSnippet } = await import(pathToFileURL(join(output, "citationParts.js")));
const { sendMessageStream } = await import(pathToFileURL(compiledApi));
const { ProgressiveText } = await import(pathToFileURL(join(output, "progressiveText.js")));

const citation = {
  document_id: "brochure", filename: "brochure.pdf", pdf_page: 8,
  excerpt: "Original text", chunk_id: "p8:c1", url: "/api/documents/brochure.pdf#page=8",
};
const response = {
  answer: "这是完整回答。 [1]", session_id: "a".repeat(32),
  citations: [citation], status: "answered", mode: "llm",
};
const event = (name, data) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
const byteStream = text => {
  const bytes = new TextEncoder().encode(text);
  let position = 0;
  return new ReadableStream({ pull(controller) {
    if (position < bytes.length) controller.enqueue(bytes.slice(position, ++position));
    else controller.close();
  } });
};

test("SSE framing handles split CRLF, comments and multiline data", () => {
  const received = [];
  const parser = new SseParser((name, data) => received.push({ name, data }));
  const text = ': keepalive\r\nevent: delta\r\ndata: {"text":\r\ndata: "中文"}\r\n\r\n';
  for (const character of text) parser.push(character);
  assert.deepEqual(received, [{ name: "delta", data: '{"text":\n"中文"}' }]);
});

test("SSE decodes fragmented Chinese bytes and completes on the final result", async () => {
  const received = [];
  const stream = byteStream(
    event("start", { session_id: response.session_id, mode: response.mode }) +
    event("status", { phase: "generating" }) +
    event("delta", { text: response.answer, citations: [citation] }) +
    event("result", response),
  );
  assert.deepEqual(await readChatStream(stream, update => received.push(update)), response);
  assert.equal(received[2].data.text, response.answer);
  assert.equal(stream.locked, false);
});

test("SSE accepts CR-only line endings at EOF", async () => {
  const stream = byteStream(event("result", response).replaceAll("\n", "\r"));
  assert.deepEqual(await readChatStream(stream, () => {}), response);
});

test("a reset and insufficient final result replace provisional content", async () => {
  const received = [];
  const final = { ...response, answer: "证据不足。", citations: [], status: "insufficient_evidence" };
  const stream = byteStream(
    event("delta", { text: "Provisional claim [1]", citations: [citation] }) +
    event("reset", {}) + event("result", final),
  );
  assert.deepEqual(await readChatStream(stream, update => received.push(update)), final);
  assert.deepEqual(received.map(update => update.event), ["delta", "reset", "result"]);
});

test("a final result completes without waiting for a server to close the body", async () => {
  let cancelled = false;
  const stream = new ReadableStream({
    start(controller) { controller.enqueue(new TextEncoder().encode(event("result", response))); },
    cancel() { cancelled = true; },
  });
  assert.deepEqual(await readChatStream(stream, () => {}), response);
  assert.equal(cancelled, true);
  assert.equal(stream.locked, false);
});

test("an interrupted stream is a failure, rather than a finished partial answer", async () => {
  const stream = byteStream(event("delta", { text: "Partial", citations: [] }));
  await assert.rejects(readChatStream(stream, () => {}), /ended before completion/);
  assert.equal(stream.locked, false);
});

test("server errors and malformed events fail and release the body", async () => {
  for (const [text, expected] of [
    [event("delta", { text: "Partial", citations: [] }) + event("error", { detail: "Model unavailable" }), /Model unavailable/],
    ['event: delta\ndata: {broken}\n\n', /JSON/],
    [event("delta", { text: 123, citations: [] }), /invalid update/],
  ]) {
    const stream = byteStream(text);
    await assert.rejects(readChatStream(stream, () => {}), expected);
    assert.equal(stream.locked, false);
  }
});

test("cancellation stops a pending read and releases its reader", async () => {
  const controller = new AbortController();
  let cancelled = false;
  const stream = new ReadableStream({ cancel() { cancelled = true; } });
  const pending = readChatStream(stream, () => {}, controller.signal);
  controller.abort();
  await assert.rejects(pending, error => error.name === "AbortError");
  assert.equal(cancelled, true);
  assert.equal(stream.locked, false);
});

test("API cancellation forwards to fetch and releases an active response stream", { timeout: 2000 }, async () => {
  const originalFetch = globalThis.fetch;
  const originalWindow = globalThis.window;
  let fetchSignal;
  let cancelled = false;
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(event("start", {
        session_id: response.session_id, mode: response.mode,
      })));
      // Hold the response open while the next model claim is pending.
    },
    cancel() { cancelled = true; },
  });
  try {
    globalThis.fetch = async (_url, options) => {
      assert.deepEqual(JSON.parse(options.body), { message: "Explain the guarantee", ui_language: "en" });
      fetchSignal = options.signal;
      return new Response(stream, { headers: { "Content-Type": "text/event-stream" } });
    };
    globalThis.window = globalThis;
    const controller = new AbortController();
    let firstEvent;
    const started = new Promise(resolve => { firstEvent = resolve; });
    const pending = sendMessageStream({ message: "Explain the guarantee", ui_language: "en" }, () => firstEvent(), controller.signal);
    await started;
    controller.abort();
    await assert.rejects(pending, error => error.name === "AbortError");
    assert.equal(fetchSignal.aborted, true);
    assert.equal(cancelled, true);
    assert.equal(stream.locked, false);
  } finally {
    globalThis.fetch = originalFetch;
    globalThis.window = originalWindow;
  }
});

test("known adjacent citations are resolved while unknown markers remain text", () => {
  const second = { ...citation, pdf_page: 16, url: "/api/documents/brochure.pdf#page=16" };
  const parts = citationParts("Answer [1][2]. Unknown [3], [0].", [citation, second]);
  assert.deepEqual(parts.map(part => part.type), ["text", "citation", "citation", "text"]);
  assert.deepEqual(parts.filter(part => part.type === "citation").map(part => part.number), [1, 2]);
  assert.equal(parts.at(-1).text, ". Unknown [3], [0].");
});

test("source snippets join PDF line wraps without changing language or policy figures", () => {
  assert.equal(sourceSnippet("The Account Value\n is guaranteed\t after 15 years.\n\nRate: 2.5% p.a."),
    "The Account Value is guaranteed after 15 years. Rate: 2.5% p.a.");
  assert.equal(sourceSnippet("保 單生效\n滿15年 , 賬戶價\n值包含 2.5% 及 HKD 48,000。"),
    "保單生效滿15年 , 賬戶價值包含 2.5% 及 HKD 48,000。");
  assert.equal(sourceSnippet("1 4% 0.25%\n2 4% 0.25%"), "1 4% 0.25%\n2 4% 0.25%");
});

test("long snippets mark truncation without splitting Unicode or monetary values", () => {
  const prefix = "文".repeat(415) + " ";
  assert.equal(sourceSnippet(prefix + "48,000.25% more text"), "文".repeat(415) + "…");
  const sentence = "文".repeat(300) + "。";
  assert.equal(sourceSnippet(sentence + "文".repeat(200)), sentence + "…");
  assert.equal(sourceSnippet("文".repeat(419) + "𠮷" + "文".repeat(20)), "文".repeat(419) + "𠮷…");
});

test("citation parsing preserves plain text and rejects non-document URLs", () => {
  const text = '<script>alert("x")</script> [1]';
  assert.deepEqual(citationParts(text, [{ ...citation, url: "javascript:alert(1)" }]), [{ type: "text", text }]);
  assert.deepEqual(citationParts("No references.", []), [{ type: "text", text: "No references." }]);
  assert.deepEqual(citationParts("", [citation]), []);
});

function playbackHarness() {
  let next = 0, time = 0;
  const frames = new Map(), updates = [];
  const playback = new ProgressiveText(text => updates.push(text),
    callback => { frames.set(++next, callback); return next; }, id => frames.delete(id));
  function tick() {
    time += 16;
    const pending = [...frames.values()]; frames.clear();
    pending.forEach(callback => callback(time));
  }
  function finish() {
    for (let i = 0; frames.size && i < 10000; i++) tick();
    assert.equal(frames.size, 0);
  }
  return { playback, updates, frames, tick, finish };
}

test("playback progressively reveals live appends and the canonical final suffix", () => {
  const h = playbackHarness();
  const first = "The cooling-off period is 21 days. [1]";
  h.playback.update(first); h.tick();
  assert.ok(h.updates[0].length > 0 && h.updates[0].length < first.length);
  const final = first + "\n\nPremiums are refunded subject to conditions. [2]\n\nDisclaimer.";
  h.playback.update(final); h.finish();
  assert.equal(h.updates.at(-1), final);
  for (let i = 1; i < h.updates.length; i++) assert.ok(h.updates[i].startsWith(h.updates[i - 1]));
});

test("playback preserves Unicode and exposes citation markers atomically", () => {
  const h = playbackHarness();
  const final = "保單𠮷😀冷靜期為21天。 [12][1]";
  h.playback.update(final); h.finish();
  assert.equal(h.updates.at(-1), final);
  for (const text of h.updates) {
    assert.equal(text.isWellFormed(), true);
    assert.ok(!/\[\d*$/.test(text));
  }
});

test("rejection/reset replaces queued text immediately and discards pending animation", () => {
  const h = playbackHarness();
  h.playback.update("A supported preview that is still typing. [1]"); h.tick();
  h.playback.update("Insufficient evidence.", false);
  assert.equal(h.updates.at(-1), "Insufficient evidence.");
  assert.equal(h.frames.size, 0); h.tick();
  assert.equal(h.updates.at(-1), "Insufficient evidence.");
  h.playback.update("", false);
  h.playback.update("A repaired answer. [2]"); h.finish();
  assert.equal(h.updates.at(-1), "A repaired answer. [2]");
});

test("disposing canceled playback prevents late text updates", () => {
  const h = playbackHarness();
  h.playback.update("An answer is arriving."); h.tick();
  const count = h.updates.length;
  h.playback.dispose(); h.tick(); h.playback.update("Late result.");
  assert.equal(h.updates.length, count);
  assert.equal(h.frames.size, 0);
});
