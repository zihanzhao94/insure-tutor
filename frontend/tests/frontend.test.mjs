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
  join(frontend, "src", "stream.ts"), join(frontend, "src", "citationParts.ts"), join(frontend, "src", "api.ts"),
]);
// The browser's bundler resolves extensionless imports; native Node ESM needs .js.
const compiledApi = join(output, "api.js");
writeFileSync(compiledApi, readFileSync(compiledApi, "utf8").replace('from "./stream"', 'from "./stream.js"'));
const { SseParser, readChatStream } = await import(pathToFileURL(join(output, "stream.js")));
const { citationParts } = await import(pathToFileURL(join(output, "citationParts.js")));
const { sendMessageStream } = await import(pathToFileURL(compiledApi));

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

test("known adjacent citations become links while unknown markers remain text", () => {
  const second = { ...citation, pdf_page: 16, url: "/api/documents/brochure.pdf#page=16" };
  const parts = citationParts("Answer [1][2]. Unknown [3], [0].", [citation, second]);
  assert.deepEqual(parts.map(part => part.type), ["text", "citation", "citation", "text"]);
  assert.deepEqual(parts.filter(part => part.type === "citation").map(part => part.number), [1, 2]);
  assert.equal(parts.at(-1).text, ". Unknown [3], [0].");
});

test("citation parsing preserves plain text and rejects non-document URLs", () => {
  const text = '<script>alert("x")</script> [1]';
  assert.deepEqual(citationParts(text, [{ ...citation, url: "javascript:alert(1)" }]), [{ type: "text", text }]);
  assert.deepEqual(citationParts("No references.", []), [{ type: "text", text: "No references." }]);
  assert.deepEqual(citationParts("", [citation]), []);
});
