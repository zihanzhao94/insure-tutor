export default function Chat() {
  // TODO: Add message history, request state, sendMessage(), and source cards.
  return (
    <section>
      <h2>Insurance plan questions</h2>
      <p>Project scaffold. Chat and document retrieval are not implemented yet.</p>
      <label htmlFor="question">Your question</label>
      <textarea id="question" rows={4} disabled />
      <button type="button" disabled>Send</button>
    </section>
  );
}
