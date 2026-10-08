type Schedule = (callback: FrameRequestCallback) => number;

/** Smooth validated text updates; corrected results replace queued previews immediately. */
export class ProgressiveText {
  private target = "";
  private visible = "";
  private frame: number | undefined;
  private previousTime: number | undefined;
  private disposed = false;

  constructor(
    private publish: (text: string) => void,
    private schedule: Schedule = callback => requestAnimationFrame(callback),
    private cancel: (id: number) => void = id => cancelAnimationFrame(id),
  ) {}

  update(text: string, animate = true) {
    if (this.disposed) return;
    this.target = text;
    if (!animate || !text.startsWith(this.visible)) {
      this.clearFrame();
      this.visible = text;
      this.publish(text);
    } else if (this.visible !== text && this.frame === undefined) {
      this.frame = this.schedule(this.tick);
    }
  }

  private tick = (time: number) => {
    this.frame = undefined;
    if (this.disposed) return;
    const elapsed = this.previousTime === undefined ? 16 : Math.min(50, time - this.previousTime);
    this.previousTime = time;
    const pending = Array.from(this.target.slice(this.visible.length));
    const speed = Math.min(240, 80 + pending.length / 8);
    const count = Math.max(1, Math.floor(elapsed * speed / 1000));
    let end = this.visible.length + pending.slice(0, count).join("").length;
    // Citation markers appear as a complete button, never as a partial "[1".
    for (const match of this.target.matchAll(/\[[1-9]\d*\]/g)) {
      if (match.index < end && end < match.index + match[0].length) end = match.index + match[0].length;
    }
    this.visible = this.target.slice(0, end);
    this.publish(this.visible);
    if (this.visible !== this.target) this.frame = this.schedule(this.tick);
    else this.previousTime = undefined;
  };

  private clearFrame() {
    if (this.frame !== undefined) this.cancel(this.frame);
    this.frame = undefined;
    this.previousTime = undefined;
  }

  dispose() {
    this.disposed = true;
    this.clearFrame();
  }
}
