export interface ServerSentEvent<T = unknown> {
  event: string;
  data: T;
}

export function parseSseBlock(block: string): ServerSentEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return null;
  try {
    return { event, data: JSON.parse(data.join("\n")) as unknown };
  } catch {
    return null;
  }
}

export function drainSseBuffer(buffer: string): { events: ServerSentEvent[]; remainder: string } {
  const normalized = buffer.replaceAll("\r\n", "\n");
  const blocks = normalized.split("\n\n");
  const remainder = blocks.pop() ?? "";
  return {
    events: blocks.map(parseSseBlock).filter((event): event is ServerSentEvent => event !== null),
    remainder,
  };
}
