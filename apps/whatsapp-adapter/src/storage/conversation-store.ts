import { randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";

export type ConversationMessage = {
  role: "user" | "assistant";
  content: string;
};

type StoredConversation = {
  chatId: string;
  messages: Array<{
    role: "user" | "assistant";
    text: string;
    at: string;
  }>;
};

function toConversationMessage(
  value: unknown
): ConversationMessage | undefined {
  if (!value || typeof value !== "object") return undefined;
  const candidate = value as {
    role?: unknown;
    text?: unknown;
    content?: unknown;
  };
  if (candidate.role !== "user" && candidate.role !== "assistant") {
    return undefined;
  }
  const content =
    typeof candidate.text === "string"
      ? candidate.text
      : typeof candidate.content === "string"
        ? candidate.content
        : undefined;
  return content === undefined ? undefined : { role: candidate.role, content };
}

function safeContactId(contactId: string): string {
  return contactId.replace(/[^a-zA-Z0-9_-]/g, "_");
}

/**
 * Histórico curto e persistente por contato.
 * O limite é por mensagem individual (user ou assistant), sem contar o system prompt.
 */
export class ConversationStore {
  private readonly queues = new Map<string, Promise<void>>();

  constructor(
    private readonly dataDir: string,
    readonly historyLimit: number
  ) {
    if (!Number.isInteger(historyLimit) || historyLimit < 1) {
      throw new Error("historyLimit deve ser um inteiro positivo");
    }
  }

  private filePath(contactId: string): string {
    return path.join(
      this.dataDir,
      "contacts",
      safeContactId(contactId),
      "conversation",
      "history.json"
    );
  }

  async loadRecent(contactId: string): Promise<ConversationMessage[]> {
    try {
      const raw = await fs.readFile(this.filePath(contactId), "utf8");
      const parsed = JSON.parse(raw) as Partial<StoredConversation>;
      if (!Array.isArray(parsed.messages)) return [];
      return parsed.messages
        .map(toConversationMessage)
        .filter((message): message is ConversationMessage => Boolean(message))
        .slice(-this.historyLimit);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw error;
    }
  }

  async saveRecent(
    contactId: string,
    messages: ConversationMessage[]
  ): Promise<void> {
    const destination = this.filePath(contactId);
    await fs.mkdir(path.dirname(destination), { recursive: true });
    const temporary = `${destination}.${randomUUID()}.tmp`;
    const conversation: StoredConversation = {
      chatId: contactId,
      messages: messages.slice(-this.historyLimit).map((message) => ({
        role: message.role,
        text: message.content,
        at: new Date().toISOString(),
      })),
    };

    try {
      await fs.writeFile(temporary, JSON.stringify(conversation, null, 2));
      await fs.rename(temporary, destination);
    } catch (error) {
      await fs.rm(temporary, { force: true });
      throw error;
    }
  }

  /** Serializa mensagens simultâneas do mesmo contato sem bloquear outros contatos. */
  async runExclusive<T>(
    contactId: string,
    operation: () => Promise<T>
  ): Promise<T> {
    const previous = this.queues.get(contactId) ?? Promise.resolve();
    let release = (): void => undefined;
    const current = new Promise<void>((resolve) => {
      release = resolve;
    });
    const tail = previous.catch(() => undefined).then(() => current);
    this.queues.set(contactId, tail);

    await previous.catch(() => undefined);
    try {
      return await operation();
    } finally {
      release();
      if (this.queues.get(contactId) === tail) {
        this.queues.delete(contactId);
      }
    }
  }
}
