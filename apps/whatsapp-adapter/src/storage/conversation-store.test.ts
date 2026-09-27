import { afterEach, describe, expect, it } from "vitest";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { ConversationStore } from "./conversation-store.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) =>
      fs.rm(directory, { recursive: true, force: true })
    )
  );
});

async function makeStore(limit = 10): Promise<ConversationStore> {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "wa-store-"));
  temporaryDirectories.push(directory);
  return new ConversationStore(directory, limit);
}

describe("ConversationStore", () => {
  it("persiste somente as mensagens mais recentes", async () => {
    const store = await makeStore(10);
    const messages = Array.from({ length: 14 }, (_, index) => ({
      role: index % 2 === 0 ? ("user" as const) : ("assistant" as const),
      content: `mensagem-${index}`,
    }));

    await store.saveRecent("5521999999999", messages);

    const loaded = await store.loadRecent("5521999999999");
    expect(loaded).toHaveLength(10);
    expect(loaded[0]?.content).toBe("mensagem-4");
    expect(loaded.at(-1)?.content).toBe("mensagem-13");
  });

  it("isola o histórico por contato", async () => {
    const store = await makeStore();
    await store.saveRecent("contato-a", [{ role: "user", content: "A" }]);
    await store.saveRecent("contato-b", [{ role: "user", content: "B" }]);

    await expect(store.loadRecent("contato-a")).resolves.toEqual([
      { role: "user", content: "A" },
    ]);
    await expect(store.loadRecent("contato-b")).resolves.toEqual([
      { role: "user", content: "B" },
    ]);
  });

  it("lê o formato de histórico legado", async () => {
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), "wa-legacy-"));
    temporaryDirectories.push(directory);
    const historyDirectory = path.join(
      directory,
      "contacts",
      "5521999999999",
      "conversation"
    );
    await fs.mkdir(historyDirectory, { recursive: true });
    await fs.writeFile(
      path.join(historyDirectory, "history.json"),
      JSON.stringify({
        chatId: "5521999999999",
        messages: [
          {
            role: "user",
            text: "mensagem antiga",
            at: "2026-09-25T00:00:00.000Z",
          },
        ],
      })
    );

    const store = new ConversationStore(directory, 10);
    await expect(store.loadRecent("5521999999999")).resolves.toEqual([
      { role: "user", content: "mensagem antiga" },
    ]);
  });
});
