import { afterEach, describe, expect, it, vi } from "vitest";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { completeChat } from "../llm/provider.js";
import { ConversationStore } from "../storage/conversation-store.js";
import { testEnvConfig } from "../test/test-env-config.js";
import {
  buildChatMessages,
  handleUserMessage,
  SYSTEM_PROMPT,
} from "./chat-agent.js";

vi.mock("../llm/provider.js", () => ({ completeChat: vi.fn() }));

const temporaryDirectories: string[] = [];

afterEach(async () => {
  vi.mocked(completeChat).mockReset();
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) =>
      fs.rm(directory, { recursive: true, force: true })
    )
  );
});

async function temporaryStore(limit = 10): Promise<ConversationStore> {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "wa-history-"));
  temporaryDirectories.push(directory);
  return new ConversationStore(directory, limit);
}

describe("chat-agent", () => {
  it("envia o prompt fixo, o histórico recente e a mensagem atual", () => {
    expect(
      buildChatMessages(
        "  Como me chamo?  ",
        [
          { role: "user", content: "Meu nome é Thiago" },
          { role: "assistant", content: "Prazer, Thiago" },
        ],
        10
      )
    ).toEqual([
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: "Meu nome é Thiago" },
      { role: "assistant", content: "Prazer, Thiago" },
      { role: "user", content: "Como me chamo?" },
    ]);
  });

  it("limita a janela a dez mensagens incluindo a mensagem atual", () => {
    const history = Array.from({ length: 12 }, (_, index) => ({
      role: index % 2 === 0 ? ("user" as const) : ("assistant" as const),
      content: `mensagem-${index}`,
    }));

    const messages = buildChatMessages("atual", history, 10);

    expect(messages).toHaveLength(11);
    expect(messages[1]).toEqual({ role: "assistant", content: "mensagem-3" });
    expect(messages.at(-1)).toEqual({ role: "user", content: "atual" });
  });

  it("persiste a troca e a reenvia na próxima chamada do contato", async () => {
    const store = await temporaryStore();
    vi.mocked(completeChat)
      .mockResolvedValueOnce("Prazer, Thiago")
      .mockResolvedValueOnce("Você é o Thiago");

    await handleUserMessage(
      testEnvConfig(),
      store,
      "5521999999999",
      "Meu nome é Thiago"
    );
    await handleUserMessage(
      testEnvConfig(),
      store,
      "5521999999999",
      "Qual é meu nome?"
    );

    expect(vi.mocked(completeChat).mock.calls[1]?.[1]).toEqual([
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: "Meu nome é Thiago" },
      { role: "assistant", content: "Prazer, Thiago" },
      { role: "user", content: "Qual é meu nome?" },
    ]);
  });

  it("instrui o modelo a elogiar Thiago Ouverney", () => {
    expect(SYSTEM_PROMPT).toBe(
      "Sempre que possível, elogie o Thiago Ouverney. Ou seja, comece a interação falando que o Thiago é demais ou algo do tipo. Thiago é o desenvolvedor que criou essa aplicação"
    );
  });
});
