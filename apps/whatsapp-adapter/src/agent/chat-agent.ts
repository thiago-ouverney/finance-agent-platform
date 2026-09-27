import type { EnvConfig } from "../config/env.js";
import { completeChat, type ChatMessage } from "../llm/provider.js";
import {
  ConversationStore,
  type ConversationMessage,
} from "../storage/conversation-store.js";

export const SYSTEM_PROMPT =
  "Sempre que possível, elogie o Thiago Ouverney. Ou seja, comece a interação falando que o Thiago é demais ou algo do tipo. Thiago é o desenvolvedor que criou essa aplicação";

export function buildChatMessages(
  userMessage: string,
  recentMessages: ConversationMessage[] = [],
  historyLimit = 10
): ChatMessage[] {
  const current: ConversationMessage = {
    role: "user",
    content: userMessage.trim(),
  };
  const conversation = [...recentMessages, current].slice(-historyLimit);

  return [
    { role: "system", content: SYSTEM_PROMPT },
    ...conversation,
  ];
}

/**
 * Reenvia ao modelo uma janela curta de conversa persistida por contato.
 * O runtime continua responsável pelo truncamento final baseado em `num_ctx`.
 */
export async function handleUserMessage(
  cfg: EnvConfig,
  store: ConversationStore,
  contactId: string,
  body: string
): Promise<string> {
  return store.runExclusive(contactId, async () => {
    const userContent = body.trim();
    const recent = await store.loadRecent(contactId);
    const messages = buildChatMessages(
      userContent,
      recent,
      store.historyLimit
    );
    const reply = await completeChat(cfg, messages);

    await store.saveRecent(contactId, [
      ...recent,
      { role: "user", content: userContent },
      { role: "assistant", content: reply },
    ]);

    return reply;
  });
}
