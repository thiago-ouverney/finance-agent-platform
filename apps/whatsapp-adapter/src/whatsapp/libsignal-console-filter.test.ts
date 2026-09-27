import { describe, expect, it } from "vitest";
import { isLibsignalSessionDecryptNoiseArgs } from "./libsignal-console-filter.js";

describe("isLibsignalSessionDecryptNoiseArgs", () => {
  it("detecta mensagem fixa do session_cipher", () => {
    expect(
      isLibsignalSessionDecryptNoiseArgs([
        "Failed to decrypt message with any known session...",
      ])
    ).toBe(true);
  });

  it("detecta Session error com Bad MAC no primeiro arg", () => {
    expect(
      isLibsignalSessionDecryptNoiseArgs([
        "Session error:Error: Bad MAC Error: Bad MAC",
        "    at Object.verifyMAC (.../libsignal/src/crypto.js:87:15)\n",
      ])
    ).toBe(true);
  });

  it("não suprime erro genérico de app", () => {
    expect(
      isLibsignalSessionDecryptNoiseArgs(["[whatsapp] Erro ao processar mensagem:", new Error("x")])
    ).toBe(false);
  });
});
