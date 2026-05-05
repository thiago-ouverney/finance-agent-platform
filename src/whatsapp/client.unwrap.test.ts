import { describe, expect, it } from "vitest";
import { unwrapMessage } from "../whatsapp/client.js";
import { proto } from "@whiskeysockets/baileys";

describe("unwrapMessage", () => {
  it("desembrulha documentWithCaptionMessage", () => {
    const inner: proto.IMessage = {
      documentMessage: {
        fileName: "x.csv",
        caption: "legenda",
      },
    };
    const wrapped: proto.IMessage = {
      documentWithCaptionMessage: { message: inner },
    };
    const out = unwrapMessage(wrapped);
    expect(out?.documentMessage?.fileName).toBe("x.csv");
  });

  it("desembrulha ephemeralMessage", () => {
    const inner: proto.IMessage = { conversation: "oi" };
    const wrapped: proto.IMessage = {
      ephemeralMessage: { message: inner },
    };
    expect(unwrapMessage(wrapped)?.conversation).toBe("oi");
  });
});
