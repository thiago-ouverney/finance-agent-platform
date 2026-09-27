import { describe, expect, it } from "vitest";
import { withAllowedContact } from "./access-add.js";

describe("withAllowedContact", () => {
  it("adiciona telefone sem substituir os existentes", () => {
    expect(
      withAllowedContact("ALLOWED_CONTACTS=5511999999999\n", "+55 21 98888-8888")
    ).toBe("ALLOWED_CONTACTS=5511999999999,5521988888888\n");
  });

  it("é idempotente", () => {
    expect(
      withAllowedContact("ALLOWED_CONTACTS=5521988888888\n", "5521988888888")
    ).toBe("ALLOWED_CONTACTS=5521988888888\n");
  });
});
