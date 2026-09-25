import { describe, expect, it } from "vitest";
import {
  checkAccess,
  checkAccessCandidates,
  extractSenderCandidates,
  extractSenderDigits,
} from "../whatsapp/access-control.js";
import type { EnvConfig } from "../config/env.js";
import { testEnvConfig } from "../test/test-env-config.js";

function cfg(contacts: string[]): EnvConfig {
  return testEnvConfig({ allowedContacts: contacts });
}

describe("extractSenderCandidates", () => {
  it("extrai dígitos do remoteJid em PV", () => {
    const msg = {
      key: { remoteJid: "5511999999999@s.whatsapp.net", participant: undefined },
    };
    expect(extractSenderCandidates(msg as never)).toContain("5511999999999");
  });

  it("ignora sufixo de device no JID (5511...:0)", () => {
    const msg = {
      key: {
        remoteJid: "5511999999999:0@s.whatsapp.net",
        participant: undefined,
      },
    };
    expect(extractSenderCandidates(msg as never)).toContain("5511999999999");
    expect(extractSenderDigits(msg as never)).toBe("5511999999999");
  });

  it("usa senderPn quando disponível (fallback LID)", () => {
    const msg = {
      key: {
        remoteJid: "12345678901234@lid",
        participant: undefined,
        senderPn: "+55 11 98888-8888",
      },
    };
    expect(extractSenderCandidates(msg as never)).toContain("5511988888888");
  });

  it("prioriza participant em grupo", () => {
    const msg = {
      key: {
        remoteJid: "120363@g.us",
        participant: "5511888888888@s.whatsapp.net",
      },
    };
    expect(extractSenderCandidates(msg as never)).toContain("5511888888888");
  });
});

describe("checkAccessCandidates", () => {
  it("permite se qualquer candidato está na lista", () => {
    expect(
      checkAccessCandidates(
        ["5511999999999", "5511888888888"],
        cfg(["5511999999999"])
      )
    ).toEqual({ allowed: true, authorizedContactId: "5511999999999" });
  });

  it("nega se nenhum candidato está na lista", () => {
    expect(
      checkAccessCandidates(["5511777777777"], cfg(["5511999999999"]))
    ).toEqual({ allowed: false, reason: "not_in_allowlist" });
  });
});

describe("checkAccess", () => {
  it("permite número na lista", () => {
    expect(checkAccess("5511999999999", cfg(["5511999999999"]))).toEqual({
      allowed: true,
      authorizedContactId: "5511999999999",
    });
  });

  it("nega número fora da lista", () => {
    expect(checkAccess("5511999999999", cfg(["5511888888888"]))).toEqual({
      allowed: false,
      reason: "not_in_allowlist",
    });
  });

  it("nega lista vazia", () => {
    expect(checkAccess("5511999999999", cfg([]))).toEqual({
      allowed: false,
      reason: "not_in_allowlist",
    });
  });
});
