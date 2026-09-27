import { describe, expect, it } from "vitest";
import {
  isPermanentFatalDisconnect,
  MAX_RECONNECT_SCHEDULES_DEFAULT,
  shouldReconnectAfterStatus,
  shouldScheduleAnotherReconnect,
} from "./reconnect-policy.js";

describe("reconnect-policy", () => {
  it("shouldScheduleAnotherReconnect respeita o teto", () => {
    expect(shouldScheduleAnotherReconnect(0)).toBe(true);
    expect(
      shouldScheduleAnotherReconnect(MAX_RECONNECT_SCHEDULES_DEFAULT - 1)
    ).toBe(true);
    expect(
      shouldScheduleAnotherReconnect(MAX_RECONNECT_SCHEDULES_DEFAULT)
    ).toBe(false);
  });

  it("isPermanentFatalDisconnect detecta erros de config/runtime", () => {
    expect(isPermanentFatalDisconnect("ReferenceError: crypto is not defined")).toBe(
      true
    );
    expect(isPermanentFatalDisconnect("temporary timeout")).toBe(false);
    expect(isPermanentFatalDisconnect("logged out")).toBe(false);
  });

  it("não reconecta quando outra instância substitui a sessão", () => {
    expect(shouldReconnectAfterStatus(440)).toBe(false);
    expect(shouldReconnectAfterStatus(401)).toBe(false);
    expect(shouldReconnectAfterStatus(500)).toBe(false);
    expect(shouldReconnectAfterStatus(408)).toBe(true);
  });
});
