import { describe, expect, it } from "vitest";
import {
  isPermanentFatalDisconnect,
  MAX_RECONNECT_SCHEDULES_DEFAULT,
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
    expect(
      isPermanentFatalDisconnect(
        "error: 'better-sqlite3' is not yet supported in Bun."
      )
    ).toBe(true);
    expect(isPermanentFatalDisconnect("logged out")).toBe(false);
  });
});
