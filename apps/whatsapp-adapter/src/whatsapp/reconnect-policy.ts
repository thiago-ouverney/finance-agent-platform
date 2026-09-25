/** Número máximo de agendamentos de reconexão sem `connection === "open"` no meio. */
export const MAX_RECONNECT_SCHEDULES_DEFAULT = 12;

const PERMANENT_FATAL_SUBSTRINGS = [
  "crypto is not defined",
  "not yet supported in bun",
  "better-sqlite3",
  "err_dlopen_failed",
] as const;

/**
 * Se false, não se deve agendar outra reconexão (circuit breaker por contagem).
 */
export function shouldScheduleAnotherReconnect(
  schedulesSoFar: number,
  maxSchedules: number = MAX_RECONNECT_SCHEDULES_DEFAULT
): boolean {
  return schedulesSoFar < maxSchedules;
}

/**
 * Erros que não melhoram com reconexão — exigem mudança de runtime/config.
 */
export function isPermanentFatalDisconnect(reason: string): boolean {
  const r = reason.toLowerCase();
  return PERMANENT_FATAL_SUBSTRINGS.some((s) => r.includes(s));
}
