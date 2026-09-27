import { DisconnectReason } from "@whiskeysockets/baileys";

/** Número máximo de agendamentos de reconexão sem `connection === "open"` no meio. */
export const MAX_RECONNECT_SCHEDULES_DEFAULT = 12;

const PERMANENT_FATAL_SUBSTRINGS = [
  "crypto is not defined",
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
 * Não reconecta quando a sessão exige intervenção ou foi assumida por outra
 * instância. Em especial, repetir após `connectionReplaced` cria um loop 440.
 */
export function shouldReconnectAfterStatus(
  statusCode: number | undefined
): boolean {
  return (
    statusCode !== DisconnectReason.loggedOut &&
    statusCode !== DisconnectReason.badSession &&
    statusCode !== DisconnectReason.connectionReplaced
  );
}

/**
 * Erros que não melhoram com reconexão — exigem mudança de runtime/config.
 */
export function isPermanentFatalDisconnect(reason: string): boolean {
  const r = reason.toLowerCase();
  return PERMANENT_FATAL_SUBSTRINGS.some((s) => r.includes(s));
}
