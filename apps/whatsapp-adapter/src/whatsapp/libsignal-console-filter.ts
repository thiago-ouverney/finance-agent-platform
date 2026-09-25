/**
 * libsignal (session_cipher.js) chama console.error antes de lançar em falhas
 * de descriptografia — isso não passa pelo logger Pino do Baileys.
 */

export function isLibsignalSessionDecryptNoiseArgs(
  args: readonly unknown[]
): boolean {
  const head = args[0];
  if (typeof head !== "string") return false;
  if (head.includes("Failed to decrypt message with any known session"))
    return true;
  if (!head.startsWith("Session error:")) return false;
  const rest = args
    .slice(1)
    .map((a) =>
      typeof a === "string"
        ? a
        : a instanceof Error
          ? a.stack ?? String(a)
          : ""
    )
    .join("\n");
  return (
    rest.includes("libsignal") ||
    rest.includes("session_cipher") ||
    head.includes("Bad MAC")
  );
}

let patched = false;
let origConsoleError: typeof console.error;

/**
 * Com Baileys em `silent`, suprime ruído conhecido do libsignal em `console.error`.
 * Com níveis verbosos, restaura o `console.error` original se estava patchado.
 */
export function applyBaileysDecryptConsoleFilter(silent: boolean): void {
  if (silent) {
    if (patched) return;
    origConsoleError = console.error.bind(console);
    patched = true;
    console.error = (...args: unknown[]) => {
      if (isLibsignalSessionDecryptNoiseArgs(args)) return;
      origConsoleError(...args);
    };
  } else if (patched) {
    console.error = origConsoleError;
    patched = false;
  }
}
