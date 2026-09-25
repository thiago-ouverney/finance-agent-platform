/**
 * Node 18 não define `globalThis.crypto` (Web Crypto) por padrão; o Baileys usa `crypto.subtle`.
 * Node 20+ costuma expor globalmente. Este módulo é idempotente.
 */
import { webcrypto } from "node:crypto";

const g = globalThis as typeof globalThis & { crypto?: typeof webcrypto };

if (typeof g.crypto === "undefined") {
  Object.defineProperty(g, "crypto", {
    value: webcrypto,
    configurable: true,
    enumerable: true,
    writable: false,
  });
}
