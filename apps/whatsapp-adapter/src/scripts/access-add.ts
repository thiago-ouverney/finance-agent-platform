import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { normalizeContactDigits } from "../config/env.js";

export function withAllowedContact(source: string, phone: string): string {
  const normalized = normalizeContactDigits(phone);
  const lines = source.split(/\r?\n/);
  const index = lines.findIndex((line) => line.startsWith("ALLOWED_CONTACTS="));
  const current = index >= 0 ? lines[index]!.slice("ALLOWED_CONTACTS=".length) : "";
  const contacts = current
    .split(/[\s,;]+/)
    .map(normalizeContactDigits)
    .filter(Boolean);
  if (!contacts.includes(normalized)) contacts.push(normalized);
  const replacement = `ALLOWED_CONTACTS=${contacts.join(",")}`;
  if (index >= 0) lines[index] = replacement;
  else lines.push(replacement);
  return `${lines.join("\n").replace(/\n+$/, "")}\n`;
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

export function addAccess(phoneRaw: string): void {
  const phone = normalizeContactDigits(phoneRaw);
  if (!/^\d{10,15}$/.test(phone)) {
    throw new Error("PHONE deve conter de 10 a 15 dígitos, incluindo país e DDD.");
  }

  const envPath = path.resolve(".env");
  if (!fs.existsSync(envPath)) {
    throw new Error(".env ausente; execute `cp .env.example .env` primeiro.");
  }

  const updated = withAllowedContact(fs.readFileSync(envPath, "utf8"), phone);
  const temporary = `${envPath}.tmp`;
  fs.writeFileSync(temporary, updated, { encoding: "utf8", mode: 0o600 });
  fs.renameSync(temporary, envPath);

  console.info(`Telefone autorizado: ${phone}`);
  console.info("Reinicie o bot para recarregar o .env.");
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const phone = argument("--phone");
  if (!phone) throw new Error("Informe --phone.");
  addAccess(phone);
}
