import http from "node:http";
import { URL } from "node:url";
import type { EnvConfig } from "../config/env.js";
import { normalizeContactDigits } from "../config/env.js";
import {
  allFeatureKeys,
  FEATURE_CATALOG,
  isKnownFeatureKey,
} from "../policy/feature-registry.js";
import type { PolicyService } from "../policy/resolver.js";
import type { MessageTemplateKey } from "../policy/types.js";
import { ADMIN_SINGLE_PAGE_HTML } from "./admin-page-html.js";

function json(
  res: http.ServerResponse,
  status: number,
  body: unknown
): void {
  const s = JSON.stringify(body);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(s),
  });
  res.end(s);
}

function unauthorized(res: http.ServerResponse): void {
  res.writeHead(401, { "WWW-Authenticate": 'Bearer realm="admin"' });
  res.end("Unauthorized");
}

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => {
      chunks.push(Buffer.isBuffer(c) ? c : Buffer.from(c));
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function authOk(
  cfg: EnvConfig,
  req: http.IncomingMessage
): boolean {
  const tok = cfg.adminToken;
  if (!tok) return false;
  const h = req.headers.authorization;
  if (!h?.startsWith("Bearer ")) return false;
  return h.slice(7).trim() === tok;
}

function validateModel(cfg: EnvConfig, model: string): boolean {
  const m = model.trim();
  return cfg.allowedOpenaiModels.includes(m);
}

export function startAdminServer(
  cfg: EnvConfig,
  policySvc: PolicyService
): http.Server {
  const db = policySvc.db;

  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? "/", `http://${req.headers.host}`);
      const path = url.pathname;

      if (path === "/" && req.method === "GET") {
        res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
        res.end(ADMIN_SINGLE_PAGE_HTML);
        return;
      }

      if (!path.startsWith("/api/")) {
        res.writeHead(404);
        res.end();
        return;
      }

      if (!authOk(cfg, req)) {
        unauthorized(res);
        return;
      }

      if (path === "/api/health" && req.method === "GET") {
        json(res, 200, { ok: true });
        return;
      }

      if (path === "/api/feature-catalog" && req.method === "GET") {
        json(res, 200, {
          features: [...FEATURE_CATALOG],
        });
        return;
      }

      if (path === "/api/models" && req.method === "GET") {
        json(res, 200, { models: cfg.allowedOpenaiModels });
        return;
      }

      if (path === "/api/policy/reload" && req.method === "POST") {
        policySvc.invalidateCache();
        json(res, 200, { ok: true });
        return;
      }

      if (path === "/api/identities" && req.method === "GET") {
        const rows = db.listIdentities();
        const models = new Map(
          db.getLlmPolicies().map((p) => [p.contact_digits, p.model_id])
        );
        const feats = db.getFeatureMatrix();
        const byContact = new Map<
          string,
          Record<string, boolean>
        >();
        for (const f of feats) {
          if (!byContact.has(f.contact_digits)) {
            byContact.set(f.contact_digits, {});
          }
          const rec = byContact.get(f.contact_digits)!;
          rec[f.feature_key] = f.enabled === 1;
        }
        json(
          res,
          200,
          rows.map((r) => ({
            contact_digits: r.contact_digits,
            display_label: r.display_label,
            enabled: r.enabled === 1,
            model_id: models.get(r.contact_digits) ?? cfg.openaiModel,
            features: byContact.get(r.contact_digits) ?? {},
          }))
        );
        return;
      }

      if (path === "/api/templates" && req.method === "GET") {
        json(res, 200, { templates: db.listTemplates() });
        return;
      }

      const tplPut = path.match(/^\/api\/templates\/([^/]+)$/);
      if (tplPut && req.method === "PUT") {
        const key = decodeURIComponent(tplPut[1] ?? "") as MessageTemplateKey;
        const allowed: MessageTemplateKey[] = [
          "access_denied_not_registered",
          "access_denied_disabled",
          "registered_no_features",
          "feature_disabled",
        ];
        if (!allowed.includes(key)) {
          json(res, 400, { error: "template_key inválida" });
          return;
        }
        const raw = await readBody(req);
        const parsed = JSON.parse(raw) as { body?: string };
        if (typeof parsed.body !== "string") {
          json(res, 400, { error: "body string obrigatório" });
          return;
        }
        db.setTemplateBody(key, parsed.body);
        policySvc.invalidateCache();
        json(res, 200, { ok: true });
        return;
      }

      if (path === "/api/identities" && req.method === "POST") {
        const raw = await readBody(req);
        const b = JSON.parse(raw) as {
          contact_digits?: string;
          display_label?: string | null;
          enabled?: boolean;
          model_id?: string;
          features?: Record<string, boolean>;
        };
        const id = normalizeContactDigits(b.contact_digits ?? "");
        if (!id) {
          json(res, 400, { error: "contact_digits obrigatório" });
          return;
        }
        if (b.model_id && !validateModel(cfg, b.model_id)) {
          json(res, 400, { error: "model_id não permitido" });
          return;
        }
        db.upsertIdentity(id, {
          displayLabel: b.display_label ?? null,
          enabled: b.enabled !== false,
          defaultModelIfNew: b.model_id ?? cfg.openaiModel,
        });
        if (b.model_id) {
          db.setModel(id, b.model_id.trim());
        }
        const featMap: Record<string, boolean> = {};
        if (b.features && typeof b.features === "object") {
          for (const k of Object.keys(b.features)) {
            if (isKnownFeatureKey(k)) featMap[k] = Boolean(b.features[k]);
          }
        } else {
          for (const k of allFeatureKeys()) featMap[k] = true;
        }
        db.replaceFeatureGrants(id, featMap);
        policySvc.invalidateCache();
        json(res, 200, { ok: true });
        return;
      }

      const idPatch = path.match(/^\/api\/identities\/([^/]+)$/);
      if (idPatch && req.method === "PATCH") {
        const id = normalizeContactDigits(decodeURIComponent(idPatch[1] ?? ""));
        if (!id || !db.getIdentity(id)) {
          json(res, 404, { error: "não encontrado" });
          return;
        }
        const raw = await readBody(req);
        const b = JSON.parse(raw) as {
          display_label?: string | null;
          enabled?: boolean;
          model_id?: string;
          features?: Record<string, boolean>;
        };
        if (b.model_id !== undefined && !validateModel(cfg, b.model_id)) {
          json(res, 400, { error: "model_id não permitido" });
          return;
        }
        const cur = db.getIdentity(id)!;
        db.upsertIdentity(id, {
          displayLabel:
            b.display_label !== undefined ? b.display_label : cur.display_label,
          enabled:
            b.enabled !== undefined ? Boolean(b.enabled) : cur.enabled === 1,
          defaultModelIfNew: cfg.openaiModel,
        });
        if (b.model_id) {
          db.setModel(id, b.model_id.trim());
        }
        if (b.features && typeof b.features === "object") {
          const current = db.getFeatureGrantMap(id);
          for (const k of Object.keys(b.features)) {
            if (isKnownFeatureKey(k)) current[k] = Boolean(b.features[k]);
          }
          db.replaceFeatureGrants(id, current);
        }
        policySvc.invalidateCache();
        json(res, 200, { ok: true });
        return;
      }

      if (idPatch && req.method === "DELETE") {
        const id = normalizeContactDigits(decodeURIComponent(idPatch[1] ?? ""));
        if (!id) {
          json(res, 400, { error: "id inválido" });
          return;
        }
        db.deleteIdentity(id);
        policySvc.invalidateCache();
        json(res, 200, { ok: true });
        return;
      }

      json(res, 404, { error: "rota não encontrada" });
    } catch (e) {
      json(res, 500, {
        error: e instanceof Error ? e.message : String(e),
      });
    }
  });

  server.listen(cfg.adminPort, cfg.adminHost, () => {
    console.info(
      `[admin] http://${cfg.adminHost}:${cfg.adminPort}/ (Bearer ADMIN_TOKEN)`
    );
  });

  return server;
}
