import type { FeatureKey } from "./feature-registry.js";

export type PolicySourceMode = "hybrid" | "db_only";

export type AccessDeniedReason = "not_registered" | "disabled";

export type AccessResolution =
  | {
      kind: "denied";
      reason: AccessDeniedReason;
      /** Primeiro candidato normalizado usado na decisão, se houver */
      contactDigits?: string;
    }
  | {
      kind: "blocked_no_features";
      contactId: string;
      identitySource: "db";
    }
  | {
      kind: "allowed";
      contactId: string;
      identitySource: "db" | "env";
      model: string;
      /** Interseção entre registry e grants (ou todas no modo env implícito) */
      features: ReadonlySet<FeatureKey>;
    };

export type EffectivePolicy = {
  model: string;
  features: ReadonlySet<FeatureKey>;
  identitySource: "db" | "env";
};

export type MessageTemplateKey =
  | "access_denied_not_registered"
  | "access_denied_disabled"
  | "registered_no_features"
  | "feature_disabled";
