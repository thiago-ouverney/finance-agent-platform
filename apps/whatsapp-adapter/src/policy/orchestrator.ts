import { isKnownFeatureKey, type FeatureKey } from "./feature-registry.js";

/**
 * Interseção entre grants persistidos e o registry versionado no código.
 * Base para evolução multi-agente: cada agente pode exigir um subconjunto de `FeatureKey`.
 */
export function intersectGrantedWithRegistry(
  rawGranted: ReadonlySet<string>
): Set<FeatureKey> {
  const out = new Set<FeatureKey>();
  for (const k of rawGranted) {
    if (isKnownFeatureKey(k)) out.add(k);
  }
  return out;
}
