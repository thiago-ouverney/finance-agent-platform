import json
import tempfile
import unittest
from pathlib import Path

from analytics.src.eaas_business_model_canvas import (
    CANVAS_BLOCKS,
    build_bmc_prompt,
    build_canvas_table,
    load_eaas_records,
    normalize_eaas_record,
    parse_bmc_response,
    tags_for_company,
)


class EaasBusinessModelCanvasTests(unittest.TestCase):
    def test_normalizes_bigquery_payload_and_builds_canvas(self):
        record = {
            "cnpj": "12345678000190",
            "payload": json.dumps(
                {
                    "domain": "acme.example",
                    "name": "Acme",
                    "description": "Plataforma de automacao comercial.",
                    "estimated_segment": "software e cloud",
                    "is_ecommerce": "false",
                    "technologies": ["React", "PostgreSQL"],
                    "cb_tags": ["Technology"],
                    "mopep": {
                        "business_model": "B2B SaaS",
                        "company_tags": "['SaaS', 'B2B']",
                    },
                    "company_size": {"company_size_category": "SMB"},
                    "employees": "50-100",
                }
            ),
        }

        company = normalize_eaas_record(record)
        self.assertEqual(company["cnpj"], "12345678000190")
        self.assertEqual(company["business_model"], "B2B SaaS")
        self.assertEqual(company["tags"], ["Technology", "SaaS", "B2B"])
        self.assertIs(company["is_ecommerce"], False)

        rows = build_canvas_table([company])
        self.assertEqual(len(rows), 1)
        self.assertIn("Assinaturas recorrentes", rows[0]["revenue_streams"])
        self.assertTrue(rows[0]["needs_review"])
        self.assertIn("business-model:b2b-saas", rows[0]["tags"])
        self.assertIn("ecommerce:false", rows[0]["tags"])
        self.assertNotIn("segment:none", rows[0]["tags"])

    def test_normalizes_graphql_response(self):
        record = {
            "data": {
                "company": {
                    "cnpj": "00999999000100",
                    "domain": "loja.example",
                    "identity": {
                        "name": "Loja Exemplo",
                        "description": "Venda de produtos para consumidores.",
                    },
                    "digitalPresence": {
                        "domainDetails": {
                            "hasBlog": True,
                            "technologies": ["Shopify"],
                        },
                        "socialMedia": {"instagram": "https://instagram.com/loja"},
                    },
                    "businessIntelligence": {
                        "estimated": {"segment": "ecommerce", "employees": "10"},
                        "tags": ["Retail"],
                    },
                    "scores": {
                        "porte": {"category": "Micro"},
                        "mopep": {
                            "businessModel": "B2C",
                            "companyTags": ["Online"],
                        },
                    },
                }
            }
        }

        company = normalize_eaas_record(record)
        self.assertEqual(company["name"], "Loja Exemplo")
        self.assertTrue(company["is_ecommerce"])
        self.assertEqual(company["company_size"], "Micro")
        self.assertIn("channel:instagram", tags_for_company(company))

    def test_loads_jsonl_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "enrichment.jsonl"
            source.write_text('{"cnpj":"1"}\n{"domain":"example.com"}\n', encoding="utf-8")
            self.assertEqual(len(load_eaas_records(source)), 2)

    def test_prompt_excludes_contacts_and_parser_checks_all_blocks(self):
        prompt = build_bmc_prompt(
            {
                "name": "Acme",
                "description": "Exemplo",
                "contacts": [{"email": "pessoa@example.com"}],
            }
        )
        self.assertNotIn("pessoa@example.com", prompt)

        payload = {block: "Hipotese" for block in CANVAS_BLOCKS}
        payload.update({"tags": ["b2b"], "assumptions": [], "confidence": 0.6})
        parsed = parse_bmc_response(f"```json\n{json.dumps(payload)}\n```")
        self.assertEqual(parsed["confidence"], 0.6)


if __name__ == "__main__":
    unittest.main()
