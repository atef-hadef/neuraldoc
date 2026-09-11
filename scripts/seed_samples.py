"""Generate sample documents for demos and manual QA."""

from pathlib import Path

SAMPLES = {
    "service_agreement.txt": (
        "Service Agreement. The provider guarantees 99.9 percent uptime, "
        "measured monthly excluding scheduled maintenance. Termination requires "
        "30 days written notice. Penalties apply for material breach."
    ),
    "refund_policy.txt": (
        "Refund Policy. Customers may request a refund within 30 days of purchase. "
        "Refunds go to the original payment method. Shipping fees are non-refundable "
        "unless the item arrived damaged."
    ),
    "security_overview.txt": (
        "Security Overview. All data is encrypted at rest with AES-256 and in transit "
        "with TLS 1.2 or higher. Access follows least-privilege with quarterly reviews. "
        "Backups are retained for 30 days."
    ),
    "outlier_cooking_notes.txt": (
        "Cooking Notes. Whisk three eggs with a pinch of salt. Simmer the tomato sauce "
        "for twenty minutes. This document is intentionally off-topic to demonstrate "
        "anomaly detection on document collections."
    ),
}


def main() -> None:
    out = Path("data/samples")
    out.mkdir(parents=True, exist_ok=True)
    for name, text in SAMPLES.items():
        (out / name).write_text(text, encoding="utf-8")
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
