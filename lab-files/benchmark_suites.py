"""Shared benchmark-suite corrections for new experiments."""

SUPERSEDED_OTHERS_DOMAINS = frozenset({"petri-net"})


def exclude_superseded_others_domains(domains):
    """Return the current Others suite without superseded domain encodings."""
    domains = list(domains)
    for domain in SUPERSEDED_OTHERS_DOMAINS:
        count = domains.count(domain)
        if count != 1:
            raise ValueError(
                f"expected exactly one {domain!r} suite, found {count}"
            )
    return [
        domain
        for domain in domains
        if domain not in SUPERSEDED_OTHERS_DOMAINS
    ]
