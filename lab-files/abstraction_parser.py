import ast
import re

from lab.parser import Parser


COLLECTION_RE = re.compile(
    r"Abstraction collection: abstractions=(\d+), total_states=(\d+), "
    r"states=(\[[^\n]*\]), domain_abstract_operators=(\d+), "
    r"cartesian_transitions=(\d+), construction_time=([0-9.]+)s"
)


def parse_abstraction_statistics(content, props):
    matches = COLLECTION_RE.findall(content)
    if len(matches) > 1:
        props.add_unexplained_error("multiple abstraction collection summaries")
        return
    if matches:
        (
            count,
            total_states,
            states,
            domain_operators,
            cartesian_transitions,
            construction_time,
        ) = matches[0]
        props["abstraction_count"] = int(count)
        props["abstraction_total_states"] = int(total_states)
        props["abstraction_states"] = ast.literal_eval(states)
        props["domain_abstract_operators"] = int(domain_operators)
        props["cartesian_transitions"] = int(cartesian_transitions)
        props["abstraction_construction_time"] = float(construction_time)

    initial_values = re.findall(
        r"Initial heuristic value for [^:]+: ([-+]?(?:\d+(?:\.\d*)?|\.\d+)|inf(?:inity)?)$",
        content,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if len(initial_values) > 1:
        props.add_unexplained_error("multiple initial heuristic values")
    elif initial_values:
        value = initial_values[0].lower()
        props["initial_h_value_float"] = (
            float("inf") if value.startswith("inf") else float(value)
        )

    portfolio = re.findall(
        r"offline diversification retained (\d+) of (\d+) evaluated partitions "
        r"over (\d+) samples \((\d+) KiB\)",
        content,
    )
    if portfolio:
        retained, evaluated, samples, size_kib = portfolio[-1]
        props["scp_partitions_retained"] = int(retained)
        props["scp_partitions_evaluated"] = int(evaluated)
        props["scp_diversification_samples"] = int(samples)
        props["scp_table_size_kib"] = int(size_kib)


class AbstractionParser(Parser):
    def __init__(self):
        super().__init__()
        self.add_function(parse_abstraction_statistics)
