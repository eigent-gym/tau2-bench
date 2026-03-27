"""Hardcoded text-mode baseline scores for comparison with voice results.

Scores are pass^1 values as fractions (0-1), sourced from tau-bench
leaderboard submissions.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextBaseline:
    """A text-mode baseline model with per-domain pass^1 scores."""

    model_name: str
    display_name: str
    retail: float
    airline: float
    telecom: float

    @property
    def overall(self) -> float:
        return (self.retail + self.airline + self.telecom) / 3

    def get_scores_dict(
        self, domains: list[str], capitalize: bool = True
    ) -> dict[str, float]:
        """Return {category: score} dict matching the plot/table format.

        Args:
            domains: List of domain names to include.
            capitalize: If True, keys are "All"/"Retail"/etc.
                        If False, keys are "all"/"retail"/etc.
        """
        domain_scores = {
            "retail": self.retail,
            "airline": self.airline,
            "telecom": self.telecom,
        }
        all_key = "All" if capitalize else "all"
        scores = {all_key: self.overall}
        for d in domains:
            key = d.lower()
            if key in domain_scores:
                out_key = d.capitalize() if capitalize else key
                scores[out_key] = domain_scores[key]
        return scores


# TEXT_SOTA = TextBaseline(
#     model_name="GPT-5",
#     display_name="GPT-5",
#     retail=0.8158,
#     airline=0.6250,
#     telecom=0.9583,
# )
TEXT_SOTA = TextBaseline(
    model_name="GPT-5.2",
    display_name="GPT-5.2",
    retail=0.81,
    airline=0.83,
    telecom=0.90,
)

# TEXT_SOTA_NONTHINKING = TextBaseline(
#     model_name="GPT-4.1",
#     display_name="GPT-4.1",
#     retail=0.74,
#     airline=0.56,
#     telecom=0.34,
# )
TEXT_SOTA_NONTHINKING = TextBaseline(
    model_name="GPT-4.1",
    display_name="GPT-4.1",
    retail=0.76,
    airline=0.53,
    telecom=0.34,
)

DEFAULT_TEXT_BASELINES = [TEXT_SOTA, TEXT_SOTA_NONTHINKING]
