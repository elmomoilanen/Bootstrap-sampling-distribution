"""Implements plotting functionality."""
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt  # type: ignore[import]


class Plotting:
    """Plots bootstrap distribution with its associated statistics.

    Parameters
    ----------
    plot_style_sheet : str
        Pyplot style sheet to be used in histogram plots.
    """

    allowed_styles = (
        "default",
        "seaborn",
        "Solarize_Light2",
        "ggplot",
    )

    def __init__(self, plot_style_sheet: str = "default") -> None:
        self.style_sheet = plot_style_sheet
        self._check_style_validity()

    def _check_style_validity(self) -> None:
        if self.style_sheet not in self.allowed_styles:
            allowed = ", ".join(self.allowed_styles)
            raise ValueError(f"Style {self.style_sheet} not in allowed styles list: {allowed}")

    @staticmethod
    def _compute_percentile_ci(data: np.ndarray, alpha: float) -> Any:
        alpha_upper = 100 - (100 - alpha) / 2
        alpha_lower = 0 + (100 - alpha) / 2
        return np.percentile(data, [alpha_lower, alpha_upper])

    @staticmethod
    def _generate_font_family() -> Dict[str, Dict[str, Any]]:
        return {
            "label_fonts": dict(fontsize=10, color="black", fontname="serif"),
            "header_fonts": dict(fontsize=12, color="black", fontname="sans-serif"),
            "data_fonts": dict(
                fontsize=8,
                color="black",
                fontname="arial",
                fontweight="semibold",
                backgroundcolor="white",
            ),
            "bca_ci_data_fonts": dict(
                fontsize=8,
                color="black",
                fontname="arial",
                fontweight="semibold",
                backgroundcolor="orangered",
            ),
            "perc_ci_data_fonts": dict(
                fontsize=8,
                color="black",
                fontname="arial",
                fontweight="semibold",
                backgroundcolor="coral",
            ),
        }

    @staticmethod
    def _set_text_field(ax: plt.Axes, config: Dict[str, Any]) -> None:
        ax.text(
            config["x"],
            config["y"],
            config["text"],
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            fontdict=config["font"],
            alpha=0.8,
        )

    @staticmethod
    def _set_annotation(ax: plt.Axes, config: Dict[str, Any]) -> None:
        arrow_points = 1 if config["arrow_direction"] == "down" else -1

        ax.annotate(
            config["text"],
            fontsize=6,
            color=config["color"],
            xy=(config["value"], 0),
            xycoords="data",
            xytext=(0, arrow_points * 25),
            textcoords="offset points",
            arrowprops=dict(color=config["color"], shrink=0.1),
            horizontalalignment="center",
        )

    def _draw_histogram(
        self, plot_data: Dict[str, Any], plot_config: Dict[str, Any], plot_comparison: bool
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        _ = ax.hist(plot_data["b_stats"], bins=plot_config["bins"], color="silver", alpha=0.75)

        fonts = self._generate_font_family()

        ax.tick_params(axis="x", which="major", pad=25)
        ax.set_xlabel(f"statistic ({plot_config['statistic']})", fontdict=fonts["label_fonts"])
        ax.set_ylabel("frequency", fontdict=fonts["label_fonts"])
        _ = ax.set_title("Bootstrap distribution", fontdict=fonts["header_fonts"])

        conf_int = plot_data["ci"]
        obs_val = plot_data["actual_stat"]
        alpha = plot_data["alpha"]

        data_text = dict(
            observed_value=f"obs {obs_val:.2f}",
            confidence_interval_bca=f"{alpha} bca ci [{conf_int[0]:.2f},{conf_int[1]:.2f}]",
            standard_error=f"se {plot_data['se']:.2f}",
        )

        self._set_text_field(
            ax,
            dict(x=1.20, y=0.95, text=data_text["observed_value"], font=fonts["data_fonts"]),
        )
        self._set_text_field(
            ax,
            dict(x=1.20, y=0.90, text=data_text["standard_error"], font=fonts["data_fonts"]),
        )
        self._set_text_field(
            ax,
            dict(
                x=1.20,
                y=0.85,
                text=data_text["confidence_interval_bca"],
                font=fonts["bca_ci_data_fonts"],
            ),
        )
        if plot_comparison:
            conf_perc = plot_data["ci_perc"]
            text = f"{alpha} perc ci [{conf_perc[0]:.2f},{conf_perc[1]:.2f}]"

            self._set_text_field(
                ax,
                dict(x=1.20, y=0.80, text=text, font=fonts["perc_ci_data_fonts"]),
            )
            ax.axvline(x=obs_val, alpha=0.5, color="black", linestyle="--", linewidth=0.75)

        if not plot_comparison:
            # Only plot obs arrow when not plotting confidence interval comparison
            self._set_annotation(
                ax, dict(text="obs", color="black", value=obs_val, arrow_direction="down")
            )

        alphas = (0 + (100 - plot_data["alpha"]) / 2, 100 - (100 - plot_data["alpha"]) / 2)

        for bca_value, alpha, index in zip(conf_int, alphas, range(2)):
            self._set_annotation(
                ax, dict(text=f"{alpha}", color="orangered", value=bca_value, arrow_direction="up")
            )
            if plot_comparison:
                perc_value = plot_data["ci_perc"][index]
                self._set_annotation(
                    ax,
                    dict(text=f"{alpha}", color="coral", value=perc_value, arrow_direction="down"),
                )

        ax.grid()
        fig.tight_layout()

        plt.show()
        plt.close(fig)

    def plot_estimates(
        self, plot_data: Dict[str, Any], plot_config: Dict[str, Any], plot_comparison: bool = False
    ) -> None:
        """Plot bootstrap distribution as a histrogram with SE, observed value and BCa CIs.

        This method can be used directly but it is more convenient to use via the
        SampDist class, see its `plot` method for more information.

        To call this method properly, run the `estimate` method for an instantiated
        SampDist object from the sampling module and pass the results here in a manner
        described below.

        Parameters
        ----------
        plot_data : Dict[str, Any]
            Results of the estimation process completed in the sampling module.
            Must contain keys `b_stats`, `se`, `ci` and `actual_stat` having
            NumPy arrays as values and key `alpha` with a numerical value.

        plot_config : Dict[str, Any]
            Settings for the histogram plot. Must contain key `bins` with a positive
            integer or string value. If integer, it must between one and the count
            of observations present in data. If string, it must be a name accepted
            by the pyplot library and its histogram method. Furthermore, key `statistic`
            with a string value must be present (basically name of the statistic).

        plot_comparison : bool
            If True, naive percentile based confidence interval is included in the plot
            alongside the BCa CI. Otherwise and as the default case, it's not included.
        """
        if plot_comparison:
            plot_data["ci_perc"] = self._compute_percentile_ci(
                plot_data["b_stats"], plot_data["alpha"]
            )

        with plt.style.context(self.style_sheet):
            self._draw_histogram(plot_data, plot_config, plot_comparison)
