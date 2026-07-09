"""Reference-synthetic variant of the MALLM-GAN pipeline.

This module keeps the original ``model_glm.py`` unchanged. It implements
direction 2: use existing generated data as the reference source instead of
reading real training samples into the generation prompt.

Example notebook usage:

    from model_glm_reference_synthetic import (
        ReferenceSyntheticGAN,
        load_reference_synthetic_data,
    )

    reference_data = load_reference_synthetic_data("gen/adult/100/df_0.csv", cols)
    magan = ReferenceSyntheticGAN(
        gen_client,
        opt_client,
        gen_model_nm,
        opt_model_nm,
        params,
        reference_data,
        cols,
        y_col,
        num_var,
        metadata,
        cate_desc,
        data_desc,
        log_file,
        opt_temperature=1,
        real_samples_num=5,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from model_glm import MultiAgentGAN


def load_reference_synthetic_data(
    csv_path: str | Path,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load an existing generated CSV to use as the pipeline reference data.

    Parameters
    ----------
    csv_path:
        Path to a generated CSV, for example ``gen/adult/100/df_0.csv``.
    columns:
        Optional ordered columns required by the generation pipeline. Pass
        ``cols`` from the Adult notebook to preserve the original schema order.

    Returns
    -------
    pandas.DataFrame
        Reference synthetic data ready to pass into ``ReferenceSyntheticGAN``.
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Reference synthetic CSV does not exist: {csv_path}")

    reference_data = pd.read_csv(csv_path)

    if columns is not None:
        missing = [col for col in columns if col not in reference_data.columns]
        if missing:
            raise ValueError(
                f"Reference synthetic CSV is missing required columns: {missing}"
            )
        reference_data = reference_data.loc[:, list(columns)].copy()

    return reference_data.reset_index(drop=True)


def load_combined_reference_synthetic_data(
    csv_paths: Sequence[str | Path],
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load and concatenate several generated CSV files as reference data."""

    if not csv_paths:
        raise ValueError("At least one reference synthetic CSV path is required.")

    frames = [
        load_reference_synthetic_data(csv_path, columns=columns)
        for csv_path in csv_paths
    ]
    return pd.concat(frames, ignore_index=True)


class ReferenceSyntheticGAN(MultiAgentGAN):
    """MALLM-GAN pipeline using generated data as its reference source.

    The parent class still stores the input dataframe in ``self.real_data`` for
    compatibility with the original implementation. In this subclass, that
    dataframe should be existing generated data, not raw real samples.
    """

    def __init__(
        self,
        gen_client,
        opt_client,
        gen_model_nm,
        opt_model_nm,
        params,
        reference_data: pd.DataFrame,
        cols,
        y_col,
        num_cols,
        meta_data,
        cate_desc,
        data_desc,
        logfile,
        gen_temperature=0.5,
        opt_temperature=0.5,
        use_fuzzy_samples=False,
        fuzzy_samples_num=2,
        num_score_pairs=3,
        real_samples_num=2,
        methodtype="hc",
        scoretype="bic",
        use_causal_graph=True,
    ) -> None:
        super().__init__(
            gen_client=gen_client,
            opt_client=opt_client,
            gen_model_nm=gen_model_nm,
            opt_model_nm=opt_model_nm,
            params=params,
            real_data=reference_data,
            cols=cols,
            y_col=y_col,
            num_cols=num_cols,
            meta_data=meta_data,
            cate_desc=cate_desc,
            data_desc=data_desc,
            logfile=logfile,
            gen_temperature=gen_temperature,
            opt_temperature=opt_temperature,
            use_fuzzy_samples=use_fuzzy_samples,
            fuzzy_samples_num=fuzzy_samples_num,
            num_score_pairs=num_score_pairs,
            real_samples_num=real_samples_num,
            methodtype=methodtype,
            scoretype=scoretype,
            use_causal_graph=use_causal_graph,
        )
        self.reference_data_source = "synthetic"

    def instruction(self, sample, refined_prompt, cond=None):
        prompt_sys = (
            "You are a skilled data generation model. Your task is to "
            "understand the instructions below and generate tabular data.\n"
        )
        prompt_sys += "<Data description>" + self.data_desc + "</Data description>\n\n"
        prompt_sys += "<Data schema>" + str(self.meta_data) + "</Data schema>\n\n"
        prompt_sys += "Categorical variables and their available categories:\n"
        prompt_sys += (
            "<Categorical variables>"
            + str(self.cate_desc)
            + "<\\Categorical variables>\n\n"
        )

        if self.use_causal_graph:
            prompt_sys += refined_prompt
        else:
            prompt_sys += """
<Task> The ultimate goal is to produce accurate and convincing synthetic
data given the provided reference synthetic samples. </Task>"""

        prompt_user = f"""<example>Here are examples from existing generated reference data:
{sample}

<\\example>
        """
        if cond:
            prompt_user += f"""
<Instruction>Generate {self.real_samples_num} synthetic samples with {cond}. Response should be formatted strictly as a list in JSON format, suitable for direct use in data processing scripts such as conversion to a DataFrame in Python. No additional text or numbers should precede the JSON data.</Instruction>"""
        else:
            prompt_user += f"<Instruction>Generate {self.real_samples_num} synthetic samples that mimic the provided reference synthetic samples. DO NOT COPY the samples. The response should be formatted strictly as a list in JSON format, which is suitable for direct use in data processing scripts such as conversion to a DataFrame in Python. No additional text or numbers should precede the JSON data. <\\Instruction>"
        return prompt_sys, prompt_user
