"""Dataset generator for guided LLM experiments.

Creates a streaming DataPool from a HuggingFace dataset. The DataPool
provides the raw data source; the loader wraps it into a FixedDataPool
for deterministic, step-seeded sampling.
"""

from framework import DatasetGenerator, DataPool


class GuidedLLMDatasetGenerator(DatasetGenerator):
    """Generates a streaming DataPool from HuggingFace data.

    Handles dataset configuration, content filtering, and optional
    content analysis for the first condition.
    """

    def __init__(self, console=None):
        self.console = console

    def generate(self, config, **kwargs):
        """Return a DataPool for gradient-aligned training.

        Expects kwargs:
            dataset_config: Dict from DATASET_CONFIGS
            filter_terms: Optional list of terms to filter out
            analysis_terms: Optional list of terms for content analysis
            is_first_condition: Whether to show content analysis
        """
        from framework import DATASET_CONFIGS

        dataset_config = kwargs.get('dataset_config', DATASET_CONFIGS[config.dataset])
        filter_terms = kwargs.get('filter_terms', [])
        analysis_terms = kwargs.get('analysis_terms', [])
        is_first_condition = kwargs.get('is_first_condition', False)

        # Create streaming DataPool
        data_pool = DataPool(
            dataset_config,
            filter_terms=filter_terms,
            buffer_size=50000,
            min_length=config.min_text_length,
            console=self.console,
        )

        # Content analysis (only for first condition)
        if is_first_condition and analysis_terms:
            data_pool.analyze_content(analysis_terms)

        return data_pool
