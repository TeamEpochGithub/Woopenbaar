import unittest
from unittest.mock import Mock

from marker.builders.llm_layout import LLMLayoutBuilder
from marker.processors.llm.llm_complex import LLMComplexRegionProcessor
from marker.processors.llm.llm_image_description import LLMImageDescriptionProcessor

from standard_data_format.src.custom_marker import (
    custom_complex_region_prompt,
    custom_complex_relabeling_prompt,
    custom_image_description_prompt,
    custom_surya_layout,
    custom_topk_relabelling_prompt,
)


class TestCustomMarker(unittest.TestCase):
    """Tests for custom marker implementations."""

    def test_custom_prompts_applied(self):
        """Test that custom prompts are correctly assigned to their classes."""
        # Check that the prompts have been assigned to the correct classes
        self.assertEqual(
            LLMLayoutBuilder.topk_relabelling_prompt, custom_topk_relabelling_prompt
        )
        self.assertEqual(
            LLMLayoutBuilder.complex_relabeling_prompt, custom_complex_relabeling_prompt
        )
        self.assertEqual(
            LLMImageDescriptionProcessor.image_description_prompt,
            custom_image_description_prompt,
        )
        self.assertEqual(
            LLMComplexRegionProcessor.complex_region_prompt,
            custom_complex_region_prompt,
        )

    def test_custom_prompts_content(self):
        """Test that custom prompts contain expected content."""
        # Check that the topk relabelling prompt contains key elements
        self.assertIn("U bent een lay-outexpert", custom_topk_relabelling_prompt)

        # Check that the complex relabeling prompt contains key elements
        self.assertIn("U bent een lay-outexpert", custom_complex_relabeling_prompt)
        self.assertIn("Reageer alleen met een van", custom_complex_relabeling_prompt)

        # Check that the image description prompt contains key elements
        self.assertIn(
            "documentanalyse-expert voor de Nederlandse overheid",
            custom_image_description_prompt,
        )
        self.assertIn("grijs gelakte tekst", custom_image_description_prompt)
        self.assertIn("5.1.2e", custom_image_description_prompt)

        # Check that the complex region prompt contains key elements
        self.assertIn("tekstformatteringsspecialist", custom_complex_region_prompt)
        self.assertIn("Formatteringsrichtlijnen", custom_complex_region_prompt)

    def test_custom_surya_layout_function(self):
        """Test the custom_surya_layout function."""
        # Create a mock instance with layout model
        mock_instance = Mock()
        mock_instance.layout_model = Mock()
        mock_instance.batch_size = 2
        mock_instance.disable_tqdm = True

        # Create mock pages
        mock_pages = [Mock(), Mock()]
        for page in mock_pages:
            page.get_image = Mock(return_value="image_data")

        # Call the custom layout function
        result = custom_surya_layout(mock_instance, mock_pages)

        # Verify that the layout model was called with the correct parameters
        mock_instance.layout_model.assert_called_once()
        # Check all pages had get_image called with highres=True
        for page in mock_pages:
            page.get_image.assert_called_once_with(highres=True)
        # Check disable_tqdm was set
        self.assertEqual(mock_instance.layout_model.disable_tqdm, True)

        # Result should be what the layout model returns
        self.assertEqual(result, mock_instance.layout_model.return_value)

    def test_surya_layout_override(self):
        """Test that the original surya_layout has been overridden."""
        # The static method should have been replaced
        self.assertEqual(LLMLayoutBuilder.surya_layout, custom_surya_layout)
