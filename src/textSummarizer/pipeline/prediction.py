from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, pipeline


class PredictionPipeline:
    """
    A class for generating summaries using a pre-trained model.

    Attributes:
        config (dict): Model evaluation configuration settings.
    """

    def __init__(self):
        """
        Initializes an instance of the PredictionPipeline class.
        """
        config_manager = ConfigurationManager()
        self.config = config_manager.get_model_evaluation_config()

    def predict(self, text):
        """
        Generates a summary for the given input text.

        Args:
            text (str): The input text to be summarized.

        Returns:
            str: The generated summary.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}
        pipe = pipeline("summarization", model=self.config.model_path, tokenizer=tokenizer)  # type: ignore

        print("Input Text: ")
        print(text)

        output = pipe(text, **gen_kwargs)[0]["summary_text"]  # type: ignore
        print("\nModel Summary:")
        print(output)

        return output


# Example usage:
if __name__ == "__main__":
    prediction_pipeline = PredictionPipeline()
    input_text = input("Please enter the text you want to summarize: ")
    summary = prediction_pipeline.predict(input_text)
