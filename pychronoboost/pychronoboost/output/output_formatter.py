import pandas as pd

class OutputFormatter:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the OutputFormatter object.

        :param data: The processed time series data as a Pandas DataFrame.
        """
        self.data = data

    def format_output(self):
        """
        Format the output data. Modify this method if any specific formatting is required.

        :return: Formatted DataFrame.
        """
        # Implement any specific formatting if needed
        return self.data

    def save_to_csv(self, file_path: str):
        """
        Save the formatted data to a CSV file.

        :param file_path: The file path where the CSV will be saved.
        """
        formatted_data = self.format_output()
        formatted_data.to_csv(file_path, index=False)

# Example usage
# output_formatter = OutputFormatter(processed_data)
# output_formatter.save_to_csv("output.csv")
