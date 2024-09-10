# Docuquery-AI

Docuquery-AI is a powerful tool designed to provide answers to user queries by leveraging state-of-the-art language models and vector databases. This README will guide you through the setup and usage of Docuquery-AI.



## Prerequisites

Before you can start using Docuquery-AI, make sure you have the following prerequisites installed on your system:

- Python 3.6 or higher
- Required Python packages (you can install them using pip):
    - langchain
    - sentence-transformers
    - faiss
    - PyPDF2 (for PDF document loading)
    - python-dotenv (for managing environment variables)

## Installation

1. Clone this repository to your local machine.

    ```bash
    git clone https://github.com/your-username/docuquery-ai.git
    cd docuquery-ai
    ```

2. Create a Python virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Download the required language models and data. Please refer to the Langchain documentation for specific instructions on how to download and set up the language model and vector store.

5. Set up the necessary paths and configurations in your project, including the `DB_FAISS_PATH` variable and other configurations as per your needs.

## Getting Started

To get started with Docuquery-AI, you need to:

1. Set up your environment and install the required packages as described in the Installation section.

2. Configure your project by updating the `DB_FAISS_PATH` variable and any other custom configurations in the code.

3. Prepare the language model and data as per the Langchain documentation.

4. Start the bot by running the provided Python script or integrating it into your application.

## Usage

Docuquery-AI can be used for answering document-related queries. To use the bot, you can follow these steps:

1. Start the bot by running your application or using the provided Python script.

2. Send a query related to the documents.

3. The bot will provide a response based on the information available in its database.

4. If sources are found, they will be provided alongside the answer.

5. The bot can be customized to return specific information based on the query and context provided.

## Contributing

Contributions to Docuquery-AI are welcome! If you'd like to contribute to the project, please follow these steps:

1. Fork the repository to your own GitHub account.

2. Create a new branch for your feature or bug fix.

3. Make your changes and ensure that the code passes all tests.

4. Create a pull request to the main repository, explaining your changes and improvements.

5. Your pull request will be reviewed, and if approved, it will be merged into the main codebase.

## License

This project is licensed under the MIT License.

---

For more information on how to use, configure, and extend Docuquery-AI, please refer to the Langchain documentation or contact the project maintainers.

Happy coding with Docuquery-AI! 🚀
