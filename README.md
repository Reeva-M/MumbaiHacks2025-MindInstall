# MumbaiHacks2025-MindInstall
Smart insurance comparison tool using AI. Upload PDFs, extract policy details, compare premiums &amp; coverage, view charts, get recommendations, and chat with an AI advisor. Built with Flask, LLaMA, and PDF parsing.


# CredenceX — Your AI Insurance Co-Pilot
### MumbaiHacks 2025 - Team MindInstall

Buying insurance is often complicated. Between the dense legal jargon, varying coverage limits, and hidden clauses, making the right choice feels like a gamble. **CredenceX** was built to solve this. It is an intelligent insurance analysis platform that acts as a neutral, data-driven co-pilot. By leveraging Large Language Models (LLaMA) and advanced PDF parsing, CredenceX transforms complex policy documents into clear, comparable insights.

## The Workflow: How CredenceX Works

We designed the user experience to be seamless, moving from raw documents to actionable advice in seconds. Here is the end-to-end flow of the application:

### 1. Context Setting
The journey begins with the user defining their specific needs. Insurance isn't one-size-fits-all, so CredenceX adapts its interface based on the category chosen:
*   **Health:** Asks for the proposer's age and number of family members.
*   **Life:** shifts focus to current age and annual income.
*   **Motor/Home:** Requests asset valuations.
This ensures the AI evaluates the policies against the user's actual life situation, not just generic benchmarks.

### 2. Document Ingestion
Users upload the raw policy wordings or brochures in PDF format. The system is designed to handle multiple files simultaneously, allowing for a direct head-to-head comparison between different insurers.

### 3. The Intelligence Layer (Processing)
Once the "Initialize Analysis" button is clicked, the backend springs into action:
*   **Text Extraction:** A parsing engine reads the PDFs, stripping away formatting to access the raw text data.
*   **LLM Analysis:** The extracted text is fed into a LLaMA-based model. The AI doesn't just read; it interprets. It looks for specific parameters like "Waiting Period," "Co-payment clauses," "No Claim Bonus," and "Exclusions."
*   **Scoring Algorithm:** The system assigns a proprietary "Credence Score" to each policy based on how well its features align with the user's inputs (e.g., a policy with a 4-year waiting period gets a lower score for a 45-year-old user compared to a 2-year waiting period).

### 4. Visual Comparison Dashboard
The user is presented with a futuristic, "Dark Mode" dashboard that visualizes the data:
*   **Bar Charts:** A clean visual representation comparing Premiums, Sum Insured, and the AI Suitability Score side-by-side.
*   **Structured Data:** A detailed table breaks down every critical metric, allowing the user to scan through coverage details without opening the PDFs again.

### 5. Interactive Consultation
Data tables can only tell half the story. The **Policy Assistant AI** (Chatbot) serves as the final layer of clarification. Users can open the chat interface to ask specific questions like:
*   "Which policy covers maternity expenses?"
*   "Does the second policy cover pre-existing diabetes?"
*   "Explain the co-pay clause in simple terms."

The AI answers based strictly on the context of the uploaded documents, preventing hallucinations and ensuring accuracy.

## Key Features

*   **Intelligent Parsing:** Capable of reading unstructured PDF layouts and converting them into structured JSON data.
*   **Context-Aware Scoring:** Policies are ranked not just by price, but by how well they fit the user's demographic profile.
*   **Visual Analytics:** dynamic charts that make financial differences immediately apparent.
*   **RAG-Based Chatbot:** A Retrieval-Augmented Generation chat system that allows users to "talk" to their insurance documents.
*   **Modern UI:** A fully responsive, glassmorphism-based design with a focus on readability and user experience.

## Tech Stack

*   **Backend:** Python (Flask)
*   **AI & Logic:** LLaMA (Large Language Model), PDFPlumber (Text Extraction)
*   **Frontend:** HTML5, CSS3 (Custom Glassmorphism), JavaScript (Vanilla)
*   **Visualization:** Chart.js
*   **Styling:** Bootstrap 5


## Future Roadmap

*   **OCR Integration:** To handle scanned physical policy documents.
*   **Premium Prediction:** Using historical data to predict premium hikes in future years.
*   **Regional Language Support:** Making insurance simplified for non-English speakers.
