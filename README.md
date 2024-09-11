# Vision is All You Need: V-RAG (Vision RAG) Demo

This is a demo of the Vision RAG (V-RAG) architecture.

The V-RAG architecture utilizes a vision language model (VLM) to embed pages of PDF files (or any other document) as vectors directly, without the tedious chunking process.

# How does V-RAG work?

1. The pages of a PDF file are converted to images.
   - `pypdfium` is used to convert the PDF pages to images
2. The images are passed through a VLM to get the embeddings.
   - ColPali is used as the VLM in this demo
3. The embeddings are stored in a database
   - QDrant is used as the vector database in this demo
4. The user passes a query to the V-RAG system
5. The query is passed through the VLM to get the query embedding
6. The query embedding is used to search the vector database for similar embeddings
7. The user query and images of the best matches from the search are passed again to a model that can understand images
   - we use GPT4o or GPT4-mini in this demo
8. The model generates a response based on the query and the images

# How to run the demo?

Make sure tou have an account in Hugging Face. You need to ask for access to the PaliGemma model here: https://huggingface.co/google/paligemma-3b-mix-448

Then, make sure you are logged into Hugging Face using `transformers-cli login`.
For OpenAI API, you need to have an API key. You can get it from here: https://platform.openai.com/account/api-keys

You can place the keys to the dotenv file:

```
OPENAI_API_KEY=
HF_TOKEN=
```

Then, you can run the demo by following these steps:

1. Install Python 3.11 or higher
2. `pip install -r requirements.txt`
3. `fastapi dev` or `fastapi run`

# How to use the demo?

1. Open your browser and go to `http://localhost:8000/docs`
2. Click on the `POST /files` endpoint
3. Click on the `Try it out` button
4. Upload a PDF file
5. Click on the `Execute` button

This will index the PDF file in to in-memory vector database. This will take some time depending on the size of the PDF file and your computer specs. On CPU this takes very long time. A CUDA GPU with at least 8GB of VRAM is recommended, and make sure you have CUDA-enabled PyTorch installed in your environment.

You can now search for similar pages using the `POST /search` endpoint.
