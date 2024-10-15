# Vision is All You Need: V-RAG (Vision RAG) Demo

This is a demo of the Vision RAG (V-RAG) architecture.

The V-RAG architecture utilizes a vision language model (VLM) to embed pages of PDF files (or any other document) as vectors directly, without the tedious chunking process.

![Screenshot 2024-10-15 145217](https://github.com/user-attachments/assets/ec153fe7-35eb-4064-a587-3bd1c16b6c79)

# How does V-RAG work?

1. The pages of a PDF file are converted to images. 
   - In theory these images can be anything, but the current demo uses PDF files since the underlying model has been trained on PDF files
   - `pypdfium` is used to convert the PDF pages to images
2. The images are passed through a VLM to get the embeddings.
   - ColPali is used as the VLM in this demo
3. The embeddings are stored in a database
   - QDrant is used as the vector database in this demo
4. The user passes a query to the V-RAG system
5. The query is passed through the VLM to get the query embedding
6. The query embedding is used to search the vector database for similar embeddings
7. The user query and images of the best matches from the search are passed again to a model that can understand images
   - we use GPT4o or GPT4o-mini in this demo
8. The model generates a response based on the query and the images

# How to run the demo?

Make sure tou have an account in Hugging Face. Make sure you are logged into Hugging Face using `transformers-cli login`.

For OpenAI API, you need to have an API key. You can get it from here: https://platform.openai.com/account/api-keys

You can place the keys to the dotenv file:

```
OPENAI_API_KEY=
HF_TOKEN=
```

Then, you can run the demo by following these steps:

1. Install Python 3.11 or higher
2. `pip install modal`
3. `modal setup`
4. `modal serve .\main.py`

# How to use the demo from the provided API?

1. Open your browser and go to the url provided by Modal and append `/docs` to the url
2. Click on the `POST /collections` endpoint
3. Click on the `Try it out` button
4. Upload a PDF file
5. Click on the `Execute` button

This will index the PDF file in to in-memory vector database. This will take some time depending on the size of the PDF file and the GPU you are using in Modal. The current demo is using a A10G GPU.

You can now search for similar pages using the `POST /search` endpoint.

The endpoint send the page images and the query to the OpenAI API and returns the response. 

# Frontend

You can also use the frontend to interact with the API. To setup the frontend for local development, follow these steps:

1. Install Node.js
2. `cd frontend`
   - modify you `.env.development` file and add your `VITE_BACKEND_URL`
3. `npm install`
4. `npm run dev`

This will start the frontend on `http://localhost:5173`

# How to deploy the demo?

You can deploy the demo to Modal using the following steps:

1. Modify you `.env.production` file and add your `VITE_BACKEND_URL` for the production environment
2. Build the frontend `npm run build` - this will create a `dist` folder
3. `modal deploy .\main.py`
