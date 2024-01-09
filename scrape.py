import os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.utilities import ApifyWrapper

# Load environment variables from a .env file
load_dotenv()

if __name__ == '__main__':
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    APIFY_API_TOKEN = os.environ.get('APIFY_API_TOKEN')
    website_url1 = os.environ.get('Website_Url1')
    website_url2 = os.environ.get('Website_Url2')

    print(f'Extracting data. Please wait...')
    apify = ApifyWrapper()
    # Call the Actor to obtain text from the crawled webpages
    loader = apify.call_actor(
        actor_id="apify/website-content-crawler",
        run_input={
            "startUrls": [
                            {"url": website_url1, "url":website_url2},
                        ]
        },
        dataset_mapping_function=lambda item: Document(
            page_content=item["text"] or "", metadata={"source": item["url"]}
        ),
    )

    print(f'Loading documents. Please wait...')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embedding = OpenAIEmbeddings()
    print(f'Creating vectorDB. Please wait...')
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory='db',
    )
    vectordb.persist()
    print('All done!')

