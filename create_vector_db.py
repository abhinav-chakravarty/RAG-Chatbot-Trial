from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Function to load and split PDF documents in paragraph chunks
def split_paragraphs(rwText):
    text_splitter = CharacterTextSplitter(
        separator = '\n',
        chunk_size = 200,
        chunk_overlap = 20,
        length_function = len,
        is_separator_regex = False
    )

    return text_splitter.split_text(rwText)

def load_pdfs(pdfs):
    text_chunks = []
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            raw = page.extract_text()
            chunks = split_paragraphs(raw)
            text_chunks += chunks
    
    return text_chunks

list_of_pdfs = ['dietary_supplements.pdf']
text_chunks = load_pdfs(list_of_pdfs)

print(text_chunks[:5])  # Display the first 5 text chunks for verification

# Create a FAISS vector store from the text chunks
from langchain.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

store = FAISS.from_texts(text_chunks, embeddings)
store.save_local("./myVectorStore")
