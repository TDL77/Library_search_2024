import os
import json
import re
import mmap
import pickle
from typing import List, Tuple, Dict
from functools import cache
from pathlib import Path
from dataclasses import dataclass, field
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from loguru import logger
from diskcache import Cache

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from bert_score import score
import torch
import multiprocessing

torch.cuda.empty_cache()  # Очистка кэша GPU в начале
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# для первого раза скачивания модели эмбедингов с HF пропишите сюда свой IP ключ
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'hf_XXX'

# Configure logging
logger.add(
    "log/Simple_RAG_TXT.log",
    format="{time} {level} {message}",
    level="DEBUG",
    rotation="100 KB",
    compression="zip",
)

# Cache configuration
cache = Cache("./cache")

def evaluate_relevance(query, retrieved_doc, generated_answer):
    P, R, F1 = score([generated_answer], [retrieved_doc], lang="ru")
    return F1.mean().item()

def evaluate_relevance_score(query, doc_text):
    """ Простая функция для re-rank 
    Вычисляется среднее значение F1 для всех токенов, измеряя, насколько хорошо сгенерированный ответ 
    совпадает с найденным документом.
    При наличии норм.железа заменить на Bi/Cross-encoder или подобное
    """
    _, _, F1 = score([query], [doc_text], lang="ru")
    return F1.mean().item()

def detect_device() -> str:
    if torch.cuda.is_available():
        logger.info("CUDA device detected and will be used")
        return "cuda"
    else:
        logger.info("No CUDA device detected, using CPU")
        return "cpu"

def get_optimal_threads() -> int:
    num_cores = multiprocessing.cpu_count()
    logger.info(f"Detected {num_cores} CPU cores")
    return num_cores

@dataclass
class Config:
    embedding_model_id: str = "deepvk/USER-bge-m3"  # <<<---  подобрать модельпод домен!
    llm_model_id: str = "llama3.2:3b-instruct-fp16" # <<<---  подобрать модельпод домен!
    db_path: str = "db/vector_store"
    data_path: str = "./data/TEST/"  # <<<---  прописать основную папку данных
    chunk_size: int = 1024   # 512 / 1024
    chunk_overlap: int = 0
    relevant_chunks: int = 3  # при наличии норм.модели увеличить до 5-10чанков
    temperature: float = 0
    num_threads: int = field(default_factory=get_optimal_threads)
    # device: str = field(default_factory=detect_device)  # <<< если есть GPU
    device: str = 'cpu'
    faiss_weight: float = 0.5 # <<<---  крутить тут
    bm25_weight: float = 0.5  # <<<---  и тут
    cache_size: int = 1000
    cache_dir: str = "./cache"

    def __post_init__(self):
        logger.info(f"Using device: {self.device}")
        logger.info(f"Using {self.num_threads} threads")

class DocumentLoader:
    """Document loader with caching"""
    def __init__(self, config: Config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n"],  # Сначала разбиваем по пустым строкам, затем по строкам с "<"
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )

    @cache.memoize()
    def load_document(self, file_path: str) -> str:
        """Cached document loading for different file types"""
        logger.debug(f"Loading document: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.txt':
            return self._load_txt(file_path)
        elif file_extension == '.pdf':
            return self._load_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _load_txt(self, file_path: str) -> str:
        """Load TXT file content"""
        with open(file_path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return mm.read().decode("utf-8")

    def _load_pdf(self, file_path: str) -> str:
        """Load PDF file content"""
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            return "\n".join(page.page_content for page in pages)
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise

    def create_documents(self, file_paths: List[str]) -> List[Document]:
        """Create Langchain documents with enhanced metadata"""
        documents = []
        for file_path in file_paths:
            text = self.load_document(file_path)
            
            # Gather enhanced metadata
            metadata = {
                "source": file_path,
                "filename": Path(file_path).name,
                "extension": Path(file_path).suffix.lower(),
                "relative_path": os.path.relpath(file_path, self.config.data_path),
                "created_date": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                "modified_date": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            }
            
            # Split text into chunks and filter based on chunk size
            texts = self.text_splitter.split_text(text)
            processed_chunks = [
                chunk.strip() 
                for chunk in texts 
                if chunk.strip() and len(chunk) <= self.config.chunk_size
            ]
            
            # Create documents with enhanced metadata
            chunk_docs = []
            for i, chunk in enumerate(processed_chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(processed_chunks)
                doc = Document(page_content=chunk, metadata=chunk_metadata)
                chunk_docs.append(doc)
                
            documents.extend(chunk_docs)
            logger.debug(f"Created {len(chunk_docs)} chunks from {file_path}")
            
        return documents


    def process_documents_parallel(self, file_paths: List[str]) -> List[Document]:
        """Parallel document processing"""
        logger.info(f"Processing {len(file_paths)} documents")
        return self.create_documents(file_paths)

class ModelManager:
    """Model and retriever management"""
    def __init__(self, config: Config):
        self.config = config
        self.document_loader = DocumentLoader(config)
        self.embeddings = None
        self.db = None
        self.faiss_retriever = None
        self.bm25_retriever = None
        self.ensemble_retriever = None

    def _check_data_changes(self) -> bool:
        """Check if data directory has changed"""
        data_path = Path(self.config.data_path)
        db_path = Path(self.config.db_path)
        bm25_path = db_path / "bm25.pkl"
        
        if not db_path.exists() or not bm25_path.exists():
            logger.info("DB path or BM25 index does not exist, need to rebuild")
            return True
            
        try:
            db_files_mtime = max(p.stat().st_mtime for p in db_path.glob('**/*') if p.is_file())
            data_files_mtime = max(p.stat().st_mtime for p in data_path.glob('*.txt'))
            return data_files_mtime > db_files_mtime
        except ValueError:  # No files found
            logger.info("No existing database files found")
            return True

    def initialize(self) -> None:
        """Initialize models and retrievers"""
        logger.info("Initializing ModelManager")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_id,
            model_kwargs={"device": self.config.device},
        )

        need_rebuild = self._check_data_changes()
        if need_rebuild:
            logger.info("Data changes detected or missing indexes, rebuilding embeddings")
            self._build_new_db()
        else:
            logger.info("Loading existing database")
            self._load_existing_db()

        self.initialize_retrievers()

    def initialize_retrievers(self) -> None:
        """Initialize retrievers"""
        logger.info("Initializing retrievers")
        try:
            self.faiss_retriever = self.db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": self.config.relevant_chunks, "lambda_mult": 0.25},
            )
            self.bm25_retriever = self._load_bm25_retriever()
            
            # Create ensemble retriever
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.faiss_retriever, self.bm25_retriever],
                weights=[self.config.faiss_weight, self.config.bm25_weight],
            )
            logger.info("Successfully initialized all retrievers")
        except Exception as e:
            logger.error(f"Error initializing retrievers: {str(e)}")
            raise

    def _build_new_db(self) -> None:
        """Build new FAISS and BM25 databases"""
        logger.info("Building new databases")
        
        # Create database directory if it doesn't exist
        os.makedirs(self.config.db_path, exist_ok=True)
        
        # Load and process documents
        # file_paths = [str(p) for p in Path(self.config.data_path).glob("*.txt")]
        file_paths = [str(p) for p in Path(self.config.data_path).rglob("*") if p.suffix in [".txt", ".pdf"]]

        if not file_paths:
            raise ValueError(f"No text files found in {self.config.data_path}")
            
        documents = self.document_loader.process_documents_parallel(file_paths)
        logger.info(f"Processed {len(documents)} total chunks")
        
        # Build FAISS
        logger.info("Building FAISS index")
        self.db = FAISS.from_documents(documents, self.embeddings)
        db_path = f"{self.config.db_path}/faiss_index"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db.save_local(db_path)
        
        # Build BM25
        logger.info("Building BM25 index")
        self._create_bm25_index(documents)

    def _load_existing_db(self) -> None:
        """Load existing FAISS database"""
        db_path = f"{self.config.db_path}/faiss_index"
        self.db = FAISS.load_local(db_path, self.embeddings, allow_dangerous_deserialization=True)

    def _create_bm25_index(self, documents: List[Document]) -> None:
        """Create and save BM25 index"""
        try:
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_path = Path(self.config.db_path) / "bm25.pkl"
            
            # Ensure the directory exists
            bm25_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(bm25_path, "wb") as f:
                pickle.dump(bm25_retriever, f)
            logger.info(f"Successfully created BM25 index at {bm25_path}")
        except Exception as e:
            logger.error(f"Error creating BM25 index: {str(e)}")
            raise

    def _load_bm25_retriever(self) -> BM25Retriever:
        """Load BM25 retriever with error handling"""
        bm25_path = Path(self.config.db_path) / "bm25.pkl"
        try:
            if not bm25_path.exists():
                logger.warning("BM25 index not found, triggering rebuild")
                self._build_new_db()
            
            with open(bm25_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading BM25 retriever: {str(e)}")
            raise

class RAGSystem:
    """Main RAG system"""
    def __init__(self, config: Config):
        self.config = config
        self.model_manager = ModelManager(config)
        self.model_manager.initialize()
        self.llm = ChatOllama(model=config.llm_model_id, 
                              temperature=config.temperature , 
                              num_thread=config.num_threads
                              )

    def get_relevant_content(self, query: str) -> Tuple[str, List, float, Dict]:
        """Get relevant content for query with reranking and metadata"""
        logger.debug(f"Processing query: {query}")
        docs = self.model_manager.ensemble_retriever.invoke(query)
        
        # Rerank documents based on BERTScore
        reranked_docs = sorted(
            docs,
            key=lambda doc: evaluate_relevance_score(query, doc.page_content),
            reverse=True
        )
        
        generated_answer = self.generate_response(query, reranked_docs[0].page_content)
        confidence = evaluate_relevance(query, reranked_docs[0].page_content, generated_answer) * 100
        
        # Extract metadata from most relevant document
        metadata = reranked_docs[0].metadata if reranked_docs else {}
        
        logger.debug("Retrieved and reranked content with metadata:")
        logger.debug(self._format_context(reranked_docs))
        
        return self._format_context(reranked_docs), reranked_docs, confidence, metadata

    def _format_context(self, docs: List) -> str:
        """Format found documents"""
        return re.sub(
            r"\n{2}",
            " ",
            "\n ".join(
                [
                    f"\n#### {i+1} Relevant chunk ####\n{doc.metadata}\n{doc.page_content}\n"
                    for i, doc in enumerate(docs)
                ]
            ),
        )

    def generate_response(self, query: str, context: str) -> str:
        """Generate response"""
        prompt = self._create_prompt(context, query)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def _create_prompt(self, context: str, query: str) -> str:
        """Create prompt for LLM"""
        return f"""Ты - профессиональный ассистент для ответов на вопросы.
        
        Контекст для ответа:
        {context}
        
        Вопрос пользователя:
        {query}
        
        Дай ответ на этот вопрос, используя только предоставленный контекст.
        Используй не более трех предложений и будь лаконичен в ответе.
        Отвечай строго на русском языке.
        
        Ответ:"""

def main():
    """Main function"""
    config = Config()
    rag_system = RAGSystem(config)
    
    # **********************************************************************
    # query = "Чем занимается платформа MEDIADESK?"
    # query = "Чем отличаются возможности слабого (NARROW AI), сильного (AI) и супер-ИИ (SUPER AI)?"
    query = "Сколько процентов рутинных процессов отдаются искусственному интеллекту?"
    # query = "Как усложняться механики кампаний при использовани ИИ?"
    # **********************************************************************
    
    context, docs, confidence, metadata = rag_system.get_relevant_content(query)
    response = rag_system.generate_response(query, context)
    
    # Format output as JSON with metadata
    output = {
        "query": query,
        "confidence": round(confidence, 2),
        "response": response,
        "most_relevant_chunk": docs[0].page_content,
        "metadata": metadata  # Include metadata in output
    }
    
    # Вывод результатов и метрик
    os.system("clear")
    print(f"Вопрос: {query}")
    print(f"Уверенность в ответе: {confidence}%")
    print(f"Ответ: {response}")
    print("=" * 100)
    print("\nНаиболее релевантный фрагмент:")
    print(docs[0].page_content)
    # print("=" * 100)
    # print(docs)

    print("\nМетаданные документа:")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    print(json.dumps(output, ensure_ascii=False, indent=2))
    print("=" * 100)
    print(metadata['source'])

if __name__ == "__main__":
    main()