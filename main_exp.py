import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import diskcache
from concurrent.futures import ThreadPoolExecutor
import json
import os

@dataclass
class Document:
    content: str
    timestamp: datetime
    filepath: Path
    metadata: Dict = None

class Config:
    def __init__(self):
        self.decay_factor = 0.1
        self.max_chunk_size = 512
        self.min_chunk_size = 128
        self.time_weight_factor = 0.3
        self.cache_dir = Path("./cache")
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.relevance_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.use_gpu = torch.cuda.is_available()
        self.num_threads = 4
        self.top_k_documents = 5
        self.cache_ttl = 86400  # 24 hours

class RAGSystem:
    def __init__(self, config: Config):
        self.config = config
        self.cache = diskcache.Cache(str(config.cache_dir))
        
        # Инициализация моделей
        self.embedding_model = SentenceTransformer(config.embedding_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.relevance_model)
        self.relevance_model = AutoModelForSequenceClassification.from_pretrained(
            config.relevance_model
        )
        
        if config.use_gpu:
            self.embedding_model.to('cuda')
            self.relevance_model.to('cuda')
            
        # Инициализация индексов
        self.document_store: List[Document] = []
        self.chunks: List[Tuple[str, float]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.IndexFlatL2] = None
        
        logger.info("RAG система инициализирована")

    def add_document(self, filepath: Union[str, Path], content: str, 
                    timestamp: Optional[datetime] = None,
                    metadata: Optional[Dict] = None) -> None:
        """Добавляет документ в систему"""
        if timestamp is None:
            timestamp = datetime.now()
            
        filepath = Path(filepath)
        doc = Document(content=content, timestamp=timestamp, 
                      filepath=filepath, metadata=metadata)
        
        self.document_store.append(doc)
        self._update_indices()
        logger.info(f"Добавлен документ: {filepath}")

    def _calculate_time_weight(self, timestamp: datetime) -> float:
        """Вычисляет временной вес документа"""
        time_diff = (datetime.now() - timestamp).days
        return np.exp(-self.config.decay_factor * time_diff)

    def _create_weighted_chunks(self) -> List[Tuple[str, float]]:
        """Создает взвешенные чанки с учетом времени"""
        weighted_chunks = []
        
        for doc in sorted(self.document_store, 
                         key=lambda x: x.timestamp, reverse=True):
            time_weight = self._calculate_time_weight(doc.timestamp)
            chunks = self._split_text(doc.content)
            
            for chunk in chunks:
                content_weight = len(chunk) / self.config.max_chunk_size
                final_weight = (
                    time_weight * self.config.time_weight_factor + 
                    content_weight * (1 - self.config.time_weight_factor)
                )
                weighted_chunks.append((chunk, final_weight))
                
        return sorted(weighted_chunks, key=lambda x: x[1], reverse=True)

    def _split_text(self, text: str) -> List[str]:
        """Разбивает текст на чанки"""
        chunks = []
        words = text.split()
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            current_length = len(' '.join(current_chunk))
            
            if current_length >= self.config.max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                
        if current_chunk and len(' '.join(current_chunk)) >= self.config.min_chunk_size:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def _update_indices(self) -> None:
        """Обновляет поисковые индексы"""
        self.chunks = self._create_weighted_chunks()
        chunk_texts = [chunk[0] for chunk in self.chunks]
        
        # Обновление BM25
        tokenized_chunks = [text.split() for text in chunk_texts]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        # Обновление FAISS
        embeddings = self.embedding_model.encode(
            chunk_texts, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
            
        if self.config.use_gpu:
            embeddings = embeddings.cpu()
            
        self.faiss_index.reset()
        self.faiss_index.add(embeddings.numpy())
        
        logger.info("Индексы обновлены")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Выполняет поиск по запросу
        
        Returns:
            List[Dict]: Список найденных чанков с метаданными
        """
        cache_key = f"search_{query}_{top_k}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            return cached_result
            
        # Гибридный поиск
        bm25_scores = self.bm25.get_scores(query.split())
        
        query_embedding = self.embedding_model.encode(
            query, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        if self.config.use_gpu:
            query_embedding = query_embedding.cpu()
            
        faiss_scores, faiss_indices = self.faiss_index.search(
            query_embedding.numpy().reshape(1, -1), 
            top_k
        )
        
        # Комбинируем результаты с учетом временных весов
        combined_scores = []
        for idx, (chunk, time_weight) in enumerate(self.chunks):
            score = (
                0.4 * bm25_scores[idx] +
                0.3 * (1 - faiss_scores[0][faiss_indices[0] == idx][0] 
                       if idx in faiss_indices[0] else 0) +
                0.3 * time_weight
            )
            combined_scores.append((chunk, score, idx))
            
        # Сортируем и возвращаем топ результаты
        results = []
        for chunk, score, idx in sorted(
            combined_scores, 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]:
            # Находим исходный документ
            doc_idx = next(
                (i for i, doc in enumerate(self.document_store) 
                 if chunk in doc.content), 
                None
            )
            
            if doc_idx is not None:
                doc = self.document_store[doc_idx]
                results.append({
                    'chunk': chunk,
                    'score': float(score),
                    'filepath': str(doc.filepath),
                    'timestamp': doc.timestamp.isoformat(),
                    'metadata': doc.metadata
                })
                
        self.cache.set(cache_key, results, expire=self.config.cache_ttl)
        return results

    def save_state(self, filepath: Union[str, Path]) -> None:
        """Сохраняет состояние системы"""
        state = {
            'documents': [
                {
                    'content': doc.content,
                    'timestamp': doc.timestamp.isoformat(),
                    'filepath': str(doc.filepath),
                    'metadata': doc.metadata
                }
                for doc in self.document_store
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Состояние сохранено в {filepath}")

    def load_state(self, filepath: Union[str, Path]) -> None:
        """Загружает состояние системы"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
            
        self.document_store = []
        for doc_data in state['documents']:
            doc = Document(
                content=doc_data['content'],
                timestamp=datetime.fromisoformat(doc_data['timestamp']),
                filepath=Path(doc_data['filepath']),
                metadata=doc_data['metadata']
            )
            self.document_store.append(doc)
            
        self._update_indices()
        logger.info(f"Состояние загружено из {filepath}")

def main():
    """Main function"""
    config = Config()
    rag_system = RAGSystem(config)
    
    # **********************************************************************
    # query = "Какая тенденция наблюдается к выбору сотрудничества?"
    # query = "какой внешний вид сотрудника в части волос?"
    query = "какая зона ответственности продавца?"
    # query = "как устроено обслуживание покупателей?"
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
    # print(f"Уверенность в ответе: {confidence}%")
    # print(f"Ответ: {response}")
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