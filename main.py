from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import TypedDict, List, Optional
import os
import re
import json
from langchain_openai import ChatOpenAI
import threading
from collections import OrderedDict
import hashlib
from langchain_community.document_loaders import CSVLoader
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ── LLM & Embeddings ──────────────────────────────────────────────────────────

load_dotenv()

def _to_text(content) -> str:
    
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type", "text") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content or "")




llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)



embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


RERANKER_MODEL = os.getenv("OMNIRAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoder(RERANKER_MODEL)

MAX_RETRIEVAL_ATTEMPTS = 2
MAX_CRITIC_ATTEMPTS = 2

GRADER_MAX_CONCURRENCY = 8

GRAPH_RECURSION_LIMIT = 60

# ── Hybrid retrieval + reranking tunables ─────────────────────────────────────
VECTOR_CANDIDATES = int(os.getenv("OMNIRAG_VECTOR_CANDIDATES", "30"))   
LEXICAL_CANDIDATES = int(os.getenv("OMNIRAG_LEXICAL_CANDIDATES", "30"))  
RRF_K = int(os.getenv("OMNIRAG_RRF_K", "60"))                            
RERANK_TOP_N = int(os.getenv("OMNIRAG_RERANK_TOP_N", "8"))               

# ── Data directories ──────────────────────────────────────────────
DATA_DIR = os.getenv(
    "OMNIRAG_DATA_DIR",
    os.path.dirname(os.path.abspath(__file__))
)

CACHE_DIR = os.path.join(DATA_DIR, "vectorstore_cache")
COLLECTIONS_DIR = os.path.join(DATA_DIR, "knowledge_base_collections")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(COLLECTIONS_DIR, exist_ok=True)

_COLLECTIONS_LOCK = threading.Lock()
_COLLECTION_LOCKS: dict[str, threading.RLock] = {}

def _validate_collection_name(name: str) -> str:
    name = (name or "").strip()
    if not name or len(name) > 64 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]*", name):
        raise ValueError("Collection name must be 1-64 characters and use only letters, numbers, spaces, dots, underscores, or hyphens; it must start with a letter or number.")
    return name


def _collection_id(name: str) -> str:
    return hashlib.sha256(_validate_collection_name(name).casefold().encode("utf-8")).hexdigest()[:32]


def _collection_path(name: str) -> str:
    return os.path.join(COLLECTIONS_DIR, _collection_id(name))


def _collection_lock(name: str):
    key = _collection_id(name)
    with _COLLECTIONS_LOCK:
        return _COLLECTION_LOCKS.setdefault(key, threading.RLock())


def ensure_knowledge_collection(name: str) -> dict:
    """Create a collection manifest without altering the retrieval architecture."""
    name = _validate_collection_name(name)
    path = _collection_path(name)
    os.makedirs(path, exist_ok=True)
    manifest_path = os.path.join(path, "collection.json")
    with _collection_lock(name):
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        else:
            manifest = {"name": name, "files": [], "version": 1}
            _write_collection_manifest(manifest_path, manifest)
    return {"name": manifest.get("name", name), "file_count": len(manifest.get("files", [])), "path": path}


def _write_collection_manifest(path: str, manifest: dict) -> None:
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def list_knowledge_collections() -> list[dict]:
    """Return persisted collection names and basic inventory metadata."""
    result = []
    try:
        entries = os.listdir(COLLECTIONS_DIR)
    except OSError:
        return result
    for entry in entries:
        manifest_path = os.path.join(COLLECTIONS_DIR, entry, "collection.json")
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            result.append({
                "name": manifest.get("name", entry),
                "file_count": len(manifest.get("files", [])),
                "has_index": os.path.isfile(os.path.join(COLLECTIONS_DIR, entry, "index.faiss")),
            })
        except (OSError, ValueError, TypeError):
            continue
    return sorted(result, key=lambda item: item["name"].casefold())


def get_collection_vectorstore(name: str):
    """Load a named collection's persistent FAISS index, using the existing LRU."""
    name = _validate_collection_name(name)
    path = _collection_path(name)
    if not os.path.isfile(os.path.join(path, "index.faiss")):
        raise ValueError(f"Collection '{name}' has no indexed documents yet. Upload at least one file first.")
    cache_key = "collection:" + _collection_id(name)
    vectorstore = _recall_vectorstore(cache_key)
    if vectorstore is None:
        vectorstore = FAISS.load_local(
            path, embeddings, allow_dangerous_deserialization=True
        )
        _remember_vectorstore(cache_key, vectorstore)
    return vectorstore

# In-memory layer on top of the on-disk FAISS cache: a repeat question about the
# same files skips even the disk load. Small LRU so memory stays bounded.
_VS_MEMORY: "OrderedDict[str, FAISS]" = OrderedDict()
_VS_MEMORY_MAX = int(os.getenv("OMNIRAG_MEMORY_INDEXES", "8"))
_VS_LOCK = threading.Lock()


def _remember_vectorstore(file_hash: str, vectorstore) -> None:
    with _VS_LOCK:
        _VS_MEMORY[file_hash] = vectorstore
        _VS_MEMORY.move_to_end(file_hash)
        while len(_VS_MEMORY) > _VS_MEMORY_MAX:
            _VS_MEMORY.popitem(last=False)


def _recall_vectorstore(file_hash: str):
    with _VS_LOCK:
        vectorstore = _VS_MEMORY.get(file_hash)
        if vectorstore is not None:
            _VS_MEMORY.move_to_end(file_hash)
        return vectorstore


# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    filetype: str
    question: str
    answer: str
    search_query: str
    file_paths: List[str]
    collection_name: str
    file_hash: str
    from_cache: bool
    documents: List[Document]
    skipped_files: List[str]
    pdf_type: str
    chunks: List[Document]
    vectorstore: object
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    filtered_docs: List[Document]
    retrieval_attempts: int
    sufficient: str
    critic_verdict: str
    critic_reason: str
    critic_attempts: int
    evaluation: dict
    final_verdict: str
    final_reason: dict


# ── Cache lookup (runs before any loading/chunking/embedding) ────────────────

def compute_hash_node(state: State):
    
    if state.get("collection_name"):
        return {"file_hash": "collection:" + _collection_id(state["collection_name"])}

    per_file_hashes = []
    for file_path in state["file_paths"]:
        with open(file_path, "rb") as f:
            content = f.read()
        per_file_hashes.append(hashlib.md5(content).hexdigest())

    combined = "docling-hybrid-chunks-v1|" + "|".join(sorted(per_file_hashes))
    file_hash = hashlib.md5(combined.encode("utf-8")).hexdigest()

    return {"file_hash": file_hash}


def _cache_path(file_hash: str) -> str:
    return os.path.join(CACHE_DIR, file_hash)


def route_cache(state: State):
    if state.get("collection_name"):
        index_file = os.path.join(_collection_path(state["collection_name"]), "index.faiss")
    else:
        index_file = os.path.join(_cache_path(state["file_hash"]), "index.faiss")

    if os.path.exists(index_file):
        return "load_cache_node"

    return "load_documents_node"


def load_cache_node(state: State):
    if state.get("collection_name"):
        vectorstore = get_collection_vectorstore(state["collection_name"])
    else:
        vectorstore = _recall_vectorstore(state["file_hash"])

        if vectorstore is None:
            vectorstore = FAISS.load_local(
                _cache_path(state["file_hash"]),
                embeddings,
                allow_dangerous_deserialization=True
            )
            _remember_vectorstore(state["file_hash"], vectorstore)

    return {
        "vectorstore": vectorstore,
        "from_cache": True
    }


def save_cache_node(state: State):
    state["vectorstore"].save_local(_cache_path(state["file_hash"]))
    _remember_vectorstore(state["file_hash"], state["vectorstore"])

    return {"from_cache": False}


# ── File type detection & per-file loading ────────────────────────────────────

DOCLING_EXTENSIONS = {
    # Documents
    ".pdf", ".docx", ".doc", ".odt", ".epub",

    # Spreadsheets
    ".xlsx", ".xls", ".ods",

    # Presentations
    ".pptx", ".ppt", ".odp",

    # Images
    ".png", ".jpeg", ".jpg", ".tiff", ".tif", ".bmp", ".webp",

    # Web
    ".html", ".xhtml",

    # Markup / text
    ".md", ".markdown", ".adoc", ".asciidoc", ".tex", ".latex",

    # Email
    ".eml", ".msg",

    # Captions
    ".vtt",
}


def _detect_filetype(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return "csv"
    elif ext in DOCLING_EXTENSIONS:
        return "docling"
    return "unsupported"


def _load_csv(file_path: str) -> List[Document]:
    loader = CSVLoader(
        file_path=file_path,
        encoding="utf-8"
    )
    return loader.load()


_DOCLING_CONVERTER = None
_DOCLING_LOCK = threading.Lock()
_DOCLING_CHUNKER = None
_DOCLING_CHUNKER_LOCK = threading.Lock()


def _get_docling_converter():
    """DocumentConverter is expensive to build; create it once and reuse it."""
    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is None:
        with _DOCLING_LOCK:
            if _DOCLING_CONVERTER is None:
                _DOCLING_CONVERTER = DocumentConverter()
    return _DOCLING_CONVERTER


def _get_docling_chunker():
    """Create Docling's structure-aware chunker once and reuse it."""
    global _DOCLING_CHUNKER
    if _DOCLING_CHUNKER is None:
        with _DOCLING_CHUNKER_LOCK:
            if _DOCLING_CHUNKER is None:
                try:
                    from docling.chunking import HybridChunker
                except ImportError:
                    # Support installations that expose HybridChunker from docling-core.
                    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
                _DOCLING_CHUNKER = HybridChunker(merge_peers=True)
    return _DOCLING_CHUNKER


def _load_docling(file_path: str) -> List[Document]:
    """Convert with Docling and preserve structure/provenance per chunk.

    HybridChunker retains heading/table context and source provenance. Exporting
    the whole document to Markdown before chunking would lose page-level links.
    """
    converter = _get_docling_converter()
    result = converter.convert(file_path)
    chunker = _get_docling_chunker()
    documents: List[Document] = []

    for chunk_index, chunk in enumerate(chunker.chunk(result.document), start=1):
        content = (chunker.contextualize(chunk) or getattr(chunk, "text", "") or "").strip()
        if not content:
            continue

        chunk_meta = getattr(chunk, "meta", None)
        page_numbers = set()
        for item in (getattr(chunk_meta, "doc_items", None) or []):
            for provenance in (getattr(item, "prov", None) or []):
                page_number = getattr(provenance, "page_no", None)
                if page_number is not None:
                    page_numbers.add(page_number)

        # Some Docling versions also expose a page number on origin.
        origin_page = getattr(getattr(chunk_meta, "origin", None), "page_no", None)
        if not page_numbers and origin_page is not None:
            page_numbers.add(origin_page)

        headings = list(getattr(chunk_meta, "headings", None) or [])
        documents.append(Document(
            page_content=content,
            metadata={
                "source": os.path.basename(file_path),
                "page_numbers": sorted(
                    page_numbers,
                    key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)),
                ),
                "headings": headings,
                "chunk_id": chunk_index,
                "_docling_hybrid_chunk": True,
            },
        ))

    return documents


_LOADERS = {
    "csv": _load_csv,
    "docling": _load_docling,
}


def load_documents_node(state: State):
    
    all_docs: List[Document] = []
    skipped: List[str] = []

    for file_path in state["file_paths"]:
        name = os.path.basename(file_path)
        filetype = _detect_filetype(file_path)
        loader = _LOADERS.get(filetype)

        if loader is None:
            skipped.append(f"{name} (unsupported format)")
            continue

        try:
            docs = loader(file_path)
        except Exception as e:
            skipped.append(f"{name} (failed to load: {e})")
            continue

        if not docs or not any(d.page_content.strip() for d in docs):
            skipped.append(f"{name} (no content extracted)")
            continue

        for doc in docs:
            doc.metadata["source"] = name

        all_docs.extend(docs)

    return {
        "documents": all_docs,
        "skipped_files": skipped,
    }


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunking_node(state: State):
    
    documents = state["documents"]
    hybrid_chunks = [doc for doc in documents if doc.metadata.get("_docling_hybrid_chunk")]
    plain_documents = [doc for doc in documents if not doc.metadata.get("_docling_hybrid_chunk")]

    chunks = list(hybrid_chunks)
    if plain_documents:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks.extend(splitter.split_documents(plain_documents))

    return {"chunks": chunks}


def ingest_files_into_collection(collection_name: str, file_paths: List[str], file_names: List[str]) -> dict:
    """Add only previously unseen file contents to a named persistent collection.

    Document extraction and chunking reuse the same loaders and chunking node as
    the normal pipeline; existing FAISS vectors are appended to, not rebuilt.
    """
    collection = ensure_knowledge_collection(collection_name)
    collection_name = collection["name"]
    path = _collection_path(collection_name)
    manifest_path = os.path.join(path, "collection.json")
    index_path = os.path.join(path, "index.faiss")
    file_names = list(file_names or [])
    file_paths = list(file_paths or [])
    if len(file_paths) != len(file_names):
        raise ValueError("Uploaded file paths and names do not match.")

    with _collection_lock(collection_name):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        known_hashes = {item.get("sha256") for item in manifest.get("files", [])}
        existing_vs = None
        cache_key = "collection:" + _collection_id(collection_name)
        if os.path.isfile(index_path):
            existing_vs = _recall_vectorstore(cache_key)
            if existing_vs is None:
                existing_vs = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
                _remember_vectorstore(cache_key, existing_vs)

        docs_to_add: List[Document] = []
        manifest_additions: list[dict] = []
        added_names: list[str] = []
        already_present: list[str] = []
        skipped: list[str] = []
        for file_path, display_name in zip(file_paths, file_names):
            digest = hashlib.sha256()
            with open(file_path, "rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(block)
            file_digest = digest.hexdigest()
            if file_digest in known_hashes:
                already_present.append(display_name)
                continue

            filetype = _detect_filetype(file_path)
            loader = _LOADERS.get(filetype)
            if loader is None:
                skipped.append(f"{display_name} (unsupported format)")
                continue
            try:
                file_docs = loader(file_path)
            except Exception as exc:
                skipped.append(f"{display_name} (failed to load: {type(exc).__name__}: {exc})")
                continue
            if not file_docs or not any(doc.page_content.strip() for doc in file_docs):
                skipped.append(f"{display_name} (no content extracted)")
                continue

            for doc in file_docs:
                doc.metadata["source"] = display_name
                doc.metadata.setdefault("page_numbers", [])
                doc.metadata.setdefault("headings", [])
            docs_to_add.extend(file_docs)
            manifest_additions.append({"name": display_name, "sha256": file_digest})
            added_names.append(display_name)
            known_hashes.add(file_digest)

        chunks = chunking_node({"documents": docs_to_add}).get("chunks", []) if docs_to_add else []
        if chunks:
            if existing_vs is None:
                existing_vs = FAISS.from_documents(chunks, embeddings)
            else:
                existing_vs.add_documents(chunks)
                # BM25 is cached on the FAISS object; invalidate after appending.
                if hasattr(existing_vs, "_omnirag_bm25"):
                    try:
                        delattr(existing_vs, "_omnirag_bm25")
                    except Exception:
                        setattr(existing_vs, "_omnirag_bm25", None)
            existing_vs.save_local(path)
            _remember_vectorstore(cache_key, existing_vs)

        if existing_vs is None:
            if skipped:
                raise ValueError("No usable files were added. " + "; ".join(skipped))
            raise ValueError(f"Collection '{collection_name}' has no indexed documents yet. Upload at least one file first.")

        if manifest_additions:
            manifest.setdefault("files", []).extend(manifest_additions)
            _write_collection_manifest(manifest_path, manifest)

        return {
            "name": collection_name,
            "files_added": added_names,
            "files_already_present": already_present,
            "skipped_files": skipped,
            "new_chunks": len(chunks),
            "total_files": len(manifest.get("files", [])),
        }


# ── Storing embeddings in vector store ────────────────────────────────────────

def vectorstore_node(state: State):
    vectorstore = FAISS.from_documents(
        documents=state["chunks"],
        embedding=embeddings
    )

    return {
        "vectorstore": vectorstore
    }


# ── Hybrid retrieval (dense vector + sparse lexical, fused with RRF) ─────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _all_indexed_docs(vectorstore: FAISS) -> List[Document]:
    """Pull every chunk out of the FAISS docstore.

    Works whether we just built the vectorstore this request or loaded it
    from the on-disk cache (in which case we never had the raw `chunks` list
    in memory), so BM25 always has the full corpus to search.
    """
    docstore = getattr(vectorstore, "docstore", None)
    data = getattr(docstore, "_dict", None)
    if isinstance(data, dict):
        return list(data.values())
    return []


def _get_bm25(vectorstore: FAISS):
    """Build the BM25 index once per vectorstore instead of on every retrieval
    pass (the graph can retrieve up to 3-5 times per question)."""
    cached = getattr(vectorstore, "_omnirag_bm25", None)
    if cached is None:
        docs = _all_indexed_docs(vectorstore)
        tokenized_corpus = [_tokenize(doc.page_content) for doc in docs]
        bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        cached = (docs, bm25)
        try:
            setattr(vectorstore, "_omnirag_bm25", cached)
        except Exception:
            pass
    return cached


def _bm25_search(query: str, docs: List[Document], top_n: int, bm25=None) -> List[Document]:
    if not docs:
        return []
    if bm25 is None:
        tokenized_corpus = [_tokenize(doc.page_content) for doc in docs]
        bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in ranked[:top_n] if scores[i] > 0]


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    limit: int,
    k: int = RRF_K,
) -> List[Document]:
    """Merge several ranked document lists into one ranking.
    """
    scores: dict[str, float] = {}
    objects: dict[str, Document] = {}
    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs, start=1):
            key = doc.metadata.get("source", "") + f"::{hash(doc.page_content)}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            objects[key] = doc
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [objects[key] for key, _ in ordered[:limit]]


def retrieval_node(state: State):

    search_query = (state.get("search_query") or state["question"])


    retrieval_attempts = state.get(
        "retrieval_attempts",
        0
    )

    critic_attempts = state.get(
        "critic_attempts",
        0
    )

    total_attempts = (
        retrieval_attempts +
        critic_attempts
    )

    k = min(
        5 + total_attempts * 5,
        20
    )

    # Dense/vector leg: FAISS with MMR for some built-in diversity.
    retriever = state["vectorstore"].as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": min(max(VECTOR_CANDIDATES, k * 3), 50)
        }
    )
    vector_docs = retriever.invoke(search_query)

    # Sparse/lexical leg: BM25 over the full indexed corpus.
    all_docs, bm25 = _get_bm25(state["vectorstore"])
    lexical_docs = _bm25_search(search_query, all_docs, min(max(LEXICAL_CANDIDATES, k * 2), 50), bm25)

    # Fuse the two rankings instead of just concatenating or picking one.
    fused_docs = _reciprocal_rank_fusion([vector_docs, lexical_docs], limit=max(k, RERANK_TOP_N))

    return {
        "retrieved_docs": fused_docs
    }


# ── Reranking ──────────────────────────────────────────────────────────────────
#
# The fused hybrid list is still ranked by cheap proxies (embedding distance,
# BM25 term overlap). A cross-encoder jointly reads (question, chunk) pairs
# and scores relevance directly, which is slower but meaningfully more
# accurate, so we use it to reorder/trim before the expensive LLM grading step.

def rerank_node(state: State):
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"reranked_docs": []}

    pairs = [(state["question"], doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    top_docs = [doc for doc, _ in ranked[:RERANK_TOP_N]]

    return {
        "reranked_docs": top_docs
    }


class GradeDocuments(BaseModel):
    """Binary score for relevance check."""

    binary_score: str = Field(
        description="Document is relevant to the question, 'yes' or 'no'"
    )


structured_llm = llm.with_structured_output(GradeDocuments)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a relevance grader.

            Your task is to determine whether a retrieved document
            contains information that can help answer the user's question.

            It does NOT need to contain the exact answer.
            It only needs to be relevant.

            Respond with:
            - yes -> relevant
            - no -> irrelevant
            """
        ),
        (
            "human",
            """
            Question:
            {question}

            Retrieved Document:
            {document}
            """
        ),
    ]
)

retrieval_grader = grade_prompt | structured_llm


def filter_chunks_node(state: State):
    question = state["question"]
    docs = state.get("reranked_docs") or state["retrieved_docs"]

    if not docs:
        return {"filtered_docs": []}

    # Grade every retrieved chunk CONCURRENTLY instead of one-by-one in a
    # for-loop. This is the single biggest latency fix: 20 sequential LLM
    # calls (~20-40s) becomes a handful of parallel batches.
    inputs = [
        {"question": question, "document": doc.page_content}
        for doc in docs
    ]

    results = retrieval_grader.batch(
        inputs,
        config={"max_concurrency": GRADER_MAX_CONCURRENCY}
    )

    filtered_docs = [
        doc for doc, result in zip(docs, results)
        if result.binary_score.lower() == "yes"
    ]

    return {
        "filtered_docs": filtered_docs
    }


# ── Checking if there is enough context to generate an answer ────────────────

class SufficiencyGrade(BaseModel):

    sufficient: str = Field(
        description="Whether the provided documents are sufficient to answer the question. Return only 'yes' or 'no'."
    )


structured_llm2 = llm.with_structured_output(
    SufficiencyGrade
)

sufficiency_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a retrieval sufficiency grader.

            Determine whether the retrieved documents contain
            enough information to answer the user's question.

            Consider:

            - Completeness
            - Coverage
            - Missing information

            Respond only:

            yes
            no
            """
        ),
        (
            "human",
            """
            Question:
            {question}

            Retrieved Documents:
            {documents}
            """
        )
    ]
)

sufficiency_chain = (
    sufficiency_prompt
    | structured_llm2
)


def check_context_sufficiency_node(state: State):

    docs_text = "\n\n".join(
        doc.page_content
        for doc in state["filtered_docs"]
    )

    result = sufficiency_chain.invoke(
        {
            "question": state["question"],
            "documents": docs_text
        }
    )

    return {
        "sufficient": result.sufficient
    }


def route_after_sufficiency(state):

    sufficient = state["sufficient"]
    attempts = state.get(
        "retrieval_attempts", 0)

    if sufficient.lower() == "yes":
        return "generate_answer"

    if attempts >= MAX_RETRIEVAL_ATTEMPTS:
        return "generate_answer"

    return "increase_attempts_node"


def increase_attempts_node(state: State):

    return {
        "retrieval_attempts":
            state.get(
                "retrieval_attempts",
                0
            ) + 1
    }


# ── Query Rewriting ─────────────────────────────────────────────

query_rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a query rewriting component in a RAG system.

        Rewrite the user's question into a clear, specific,
        standalone search query suitable for hybrid retrieval.

        Rules:
        - Preserve the original intent.
        - Resolve ambiguity where possible.
        - Include important keywords and entities.
        - Do not answer the question.
        - Return only the rewritten query.
        """
    ),
    (
        "human",
        """
        Original question:
        {question}

        Retrieved context was insufficient.
        Produce an improved search query.
        """
    )
])

query_rewrite_chain = query_rewrite_prompt | llm | StrOutputParser()


def query_rewriting_node(state: State):
    rewritten_query = query_rewrite_chain.invoke({
        "question": state["question"]
    }).strip()

    return {
        "search_query": rewritten_query or state["question"]
    }


# ── Answer generation ──────────────────────────────────────────────────────────

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.

            Answer the question using ONLY the provided context.

            Rules:
            - Use only information from the context.
            - Do not make up facts.
            - If the context does not contain enough information,
              clearly say so.
            - Cite supporting chunks with the source IDs exactly as supplied,
              for example [S1] or [S2]. Never invent source IDs.
            - Add source citations to factual claims where supported by context.
            - Be concise and accurate.
            """
        ),
        (
            "human",
            """
            Context:
            {context}

            Question:
            {question}
            """
        )
    ]
)
answer_chain = answer_prompt | llm


def _format_context_with_citations(docs: List[Document]) -> str:
    """Label context chunks with source IDs used by the answer and Sources UI."""
    formatted = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source") or "Unknown source"
        pages = metadata.get("page_numbers") or []
        if not pages and metadata.get("page_number") is not None:
            pages = [metadata["page_number"]]
        headings = metadata.get("headings") or []

        location_parts = []
        if pages:
            location_parts.append("page " + ", ".join(str(page) for page in pages))
        if headings:
            location_parts.append("section " + " > ".join(str(item) for item in headings))
        location = f" ({'; '.join(location_parts)})" if location_parts else ""
        formatted.append(
            f"[S{index}] Source: {source}{location}\nContent:\n{doc.page_content}"
        )
    return "\n\n".join(formatted)


def generation_node(state: State):
    """Generate the answer and publish each text chunk to LangGraph.

    Returning the joined text keeps the graph state unchanged, while the
    custom stream writer lets the API send text to the UI as LLM produces
    it instead of waiting for a completed ``invoke()`` response.
    """
    context = _format_context_with_citations(state.get("filtered_docs", []))

    writer = get_stream_writer()
    answer_parts: List[str] = []
    for chunk in answer_chain.stream(
        {
            "question": state["question"],
            "context": context,
        }
    ):
        text = _to_text(chunk.content)
        if not text:
            continue
        answer_parts.append(text)
        writer({"type": "token", "text": text})

    return {
        "answer": "".join(answer_parts)
    }


class CriticGrade(BaseModel):

    verdict: str = Field(
        description="""
        pass or fail
        """
    )

    reason: str = Field(
        description="""
        One of:

        sufficient_context
        insufficient_context
        hallucination
        irrelevant_answer
        """
    )


critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a critic agent for a RAG system.

            Evaluate whether the answer is supported
            by the provided context and answers
            the user's question.

            Return:

            pass
            fail

            Reasons:

            sufficient_context
                -> answer is correct

            insufficient_context
                -> context does not contain enough information

            hallucination
                -> answer contains unsupported facts

            irrelevant_answer
                -> answer does not answer the question
            """
        ),
        (
            "human",
            """
            Question:
            {question}

            Context:
            {context}

            Answer:
            {answer}
            """
        )
    ]
)
critic_chain = (
    critic_prompt
    | llm.with_structured_output(
        CriticGrade
    )
)


def critic_node(state):

    context = _format_context_with_citations(state.get("filtered_docs", []))

    result = critic_chain.invoke(
        {
            "question": state["question"],
            "context": context,
            "answer": state["answer"]
        }
    )

    return {
        "critic_verdict": result.verdict,
        "critic_reason": result.reason
    }


def route_after_critic(state: State):

    verdict = (state["critic_verdict"] or "").strip().lower()

    if verdict == "pass":
        return "finish"

    if state.get("critic_attempts", 0) >= MAX_CRITIC_ATTEMPTS:
        return "answer_not_found"

    return "increase_critic_attempts"


def increase_critic_attempts(state: State):

    return {
        "critic_attempts":
            state.get(
                "critic_attempts",
                0
            ) + 1
    }


def _not_found_message(state: State) -> str:
    skipped = state.get("skipped_files") or []
    if skipped and not state.get("documents"):
        return "None of the uploaded files could be read: " + "; ".join(skipped)
    return "Answer not found in the uploaded documents."


def answer_not_found_node(state: State):

    return {
        "answer": _not_found_message(state),

        "final_verdict": "answer_not_found",

        "final_reason": {
            "critic_attempts": state.get("critic_attempts", 0),
            "last_critic_reason": state.get(
                "critic_reason",
                "unknown"
            )
        }
    }


# ── Graph construction ─────────────────────────────────────────────────────────

def route_after_load(state: State):
    # Every file unsupported/unreadable -> stop here instead of crashing FAISS
    # with an empty document list.
    if state.get("documents"):
        return "chunking_node"

    return "answer_not_found"


builder = StateGraph(State)

# Cache check (new)
builder.add_node("compute_hash_node", compute_hash_node)
builder.add_node("load_cache_node", load_cache_node)
builder.add_node("save_cache_node", save_cache_node)

builder.add_node("load_documents_node", load_documents_node)

builder.add_node(
    "chunking_node",
    chunking_node
)

builder.add_node(
    "vectorstore_node",
    vectorstore_node
)

builder.add_node(
    "retrieval_node",
    retrieval_node
)

builder.add_node(
    "rerank_node",
    rerank_node
)

builder.add_node(
    "filter_chunks_node",
    filter_chunks_node
)

builder.add_node(
    "check_context_sufficiency_node",
    check_context_sufficiency_node
)

builder.add_node(
    "increase_attempts_node",
    increase_attempts_node
)

builder.add_node(
    "query_rewriting_node",
    query_rewriting_node
)

builder.add_node(
    "generation_node",
    generation_node
)

builder.add_node(
    "critic_node",
    critic_node
)

builder.add_node(
    "increase_critic_attempts",
    increase_critic_attempts
)

builder.add_node(
    "answer_not_found",
    answer_not_found_node
)

# ── Edges ────────────────────────────────────────────────────────────────────

builder.add_edge(START, "compute_hash_node")

# Cache branch: skip load/chunk/embed entirely on a cache hit.
builder.add_conditional_edges(
    "compute_hash_node",
    route_cache,
    {
        "load_cache_node": "load_cache_node",
        "load_documents_node": "load_documents_node"
    }
)

builder.add_edge("load_cache_node", "retrieval_node")
builder.add_edge("retrieval_node", "rerank_node")
builder.add_edge("rerank_node", "filter_chunks_node")

builder.add_conditional_edges(
    "load_documents_node",
    route_after_load,
    {
        "chunking_node": "chunking_node",
        "answer_not_found": "answer_not_found"
    }
)
builder.add_edge(
    "chunking_node",
    "vectorstore_node"
)

builder.add_edge(
    "vectorstore_node",
    "save_cache_node"
)

builder.add_edge(
    "save_cache_node",
    "retrieval_node"
)

builder.add_edge(
    "filter_chunks_node",
    "check_context_sufficiency_node"
)
builder.add_conditional_edges(
    "check_context_sufficiency_node",
    route_after_sufficiency,
    {
        "generate_answer":
            "generation_node",

        "increase_attempts_node":
            "increase_attempts_node"
    }
)
builder.add_edge(
    "increase_attempts_node",
    "query_rewriting_node"
)

builder.add_edge(
    "query_rewriting_node",
    "retrieval_node"
)
builder.add_edge(
    "generation_node",
    "critic_node"
)
builder.add_conditional_edges(
    "critic_node",
    route_after_critic,
    {
        "finish":
            END,

        "increase_critic_attempts":
            "increase_critic_attempts",

        "answer_not_found":
            "answer_not_found"
    }
)
builder.add_edge(
    "increase_critic_attempts",
    "retrieval_node"
)


builder.add_edge(
    "answer_not_found",
    END
)
graph = builder.compile()


# ── DeepEval Evaluation ──────────────────────────────────────────────────────

# DeepEval's metrics default to an OpenAI judge. 
# The three metrics are independent of each other,so they run in parallel threads rather than one after another.

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from concurrent.futures import ThreadPoolExecutor, as_completed
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

EVAL_THRESHOLD = 0.7
EVAL_METRICS = ["faithfulness", "answer_relevancy", "contextual_relevancy"]


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


class OpenAIJuddge(DeepEvalBaseLLM):
    """Lets DeepEval use the pipeline's own LangChain chat model as its judge."""

    def __init__(self, chat_model, name: str = "gpt-4o-mini"):
        self.chat_model = chat_model
        super().__init__(name)

    def load_model(self):
        return self.chat_model

    def generate(self, prompt: str, schema: Optional[type] = None):
        if schema is None:
            return _to_text(self.chat_model.invoke(prompt).content)
        try:
            return self.chat_model.with_structured_output(schema).invoke(prompt)
        except Exception:
            text = _to_text(self.chat_model.invoke(prompt).content)
            return schema(**_extract_json(text))

    async def a_generate(self, prompt: str, schema: Optional[type] = None):
        if schema is None:
            return _to_text((await self.chat_model.ainvoke(prompt)).content)
        try:
            return await self.chat_model.with_structured_output(schema).ainvoke(prompt)
        except Exception:
            text = _to_text((await self.chat_model.ainvoke(prompt)).content)
            return schema(**_extract_json(text))

    def get_model_name(self):
        return "gpt-4o-mini"


def eval_answer(state: State, progress_cb=None):
    """
    Evaluate an already generated RAG answer using DeepEval.

    This node is called separately after the normal RAG workflow
    has returned the answer. 
    """

    question = state["question"]
    answer = state["answer"]

    # Same context used during answer generation
    filtered_docs = state.get("filtered_docs", [])

    retrieval_context = [
        doc.page_content
        for doc in filtered_docs
        if hasattr(doc, "page_content")
    ]

    
    # DeepEval test case

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=retrieval_context,
    )
    # Metrics (one judge, sync mode so each runs cleanly in its own thread)
    

    judge = OpenAIJuddge(llm)

    metrics = {
        "faithfulness": FaithfulnessMetric(
            threshold=EVAL_THRESHOLD, model=judge,
            include_reason=True, async_mode=False,
        ),
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=EVAL_THRESHOLD, model=judge,
            include_reason=True, async_mode=False,
        ),
        "contextual_relevancy": ContextualRelevancyMetric(
            threshold=EVAL_THRESHOLD, model=judge,
            include_reason=True, async_mode=False,
        ),
    }

    def _run(name, metric):
        metric.measure(test_case, _show_indicator=False)
        score = metric.score
        return {
            "score": score,
            "reason": metric.reason,
            "passed": score is not None and score >= EVAL_THRESHOLD,
        }
    # Run all metrics in parallel

    results = {}
    with ThreadPoolExecutor(max_workers=len(metrics)) as pool:
        futures = {pool.submit(_run, n, m): n for n, m in metrics.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "score": None,
                    "reason": f"Evaluation failed: {type(exc).__name__}: {exc}",
                    "passed": False,
                    "error": True,
                }
            results[name] = result
            if progress_cb:
                progress_cb(name, result)

    # Collect score + reason 

    evaluation = {name: results[name] for name in EVAL_METRICS}

    # Overall score

    scores = [r["score"] for r in evaluation.values() if r["score"] is not None]
    overall_score = sum(scores) / len(scores) if scores else None

    evaluation["overall"] = {
        "score": overall_score,
        "passed": overall_score is not None and overall_score >= EVAL_THRESHOLD,
    }

    return {
        "evaluation": evaluation
    }


# ══════════════════════════════════════════════════════════════════════════════
# Direct-call service layer (used by app.py; no HTTP/API in between)
# ══════════════════════════════════════════════════════════════════════════════

import queue
import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterator

MAX_UPLOAD_FILE_BYTES = int(os.getenv("OMNIRAG_MAX_UPLOAD_FILE_BYTES", str(25 * 1024 * 1024)))
MAX_UPLOAD_TOTAL_BYTES = int(os.getenv("OMNIRAG_MAX_UPLOAD_TOTAL_BYTES", str(100 * 1024 * 1024)))
MAX_FILES_PER_REQUEST = int(os.getenv("OMNIRAG_MAX_FILES_PER_REQUEST", "20"))

SUPPORTED_EXTENSIONS = sorted({".csv", *DOCLING_EXTENSIONS})


def warmup() -> None:
    """Pay the model-loading cost once, before the first question."""
    try:
        embeddings.embed_query("warmup")
        reranker.predict([("warmup", "warmup")])
        converter = _get_docling_converter()
        try:
            from docling.datamodel.base_models import InputFormat

            converter.initialize_pipeline(InputFormat.PDF)
        except Exception:
            pass  # older docling versions: model loads lazily on first PDF
    except Exception as exc:  # warm-up must never break the app
        print(f"[omnirag] warm-up skipped: {type(exc).__name__}: {exc}")


def save_uploads(files: list[tuple[str, bytes]]) -> tuple[str, list[str], list[str]]:
    """Write (filename, bytes) uploads to a temp dir, enforcing size/count/type limits."""
    if len(files) > MAX_FILES_PER_REQUEST:
        raise ValueError(f"Too many files. Maximum allowed per request is {MAX_FILES_PER_REQUEST}.")
    workdir = tempfile.mkdtemp(prefix="omnirag_")
    paths: list[str] = []
    names: list[str] = []
    total_bytes = 0
    try:
        for i, (filename, data) in enumerate(files):
            if not filename:
                raise ValueError("Uploaded file must have a filename.")
            suffix = Path(filename).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                supported = ", ".join(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)
                raise ValueError(f"Unsupported file type '{suffix or 'unknown'}'. Supported: {supported}")
            original = Path(filename).name
            if len(data) > MAX_UPLOAD_FILE_BYTES:
                raise ValueError(f"File '{original}' exceeds the {MAX_UPLOAD_FILE_BYTES // (1024 * 1024)} MB per-file limit.")
            total_bytes += len(data)
            if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
                raise ValueError(f"Combined upload exceeds the {MAX_UPLOAD_TOTAL_BYTES // (1024 * 1024)} MB total limit.")
            safe = re.sub(r"[^\w.\- ]", "_", original) or f"file_{i}"
            target = Path(workdir) / safe
            if target.exists():
                target = Path(workdir) / f"{i}_{safe}"
            target.write_bytes(data)
            paths.append(str(target))
            names.append(original)
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    return workdir, paths, names


# Which node runs after which (so the UI can show "what's happening now").
_STATIC_NEXT = {
    "load_cache_node": "retrieval_node",
    "chunking_node": "vectorstore_node",
    "vectorstore_node": "save_cache_node",
    "save_cache_node": "retrieval_node",
    "retrieval_node": "rerank_node",
    "rerank_node": "filter_chunks_node",
    "filter_chunks_node": "check_context_sufficiency_node",
    "increase_attempts_node": "query_rewriting_node",
    "query_rewriting_node": "retrieval_node",
    "generation_node": "critic_node",
    "increase_critic_attempts": "retrieval_node",
    "answer_not_found": None,
}
_SUFFICIENCY_NEXT = {
    "generate_answer": "generation_node",
    "increase_attempts_node": "increase_attempts_node",
}
_CRITIC_NEXT = {
    "finish": None,
    "increase_critic_attempts": "increase_critic_attempts",
    "answer_not_found": "answer_not_found",
}


def _next_node(node: str, state: dict) -> Optional[str]:
    """Reuse the graph's own routers so the prediction can't drift from the graph."""
    try:
        if node == "compute_hash_node":
            return route_cache(state)
        if node == "load_documents_node":
            return route_after_load(state)
        if node == "check_context_sufficiency_node":
            return _SUFFICIENCY_NEXT.get(route_after_sufficiency(state))
        if node == "critic_node":
            return _CRITIC_NEXT.get(route_after_critic(state))
    except Exception:
        return None
    return _STATIC_NEXT.get(node)


def _step_detail(node: str, state: dict) -> str:
    n = lambda key: len(state.get(key) or [])  # noqa: E731
    if node == "load_cache_node":
        return "cache hit"
    if node == "load_documents_node":
        text = f"{n('documents')} document(s) loaded"
        if state.get("skipped_files"):
            text += f", {n('skipped_files')} skipped"
        return text
    if node == "chunking_node":
        return f"{n('chunks')} chunks"
    if node == "retrieval_node":
        return f"{n('retrieved_docs')} candidates"
    if node == "rerank_node":
        return f"top {n('reranked_docs')} kept"
    if node == "filter_chunks_node":
        return f"{n('filtered_docs')} of {n('reranked_docs')} relevant"
    if node == "check_context_sufficiency_node":
        ok = str(state.get("sufficient", "")).lower() == "yes"
        return "enough context" if ok else "needs more context"
    if node == "increase_attempts_node":
        return f"attempt {state.get('retrieval_attempts', 0)}"
    if node == "critic_node":
        verdict = str(state.get("critic_verdict", "")).lower()
        return f"{verdict} ({state.get('critic_reason', '')})"
    if node == "increase_critic_attempts":
        return f"attempt {state.get('critic_attempts', 0)}"
    return ""


def _run_pipeline(
    file_paths: list[str], file_names: list[str], question: str,
    collection_name: Optional[str] = None,
) -> Iterator[dict]:
    """Run the LangGraph pipeline against a named persistent collection."""
    started = time.perf_counter()
    final: dict = {"file_paths": file_paths, "question": question}
    if collection_name:
        final["collection_name"] = collection_name
    executed: list[str] = []
    last_t = 0.0
    generation_done = False

    yield {"type": "start", "file_names": file_names}

    for mode, payload in graph.stream(
        dict(final),
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        stream_mode=["updates", "custom"],
    ):
        if mode == "custom":
            if not isinstance(payload, dict) or payload.get("type") != "token":
                continue
            if generation_done:  # critic sent us back: a fresh draft is starting
                generation_done = False
                yield {"type": "reset"}
            yield payload
            continue

        for node, update in payload.items():
            if isinstance(update, dict):
                final.update(update)
            now = time.perf_counter() - started
            executed.append(node)
            if node == "generation_node":
                generation_done = True
            yield {
                "type": "step",
                "node": node,
                "detail": _step_detail(node, final),
                "next": _next_node(node, final),
                "t": round(now, 2),
                "dt": round(now - last_t, 2),
            }
            last_t = now

    docs = final.get("filtered_docs") or []
    sources = sorted({d.metadata.get("source") for d in docs if d.metadata.get("source")})
    source_chunks = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        pages = metadata.get("page_numbers") or []
        if not pages and metadata.get("page_number") is not None:
            pages = [metadata["page_number"]]
        source_chunks.append({
            "id": f"S{index}",
            "citation": f"[S{index}]",
            "source": metadata.get("source") or "Unknown source",
            "page_numbers": pages,
            "headings": metadata.get("headings") or [],
            "chunk_id": metadata.get("chunk_id"),
            "content": doc.page_content,
        })

    yield {
        "type": "final",
        "success": True,
        "question": question,
        "file_names": file_names,
        "collection_name": collection_name or "",
        "answer": final.get("answer", "No answer returned."),
        "critic_verdict": final.get("critic_verdict", "unknown"),
        "critic_reason": final.get("critic_reason", ""),
        "retrieval_attempts": final.get("retrieval_attempts", 0),
        "critic_attempts": final.get("critic_attempts", 0),
        "final_verdict": final.get("final_verdict", ""),
        "from_cache": final.get("from_cache", False),
        "skipped_files": final.get("skipped_files", []),
        "executed_steps": executed,
        "evaluation_contexts": [d.page_content for d in docs],
        "source_chunks": source_chunks,
        "sources": sources,
        "elapsed_total": round(time.perf_counter() - started, 3),
    }


def stream_query(
    question: str,
    collection_name: str = "default",
    files: Optional[list[tuple[str, bytes]]] = None,
) -> Iterator[dict]:
    """Ingest any supplied (filename, bytes) files into the collection, then stream the answer.

    Yields event dicts: start / step / token / reset / final / error.
    """
    question = (question or "").strip()
    if not question:
        yield {"type": "error", "detail": "Question cannot be empty."}
        return

    workdir = None
    try:
        workdir, paths, names = save_uploads(files or [])
        ingestion = ingest_files_into_collection(collection_name, paths, names)
        for event in _run_pipeline([], names, question, collection_name=collection_name):
            if event.get("type") == "final":
                event["collection_ingestion"] = ingestion
            yield event
    except ValueError as exc:
        yield {"type": "error", "detail": str(exc)}
    except Exception as exc:
        yield {"type": "error", "detail": f"RAG pipeline failed: {type(exc).__name__}: {exc}"}
    finally:
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def stream_evaluation(question: str, answer: str, contexts: list[str]) -> Iterator[dict]:
    """Evaluate an answer; each metric is yielded the moment it finishes.

    DeepEval runs its metrics in worker threads, so results are handed back
    through a queue and yielded from the caller's own thread.
    """
    question = (question or "").strip()
    answer = (answer or "").strip()
    contexts = [c for c in (contexts or []) if isinstance(c, str) and c.strip()]
    if not question or not answer:
        yield {"type": "error", "detail": "Question and answer cannot be empty."}
        return
    if not contexts:
        yield {"type": "error", "detail": "No retrieved context available for evaluation."}
        return

    events: "queue.Queue[Optional[dict]]" = queue.Queue()
    started = time.perf_counter()

    def worker():
        try:
            result = eval_answer(
                {
                    "question": question,
                    "answer": answer,
                    "filtered_docs": [Document(page_content=c) for c in contexts],
                },
                progress_cb=lambda name, res: events.put({
                    "type": "metric",
                    "name": name,
                    "result": res,
                    "t": round(time.perf_counter() - started, 2),
                }),
            )
            events.put({
                "type": "final",
                "evaluation": result["evaluation"],
                "elapsed_total": round(time.perf_counter() - started, 3),
            })
        except Exception as exc:
            events.put({"type": "error", "detail": f"Evaluation failed: {type(exc).__name__}: {exc}"})
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()
    yield {"type": "start", "metrics": EVAL_METRICS}
    while True:
        item = events.get()
        if item is None:
            break
        yield item
