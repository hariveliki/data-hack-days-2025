from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import uuid
from rank_bm25 import BM25Okapi
import numpy as np

# --- Configuration ---
QDRANT_HOST = "http://localhost:6333"
COLLECTION_NAME = "job_search"
TEXT_VECTOR_NAME = "text_dense"  # Name for our dense vector in Qdrant
EMBEDDING_MODEL_NAME = (
    "tomaarsen/mpnet-base-nli-matryoshka"  # Use a real Matryoshka model
)
MATRYOSHKA_64 = "matryoshka-64dim"
MATRYOSHKA_128 = "matryoshka-128dim"
MATRYOSHKA_256 = "matryoshka-256dim"
SPARSE_VECTOR_NAME = "text_sparse"
SPARSE_VECTOR_SIZE = 635

# --- Sample Data ---
sample_documents = [
    "# Business Analysis & Project Management Officer 60–100% (f/m/d)At Julius Baer, we celebrate and value the individual qualities you bring, enabling you to be impactful, to be entrepreneurial, to be empowered, and to create value beyond wealth. Let's shape the future of wealth management together.We are looking for a highly motivated and experienced Business Analysis and Project Management Officer (PMO), who is passionate about building applications in wealth management. The focus will be on a Sales Tool integration Project, to streamline our system landscape and create added value for relationship managers by reducing the number of applications in use. You will join a dynamic and experienced agile team of dedicated and passionate specialists and work with inspiring colleagues from several functions within the bank.## Your Challenge- Business Analysis and Project Management Officer (PMO) for MIS Sales Forecasting and Pipeline Management Tools including related Financials.- Acting as finance business partner between the business and IT developers.- Coordinate various stakeholders to collect requirements and translate them into technical requirements for SAP solutions.- Develop and execute test cases to ensure delivery quality.- Prepare business analysis and all project documentation for business workshops, STC meetings, budget monitoring, and change management activities.- Deputize the Product Owner when needed.- Collaborate with cross-functional teams.## Your Profile- Minimum 2 years of experience in a business analyst role in the financial service industry, ideally in wealth management.- Master's degree in Economics, Banking, Computer Science, Data Science, or a related field, ideally with banking experience in Wealth Management.- SAP certification (SAP BW and SAP Hana) is an asset in this role.- Knowledge of Python, SQL, R, or other analytical programs/software is a plus.- Excellent problem-solving skills with strong attention to detail.- Entrepreneurial mindset, coupled with the willingness to take responsibility.- Independent, resilient, and team-oriented personality.- Strong communication and interpersonal skills.- Fluent in English; German is a plus.",
    "Introduction  *Career and master's degree? At IBM, you can combine a promising work environment for the start of your career with a part-time master's course at a renowned university of your choice!* IBM is one of the world's largest providers of information technology solutions and business consultancy. Our vision is to bring a new level of ,smart' to how the world works. A place where people, businesses, governments and systems are interconnected and work together. A place where billions of people will live and work better.  *What does Master@IBM offer you?* Within the scope of the 2-3 year Master@IBM program, we offer valuable professional experience in the area of a sales department, which you will gain in parallel to your studies. You can complete the master study course at a university (or university of applied sciences) of your choice. The duration of studies depends on the respective university and is between two and three years. The volume of work allows you to focus sufficiently on your master's degree. Nevertheless, you can expect challenging assignments in a team-oriented and dynamic environment as well as many opportunities for active co-determination. You will also write your master's thesis in collaboration with your department.  *Who are we looking for?* For our in-job master's program, we are looking for highly-motivated students from Information Technology or Management courses. If you like to work in a team, work solution oriented and have good analytical skills and last but not least love to be challenged and learn something new every day then we are looking forward to receiving your application.  Your role and responsibilities  *What can you expect from this position?* In this role, you'll support the Data Platform sales organization to position IBM's AI solution portfolio (e.g. watsonx) in the Swiss market. You'll work closely with sales and technical sales organizations to define, execute and monitor software product related Go-To-Market (GTM) initiatives. You'll also help to plan and run external facing events promoting IBM's AI portfolio and related capabilities.  *Key Responsibilities:* • Work with Sales Management and sales/technical functions to plan, execute and monitor GTM initiatives for IBM's AI solutions • Plan and run external facing events (both IBM and 3rd party) • Support Sales Managers in operational tasks  *What you bring:* • Bachelor's degree in Business Informatics, Computer Science, or a related field • Deep passion for technology and innovative software solutions related to artificial intelligence and data science • Self-motivated and independent, with a knack for solving problems and embracing new challenges • Excellent communication, teamwork, and customer-oriented skills • Passionate about new technologies in the field of Data & AI and how they can provide business value to clients • Comes with a sales mindset and is keen to learn how to build compelling value propositions for clients  Required education Bachelor's Degree Required technical and professional expertise  Fluent in German & English (spoken and written), French is an advantage   ABOUT BUSINESS UNIT  IBM has a global presence, operating in more than 175 countries with a broad-based geographic distribution of revenue. The company's Global Markets organization is a strategic sales business unit that manages IBM's global footprint, working closely with dedicated country-based operating units to serve clients locally. These country teams have client relationship managers who lead integrated teams of consultants, solution specialists and delivery professionals to enable clients' growth and innovation. By complementing local expertise with global experience and digital capabilities, IBM builds deep and broad-based client relationships. This local management focus fosters speed in supporting clients, addressing new markets and making investments in emerging opportunities. Additionally, the Global Markets organization serves clients with expertise in their industry as well as through the products and services that IBM and partners supply. IBM is also expanding its reach to new and existing clients through digital marketplaces.   YOUR LIFE @ IBM  In a world where technology never stands still, we understand that, dedication to our clients success, innovation that matters, and trust and personal responsibility in all our relationships, lives in what we do as IBMers as we strive to be the catalyst that makes the world work better.  Being an IBMer means you'll be able to learn and develop yourself and your career, you'll be encouraged to be courageous and experiment everyday, all whilst having continuous trust and support in an environment where everyone can thrive whatever their personal or professional background.   Our IBMers are growth minded, always staying curious, open to feedback and learning new information and skills to constantly transform themselves and our company. They are trusted to provide on-going feedback to help other IBMers grow, as well as collaborate with colleagues keeping in mind a team focused approach to include different perspectives to drive exceptional outcomes for our customers. The courage our IBMers have to make critical decisions everyday is essential to IBM becoming the catalyst for progress, always embracing challenges with resources they have to hand, a can-do attitude and always striving for an outcome focused approach within everything that they do.   Are you ready to be an IBMer?   ABOUT IBM  IBM's greatest invention is the IBMer. We believe that through the application of intelligence, reason and science, we can improve business, society and the human condition, bringing the power of an open hybrid cloud and AI strategy to life for our clients and partners around the world.   Restlessly reinventing since 1911, we are not only one of the largest corporate organizations in the world, we're also one of the biggest technology and consulting employers, with many of the Fortune 50 companies relying on the IBM Cloud to run their business.   At IBM, we pride ourselves on being an early adopter of artificial intelligence, quantum computing and blockchain. Now it's time for you to join us on our journey to being a responsible technology innovator and a force for good in the world.  IBM is proud to be an equal-opportunity employer. All qualified applicants will receive consideration for employment without regard to race, color, religion, sex, gender, gender identity or expression, sexual orientation, national origin, caste, genetics, pregnancy, disability, neurodivergence, age, veteran status, or other characteristics. IBM is also committed to compliance with all fair employment practices regarding citizenship and immigration status.",
]
search_query = "# **Skills**  ## **Software Development**  - Python, JavaScript, SQL - Microservices, Data Pipelines - Cloud Computing (GCP) - OOP, TDD, DDD, DevOps, CI/CD - Docker, Kubernetes, Git, Pytorch  ## **Project Management**  - Agile, SCRUM, Conventional  ## **Data Science**  - Classification models, chatbots, RAG systems  ## **Business Analysis, Logistics**  - Analysing, designing and implementing business processes  # **Experience**  ## **GLOBUS**  ### **Software Engineer**  **01/2021 – 08/2024 | Zurich, Switzerland**  - Developed data pipelines and microservices in Google Cloud using Docker and Kubernetes. - Fine-tuned large language models for product data classification. - Analyzed and optimized data structures and algorithms to reduce computation time. - Collaborated in agile teams using SCRUM. - Developed software with OOP, TDD, DDD, and CI/CD methodologies.  ### **SCRUM Master / Data Analyst / Application Manager**  **12/2019 – 10/2021 | Zurich, Switzerland**  - Acted as SCRUM Master in a digital transformation project, focusing on restructuring procurement processes and enhancing logistics collaboration. - Facilitated three teams by moderating meetings, delivering weekly presentations, and ensuring adherence to Agile frameworks. - Served as Data Analyst, managing EDI traffic and automating ERP processes. - Held the role of Application Manager in a subsequent project to implement a new Product Information Management (PIM) system.  ## **Coordinator Operations E-Logistics**  **03/2018 – 12/2019 | Otelfingen, Switzerland**  - Led a team of ~10 people and was responsible for the returns department of the online store.  ## **Jakob Müller**  ### **Logistics**  **2015 – 2017 | Frick, Switzerland**  - Experience in warehouse optimization projects.  ## **ABB Turbocharging**  ### **Logistics**  **2011 – 2014 | Baden, Switzerland**  - Training in various departments such as Shipping, Quality Management, Production Planning and Incoming Goods.  # **Qualifications**  ## **BSc Data Science**  **University of Applied Sciences FHNW** 2021 – 2025 | Brugg/Windisch, Switzerland  ## **BSc Computer Science (Exchange)**  **TU Dublin** 01/2024 – 07/2024 | Dublin, Ireland  ## **Industrial Engineering**  **ABB Technical School** 2016 – 2019 | Baden, Switzerland  ## **Certificate of Proficiency in Logistics**  **Professional School Aarau** 2011 – 2014 | Aarau, Switzerland"
key_words = ["Master"]


def get_embedding_model(model_name: str):
    """Initializes and returns the sentence transformer model."""
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Embedding model loaded.")
    return model


def build_bm25_index(documents):
    """Builds a BM25 index and returns the BM25 object and tokenized corpus."""
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


def get_vocabulary(tokenized_corpus, max_vocab_size=SPARSE_VECTOR_SIZE):
    """Builds a vocabulary from the tokenized corpus, limited to max_vocab_size."""
    from collections import Counter

    all_tokens = [token for doc in tokenized_corpus for token in doc]
    most_common = Counter(all_tokens).most_common(max_vocab_size)
    vocabulary = [token for token, _ in most_common]
    return vocabulary


def setup_collection(client: QdrantClient, model: SentenceTransformer):
    """Creates or recreates the Qdrant collection with dense, matryoshka, and sparse vector configuration."""
    vector_size = model.get_sentence_embedding_dimension()
    print(f"Vector size for '{TEXT_VECTOR_NAME}': {vector_size}")
    print(f"Matryoshka vector sizes: 64, 128, 256")
    print(f"Sparse vector size: {SPARSE_VECTOR_SIZE}")

    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists. Recreating it.")
        client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' does not exist. Creating it.")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            TEXT_VECTOR_NAME: models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
            MATRYOSHKA_64: models.VectorParams(
                size=64, distance=models.Distance.COSINE
            ),
            MATRYOSHKA_128: models.VectorParams(
                size=128, distance=models.Distance.COSINE
            ),
            MATRYOSHKA_256: models.VectorParams(
                size=256, distance=models.Distance.COSINE
            ),
            SPARSE_VECTOR_NAME: models.VectorParams(
                size=SPARSE_VECTOR_SIZE, distance=models.Distance.DOT
            ),
        },
    )
    print(f"Collection '{COLLECTION_NAME}' created/recreated successfully.")


def index_documents(
    client: QdrantClient,
    model: SentenceTransformer,
    documents: list[str],
    bm25: BM25Okapi,
    vocabulary: list[str],
):
    """Generates dense, matryoshka, and sparse vectors for documents and upserts them into Qdrant."""
    print(f"Indexing {len(documents)} documents...")
    dense_embeddings = model.encode(documents)
    tokenized_corpus = [doc.lower().split() for doc in documents]
    points_to_upsert = []
    for i, doc_text in enumerate(documents):
        dense_emb = dense_embeddings[i].tolist()
        matryoshka_64 = dense_emb[:64]
        matryoshka_128 = dense_emb[:128]
        matryoshka_256 = dense_emb[:256]
        # Sparse vector: BM25 scores for the vocabulary
        doc_tokens = tokenized_corpus[i]
        sparse_vec = np.zeros(len(vocabulary))
        for idx, token in enumerate(vocabulary):
            if token in doc_tokens:
                sparse_vec[idx] = bm25.idf[token]  # Use idf as weight
        points_to_upsert.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    TEXT_VECTOR_NAME: dense_emb,
                    MATRYOSHKA_64: matryoshka_64,
                    MATRYOSHKA_128: matryoshka_128,
                    MATRYOSHKA_256: matryoshka_256,
                    SPARSE_VECTOR_NAME: sparse_vec.tolist(),
                },
                payload={"text": doc_text, "source": "sample_data"},
            )
        )
        if (i + 1) % 100 == 0:
            print(f"  Prepared {i+1} points for indexing...")
    if points_to_upsert:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_to_upsert,
            wait=True,
        )
    print(f"Successfully indexed {len(points_to_upsert)} documents.")


def matryoshka_dense_sparse_pipeline_search(
    client: QdrantClient,
    model: SentenceTransformer,
    bm25: BM25Okapi,
    vocabulary: list[str],
    query_text: str,
    key_words: list[str],
    top_k: int = 10,
):
    """Implements the pipeline as described in the Qdrant Hybrid Search documentation: matryoshka branch, dense branch, and sparse branch fused."""
    query_embedding = model.encode(query_text)
    matryoshka_query_64 = query_embedding[:64].tolist()
    matryoshka_query_128 = query_embedding[:128].tolist()
    matryoshka_query_256 = query_embedding[:256].tolist()
    dense_query = query_embedding

    # Matryoshka branch (multi-stage)
    matryoshka_prefetch = models.Prefetch(
        prefetch=[
            models.Prefetch(
                prefetch=[
                    models.Prefetch(
                        query=matryoshka_query_64,
                        using=MATRYOSHKA_64,
                        limit=100,
                    ),
                ],
                query=matryoshka_query_128,
                using=MATRYOSHKA_128,
                limit=50,
            )
        ],
        query=matryoshka_query_256,
        using=MATRYOSHKA_256,
        limit=top_k,
    )

    # Dense branch (single-stage)
    dense_prefetch = models.Prefetch(
        query=dense_query.tolist(),
        using=TEXT_VECTOR_NAME,
        limit=top_k,
    )

    # Sparse branch (BM25)
    query_tokens = [kw.lower() for kw in key_words]
    sparse_query = np.zeros(len(vocabulary))
    for idx, token in enumerate(vocabulary):
        if token in query_tokens:
            sparse_query[idx] = bm25.idf[token]
    sparse_prefetch = models.Prefetch(
        query=sparse_query.tolist(),
        using=SPARSE_VECTOR_NAME,
        limit=top_k,
    )

    # Fusion of all branches (RRF)
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[matryoshka_prefetch, dense_prefetch, sparse_prefetch],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    print("Matryoshka + Dense + Sparse Hybrid Search Results (RRF):")
    if not search_result:
        print("  No results found.")
        return
    for point in search_result.points:
        print(f"Point ID: {point.id}")
        print(f"Score: {point.score}")
        print(f"Payload: {point.payload['text'][:100]}")
        print("-" * 100)


def main():
    """Main function to run the hybrid search demo with BM25 sparse search."""
    client = QdrantClient(QDRANT_HOST)
    print(f"Qdrant client initialized (connecting to {QDRANT_HOST}).")
    embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)
    bm25, tokenized_corpus = build_bm25_index(sample_documents)
    vocabulary = get_vocabulary(tokenized_corpus)
    setup_collection(client, embedding_model)
    index_documents(client, embedding_model, sample_documents, bm25, vocabulary)
    print("\n--- Matryoshka + Dense + Sparse Hybrid Search (RRF) ---")
    matryoshka_dense_sparse_pipeline_search(
        client, embedding_model, bm25, vocabulary, search_query, key_words
    )


if __name__ == "__main__":
    main()
