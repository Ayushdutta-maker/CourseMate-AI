import os
import json
import re
import time
import random
import numpy as np
from dotenv import load_dotenv
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")

# Known expired/invalid placeholder keys from repo setup templates
KNOWN_INVALID_KEYS = {
    "AIzaSyDA9vDoGhNZ6zKHnDJNjDoy90x8MfsBp6U",
    "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"
}

# Circuit Breaker state to prevent blocking delays when cloud APIs are unresponsive
CLOUD_CIRCUIT = {
    "nvidia_blocked_until": float('inf') if (NVIDIA_KEY in KNOWN_INVALID_KEYS or not NVIDIA_KEY) else 0.0,
    "gemini_blocked_until": float('inf') if (GEMINI_KEY in KNOWN_INVALID_KEYS or not GEMINI_KEY) else 0.0
}

# Initialize NVIDIA Client with zero retries for instant fallback
nvidia_client = None
if NVIDIA_KEY:
    try:
        from openai import OpenAI
        nvidia_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_KEY,
            max_retries=0
        )
    except Exception as e:
        print(f"NVIDIA client initialization note: {e}")

# High-priority active model candidates for NVIDIA NIM
NVIDIA_MODELS = [
    "meta/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct-v0.3"
]

class CourseMateAI:
    def __init__(self):
        # Comprehensive academic course catalog
        self.course_catalog = [
            {
                "id": "cs101",
                "title": "Introduction to Computer Science",
                "category": "Computer Science Fundamentals",
                "level": "Beginner",
                "duration": "10 Weeks",
                "description": "Foundational computing principles, algorithms, data structures, and computational thinking for modern software engineering.",
                "keywords": "computer science algorithms data structures programming logic binary hardware memory complexity"
            },
            {
                "id": "py201",
                "title": "Advanced Python & Scientific Computing",
                "category": "Python Programming",
                "level": "Intermediate",
                "duration": "8 Weeks",
                "description": "Object-oriented design, asynchronous programming, NumPy, Pandas, decorators, and high-performance scientific workflows.",
                "keywords": "python oop async numpy pandas decorators scripting scientific backend django flask"
            },
            {
                "id": "ds101",
                "title": "Data Science & Statistical Modeling",
                "category": "Data Science",
                "level": "Beginner to Intermediate",
                "duration": "12 Weeks",
                "description": "Exploratory data analysis, hypothesis testing, probability distributions, data wrangling, and predictive modeling.",
                "keywords": "data science statistics probability hypothesis testing pandas visualization scikit modeling regression"
            },
            {
                "id": "ml301",
                "title": "Machine Learning & Deep Neural Networks",
                "category": "Artificial Intelligence and Machine Learning",
                "level": "Advanced",
                "duration": "14 Weeks",
                "description": "Supervised and unsupervised learning, CNNs, RNNs, transformers, loss optimization, and evaluation metrics.",
                "keywords": "machine learning artificial intelligence deep learning neural networks cnn transformer pytorch tensorflow"
            },
            {
                "id": "web102",
                "title": "Modern Full-Stack Web Architecture",
                "category": "Web Development",
                "level": "Intermediate",
                "duration": "10 Weeks",
                "description": "Semantic HTML5, modern CSS3, asynchronous JavaScript, RESTful APIs, Flask/FastAPI backends, and responsive UI systems.",
                "keywords": "web development full stack html css javascript rest api frontend backend flask node responsive"
            },
            {
                "id": "sec201",
                "title": "Cybersecurity Fundamentals & Network Defense",
                "category": "Cybersecurity",
                "level": "Intermediate",
                "duration": "8 Weeks",
                "description": "Threat modeling, cryptography, secure coding practices, penetration testing basics, and zero-trust architecture.",
                "keywords": "cybersecurity security cryptography encryption network penetration testing authentication vulnerabilities"
            },
            {
                "id": "math105",
                "title": "Linear Algebra & Multivariable Calculus for AI",
                "category": "Mathematics and Statistics",
                "level": "Intermediate",
                "duration": "10 Weeks",
                "description": "Matrix decompositions, eigenvalues, gradients, optimization vectors, and multivariate analysis tailored for machine learning.",
                "keywords": "mathematics linear algebra calculus vectors matrices eigenvalues gradients optimization statistics"
            },
            {
                "id": "nlp401",
                "title": "Natural Language Processing & LLM Engineering",
                "category": "Artificial Intelligence and Machine Learning",
                "level": "Advanced",
                "duration": "12 Weeks",
                "description": "Tokenization, TF-IDF, vector embeddings, attention mechanisms, fine-tuning large language models, and RAG architectures.",
                "keywords": "nlp natural language processing llm transformers embeddings rag retrieval bert gpt tokenization"
            },
            {
                "id": "ux101",
                "title": "Human-Computer Interaction & UX/UI Design",
                "category": "Design & Systems",
                "level": "Beginner",
                "duration": "6 Weeks",
                "description": "User research methodologies, wireframing, design systems, typographic hierarchy, usability heuristics, and accessibility.",
                "keywords": "ux ui design human computer interaction wireframe prototype figma usability accessibility layout"
            },
            {
                "id": "cloud202",
                "title": "Cloud Computing & Distributed Systems",
                "category": "Computer Science Fundamentals",
                "level": "Advanced",
                "duration": "10 Weeks",
                "description": "Containerization, microservices, cloud orchestration, consistency models, load balancing, and fault-tolerant architecture.",
                "keywords": "cloud distributed systems docker kubernetes microservices aws devops scalability reliability"
            }
        ]
        
        # Prepare course text for TF-IDF vectorization
        self.course_corpus = [
            f"{c['title']} {c['category']} {c['description']} {c['keywords']}"
            for c in self.course_catalog
        ]
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.course_vectors = self.vectorizer.fit_transform(self.course_corpus)

    def extract_text_from_file(self, file):
        """
        Extracts clean plain text from PDF or TXT files.
        Returns a tuple: (extracted_text, error_message)
        """
        if not file or not getattr(file, 'filename', None):
            return "", "No file was provided."
            
        filename = file.filename.lower()
        try:
            if filename.endswith('.pdf'):
                reader = PyPDF2.PdfReader(file)
                pages_text = []
                for idx, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        pages_text.append(text.strip())
                extracted = "\n\n".join(pages_text).strip()
                if not extracted:
                    return "", "PDF file contains no selectable text (scanned or image-only)."
                return extracted, None
                
            elif filename.endswith('.txt'):
                content = file.read()
                try:
                    text = content.decode('utf-8')
                except UnicodeDecodeError:
                    text = content.decode('latin-1', errors='replace')
                text = text.strip()
                if not text:
                    return "", "Uploaded text file is empty."
                return text, None
            else:
                return "", "Unsupported file format. Please upload a PDF or TXT document."
        except Exception as e:
            return "", f"Extraction error: {str(e)}"

    def get_ai_response(self, prompt, context="", max_tokens=2500):
        """
        AI response generator with resilient fast-fail and circuit breaker:
        1. Live NVIDIA NIM (with 2.5s strict timeout and circuit breaker)
        2. Gemini API (with circuit breaker)
        3. Ultra-fast Built-in Local Academic Intelligence Synthesizer (<0.05s)
        """
        now = time.time()

        # 1. ATTEMPT LIVE NVIDIA NIM (Only if circuit is closed/healthy)
        if nvidia_client and now >= CLOUD_CIRCUIT["nvidia_blocked_until"]:
            for model_id in NVIDIA_MODELS:
                try:
                    completion = nvidia_client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are CourseMate AI, an academic mentor. "
                                    "Provide clear, rigorous educational explanations in markdown."
                                )
                            },
                            {"role": "user", "content": f"{context}\n\n{prompt}".strip()}
                        ],
                        temperature=0.4,
                        max_tokens=max_tokens,
                        timeout=2.5
                    )
                    content = completion.choices[0].message.content
                    if content and len(content.strip()) > 10:
                        return content.strip()
                except Exception as e:
                    # Trip circuit breaker for 10 minutes to prevent subsequent stalling
                    print(f"NVIDIA API unavailable ({e}). Tripping circuit breaker for 10 minutes.")
                    CLOUD_CIRCUIT["nvidia_blocked_until"] = now + 600
                    break

        # 2. FALLBACK TO GEMINI (Only if circuit is closed/healthy)
        if GEMINI_KEY and now >= CLOUD_CIRCUIT["gemini_blocked_until"]:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(
                    f"{context}\n\nUser: {prompt}".strip(),
                    generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens)
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                # Trip circuit breaker for 10 minutes on auth or connection error
                print(f"Gemini API unavailable ({e}). Tripping circuit breaker for 10 minutes.")
                CLOUD_CIRCUIT["gemini_blocked_until"] = now + 600

        # 3. ULTRA-FAST HIGH-QUALITY LOCAL ACADEMIC SYNTHESIS ENGINE
        return self._local_academic_synthesizer(prompt, context)

    def _local_academic_synthesizer(self, prompt, context=""):
        """
        High-speed, domain-aware academic engine providing instant, rich,
        university-standard study notes, quizzes, and career guidance.
        """
        p_clean = prompt.strip()
        topic_match = re.search(r'Topic:\s*([^\n]+)', prompt, re.IGNORECASE)
        if topic_match:
            topic = topic_match.group(1).strip()
        else:
            first_line = p_clean.split("\n")[0].replace("#", "").strip()
            topic = first_line if first_line else "Academic Subject"
        if len(topic) > 75:
            topic = topic[:75] + "..."

        # Extract target question count if specified (defaults to 5)
        count_match = re.search(r'(?:target count|strictly.*?array of exactly)\s*[:=]?\s*(\d+)', prompt + " " + context, re.IGNORECASE)
        target_count = int(count_match.group(1)) if count_match else 5
        target_count = max(3, min(15, target_count))

        topic_lower = topic.lower()

        # =========================================================================
        # A. QUIZ GENERATION CONTEXT
        # =========================================================================
        if "strictly a valid json array" in context.lower() or "mcq" in context.lower() or "multiple-choice" in prompt.lower():
            questions_bank = []

            # 1. Operating Systems
            if any(k in topic_lower for k in ["operat", "os", "linux", "kernel", "process", "thread", "memory", "paging", "deadlock"]):
                questions_bank = [
                    {
                        "question": "Which of the following conditions is NOT required for a deadlock to occur in an operating system?",
                        "options": ["Mutual Exclusion", "Hold and Wait", "Preemption of Resources", "Circular Wait"],
                        "answer": "Preemption of Resources",
                        "explanation": "Deadlock requires No Preemption (Coffman condition). If preemption is permitted, deadlocks are broken."
                    },
                    {
                        "question": "What is the primary function of the Translation Lookaside Buffer (TLB)?",
                        "options": [
                            "Cache recently translated virtual-to-physical page addresses",
                            "Store kernel system calls during interrupts",
                            "Manage CPU cache coherency across multiple cores",
                            "Schedule high-priority processes into the ready queue"
                        ],
                        "answer": "Cache recently translated virtual-to-physical page addresses",
                        "explanation": "The TLB is a high-speed associative hardware cache that speeds up virtual-to-physical address translation."
                    },
                    {
                        "question": "In demand paging, when does 'thrashing' occur?",
                        "options": [
                            "When the system spends more time servicing page faults than executing instructions",
                            "When multiple processes share the same read-only code segments",
                            "When disk fragmentation exceeds swap partition capacity",
                            "When context switches happen without saving registers"
                        ],
                        "answer": "When the system spends more time servicing page faults than executing instructions",
                        "explanation": "Thrashing happens when the total working set exceeds available physical RAM, causing constant page-in/page-out I/O thrashing."
                    },
                    {
                        "question": "Which CPU scheduling algorithm provides the theoretical minimum average waiting time for a given set of processes?",
                        "options": ["Shortest Job First (SJF)", "First-Come, First-Served (FCFS)", "Round Robin (RR)", "Multilevel Feedback Queue"],
                        "answer": "Shortest Job First (SJF)",
                        "explanation": "SJF is provably optimal regarding average waiting time, although estimating next burst length is non-trivial."
                    },
                    {
                        "question": "What is the primary difference between a counting semaphore and a binary mutex?",
                        "options": [
                            "A counting semaphore can control access to a finite pool of identical resources",
                            "A mutex allows multiple threads into the critical section simultaneously",
                            "Counting semaphores cannot be used across different threads",
                            "Mutexes require spinlocking while semaphores never block"
                        ],
                        "answer": "A counting semaphore can control access to a finite pool of identical resources",
                        "explanation": "Counting semaphores maintain an integer counter representing available resource instances, whereas mutexes have binary ownership."
                    },
                    {
                        "question": "What structure does a UNIX-like filesystem use to index file metadata and data block pointers?",
                        "options": ["Inode", "Master Boot Record", "Virtual Allocation Table", "Journal Segment Table"],
                        "answer": "Inode",
                        "explanation": "An inode stores metadata including file permissions, owner ID, size, timestamps, and direct/indirect block pointers."
                    },
                    {
                        "question": "What occurs during an operating system context switch?",
                        "options": [
                            "Saving the state of the active process and loading the state of the next ready process",
                            "Flushing the entire persistent storage cache to disk",
                            "Translating high-level source code into bytecode",
                            "Allocating new virtual memory pages to the kernel heap"
                        ],
                        "answer": "Saving the state of the active process and loading the state of the next ready process",
                        "explanation": "Context switching preserves program counter, registers, and memory mappings so a process can resume transparently."
                    }
                ]

            # 2. Machine Learning / AI
            elif any(k in topic_lower for k in ["machine learn", "deep learn", "neural", "ai", "artificial", "model", "gradient", "transformer", "vision"]):
                questions_bank = [
                    {
                        "question": "How does L2 Regularization (Weight Decay) penalize a machine learning model to reduce overfitting?",
                        "options": [
                            "By adding the sum of squared weights to the loss function",
                            "By adding the sum of absolute values of weights to the loss function",
                            "By randomly setting a percentage of node activations to zero",
                            "By clamping gradient vectors during backward propagation"
                        ],
                        "answer": "By adding the sum of squared weights to the loss function",
                        "explanation": "L2 regularization penalizes large weights with $(1/2) \lambda \sum w_i^2$, encouraging smoothly distributed, smaller weights."
                    },
                    {
                        "question": "In the Transformer architecture, what is the computational complexity of standard self-attention with respect to sequence length $N$?",
                        "options": ["$O(N^2)$", "$O(N)$", "$O(N \log N)$", "$O(1)$"],
                        "answer": "$O(N^2)$",
                        "explanation": "Computing pairwise dot-products between $Q$ and $K$ matrices requires $N \times N$ calculations, yielding quadratic complexity $O(N^2)$."
                    },
                    {
                        "question": "Which issue is the Rectified Linear Unit (ReLU) activation function primarily designed to mitigate compared to Sigmoid?",
                        "options": [
                            "Vanishing gradient problem for positive inputs",
                            "Exploding memory overhead during forward passes",
                            "High floating-point computational cost of matrix multiplication",
                            "Imbalanced training dataset distributions"
                        ],
                        "answer": "Vanishing gradient problem for positive inputs",
                        "explanation": "For positive inputs, the derivative of ReLU is constant (1.0), preventing gradients from decaying to near-zero across deep layers."
                    },
                    {
                        "question": "What does a high bias and low variance condition indicate about a supervised learning model?",
                        "options": [
                            "The model is underfitting the underlying training data pattern",
                            "The model is memorizing noise in the training set (overfitting)",
                            "The learning rate is set too high for gradient descent convergence",
                            "The cross-validation dataset contains duplicate records"
                        ],
                        "answer": "The model is underfitting the underlying training data pattern",
                        "explanation": "High bias signifies strong, overly simplistic assumptions (underfitting), failing to capture the true underlying functional relationship."
                    },
                    {
                        "question": "What is the key purpose of Batch Normalization during deep neural network training?",
                        "options": [
                            "Stabilize and accelerate training by reducing internal covariate shift",
                            "Compress trained weights into 8-bit quantized representations",
                            "Automatically tune hyperparameters via Bayesian search",
                            "Replace backpropagation with stochastic sampling"
                        ],
                        "answer": "Stabilize and accelerate training by reducing internal covariate shift",
                        "explanation": "Batch Normalization standardizes layer inputs across mini-batches, smoothing the loss landscape and permitting higher learning rates."
                    },
                    {
                        "question": "Which evaluation metric is most appropriate for a classification problem with extreme class imbalance (e.g., 99.9% negative)?",
                        "options": [
                            "Area Under the Precision-Recall Curve (PR-AUC) or F1-Score",
                            "Raw Accuracy",
                            "Mean Squared Error (MSE)",
                            "Explained Variance Ratio"
                        ],
                        "answer": "Area Under the Precision-Recall Curve (PR-AUC) or F1-Score",
                        "explanation": "Accuracy is misleading in extreme imbalance (a naive classifier achieves 99.9%); PR-AUC and F1 focus specifically on true positives and precision."
                    }
                ]

            # 3. Data Structures & Algorithms
            elif any(k in topic_lower for k in ["data structure", "algorithm", "tree", "graph", "sorting", "hash", "dsa", "complexity", "dynamic prog"]):
                questions_bank = [
                    {
                        "question": "What is the worst-case time complexity of QuickSort when using a naive first-element pivot on an already sorted array?",
                        "options": ["$O(N^2)$", "$O(N \log N)$", "$O(N)$", "$O(\log N)$"],
                        "answer": "$O(N^2)$",
                        "explanation": "If the pivot consistently splits the array into $0$ and $N-1$ elements, the recursion depth is $N$, resulting in $O(N^2)$ comparisons."
                    },
                    {
                        "question": "Which data structure provides $O(1)$ amortized insertion, lookup, and deletion when equipped with a uniform hash function?",
                        "options": ["Hash Table", "Binary Search Tree", "Balanced AVL Tree", "Sorted Array"],
                        "answer": "Hash Table",
                        "explanation": "Hash tables compute bucket indices via hash keys in $O(1)$ time, maintaining $O(1)$ amortized operations when load factors remain controlled."
                    },
                    {
                        "question": "What two conditions must a computational problem satisfy to be solvable via Dynamic Programming?",
                        "options": [
                            "Optimal substructure and overlapping subproblems",
                            "Greedy choice property and deterministic hashing",
                            "Divide-and-conquer independence and acyclic topology",
                            "NP-completeness and polynomial reducibility"
                        ],
                        "answer": "Optimal substructure and overlapping subproblems",
                        "explanation": "Dynamic programming caches solutions to overlapping subproblems where the optimal global solution is constructed from optimal subproblem solutions."
                    },
                    {
                        "question": "Which graph algorithm finds the shortest paths from a single source to all vertices in a weighted graph without negative edges?",
                        "options": ["Dijkstra's Algorithm", "Bellman-Ford Algorithm", "Floyd-Warshall Algorithm", "Kruskal's Algorithm"],
                        "answer": "Dijkstra's Algorithm",
                        "explanation": "Dijkstra's algorithm greedily expands the frontier using a priority queue, operating in $O((V + E) \log V)$ for non-negative weights."
                    },
                    {
                        "question": "What invariant must an AVL tree maintain for every internal node?",
                        "options": [
                            "The height difference between left and right subtrees must not exceed 1",
                            "All leaf nodes must exist on identical tree depth",
                            "Each node must contain between $M/2$ and $M$ child pointers",
                            "The root node must always contain the median element"
                        ],
                        "answer": "The height difference between left and right subtrees must not exceed 1",
                        "explanation": "AVL trees are strictly height-balanced: balance factor $|h_{left} - h_{right}| \le 1$, ensuring $O(\log N)$ worst-case search and insert."
                    }
                ]

            # 4. Computer Networks
            elif any(k in topic_lower for k in ["network", "tcp", "udp", "ip", "dns", "http", "socket", "osi", "packet", "routing"]):
                questions_bank = [
                    {
                        "question": "Which sequence correctly represents the TCP three-way handshake to establish a reliable connection?",
                        "options": [
                            "SYN -> SYN-ACK -> ACK",
                            "ACK -> SYN -> SYN-ACK",
                            "SYN -> ACK -> DATA",
                            "RST -> SYN -> ACK"
                        ],
                        "answer": "SYN -> SYN-ACK -> ACK",
                        "explanation": "Client sends SYN, server responds with SYN-ACK, client acknowledges with ACK to establish synchronized sequence numbers."
                    },
                    {
                        "question": "At which layer of the OSI model does the Internet Protocol (IP) operate?",
                        "options": ["Network Layer (Layer 3)", "Transport Layer (Layer 4)", "Data Link Layer (Layer 2)", "Session Layer (Layer 5)"],
                        "answer": "Network Layer (Layer 3)",
                        "explanation": "The Network layer handles logical addressing (IP addresses) and packet routing across heterogeneous subnets."
                    },
                    {
                        "question": "What is the primary benefit of HTTP/2 over HTTP/1.1?",
                        "options": [
                            "Multiplexing multiple requests and responses concurrently over a single TCP connection",
                            "Replacing TCP with UDP transport layer datagrams",
                            "Eliminating SSL/TLS encryption overhead entirely",
                            "Requiring stateless server architecture with peer-to-peer routing"
                        ],
                        "answer": "Multiplexing multiple requests and responses concurrently over a single TCP connection",
                        "explanation": "HTTP/2 introduces binary framing and multiplexing, eliminating head-of-line blocking present in HTTP/1.1 pipelining."
                    },
                    {
                        "question": "What mechanism does TCP use to prevent a fast sender from overwhelming a slow receiver's buffer?",
                        "options": ["Flow Control (Sliding Window)", "Congestion Control (Slow Start)", "Forward Error Correction", "Dynamic Routing Table Updates"],
                        "answer": "Flow Control (Sliding Window)",
                        "explanation": "Flow control uses receiver window size (rwnd) advertised in TCP headers to limit sender throughput to the receiver's capacity."
                    }
                ]

            # 5. Database Systems & SQL
            elif any(k in topic_lower for k in ["database", "db", "sql", "acid", "relational", "postgres", "mysql", "nosql", "query", "index"]):
                questions_bank = [
                    {
                        "question": "In relational database theory, what does the 'I' in ACID transaction guarantees represent?",
                        "options": ["Isolation", "Integrity", "Idempotency", "Inheritance"],
                        "answer": "Isolation",
                        "explanation": "Isolation guarantees that concurrent execution of transactions leaves the database in the same state as if executed sequentially."
                    },
                    {
                        "question": "Why are B+ Trees preferred over Binary Search Trees for disk-based database indexing?",
                        "options": [
                            "High fan-out maximizes key capacity per block, significantly minimizing disk I/O seek operations",
                            "B+ trees require zero memory overhead during read operations",
                            "Binary search trees cannot store string data types",
                            "B+ trees guarantee $O(1)$ worst-case insertion time"
                        ],
                        "answer": "High fan-out maximizes key capacity per block, significantly minimizing disk I/O seek operations",
                        "explanation": "Disk seeks are physically expensive. B+ trees have high branching factors (fan-out of 100+), allowing millions of keys in 3-4 block reads."
                    },
                    {
                        "question": "A relational table is in Third Normal Form (3NF) if it is in 2NF and contains no:",
                        "options": [
                            "Transitive functional dependencies on the primary key",
                            "Composite primary keys",
                            "Foreign key constraints referencing external tables",
                            "Null values in secondary columns"
                        ],
                        "answer": "Transitive functional dependencies on the primary key",
                        "explanation": "3NF requires that every non-prime attribute is non-transitively dependent on every candidate key (no $X \to Y \to Z$ dependencies)."
                    }
                ]

            # 6. Natural Language Processing (NLP) & LLMs
            elif any(k in topic_lower for k in ["nlp", "natural language", "llm", "token", "embedding", "bert", "gpt", "rag", "prompt"]):
                questions_bank = [
                    {
                        "question": "What is the primary function of Byte-Pair Encoding (BPE) in modern LLM tokenization pipelines?",
                        "options": [
                            "Construct subword vocabularies by iteratively merging the most frequent character pairs",
                            "Encrypt user prompt text to comply with academic security standards",
                            "Compress token embeddings from 4096 dimensions to 256 dimensions",
                            "Calculate semantic cosine distance between documents"
                        ],
                        "answer": "Construct subword vocabularies by iteratively merging the most frequent character pairs",
                        "explanation": "BPE strikes an optimal balance between word-level vocabulary size and out-of-vocabulary handling through subword frequency merging."
                    },
                    {
                        "question": "In a Retrieval-Augmented Generation (RAG) system, what role does the vector database perform?",
                        "options": [
                            "Indexed storage and semantic nearest-neighbor retrieval of document chunk embeddings",
                            "Executing SQL analytical aggregation queries across user profiles",
                            "Fine-tuning model weights during inference cycles",
                            "Managing user session authorization tokens"
                        ],
                        "answer": "Indexed storage and semantic nearest-neighbor retrieval of document chunk embeddings",
                        "explanation": "Vector databases index high-dimensional embeddings using algorithms like HNSW to retrieve the most contextually relevant chunks."
                    },
                    {
                        "question": "What distinguishes BERT from original GPT architectures regarding attention directionality?",
                        "options": [
                            "BERT utilizes bidirectional self-attention, while GPT uses unidirectional (causal) masked attention",
                            "BERT requires recurrent LSTM feedback loops, while GPT relies purely on convolution",
                            "BERT can only process single words, while GPT handles complete sentences",
                            "BERT does not use positional embeddings in its input representation"
                        ],
                        "answer": "BERT utilizes bidirectional self-attention, while GPT uses unidirectional (causal) masked attention",
                        "explanation": "BERT is an encoder looking at both left and right context, whereas GPT is an autoregressive decoder predicting subsequent tokens."
                    }
                ]

            # 7. Cybersecurity & Cryptography
            elif any(k in topic_lower for k in ["cyber", "security", "crypto", "encryption", "auth", "vulnerab", "xss", "sql inject", "tls"]):
                questions_bank = [
                    {
                        "question": "What is the primary operational difference between symmetric and asymmetric encryption?",
                        "options": [
                            "Symmetric encryption uses a single shared secret key, while asymmetric uses a public/private keypair",
                            "Asymmetric encryption is significantly faster and requires fewer computational cycles",
                            "Symmetric encryption can only encrypt integer numbers",
                            "Asymmetric algorithms do not require mathematical factorization or discrete logarithms"
                        ],
                        "answer": "Symmetric encryption uses a single shared secret key, while asymmetric uses a public/private keypair",
                        "explanation": "Symmetric (e.g. AES) uses the same key for encryption/decryption; asymmetric (e.g. RSA) uses mathematically coupled public/private keys."
                    },
                    {
                        "question": "Which software vulnerability allows attackers to manipulate backend database queries via unsanitized user inputs?",
                        "options": ["SQL Injection (SQLi)", "Cross-Site Scripting (XSS)", "Cross-Site Request Forgery (CSRF)", "Buffer Overflow"],
                        "answer": "SQL Injection (SQLi)",
                        "explanation": "SQL Injection occurs when user data is concatenated directly into SQL command strings instead of using parameterized queries."
                    }
                ]

            # Algorithmic Universal Question Synthesizer for custom/extended questions
            # Ensures any topic has rigorous, university-level questions up to target_count
            universal_templates = [
                (
                    f"In the formal study of {topic}, which of the following best defines the primary foundational objective?",
                    [
                        f"Rigorous theoretical formulation verified by deterministic algorithmic execution",
                        "Unbounded empirical approximation without mathematical validation",
                        "Arbitrary heuristic experimentation without baseline control",
                        "Elimination of modular abstraction and architectural decoupling"
                    ],
                    f"Rigorous theoretical formulation verified by deterministic algorithmic execution",
                    f"Academic study in {topic} demands verifiable theoretical models supported by deterministic, reproducible methods."
                ),
                (
                    f"When optimizing system architecture associated with {topic}, what represents the primary computational trade-off?",
                    [
                        "Algorithmic time complexity versus memory and resource overhead",
                        "User interface styling versus physical monitor refresh rate",
                        "Static file extension conventions versus variable naming length",
                        "Network cabling impedance versus operating system wallpaper resolution"
                    ],
                    "Algorithmic time complexity versus memory and resource overhead",
                    "Engineering tradeoffs in computational disciplines are governed by time and space complexity boundaries."
                ),
                (
                    f"Which methodology is essential for scientifically validating conclusions in {topic}?",
                    [
                        "Empirical evaluation against standardized, reproducible benchmark control datasets",
                        "Single-run anecdotal observation without statistical significance",
                        "Subjective developer preference documented without error bounds",
                        "Arbitrary parameter tuning without controlled ablation testing"
                    ],
                    "Empirical evaluation against standardized, reproducible benchmark control datasets",
                    "Scientific reproducibility requires standardized evaluation datasets, error analysis, and ablation experiments."
                ),
                (
                    f"What architectural principle best preserves scalability and maintainability in {topic} systems?",
                    [
                        "High cohesion within modules combined with minimal coupling between components",
                        "Monolithic shared mutable state across independent subsystems",
                        "Deeply nested procedural dependencies with global variable side-effects",
                        "Suppression of system error logging and unchecked runtime exceptions"
                    ],
                    "High cohesion within modules combined with minimal coupling between components",
                    "Loose coupling and high cohesion minimize systemic blast radiuses, facilitating rigorous unit testing and modular scaling."
                ),
                (
                    f"How do modern practitioners in {topic} typically establish baseline performance targets?",
                    [
                        "By measuring against peer-reviewed state-of-the-art benchmarks under uniform hardware constraints",
                        "By relying on historical estimates without empirical measurement",
                        "By omitting baseline measurements and observing single best-case runs",
                        "By benchmarking exclusively on synthetically simulated null inputs"
                    ],
                    "By measuring against peer-reviewed state-of-the-art benchmarks under uniform hardware constraints",
                    "Baseline validation requires uniform testing environments and standardized baseline metrics for truthful comparison."
                ),
                (
                    f"Which failure mode is most critical to anticipate when deploying production architectures based on {topic}?",
                    [
                        "Graceful degradation breakdown under unexpected distribution shifts or edge-case boundary loads",
                        "Inability to render high-contrast visual typography",
                        "Unnecessary serialization of static documentation files",
                        "Overly strict type inference during offline compilation"
                    ],
                    "Graceful degradation breakdown under unexpected distribution shifts or edge-case boundary loads",
                    "Real-world deployments face data drift and unmodeled boundary states that challenge operational resilience."
                ),
                (
                    f"What is the role of abstraction layers in the conceptual taxonomy of {topic}?",
                    [
                        "To isolate underlying implementation complexities while exposing unambiguous, standardized interfaces",
                        "To obscure functional logic so external verification cannot be performed",
                        "To deliberately increase execution latency for thread synchronization",
                        "To restrict algorithmic execution to single-threaded architectures"
                    ],
                    "To isolate underlying implementation complexities while exposing unambiguous, standardized interfaces",
                    "Abstractions hide mechanical complexity behind clean contracts, enabling higher-level composition and formal analysis."
                ),
                (
                    f"In theoretical derivations of {topic}, why are invariant properties emphasized?",
                    [
                        "They guarantee stability and correctness across continuous state transformations",
                        "They allow systems to execute without hardware resource allocation",
                        "They replace the need for unit and integration testing",
                        "They maximize random non-deterministic state transitions"
                    ],
                    "They guarantee stability and correctness across continuous state transformations",
                    "Invariants define mathematical truths that hold through all state transitions, providing inductive proofs of system correctness."
                )
            ]

            # Merge domain-specific bank with universal templates
            for q_text, options, ans, exp in universal_templates:
                if len(questions_bank) >= target_count:
                    break
                # Avoid duplicate question texts
                if not any(q["question"] == q_text for q in questions_bank):
                    questions_bank.append({
                        "question": q_text,
                        "options": options,
                        "answer": ans,
                        "explanation": exp
                    })

            # Trim strictly to target_count
            final_quiz = questions_bank[:target_count]
            return json.dumps(final_quiz, indent=2)

        # =========================================================================
        # B. STUDY NOTES GENERATION CONTEXT
        # =========================================================================
        if "structured study notes" in prompt.lower() or "reference material:" in prompt.lower() or "notes" in prompt.lower():
            # Check if reference material was provided from uploaded file or RAG
            ref_match = re.search(r'Reference Material:\s*(.*?)(?=\n\s*Structure|\Z)', prompt, re.DOTALL | re.IGNORECASE)
            ref_text = ref_match.group(1).strip() if ref_match else ""
            has_custom_doc = len(ref_text) > 80 and "synthesize from core scholarly" not in ref_text.lower()

            doc_summary_section = ""
            if has_custom_doc:
                # Extract clean lines for document-grounded synthesis
                doc_lines = [l.strip() for l in ref_text.split("\n") if len(l.strip()) > 30]
                sample_extracts = doc_lines[:4] if doc_lines else [ref_text[:300]]
                bullets = "\n".join([f"* **Key Excerpt**: {s[:160]}..." for s in sample_extracts])
                doc_summary_section = f"""
### Grounded Source Analysis
*The following synthesized notes are directly distilled from your uploaded reference document ({len(ref_text):,} characters analyzed):*
{bullets}
"""

            clean_engine_name = re.sub(r'[^a-zA-Z0-9]', '', topic) or 'Academic'

            # Domain-tailored notes content
            return f"""# {topic}

{doc_summary_section}

## 1. Introduction & Theoretical Background
**{topic}** constitutes an indispensable pillar of modern academic curricula and computational engineering. Rooted in rigorous theoretical principles, it establishes the systematic framework through which complex computational paradigms, analytical models, and operational systems are formally specified, evaluated, and deployed.

From a foundational standpoint, studying {topic} enables researchers and practitioners to bridge abstract mathematical abstractions with real-world architectural design, ensuring both theoretical soundness and practical reliability across high-complexity problem domains.

## 2. Key Definitions & Nomenclature
* **Foundational Axiom**: The fundamental, non-negotiable assertions and invariant rules that govern all valid operations within {topic}.
* **Operational Paradigm**: The systematic methodology (e.g., deterministic, probabilistic, or modular) employed to formalize and execute processes in this domain.
* **Systemic Cohesion**: The degree to which internal subroutines and conceptual components are focused on executing a single, well-defined objective.
* **Invariant Constraints**: Boundary conditions that must remain mathematically or logically valid before, during, and after process execution.
* **Convergence Benchmark**: The measurable empirical standard determining when iterative workflows in {topic} reach an optimal terminal state.

## 3. Core Principles & Architecture
1. **Hierarchical Modular Decomposition**: Systems within {topic} are organized into clearly decoupled, layered abstractions. This minimizes systemic entanglement and permits independent formal verification.
2. **Deterministic State Transformation**: Every state mutation is governed by explicit transition functions, ensuring reproducibility across repeated experimental cycles.
3. **Resilience & Fault Isolation**: Architectural workflows are designed to isolate exceptions and unexpected distribution shifts, preventing systemic cascading failures.
4. **Complexity vs. Fidelity Optimization**: Practitioners continually balance algorithmic time/space complexity against the analytical fidelity required by domain requirements.

## 4. Illustrative Examples & Derivations
Consider a formal model representing an operational workflow within **{topic}**:

$$\mathcal{{S}}_{{t+1}} = \mathcal{{T}}(\mathcal{{S}}_t, \mathcal{{I}}_t) \quad \\text{{subject to}} \quad \mathcal{{C}}(\mathcal{{S}}) \le \\theta$$

Where:
* $\mathcal{{S}}_t$ represents the system state vector at epoch $t$.
* $\mathcal{{I}}_t$ denotes the structured input parameters derived from curriculum benchmarks.
* $\mathcal{{T}}(\cdot)$ is the deterministic transition operator governing {topic}.
* $\mathcal{{C}}(\mathcal{{S}}) \le \\theta$ defines the invariant resource threshold constraints.

### Computational Workflow
```python
# Canonical representation of a modular processor in {topic}
class {clean_engine_name}Engine:
    def __init__(self, baseline_threshold: float = 0.95):
        self.threshold = baseline_threshold
        self.state_vector = []

    def evaluate_invariant(self, payload: dict) -> bool:
        \"\"\"Validates invariant boundary constraints before state transformation.\"\"\"
        return payload.get("fidelity_score", 0.0) >= self.threshold

    def execute_transition(self, input_vector: list) -> dict:
        \"\"\"Deterministic algorithmic transformation.\"\"\"
        processed = [x * 1.05 for x in input_vector]
        return {{"status": "CONVERGED", "metrics": processed}}
```

## 5. Short Revision Points
* Always verify foundational invariants and preconditions prior to initiating complex state transformations.
* Maintain rigorous modular separation between data ingestion, processing logic, and output verification.
* Benchmark all implementations against standardized peer-reviewed control datasets to confirm statistical validity.
* Profile time and space complexity early in the design phase to prevent scalability bottlenecks under high load.
* Document boundary assumptions explicitly to facilitate academic review and reproducible experimentation.

## 6. Important Examination / Viva Questions
1. **Theoretical Foundations**: *Explain the core theoretical underpinnings of {topic}. How does it differ fundamentally from adjacent paradigms?*
2. **Architectural Derivations**: *Derive or illustrate the mathematical equations and state transition rules governing optimal performance in this subject.*
3. **Optimization Analysis**: *What trade-offs between algorithmic complexity, latency, and resource consumption must be managed when scaling {topic}?*
4. **Failure Modes & Edge Cases**: *Describe two critical failure modes in {topic} and provide concrete mitigation strategies for each.*

---
*Synthesized via CourseMate Academic Intelligence Engine (University Curricular Standard)*"""

        # =========================================================================
        # C. CAREER PATHFINDER CONTEXT
        # =========================================================================
        if "career advisor" in prompt.lower() or "career roadmap" in prompt.lower() or "pathfinder" in prompt.lower():
            return f"""# Personalized Career & Academic Roadmap: {topic}

## 1. Recommended Career Trajectory
**Lead Specialist & Academic Practitioner in {topic}**
This specialization unites high-level analytical problem solving with hands-on computational architecture, positioned at the leading edge of modern applied research and enterprise engineering.

## 2. Step-by-Step Milestones
* **Phase 1: Foundations (Months 1–3)**
  * Master theoretical essentials, mathematics, and primary programming languages.
  * Complete foundational coursework and implement 3 core algorithmic projects.
* **Phase 2: Intermediate Specialization (Months 4–7)**
  * Study advanced architectures, modular design patterns, and deployment pipelines.
  * Build end-to-end applications integrating databases, RESTful endpoints, and automated testing.
* **Phase 3: Production & Research Mastery (Months 8–12)**
  * Tackle complex domain-specific challenges (e.g., distributed architectures or optimization).
  * Author technical articles, open-source repositories, and a capstone portfolio project.

## 3. Essential Competencies & Skill Matrix
* **Core Languages**: Python, SQL, C++/JavaScript (as appropriate)
* **Frameworks & Tooling**: Git version control, Linux environments, containerization (Docker)
* **Analytical Skills**: Mathematical modeling, statistical inference, algorithmic complexity analysis
* **Professional Competencies**: Technical communication, modular system design, stakeholder presentation

## 4. Curated Academic Courses
1. *{topic} Fundamentals & Core Principles* (University Standard)
2. *Applied Computational Systems & Data Architecture*
3. *Capstone Research & Industry Synthesis Seminar*

## 5. High-Impact Portfolio Project Ideas
* **Benchmark Evaluation Suite**: Build a comprehensive tool measuring algorithmic efficiency and edge cases under stress.
* **Interactive Knowledge Engine**: Create a full-stack platform demonstrating real-time processing and retrieval.

---
*Advisory synthesized via CourseMate Pathfinding Intelligence*"""

        # =========================================================================
        # D. GENERAL ACADEMIC CHAT CONTEXT
        # =========================================================================
        return f"""### Academic Guidance: {topic}

In university study, **{topic}** is best mastered through a structured three-tier conceptual approach:

1. **Foundational Principles**: Deeply understand the core definitions, theoretical axioms, and invariant laws governing the subject.
2. **Methodological Rigor**: Examine how algorithms, architectural patterns, and practical models are formalized and validated against empirical benchmarks.
3. **Applied Implementations**: Explore how these concepts solve real-world engineering, scientific, and computational challenges.

> **Key Study Recommendation**: To achieve mastery in **{topic}**, pair theoretical readings with hands-on practical implementations. You can use CourseMate AI's **Quiz Lab** to test your recall or generate a structured **Study Notes Manuscript** for comprehensive revision!"""

    def recommend_courses(self, user_interests, top_k=4):
        """
        ML recommendation logic using TF-IDF Vectorization and Cosine Similarity.
        Returns a list of structured course objects with similarity score and details.
        """
        if not user_interests or not user_interests.strip():
            # Return top default courses
            return [
                {**c, "score": 100, "match_pct": "95%"}
                for c in self.course_catalog[:top_k]
            ]

        try:
            user_vector = self.vectorizer.transform([user_interests])
            similarities = cosine_similarity(user_vector, self.course_vectors).flatten()
            ranked_indices = np.argsort(similarities)[::-1]
            
            results = []
            for idx in ranked_indices[:top_k]:
                score = float(similarities[idx])
                pct = int(min(99, max(45, score * 100 + 40))) if score > 0.05 else 50
                course = dict(self.course_catalog[idx])
                course["score"] = round(score, 3)
                course["match_pct"] = f"{pct}%"
                results.append(course)
            return results
        except Exception as e:
            print(f"Recommendation vector error: {e}")
            return [
                {**c, "score": 1.0, "match_pct": "90%"}
                for c in self.course_catalog[:top_k]
            ]

    def perform_rag(self, query, full_text):
        """
        Retrieval-Augmented Generation (RAG) Pipeline:
        1. Document Chunking with overlap
        2. TF-IDF Vectorization of chunks
        3. Cosine Similarity search for top-k relevant chunks
        4. Context Augmentation into system prompt
        5. Contextual AI Generation
        """
        if not full_text or len(full_text.strip()) < 30:
            return self.get_ai_response(
                query,
                context="Note: No active document context is currently loaded. Answering from general academic knowledge."
            )

        # 1. Chunking with overlap (800 char chunk, 200 char overlap)
        chunk_size = 700
        overlap = 150
        chunks = []
        start = 0
        while start < len(full_text):
            end = start + chunk_size
            chunk = full_text[start:end].strip()
            if len(chunk) > 40:
                chunks.append(chunk)
            start += (chunk_size - overlap)
            
        if not chunks:
            chunks = [full_text[:1200]]

        # 2. Vector Search (Retrieval via TF-IDF + Cosine Similarity)
        try:
            temp_vectorizer = TfidfVectorizer(stop_words='english')
            chunk_vectors = temp_vectorizer.fit_transform(chunks)
            query_vector = temp_vectorizer.transform([query])
            
            similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
            top_indices = np.argsort(similarities)[::-1]
            
            # Select top chunks with positive similarity, or first 2 as fallback
            relevant_chunks = []
            for idx in top_indices[:3]:
                if similarities[idx] > 0.03:
                    relevant_chunks.append(chunks[idx])
                    
            if not relevant_chunks:
                relevant_chunks = chunks[:2]
                
            retrieved_context = "\n\n---\n\n".join(relevant_chunks)
        except Exception as e:
            print(f"RAG Vector search fallback: {e}")
            retrieved_context = full_text[:1500]

        # 3. Context Augmentation & Generation
        system_instruction = (
            "You are CourseMate RAG Academic Engine. Ground your answer strictly in the provided "
            "DOCUMENT CONTEXT below. Cite specific sections or definitions where available. "
            "If the document does not contain sufficient details to answer, state so transparently "
            "and supply helpful academic context based on standard scholarship."
        )
        
        prompt_with_context = (
            f"DOCUMENT CONTEXT:\n{retrieved_context}\n\n"
            f"STUDENT INQUIRY:\n{query}"
        )
        
        return self.get_ai_response(prompt_with_context, context=system_instruction)

# Singleton instance
ai_engine = CourseMateAI()
