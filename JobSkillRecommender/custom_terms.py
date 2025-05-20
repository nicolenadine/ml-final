"""
Custom Terms for Job-Resume Matching
-----------------------------------
This file contains a comprehensive dictionary of technical terms
organized by category for use in the job-resume matching system.
"""

# Dictionary of terms by category
custom_terms = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "r", "go", "golang", "ruby",
        "php", "scala", "kotlin", "swift", "objective-c", "perl", "bash", "shell", "powershell",
        "rust", "dart", "matlab", "fortran", "cobol", "assembly", "lisp", "clojure", "haskell",
        "groovy", "lua", "julia", "solidity", "vba", "apex", "abap", "pl/sql", "t-sql"
    ],

    "web_development": [
        "html", "css", "html5", "css3", "javascript", "dom", "jquery", "bootstrap", "sass", "less",
        "react", "react.js", "angular", "angular.js", "vue", "vue.js", "svelte", "backbone.js", "ember.js",
        "next.js", "gatsby", "node.js", "express", "express.js", "django", "flask", "fastapi", "spring",
        "spring boot", "asp.net", "ruby on rails", "laravel", "symfony", "php", "wordpress",
        "drupal", "magento", "shopify", "woocommerce", "webflow", "responsive design",
        "progressive web app", "pwa", "web components", "web apis", "websockets", "graphql",
        "rest api", "restful", "json", "xml", "ajax", "spa", "ssr", "web accessibility", "wai-aria"
    ],

    "mobile_development": [
        "android", "ios", "swift", "objective-c", "kotlin", "java", "react native", "flutter",
        "xamarin", "ionic", "cordova", "phonegap", "android studio", "xcode", "mobile app",
        "mobile development", "responsive design", "progressive web app", "pwa", "app store",
        "google play", "push notifications", "mobile ui", "ui/ux", "gestures", "touch interface"
    ],

    "databases": [
        "sql", "nosql", "mysql", "postgresql", "oracle", "sql server", "sqlite", "mongodb",
        "cassandra", "redis", "dynamodb", "couchdb", "neo4j", "graph database", "firebase",
        "mariadb", "snowflake", "teradata", "data warehouse", "data lake", "olap", "oltp",
        "stored procedure", "trigger", "index", "query optimization", "database design",
        "database schema", "er diagram", "database normalization", "database administration",
        "dba", "database migration", "etl", "rdbms", "acid", "transactions", "database replication",
        "database sharding", "database partitioning", "data modeling"
    ],

    "data_science": [
        "machine learning", "ml", "deep learning", "neural networks", "data mining",
        "statistical analysis", "predictive modeling", "regression", "classification",
        "clustering", "natural language processing", "nlp", "computer vision",
        "recommendation systems", "reinforcement learning", "supervised learning",
        "unsupervised learning", "semi-supervised learning", "time series analysis",
        "anomaly detection", "feature engineering", "feature selection", "dimensionality reduction",
        "pca", "svd", "t-sne", "umap", "data wrangling", "data cleansing", "data preprocessing",
        "data visualization", "exploratory data analysis", "eda", "a/b testing", "hypothesis testing",
        "statistical significance", "correlation", "causation", "data science", "data scientist",
        "sentiment analysis", "text analytics", "predictive analytics", "prescriptive analytics",
        "descriptive analytics", "analytics", "anova", "chi-square", "probability", "bayesian",
        "inference", "confidence interval", "p-value", "big data", "data pipeline", "etl"
    ],

    "machine_learning_tools": [
        "scikit-learn", "sklearn", "tensorflow", "pytorch", "keras", "theano", "mxnet",
        "caffe", "cntk", "xgboost", "lightgbm", "catboost", "h2o", "weka", "spark mllib",
        "hugging face", "transformers", "gensim", "spacy", "nltk", "openai", "gpt", "bert",
        "word2vec", "glove", "fasttext", "opencv", "pillow", "statsmodels", "prophet",
        "arima", "sarima", "automl", "hyperparameter tuning", "grid search", "random search",
        "bayesian optimization", "mlops", "model deployment", "model serving", "model monitoring",
        "ml pipeline", "experiment tracking", "mlflow", "kubeflow", "autoencoder", "gan",
        "cnn", "rnn", "lstm", "transformer", "attention mechanism", "transfer learning"
    ],

    "data_engineering": [
        "etl", "data pipeline", "data integration", "data migration", "data modeling",
        "data architecture", "data warehouse", "data lake", "data lakehouse", "data mart",
        "database design", "big data", "hadoop", "spark", "apache spark", "spark streaming",
        "kafka", "apache kafka", "airflow", "apache airflow", "luigi", "nifi", "apache nifi",
        "flume", "sqoop", "hive", "pig", "presto", "impala", "zookeeper", "hdfs", "mapreduce",
        "yarn", "hbase", "elasticsearch", "kibana", "logstash", "elk stack", "data ingestion",
        "data processing", "batch processing", "stream processing", "real-time processing",
        "data orchestration", "data governance", "data quality", "data lineage", "data catalog",
        "metadata management", "dbt", "great expectations", "prefect", "dagster", "fivetran",
        "stitch", "informatica", "talend", "aws glue", "azure data factory"
    ],

    "cloud_platforms": [
        "aws", "amazon web services", "azure", "microsoft azure", "gcp", "google cloud platform",
        "alibaba cloud", "oracle cloud", "ibm cloud", "cloud computing", "multi-cloud", "hybrid cloud",
        "cloud migration", "cloud architecture", "serverless", "cloud native", "cloud security",
        "iac", "infrastructure as code", "saas", "paas", "iaas", "faas", "cloud storage",
        "cloud database", "cloud deployment", "ec2", "s3", "lambda", "rds", "dynamodb",
        "ebs", "vpc", "iam", "route53", "cloudfront", "elasticache", "sqs", "sns", "ecs",
        "eks", "fargate", "azure vm", "azure sql", "azure cosmos db", "azure functions",
        "blob storage", "gce", "gcs", "bigquery", "cloud functions", "app engine", "cloud run",
        "firebase", "kubernetes", "heroku", "digital ocean", "linode", "openstack", "cloudflare"
    ],

    "devops_and_infrastructure": [
        "devops", "ci/cd", "continuous integration", "continuous deployment", "continuous delivery",
        "git", "github", "gitlab", "bitbucket", "jenkins", "travis ci", "circle ci", "github actions",
        "azure devops", "terraform", "ansible", "puppet", "chef", "docker", "containerization",
        "kubernetes", "k8s", "helm", "istio", "service mesh", "prometheus", "grafana", "datadog",
        "new relic", "splunk", "elk stack", "logging", "monitoring", "alerting", "observability",
        "infrastructure as code", "iac", "configuration management", "infrastructure monitoring",
        "application monitoring", "log management", "security monitoring", "vulnerability scanning",
        "infrastructure automation", "cloud automation", "network automation", "deployment automation",
        "security automation", "nginx", "apache", "load balancing", "auto scaling", "high availability",
        "fault tolerance", "disaster recovery", "backup and restore", "shell scripting", "linux", "unix"
    ],

    "software_engineering_concepts": [
        "software development", "software engineering", "software architecture", "systems design",
        "distributed systems", "microservices", "service oriented architecture", "soa",
        "monolithic architecture", "event driven architecture", "domain driven design", "ddd",
        "test driven development", "tdd", "behavior driven development", "bdd", "agile",
        "scrum", "kanban", "waterfall", "lean", "extreme programming", "xp", "pair programming",
        "code review", "code quality", "clean code", "refactoring", "technical debt", "version control",
        "git", "software testing", "unit testing", "integration testing", "functional testing",
        "end to end testing", "performance testing", "load testing", "stress testing", "regression testing",
        "automated testing", "manual testing", "qa", "quality assurance", "quality engineering",
        "debugging", "troubleshooting", "problem solving", "algorithms", "data structures",
        "object oriented programming", "oop", "functional programming", "parallel programming",
        "concurrent programming", "asynchronous programming", "api design", "rest", "graphql",
        "soap", "rpc", "grpc", "design patterns", "solid principles", "dry principle",
        "code documentation", "technical documentation", "uml", "deployment strategies"
    ],

    "data_analysis_and_visualization": [
        "data analysis", "data analytics", "business analytics", "business intelligence", "bi",
        "data visualization", "dashboard", "metrics", "kpi", "reporting", "data storytelling",
        "excel", "google sheets", "power bi", "microsoft power bi", "tableau", "looker",
        "qlik", "qlik sense", "qlikview", "domo", "metabase", "redash", "superset", "mode",
        "thoughtspot", "sisense", "matplotlib", "seaborn", "plotly", "ggplot2", "d3.js",
        "highcharts", "chart.js", "data studio", "google data studio", "power query", "dax",
        "sql", "pivot table", "olap", "bi tools", "data lake", "data warehouse", "data mart",
        "etl", "pandas", "numpy", "quantitative analysis", "qualitative analysis", "descriptive statistics",
        "inferential statistics", "data interpretation", "trend analysis", "forecasting",
        "market analysis", "sales analysis", "customer analytics", "web analytics", "google analytics",
        "adobe analytics", "marketing analytics", "financial analysis", "data modeling"
    ],

    "ai_and_emerging_tech": [
        "artificial intelligence", "ai", "generative ai", "gen ai", "large language models", "llm",
        "chatgpt", "gpt", "bert", "transformers", "neural networks", "deep learning", "nlp",
        "computer vision", "robotics", "autonomous systems", "recommendation systems",
        "reinforcement learning", "ai ethics", "responsible ai", "explainable ai", "xai",
        "mlops", "model deployment", "ai governance", "ai strategy", "prompt engineering",
        "rag", "retrieval augmented generation", "zero shot learning", "few shot learning",
        "transfer learning", "federated learning", "edge ai", "embedded ai", "ai inference",
        "ai training", "stable diffusion", "dall-e", "midjourney", "image generation",
        "text to image", "text to speech", "speech to text", "natural language generation",
        "natural language understanding", "sentiment analysis", "knowledge graphs", "ontology",
        "semantic search", "vector database", "embedding", "similarity search", "ai assistant",
        "conversational ai", "chatbot", "virtual assistant", "augmented reality", "ar",
        "virtual reality", "vr", "mixed reality", "xr", "blockchain", "smart contracts",
        "quantum computing", "internet of things", "iot", "edge computing", "5g"
    ],

    "soft_skills_and_qualities": [
        "problem solving", "analytical thinking", "critical thinking", "communication skills",
        "teamwork", "collaboration", "leadership", "project management", "time management",
        "organization", "attention to detail", "creativity", "innovation", "adaptability",
        "flexibility", "continuous learning", "self motivated", "independent", "proactive",
        "initiative", "curiosity", "passion", "domain knowledge", "business acumen",
        "customer focused", "result oriented", "agile methodology", "scrum", "kanban",
        "experience with", "proficiency in", "knowledge of", "familiarity with", "expertise in",
        "proven track record", "strong understanding", "hands on experience", "cross functional",
        "strategic thinking", "decision making", "fast paced environment", "deadline driven",
        "multitasking", "prioritization", "technical writing", "documentation", "mentoring"
    ],

    "security_and_compliance": [
        "cybersecurity", "information security", "network security", "application security",
        "cloud security", "security engineering", "security architecture", "data protection",
        "data privacy", "gdpr", "ccpa", "hipaa", "pci dss", "sox", "compliance", "risk management",
        "risk assessment", "vulnerability assessment", "penetration testing", "pen testing",
        "ethical hacking", "security audit", "security monitoring", "siem", "encryption",
        "authentication", "authorization", "identity management", "iam", "sso", "mfa",
        "zero trust", "firewall", "intrusion detection", "intrusion prevention", "ids", "ips",
        "dlp", "data loss prevention", "security protocols", "tls", "ssl", "vpn", "devsecops",
        "secure coding", "code scanning", "sast", "dast", "security testing", "security compliance",
        "security certifications", "iso 27001", "nist", "cis", "security operations", "secops",
        "security incident", "incident response", "threat intelligence", "threat modeling"
    ]
}

# Function to get all terms as a flat list
def get_all_terms():
    """Return all terms as a flat list without duplicates"""
    all_terms = []
    for category, terms in custom_terms.items():
        all_terms.extend(terms)

    # Remove duplicates while preserving order
    all_terms_unique = list(dict.fromkeys(all_terms))

    return all_terms_unique

# If this file is run directly, print some statistics
if __name__ == "__main__":
    all_terms = get_all_terms()
    print(f"Total number of unique terms: {len(all_terms)}")
    print(f"Number of categories: {len(custom_terms)}")
    for category, terms in custom_terms.items():
        print(f"  - {category}: {len(terms)} terms")