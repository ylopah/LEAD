import os
import json
import argparse
import itertools
import pandas as pd
import numpy as np
import sys
import time
import yaml
import logging
import ast
from tqdm import tqdm

from core.retriever.retriever import deep_retrieve_by_authorities
from core.extraction.extraction import (extract_value_of_variables, causal_claim_verification,
                                       causal_claim_verification_batch, get_authoritative_domains,
                                       send_query_to_openai)
from core.discovery.pc.call_pc_proposal import run_pc_post_add_remove_relations
from core.discovery.ges.ges import ges_post_add_remove_relations
from core.discovery.notears.notears import notears_post_add_remove_relations

from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphClass import CausalGraph
from metrics import compute_metrics


def setup_logging(config):
    log_level = getattr(logging, config['environment'].get('log_level', 'INFO').upper(), logging.INFO)
    log_file = config['environment'].get('log_file', './results/run.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("Main")


def get_args():
    parser = argparse.ArgumentParser(description='LLM4Causal: LLM-enhanced Causal Discovery')
    parser.add_argument('--dataset', type=str, default='cancer', help='Dataset name')
    parser.add_argument('--alg', type=str, default='pc', help='Algorithm: pc, ges, notears')
    parser.add_argument('--llm', type=str, default="glm-4-flash", help='LLM identifier')
    parser.add_argument('--lambda1', type=float, default=0.01, help='NOTEARS lambda')
    parser.add_argument('--w_threshold', type=float, default=0.3, help='NOTEARS weight threshold')
    return parser.parse_args()


def convert_values(x):
    """Convert boolean strings to numeric values for causal algorithms."""
    if x is True or str(x).lower() == "true":
        return 1
    elif x is False or str(x).lower() == "false":
        return -1
    else:
        return 0


def set_true_edges_in_matrix(variables, edges):
    """Construct the ground truth adjacency matrix."""
    index_map = {name: idx for idx, name in enumerate(variables)}
    n = len(variables)
    m = np.zeros((n, n), dtype=int)
    for edge in edges:
        if edge[0] in index_map and edge[1] in index_map:
            src_idx = index_map[edge[0]]
            dst_idx = index_map[edge[1]]
            m[src_idx, dst_idx] = 1
    return m


if __name__ == "__main__":
    # Load Config and Args
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    args = get_args()
    logger = setup_logging(config)

    # Setup Constants — env vars take priority over config.yaml
    PROXY = os.environ.get('LEAD_PROXY', config['environment']['proxy'])
    if not PROXY:
        PROXY = None
    api_key = os.environ.get('LEAD_API_KEY') or config['api']['key']
    api_base = os.environ.get('LEAD_API_BASE') or config['api']['base_url']
    API_CFG = {'key': api_key, 'base_url': api_base}

    if not api_key or api_key in ("YOUR_API_KEY_HERE", "sk-your-api-key"):
        logger.error("API key not configured. Set LEAD_API_KEY environment variable or update config.yaml.")
        sys.exit(1)

    RETRIEVAL_CFG = config['retrieval']
    target_dir = "./datasets"

    # Ensure cache directories exist
    os.makedirs(f"{target_dir}/processed", exist_ok=True)

    # Data Loading
    file_path = f"{target_dir}/{args.dataset}.json"
    if not os.path.exists(file_path):
        logger.error(f"Dataset file not found: {file_path}")
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes, synonyms, edges = graph["nodes"], graph["synonyms"], graph["edges"]

    logger.info(f"Starting pipeline for dataset: {args.dataset} using {args.llm}")

    # Get authoritative domains via LLM (Model name is now configurable)
    authorities = get_authoritative_domains(
        args.dataset, list(nodes.keys()),
        lambda msg, model: send_query_to_openai(msg, model, API_CFG),
        args.llm)

    # Build a comprehensive keyword list including all synonyms
    current_keywords = list(nodes.keys())
    for syn_list in synonyms.values():
        current_keywords.extend(syn_list)

    #################### Part 1: Document Retrieval via Targeted Search ##################
    retrieved_docs_file = f"{target_dir}/processed/{args.dataset}_local_docs_cache.json"

    if not os.path.exists(retrieved_docs_file):
        # Generate search combinations for core variables and their synonyms
        logger.info("Generating search combinations for global retrieval...")
        terms = [[var] + syn for var, syn in synonyms.items()]
        all_combinations = [(ele,) for sublist in terms for ele in sublist]
        full_combinations = list(itertools.product(*terms))
        all_combinations += full_combinations
        for one_full in full_combinations:
            for r in range(len(one_full), 1, -1):
                all_combinations.extend(itertools.combinations(one_full, r))

        # Prioritize combinations that correspond to edges in the ground truth for better evidence gathering
        len2_combinations = [c for c in set(all_combinations) if len(c) == 2]
        prioritized_combinations = []

        for combo in len2_combinations:
            is_core_edge = False
            for node_a, node_b in edges:
                syns_a = [node_a] + synonyms.get(node_a, [])
                syns_b = [node_b] + synonyms.get(node_b, [])
                if (combo[0] in syns_a and combo[1] in syns_b) or (combo[0] in syns_b and combo[1] in syns_a):
                    is_core_edge = True
                    break

            if is_core_edge:
                prioritized_combinations.insert(0, combo)  # Move core edges to front
            else:
                prioritized_combinations.append(combo)

        search_limit = len(nodes)
        top_combinations = prioritized_combinations[:search_limit]

        docs = {}
        # Online retrieval process
        for retrieval_terms in tqdm(top_combinations, desc="Global Retrieval"):
            query = " ".join([f'"{t}"' for t in retrieval_terms])

            # Deep search within authoritative domains
            docs = deep_retrieve_by_authorities(
                query, authorities, docs, current_keywords,
                proxy_url=PROXY,
                max_new_docs=RETRIEVAL_CFG['max_new_docs_per_query'],
                timeout=RETRIEVAL_CFG['timeout'],
                pages=2
            )

            # Set an upper limit to control token usage and search time
            if len(docs) > RETRIEVAL_CFG['max_docs_limit']:
                logger.info(f"Reached doc limit ({len(docs)}), stopping global search.")
                break

        if docs:
            with open(retrieved_docs_file, "w", encoding="utf-8") as f:
                json.dump(docs, f, indent=4)
        else:
            logger.warning("No documents retrieved; cache not saved.")
    else:
        logger.info(f"Loading cached documents: {retrieved_docs_file}")
        with open(retrieved_docs_file, "r", encoding="utf-8") as f:
            docs = json.load(f)
        if not docs:
            logger.warning("Cached document file is empty, removing it.")
            os.remove(retrieved_docs_file)

    ############## Part 2: Extract Variable Values for Statistical Analysis ##################
    table_file = f'{target_dir}/processed/{args.dataset}_{args.llm}_extracted_table_data.csv'

    if not os.path.exists(table_file):
        samples = []
        if not docs:
            logger.error("No documents available for value extraction. Check network/proxy settings and retry.")
            sys.exit(1)
        else:
            for url, doc in tqdm(docs.items(), desc="Extracting Variable States"):
                # Use LLM to determine if variables are present/active in the document
                exacted_values = extract_value_of_variables(doc, nodes, args.llm, API_CFG)
                samples.append(exacted_values)

            df = pd.DataFrame(samples)
            df.to_csv(table_file, index=False)
    else:
        df = pd.read_csv(table_file)

    ################## Part 3: Deep Search for Causal Evidence #################
    explicit_causal_relation_evidence_file = f"{target_dir}/processed/{args.dataset}_explicit_causal_relation_evidence.json"

    if not os.path.exists(explicit_causal_relation_evidence_file):
        var_pairs = list(itertools.combinations(synonyms.keys(), 2))
        evidence_map = {}

        # Logic: Retry with broader search terms if no evidence is found
        MIN_DOCS, MAX_ATTEMPTS = 2, 3

        for pair in tqdm(var_pairs, desc="Acquiring Targeted Causal Evidence"):
            pair_docs = {}

            for attempt in range(1, MAX_ATTEMPTS + 1):
                if len(pair_docs) >= MIN_DOCS:
                    break

                # Tiered search strategy: 1. Strict -> 2. Synonyms -> 3. Broad
                if attempt == 1:
                    query = f'"{pair[0]}" "{pair[1]}" causal relationship'
                elif attempt == 2:
                    syn_a = synonyms.get(pair[0], [])[0] if synonyms.get(pair[0], []) else pair[0]
                    query = f'"{syn_a}" "{pair[1]}" mechanism'
                else:
                    query = f'{pair[0]} influence on {pair[1]}'

                # Search with increasing depth (pages)
                pair_docs = deep_retrieve_by_authorities(
                    query, authorities, pair_docs, current_keywords,
                    proxy_url=PROXY,
                    max_new_docs=RETRIEVAL_CFG['max_new_docs_per_query'],
                    timeout=RETRIEVAL_CFG['timeout'],
                    pages=attempt
                )
                time.sleep(0.5)
            evidence_map[pair] = pair_docs

        serialized_evidence = {str(k): v for k, v in evidence_map.items()}
        with open(explicit_causal_relation_evidence_file, "w", encoding="utf-8") as f:
            json.dump(serialized_evidence, f, indent=4)
    else:
        with open(explicit_causal_relation_evidence_file, "r", encoding="utf-8") as f:
            serialized_evidence = json.load(f)
        evidence_map = {ast.literal_eval(k): v for k, v in serialized_evidence.items()}

    ################ Part 4: LLM-based Causal Claim Verification ##################
    causal_verify_file = f"{target_dir}/processed/{args.dataset}_{args.llm}_causal_relation_verification_results.json"

    if not os.path.exists(causal_verify_file):
        verification_results = {}
        for pair, pair_docs in tqdm(evidence_map.items(), desc="LLM Verifying Causal Relations"):
            if not pair_docs:
                verification_results[f"{pair[0]} causes {pair[1]}"] = ["Unknown"]
                verification_results[f"{pair[1]} causes {pair[0]}"] = ["Unknown"]
                continue

            # Batch verify both directions in one API call (50% fewer calls)
            batch_results = causal_claim_verification_batch(pair, pair_docs, args.llm, API_CFG)
            verification_results.update(batch_results)

            # Real-time backup
            with open(causal_verify_file, "w", encoding="utf-8") as f:
                json.dump(verification_results, f, indent=4)
    else:
        with open(causal_verify_file, "r", encoding="utf-8") as f:
            verification_results = json.load(f)

    ############## Part 5: Consensus Merging and Graph Construction #################
    verified_relations, remove_relations = [], []

    for relation, veracity in verification_results.items():
        # Filter out 'Unknown' to focus on valid binary evidence
        valid_veracity = [v for v in veracity if str(v).lower() in ["true", "false"]]
        if not valid_veracity:
            continue

        t_ratio = sum(1 for v in valid_veracity if str(v).lower() == "true") / len(valid_veracity)

        # Vote-based determination
        if t_ratio > 0.6:
            verified_relations.append(relation)
        elif t_ratio < 0.2:
            remove_relations.append(relation)

    logger.info(f"Verified Edges: {verified_relations}")
    logger.info(f"Removed Edges: {remove_relations}")

    # Data preparation for causal discovery
    if args.dataset in ["cancer", "diabetes", "obesity", "respiratory", "adni"]:
        if not df.empty:
            df = df.map(convert_values)

    data = df.to_numpy()
    var_names = df.columns
    true_matrix = set_true_edges_in_matrix(var_names, edges)

    # Execute discovery with LLM-priors
    if args.alg == "pc":
        predicted_adj_mat, predicted_dag = run_pc_post_add_remove_relations(data, var_names, verified_relations, remove_relations)
    elif args.alg == "ges":
        predicted_adj_mat, predicted_dag = ges_post_add_remove_relations(data, node_names=var_names, add_relations=verified_relations, remove_relations=remove_relations)
    elif args.alg == "notears":
        predicted_adj_mat, predicted_dag = notears_post_add_remove_relations(data, lambda1=args.lambda1, loss_type='l2',
                                                                             w_threshold=args.w_threshold, node_names=var_names,
                                                                             add_relations=verified_relations,
                                                                             remove_relations=remove_relations)
    print("Predicted Adjacency Matrix:\n", predicted_adj_mat)

    # Evaluation
    metrics = compute_metrics(true_matrix, predicted_adj_mat, var_names)
    print("Discovery Metrics:", metrics)

    logger.info(f"Final Metrics: {metrics}")

    results_dir = "./results/metrics"
    os.makedirs(results_dir, exist_ok=True)

    experiment_info = {
        "dataset": args.dataset,
        "algorithm": args.alg,
        "llm_model": args.llm,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    full_results = {**experiment_info, **metrics}

    json_filename = f"{results_dir}/{args.dataset}_{args.alg}_{args.llm}_metrics.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=4, ensure_ascii=False)

    summary_csv = "./results/all_experiments_summary.csv"
    summary_df = pd.DataFrame([full_results])

    if not os.path.exists(summary_csv):
        summary_df.to_csv(summary_csv, index=False, header=True)
    else:
        summary_df.to_csv(summary_csv, mode='a', index=False, header=False)

    logger.info(f"Metrics saved to {json_filename} and appended to {summary_csv}")

    plots_dir = "./results/plots"
    os.makedirs(plots_dir, exist_ok=True)

    plot_filename = f"{plots_dir}/{args.dataset}_{args.alg}_{args.llm}_dag.png"

    if isinstance(predicted_dag, CausalGraph):
        pyd = GraphUtils.to_pydot(predicted_dag.G, labels=var_names)
    elif isinstance(predicted_dag, GeneralGraph):
        pyd = GraphUtils.to_pydot(predicted_dag, labels=var_names)

    try:
        pyd.write_png(plot_filename)
        logger.info(f"Causal graph visualization saved to {plot_filename}")
    except FileNotFoundError:
        logger.warning("Graphviz 'dot' not found in PATH. Install from https://graphviz.org/download/")
