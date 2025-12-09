#!/usr/bin/env python3
"""Debug the knowledge base to see what's stored"""

from pathlib import Path

from deps.knowledge_deps import KnowledgeBase


def debug_knowledge_base():
    """Debug what's in the knowledge base"""

    # Check what's actually in the knowledge base
    kb = KnowledgeBase(db_path=Path("./infinigen_agent_knowledge.db"))

    # Direct query to see all data
    print("🔍 All data in knowledge base:")
    results = kb.conn.execute(
        "SELECT id, scene_type, parameters FROM scene_knowledge"
    ).fetchall()
    print(f"Total records: {len(results)}")
    for r in results:
        print(f"  ID {r[0]}: {r[1]} - {str(r[2])[:100]}...")

    # Test semantic search with lower threshold
    print("\n🔍 Semantic search with lower threshold:")
    results = kb.semantic_search_scenes("astronomy", limit=5, similarity_threshold=0.1)
    print(f"Found {len(results)} results")
    for r in results:
        print(f'  - {r["scene_type"]} (similarity: {r["similarity"]:.3f})')

    # Test different queries
    test_queries = ["documentation", "workflow", "nature", "config"]
    for query in test_queries:
        results = kb.semantic_search_scenes(query, limit=3, similarity_threshold=0.1)
        print(f'Query "{query}": {len(results)} results')

    kb.close()


if __name__ == "__main__":
    debug_knowledge_base()
